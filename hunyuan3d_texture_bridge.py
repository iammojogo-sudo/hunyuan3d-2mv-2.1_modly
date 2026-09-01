import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

# Register torch's DLL directory so custom_rasterizer_kernel (a prebuilt CUDA
# extension) can find cudart64_12.dll and other native libs at import time.
try:
    _torch_lib = str(Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "torch" / "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import numpy as np
import trimesh
import torch
from PIL import Image

# Shared utilities from the shape bridge
from hunyuan3d_bridge import (
    report, cleanup_cuda,
    setup_paths,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


_BRIDGE_STALL_TIMEOUT = 45
_last_activity = time.time()


def _touch_activity():
    global _last_activity
    _last_activity = time.time()


def _watchdog_loop():
    while True:
        time.sleep(60)
        if time.time() - _last_activity > _BRIDGE_STALL_TIMEOUT * 60:
            print(json.dumps({"type": "error", "message": f"No progress for {_BRIDGE_STALL_TIMEOUT} min — aborting"}), flush=True)
            os._exit(1)


def _start_watchdog():
    t = threading.Thread(target=_watchdog_loop, daemon=True)
    t.start()


def _composite_on_white(img):
    """Return an RGB image with the input composited onto a white background.

    RGBA inputs use their alpha channel. RGB inputs are passed through unless
    they look like they have a dark/solid background, in which case rembg is
    used to extract the foreground first.
    """
    if img.mode == "RGBA":
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.getchannel("A"))
        return white

    # RGB path: try to remove any existing background so the diffusion model
    # always sees the object on white rather than on the original backdrop.
    rgb = img.convert("RGB")
    try:
        from rembg import remove, new_session
        sess = new_session(providers=["CPUExecutionProvider"])
        rgba = remove(rgb, session=sess, bgcolor=[255, 255, 255, 0])
        # If rembg wiped almost everything, fall back to the original image.
        alpha = np.array(rgba.getchannel("A"))
        if alpha.max() < 20 or (alpha > 10).sum() < (rgba.width * rgba.height * 0.01):
            return rgb
        white = Image.new("RGB", rgba.size, (255, 255, 255))
        white.paste(rgba, mask=rgba.getchannel("A"))
        return white
    except Exception as _e:
        return rgb


def _subject_silhouette(img_np, tol=25):
    """Extract a foreground silhouette from an RGB image on a uniform background.

    Flood-fills the background from the 4 image corners (tolerant to mild
    shading), returns a boolean mask that is True where the subject is.
    """
    import cv2 as _cv
    h, w = img_np.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    work = img_np.astype(np.uint8)
    for pt in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        _cv.floodFill(
            work, mask, (int(pt[1]), int(pt[0])), 0,
            loDiff=(tol,) * 3, upDiff=(tol,) * 3,
            flags=8 | _cv.FLOODFILL_MASK_ONLY | (1 << 8),
        )
    return mask[1:-1, 1:-1] == 0  # unfilled region = subject


def _mesh_silhouette_from_pos(pos_img):
    """Derive the mesh silhouette from a position map image.

    Position maps render the background as white (255,255,255); mesh pixels
    carry 3D position colours. Returns a bool mask True where the mesh is.
    """
    arr = np.asarray(pos_img.convert("RGB")).astype(np.int16)
    return (np.abs(arr - 255).max(axis=-1)) > 8


def _silhouette_contour(sil_mask, n=128):
    """Outer contour of a silhouette, resampled to `n` points by arc length."""
    import cv2 as _cv
    sil_u8 = (sil_mask * 255).astype(np.uint8)
    contours, _ = _cv.findContours(sil_u8, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=_cv.contourArea).reshape(-1, 2).astype(np.float64)
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    total = cum[-1]
    if total <= 0:
        return None
    targets = np.linspace(0, total, n, endpoint=False)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(c) - 1)
    frac = np.zeros(n)
    valid = idx < len(c) - 1
    frac[valid] = (targets[valid] - cum[idx[valid]]) / np.maximum(d[np.clip(idx[valid], 0, len(d) - 1)], 1e-9)
    nxt = c[np.clip(idx + 1, 0, len(c) - 1)]
    return c[idx] + frac[:, None] * (nxt - c[idx])


def _deform_view_to_silhouette(ref_img, pos_img, out_size):
    """Warp a reference view so its subject silhouette matches the mesh's.

    Uses a thin-plate spline fit on corresponding arc-length points of the two
    outer contours, then samples the reference image through it. Output is the
    warped subject composited on white, masked to the mesh silhouette. If the
    warp can't be fit, returns the reference resized unchanged (the bake will
    still project it, just without silhouette correction).
    """
    import cv2 as _cv
    from skimage.transform import ThinPlateSplineTransform, warp as _skwarp

    ref = ref_img.convert("RGB").resize((out_size, out_size), Image.LANCZOS)
    ref_np = np.asarray(ref).astype(np.float32)

    mesh_sil = _mesh_silhouette_from_pos(pos_img)
    if (mesh_sil.shape[0], mesh_sil.shape[1]) != (out_size, out_size):
        _u8 = Image.fromarray((mesh_sil * 255).astype(np.uint8)).resize(
            (out_size, out_size), Image.NEAREST)
        mesh_sil = np.asarray(_u8) > 0
    if not mesh_sil.any():
        return ref  # no mesh visible from this camera — nothing to align to

    ref_sil = _subject_silhouette(ref_np.astype(np.uint8))
    if not ref_sil.any():
        return ref

    ref_pts = _silhouette_contour(ref_sil)
    mesh_pts = _silhouette_contour(mesh_sil)
    if ref_pts is None or mesh_pts is None or len(ref_pts) < 16 or len(mesh_pts) < 16:
        return ref

    # tps maps mesh-silhouette points -> reference-silhouette points, so
    # warp(ref, inverse_map=tps) samples the reference at each mesh pixel.
    tps = ThinPlateSplineTransform()
    try:
        tps.estimate(src=mesh_pts, dst=ref_pts)
    except Exception:
        return ref

    warped = _skwarp(ref_np, inverse_map=tps, output_shape=(out_size, out_size),
                     order=1, mode="constant", cval=255.0, clip=True)
    warped = np.clip(warped, 0, 255).astype(np.uint8)
    out = np.full((out_size, out_size, 3), 255, np.uint8)
    out[mesh_sil] = warped[mesh_sil]
    return Image.fromarray(out, "RGB")


def _deform_multiview(cond_views, position_maps, out_size):
    """Deform each reference view to match the mesh's silhouette from the same
    camera angle. cond_views and position_maps are ordered identically."""
    out = []
    for i, (cv_img, pos_img) in enumerate(zip(cond_views, position_maps)):
        _touch_activity()
        try:
            out.append(_deform_view_to_silhouette(cv_img, pos_img, out_size))
        except Exception as _e:
            print(json.dumps({"type": "log", "message": f"[deform] view {i} failed ({_e}), using raw view"}), flush=True)
            out.append(cv_img.convert("RGB").resize((out_size, out_size), Image.LANCZOS))
    return out


def _hybrid_flow_multiview(cond_views, diff_views, position_maps, out_size, flow_res=512):
    """Warp the original full-res references to match the diffusion output.

    The multiview diffusion model produces views with the CORRECT mesh geometry
    but capped at 512px (its training resolution). This helper derives a dense
    optical-flow field (Farneback) between each 512px diffusion view and the
    corresponding reference, then applies it to the full-res reference. The
    result is a full-resolution view with the diffusion model's geometry.

    cond_views[i] is the reference for camera i (reading order == candidate
    camera order). Views past the end of cond_views have no reference, so they
    keep the diffusion output as-is (resized).
    """
    import cv2 as _cv
    out = []
    for i, diff_view in enumerate(diff_views):
        _touch_activity()
        try:
            if i >= len(cond_views):
                out.append(diff_view.convert("RGB").resize((out_size, out_size), Image.LANCZOS))
                continue

            ref = cond_views[i].convert("RGB")
            diff = diff_view.convert("RGB")

            # Downscale both to flow_res for the flow computation.
            ref_low = ref.resize((flow_res, flow_res), Image.LANCZOS)
            diff_low = diff.resize((flow_res, flow_res), Image.LANCZOS)
            ref_g = np.asarray(ref_low.convert("L"), dtype=np.float32)
            diff_g = np.asarray(diff_low.convert("L"), dtype=np.float32)

            # Farneback(prev, next): next(x + flow) ~= prev(x). With
            # prev=diff, next=ref each diff pixel's source lives at ref(x+flow).
            flow = _cv.calcOpticalFlowFarneback(
                diff_g, ref_g, None,
                pyr_scale=0.5, levels=5, winsize=21, iterations=5,
                poly_n=5, poly_sigma=1.2, flags=0)

            scale = out_size / flow_res
            flow_full = _cv.resize(flow, (out_size, out_size),
                                   interpolation=_cv.INTER_LINEAR) * scale

            ref_full = np.asarray(ref.resize((out_size, out_size), Image.LANCZOS), dtype=np.uint8)
            yy, xx = np.meshgrid(np.arange(out_size), np.arange(out_size), indexing="ij")
            map_x = (xx.astype(np.float32) + flow_full[..., 0]).clip(0, out_size - 1)
            map_y = (yy.astype(np.float32) + flow_full[..., 1]).clip(0, out_size - 1)
            warped = _cv.remap(ref_full, map_x, map_y, _cv.INTER_LINEAR,
                               borderMode=_cv.BORDER_CONSTANT, borderValue=(255, 255, 255))

            # Restrict the warped content to the mesh silhouette (position map)
            # so background pixels can't bleed into the subject.
            mesh_sil = _mesh_silhouette_from_pos(position_maps[i])
            if (mesh_sil.shape[0], mesh_sil.shape[1]) != (out_size, out_size):
                _u8 = Image.fromarray((mesh_sil * 255).astype(np.uint8)).resize(
                    (out_size, out_size), Image.NEAREST)
                mesh_sil = np.asarray(_u8) > 0
            res = np.full((out_size, out_size, 3), 255, np.uint8)
            res[mesh_sil] = warped[mesh_sil]
            out.append(Image.fromarray(res, "RGB"))
        except Exception as _e:
            print(json.dumps({"type": "log", "message": f"[hybrid] view {i} failed ({_e}), using diffusion view"}), flush=True)
            out.append(diff_view.convert("RGB").resize((out_size, out_size), Image.LANCZOS))
    return out


def _split_tiled_image(image_path, count=4):
    """Split a tiled image into a grid based on the image's aspect ratio.

    Uses closest-distance matching against candidate grids (same logic as
    the shape bridge's _detect_grid) so both pipelines agree on layouts.

    Returns cropped PIL images composited onto white, up to `count`.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception:
        return [image_path]
    w, h = img.size
    ratio = w / h if h > 0 else 1.0
    _candidates = [(2, 1), (2, 2), (2, 3)]
    _best, _best_err = (2, 2), float("inf")
    for _nc, _nr in _candidates:
        _err = abs(ratio - _nc / _nr)
        if _err < _best_err:
            _best, _best_err = (_nc, _nr), _err
    _ncols, _nrows = _best
    _cw, _ch = w // _ncols, h // _nrows
    if _cw < 8 or _ch < 8:
        return [_composite_on_white(img)]
    _views = []
    for idx in range(min(count, _ncols * _nrows)):
        _ri = idx // _ncols
        _ci = idx % _ncols
        _cell = img.crop((_ci * _cw, _ri * _ch, (_ci + 1) * _cw, (_ri + 1) * _ch))
        _views.append(_composite_on_white(_cell))
    return _views


def _resolve_source_image(args):
    """Find the single image the user wired in for texturing."""
    source = args.get("image_path") or ""
    return source if source and os.path.exists(source) else ""


def _select_reference_views(args, count, input_mode="tiled"):
    """Pick `count` reference images for the diffusion model.

    count == 1: the whole wired image is used as a single reference (no split).
    count >= 2: the wired image is split into a grid (auto-detected from aspect
                ratio) and the first `count` views are returned.

    When input_mode is "single", always returns the whole image as one reference
    regardless of `count`.

    Returns a list of PIL images (length == count), or [] if nothing resolved.
    """
    source = _resolve_source_image(args)
    if not source:
        return []
    img = Image.open(source).convert("RGBA")
    img = _composite_on_white(img)

    if input_mode == "single":
        return [img]

    if count <= 1:
        return [img]

    quads = _split_tiled_image(source, count)
    if len(quads) < 2:
        return [img]
    return quads[:count]


def _bridge_first_load(args):
    """First-load bridge: link weights Modly placed anywhere (sibling node
    dirs / HF cache) into the layout hy3dgen expects, and repair out-of-extension
    files (venv site-packages) if needed. Non-fatal — logs and moves on."""
    try:
        from hunyuan3d_bootstrap import ensure_bridged
        model_dir = args.get("model_dir") or args.get("model_cache") or ""
        siblings = []
        if model_dir and os.path.isdir(os.path.dirname(model_dir)):
            _parent = os.path.dirname(model_dir)
            siblings = [os.path.join(_parent, d) for d in os.listdir(_parent)
                        if os.path.isdir(os.path.join(_parent, d)) and d != os.path.basename(model_dir)]
        ensure_bridged({
            "ext_dir": str(Path(__file__).resolve().parent),
            "model_dir": model_dir,
            "node_id": "texture",
            "siblings": siblings,
        })
    except Exception as e:
        print(json.dumps({"type": "log",
            "message": f"[bridge] first-load bootstrap skipped: {e}"}), flush=True)


def texture_mesh(args):
    """Texture an existing mesh with the Hunyuan3D-2.0 paint pipeline.

    Stages (each reported with status + subtext + percentage):
      load mesh -> decimate -> load paint models -> UV-unwrap ->
      render normal/position multiviews -> delight conditioning image ->
      multiview diffusion -> bake textures -> inpaint -> export GLB.
    """
    _bridge_first_load(args)

    mesh_path = args.get("mesh_path", "")
    if not mesh_path or not os.path.exists(mesh_path):
        print(json.dumps({"type": "error", "message": f"mesh_path not found: {mesh_path}"}), flush=True)
        return

    output_path = args.get("output_path", "output.glb")
    DECIMATE_FACES = int(args.get("decimate_faces", 40000))
    TEXTURE_SIZE = int(args.get("texture_size", 2048))
    TEXTURE_DIFFUSION_STEPS = int(args.get("texture_diffusion_steps", 30) or 30)
    if TEXTURE_DIFFUSION_STEPS < 1:
        TEXTURE_DIFFUSION_STEPS = 1
    if TEXTURE_DIFFUSION_STEPS > 100:
        TEXTURE_DIFFUSION_STEPS = 100

    # Delight (lighting normalization) is OFF by default: it re-renders the
    # subject under canonical lighting and often shifts/washes out the real
    # colors and detail. Pass delight="on" to re-enable the stock behaviour.
    delight = str(args.get("delight", "off") or "off").lower()

    # Texture generation method.
    #   diffusion: Hunyuan3D-2 multiview diffusion (512px ceiling).
    #   deform:    warp the wired reference views to the mesh silhouettes and
    #              bake them directly — no diffusion, full reference resolution.
    texture_method = str(args.get("texture_method", "diffusion") or "diffusion").lower()
    if texture_method not in ("diffusion", "deform"):
        texture_method = "diffusion"

    # How many reference images we are feeding the diffusion model. 1 = whole
    # image as a single reference; 2/3/4 = take that many quadrants from the
    # 2x2 tile in reading order [front, left, back, right].
    input_mode = str(args.get("input_mode", "tiled")).lower()
    reference_images = int(args.get("reference_images", 4) or 4)
    if input_mode == "single":
        reference_images = 1
    if reference_images < 1:
        reference_images = 1
    if reference_images > 6:
        reference_images = 6

    cond_views = _select_reference_views(args, reference_images, input_mode)
    if not cond_views:
        print(json.dumps({"type": "error", "message": "Conditioning image not found"}), flush=True)
        return

    print(json.dumps({"type": "log", "message": f"Using {len(cond_views)} reference image(s) for texturing"}), flush=True)

    # Dynamic view weights graduated to avoid competition at view boundaries.
    # Front view (index 0) gets highest weight; sides (1,3) moderate;
    # back (2), top (4), bottom (5) lower — they fill gaps without dominating.
    _known_weights = [1.0] * 6
    _dynamic_weights = [1.0] * 6
    _num_known = min(len(cond_views), 6)
    for _i in range(_num_known):
        _dynamic_weights[_i] = _known_weights[_i]
    if texture_method == "deform":
        # Deformed views are ground-truth references (not synthesized), so every
        # provided view gets full weight at the view boundaries.
        _dynamic_weights = [1.0] * 6
    elif texture_method == "hybrid":
        # Hybrid views are the same ground-truth references (flow-warped to the
        # diffusion geometry), but views past the reference count are diffusion
        # output, so they keep the graduated weights.
        _dynamic_weights = [1.0 if _i < len(cond_views) else _known_weights[_i] for _i in range(6)]
    _view_weights_override = _dynamic_weights
    print(json.dumps({"type": "log", "message": f"[texture] dynamic view weights: {_view_weights_override}"}), flush=True)

    _start_watchdog()

    report(3, "Loading input mesh", os.path.basename(mesh_path))
    mesh = trimesh.load(mesh_path)
    if mesh is None:
        print(json.dumps({"type": "error", "message": "Failed to load mesh"}), flush=True)
        return
    # trimesh may return a Scene (GLB container); extract the single geometry.
    if isinstance(mesh, trimesh.Scene):
        _geoms = [g for g in mesh.geometry.values() if hasattr(g, "faces")]
        mesh = _geoms[0] if _geoms else None
    if mesh is None or not hasattr(mesh, "faces"):
        print(json.dumps({"type": "error", "message": "Mesh has no geometry"}), flush=True)
        return
    report(8, "Mesh loaded", f"{len(mesh.vertices)} verts / {len(mesh.faces)} faces")

    # Pre-UV-unwrap simplification (cheap face reduction before the paint pass).
    if DECIMATE_FACES > 0 and hasattr(mesh, "faces") and len(mesh.faces) > DECIMATE_FACES:
        report(12, "Decimating mesh", f"target ~{DECIMATE_FACES} faces")
        try:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=int(DECIMATE_FACES))
            except TypeError:
                mesh = mesh.simplify_quadric_decimation(target_count=int(DECIMATE_FACES))
            report(18, "Decimation done", f"{len(mesh.faces)} faces")
        except Exception as _dec_err:
            print(json.dumps({"type": "log", "message": f"[warn] decimation skipped: {_dec_err}"}), flush=True)

    report(22, "Loading Hunyuan3D-2.0 paint models", "delight + multiview diffusion")
    # hy3dgen_path points at the Hunyuan3D-2 (2.0) folder so `hy3dgen.texgen`
    # resolves to the 2.0 paint pipeline.
    hy3dgen_path = args.get("hy3dgen_path", "")
    if hy3dgen_path and os.path.isdir(hy3dgen_path):
        sys.path.insert(0, hy3dgen_path)

    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
    import hy3dgen.texgen.utils.uv_warp_utils as _uvwarp_module

    # hy3dgen's stock mesh_uv_wrap(mesh) only takes the mesh and leaves every
    # xatlas option at its default — that produces hundreds of tiny single-face
    # charts and a sparse atlas full of black gaps, which the inpaint stage
    # then has to bridge (blurry). It also lacks the atlas_size / max_cost /
    # uv_stats params the texture pipeline relies on. Replace it with the
    # full-API version (patch lives here, not in site-packages, so it survives
    # venv rebuilds).
    def _mesh_uv_wrap_patched(mesh, atlas_size=0, padding=2, max_cost=8.0,
                              max_iterations=3, brute_force=False,
                              rotate_charts=True, bilinear=True):
        import trimesh as _trimesh
        import xatlas as _xatlas
        if isinstance(mesh, _trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if len(mesh.faces) > 500000000:
            raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")
        _atlas = _xatlas.Atlas()
        _atlas.add_mesh(mesh.vertices.astype('float32'), mesh.faces.astype('uint32'))
        _chart_opts = _xatlas.ChartOptions()
        _chart_opts.max_cost = max_cost
        _chart_opts.max_iterations = max_iterations
        _chart_opts.fix_winding = True
        _pack_opts = _xatlas.PackOptions()
        _pack_opts.resolution = atlas_size
        _pack_opts.padding = padding
        _pack_opts.bruteForce = brute_force
        _pack_opts.rotate_charts = rotate_charts
        _pack_opts.bilinear = bilinear
        _atlas.generate(_chart_opts, _pack_opts, verbose=False)
        vmapping, indices, uvs = _atlas.get_mesh(0)
        mesh.vertices = mesh.vertices[vmapping]
        mesh.faces = indices
        mesh.visual.uv = uvs
        try:
            mesh.metadata['uv_stats'] = {
                'charts': _atlas.chart_count,
                'width': _atlas.width,
                'height': _atlas.height,
                'utilization': float(_atlas.utilization),
            }
        except Exception:
            pass
        return mesh

    _uvwarp_module.mesh_uv_wrap = _mesh_uv_wrap_patched
    mesh_uv_wrap = _mesh_uv_wrap_patched

    # Stock hy3dgen's Multiview_Diffusion_Net.__call__ hardcodes
    # num_inference_steps=30 and accepts no such kwarg, so the user's
    # "Texture Diffusion Steps" parameter would be ignored (and the call
    # below would crash). Restore the tunable signature (patch lives here, not
    # in site-packages, so it survives venv rebuilds).
    from hy3dgen.texgen.utils import multiview_utils as _mv_module

    def _mv_call_patched(self, input_images, control_images, camera_info,
                         num_inference_steps=30):
        from typing import List as _List
        self.seed_everything(0)
        if not isinstance(input_images, _List):
            input_images = [input_images]
        input_images = [input_image.resize((self.view_size, self.view_size))
                        for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((self.view_size, self.view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))
        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]
        camera_info_gen = [camera_info]
        camera_info_ref = [[0]]
        kwargs['width'] = self.view_size
        kwargs['height'] = self.view_size
        kwargs['num_in_batch'] = num_view
        kwargs['camera_info_gen'] = camera_info_gen
        kwargs['camera_info_ref'] = camera_info_ref
        kwargs["normal_imgs"] = normal_image
        kwargs["position_imgs"] = position_image
        mvd_image = self.pipeline(input_images, num_inference_steps=num_inference_steps, **kwargs).images
        return mvd_image

    _mv_module.Multiview_Diffusion_Net.__call__ = _mv_call_patched

    # diffusers' encode_prompt returns the negative embeddings on CPU when CFG
    # is disabled (default in hy3dgen for non-turbo), while the positive
    # embeddings come back on cuda — the very next torch.cat then raises
    # "Expected all tensors to be on the same device". The old patched
    # site-packages had an explicit device sync here; stock hy3dgen removed it.
    # Re-apply it (patch lives here, not in site-packages, so it survives venv
    # rebuilds).
    import diffusers as _diffusers_cls
    _orig_encode_prompt = _diffusers_cls.StableDiffusionPipeline.encode_prompt

    def _encode_prompt_synced(self, *args, **kwargs):
        _out = _orig_encode_prompt(self, *args, **kwargs)
        if isinstance(_out, tuple) and len(_out) >= 2:
            _pe, _npe = _out[0], _out[1]
            if torch.is_tensor(_pe) and torch.is_tensor(_npe) and _npe.device != _pe.device:
                _out = (_pe, _npe.to(_pe.device)) + tuple(_out[2:])
        return _out

    _diffusers_cls.StableDiffusionPipeline.encode_prompt = _encode_prompt_synced

    # Stock hy3dgen tuned the delight stage down (cfg_image 2.5 -> 1.5, steps
    # 75 -> 50). Restore the original values so delight=on produces the same
    # results as the old patched site-packages (patch lives here, not in
    # site-packages, so it survives venv rebuilds).
    import hy3dgen.texgen.utils.dehighlight_utils as _delight_module

    _orig_delight_init = _delight_module.Light_Shadow_Remover.__init__

    def _delight_init_patched(self, config):
        _orig_delight_init(self, config)
        self.cfg_image = 2.5

    _delight_module.Light_Shadow_Remover.__init__ = _delight_init_patched

    def _delight_call_patched(self, image):
        import numpy as _np
        import cv2 as _cv2
        image = image.resize((512, 512))
        if image.mode == 'RGBA':
            image_array = _np.array(image)
            alpha_channel = image_array[:, :, 3]
            erosion_size = 3
            kernel = _np.ones((erosion_size, erosion_size), _np.uint8)
            alpha_channel = _cv2.erode(alpha_channel, kernel, iterations=1)
            image_array[alpha_channel == 0, :3] = 255
            image_array[:, :, 3] = alpha_channel
            image = Image.fromarray(image_array)
            image_tensor = torch.tensor(_np.array(image) / 255.0).to(self.device)
            alpha = image_tensor[:, :, 3:]
            rgb_target = image_tensor[:, :, :3]
        else:
            image_tensor = torch.tensor(_np.array(image) / 255.0).to(self.device)
            alpha = torch.ones_like(image_tensor)[:, :, :1]
            rgb_target = image_tensor[:, :, :3]
        image = image.convert('RGB')
        image = self.pipeline(
            prompt="",
            image=image,
            generator=torch.manual_seed(42),
            height=512,
            width=512,
            num_inference_steps=75,
            image_guidance_scale=self.cfg_image,
            guidance_scale=self.cfg_text,
        ).images[0]
        image_tensor = torch.tensor(_np.array(image) / 255.0).to(self.device)
        rgb_src = image_tensor[:, :, :3]
        image = self.recorrect_rgb(rgb_src, rgb_target, alpha)
        image = image[:, :, :3] * image[:, :, 3:] + torch.ones_like(image[:, :, :3]) * (1.0 - image[:, :, 3:])
        image = Image.fromarray((image.cpu().numpy() * 255).astype(_np.uint8))
        return image

    _delight_module.Light_Shadow_Remover.__call__ = _delight_call_patched

    # diffusers >= 0.39 refuses to execute the custom 'hunyuanpaint' pipeline
    # code (bundled with hy3dgen itself) without trust_remote_code=True, but
    # hy3dgen's Multiview_Diffusion_Net does not pass it. Inject the flag for
    # this process so the paint pipeline loads. The code being executed is the
    # installed hy3dgen dependency — trusted local code, not remote.
    import diffusers as _diffusers
    _orig_from_pretrained = _diffusers.DiffusionPipeline.from_pretrained.__func__

    def _from_pretrained_trust_remote(cls, *args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return _orig_from_pretrained(cls, *args, **kwargs)

    _diffusers.DiffusionPipeline.from_pretrained = classmethod(_from_pretrained_trust_remote)

    # Patch the multiview pipeline's RGBA->RGB conversion so transparent
    # backgrounds are composited onto WHITE instead of the stock gray (127,127,127).
    # White matches the training distribution of the multiview model and prevents
    # dark backgrounds from pulling the generated views down.
    from hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline as _HPP_cls
    from hy3dgen.texgen.hunyuanpaint import pipeline as _hunyuanpaint_pipeline
    import numpy as _np
    from PIL import Image as _Image
    def _to_rgb_white(maybe_rgba):
        if maybe_rgba.mode == 'RGB':
            return maybe_rgba
        elif maybe_rgba.mode == 'RGBA':
            rgba = maybe_rgba
            white = _np.full((rgba.size[1], rgba.size[0], 3), 255, dtype=_np.uint8)
            white = _Image.fromarray(white, 'RGB')
            white.paste(rgba, mask=rgba.getchannel('A'))
            return white
        else:
            raise ValueError("Unsupported image type.", maybe_rgba.mode)
    _hunyuanpaint_pipeline.to_rgb_image = _to_rgb_white

    # When delight is off (default), skip loading the ~1.5 GB Light_Shadow_Remover
    # entirely. The stock load_models() loads it unconditionally; we patch the
    # class so only the multiview model is built, and make cpu-offload skip the
    # missing delight model. This meaningfully lowers the VRAM/RAM ceiling on
    # 6 GB cards where the delight model is never used anyway.
    if texture_method == "deform":
        # Deform mode doesn't run diffusion at all — skip loading EVERY paint
        # model (~5 GB saved). The renderer (built in __init__) is all we need.
        def _load_models_none(self):
            torch.cuda.empty_cache()

        def _offload_none(self, gpu_id=None, device="cuda"):
            pass

        _HPP_cls.load_models = _load_models_none
        _HPP_cls.enable_model_cpu_offload = _offload_none
        print(json.dumps({"type": "log", "message": "[texture] deform mode — skipping all paint model loads"}), flush=True)
    elif delight != "on":
        from hy3dgen.texgen.utils.multiview_utils import Multiview_Diffusion_Net as _MVNet

        def _load_models_no_delight(self):
            torch.cuda.empty_cache()
            self.models['multiview_model'] = _MVNet(self.config)

        def _offload_no_delight(self, gpu_id=None, device="cuda"):
            self.models['multiview_model'].pipeline.enable_model_cpu_offload(
                gpu_id=gpu_id, device=device)

        _HPP_cls.load_models = _load_models_no_delight
        _HPP_cls.enable_model_cpu_offload = _offload_no_delight
        print(json.dumps({"type": "log", "message": "[texture] delight off — skipping delight model load (~1.5 GB saved)"}), flush=True)

    paint_model = args.get("paint_model", "")
    paint_subfolder = "hunyuan3d-paint-v2-0"  # always standard; turbo is broken
    model_cache = args.get("model_cache", "")
    # Resolve the snapshot root locally (ignore any bad path passed in). Search
    # model_cache and its sibling node dirs for the paint subfolder.
    _candidates = [model_cache] if model_cache else []
    if model_cache:
        _parent = os.path.dirname(model_cache)
        if os.path.isdir(_parent):
            _candidates += [os.path.join(_parent, d) for d in os.listdir(_parent)
                            if os.path.isdir(os.path.join(_parent, d)) and d != os.path.basename(model_cache)]
    for _c in _candidates:
        if _c and os.path.isfile(os.path.join(_c, paint_subfolder, "model_index.json")):
            paint_model = _c
            break
    if not paint_model:
        print(json.dumps({"type": "error", "message": f"Paint weights not found under {model_cache} or siblings."}), flush=True)
        return
    # Use the STANDARD (non-turbo) paint model. The turbo variant's 2p5D UNet
    # skips the `ref_scale_timing` assignment inside its reference-attention
    # block, so the reference images are effectively ignored and every output
    # view collapses to one averaged color. The standard model applies
    # ref_scale_timing = ref_scale and uses CFG (ref_scale=[0,1]), which is
    # what actually conditions the texture on the input views. This matches the
    # known-good oldmodel config (texture_variant="hunyuan3d-paint-v2-0").
    _saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        # from_pretrained(model_path, subfolder=...) expects model_path to be
        # the snapshot root and resolves delight + multiview as siblings of the
        # subfolder. If paint_model is already the snapshot root, pass subfolder.
        _snap_root = str(paint_model)
        if os.path.isfile(os.path.join(_snap_root, paint_subfolder, "model_index.json")):
            print(json.dumps({"type": "log", "message": f"[texture] loading paint model subfolder: {paint_subfolder}"}), flush=True)
            pipeline = Hunyuan3DPaintPipeline.from_pretrained(_snap_root, subfolder=paint_subfolder)
        else:
            # from_pretrained defaults subfolder='hunyuan3d-paint-v2-0-turbo' which
            # constructs a broken multiview path. Always pass the standard subfolder.
            pipeline = Hunyuan3DPaintPipeline.from_pretrained(paint_model, subfolder=paint_subfolder)
    finally:
        torch.set_default_dtype(_saved_dtype)

    # Apply dynamic view weights based on how many reference images exist.
    if '_view_weights_override' in locals():
        pipeline.config.candidate_view_weights = _view_weights_override
        print(json.dumps({"type": "log", "message": f"[texture] applied view weights: {pipeline.config.candidate_view_weights}"}), flush=True)

    try:
        pipeline.enable_model_cpu_offload()
    except Exception as _offload_err:
        print(json.dumps({"type": "log", "message": f"[warn] cpu offload skipped: {_offload_err}"}), flush=True)

    # Low-VRAM throughput optimizations (6 GB): attention slicing and VAE
    # slicing cap peak memory per forward pass so the run doesn't spill into
    # slow Windows shared GPU memory, which is what makes later steps drag.
    try:
        pipeline.enable_attention_slicing()
    except Exception as _attn_err:
        print(json.dumps({"type": "log", "message": f"[warn] attention slicing skipped: {_attn_err}"}), flush=True)
    try:
        pipeline.enable_vae_slicing()
    except Exception as _vae_err:
        print(json.dumps({"type": "log", "message": f"[warn] vae slicing skipped: {_vae_err}"}), flush=True)

    # The bake rasterizes each generated view to (re)project it onto the UVs.
    # back_project produces only RENDER_RES^2 points scattered onto the
    # TEXTURE_SIZE^2 atlas. If RENDER_RES is too small, the points are too sparse
    # -> empty bake mask -> black mesh. Keep at least a 512-pixel raster buffer so
    # the mesh always projects enough pixels, even when the requested atlas is tiny.
    # Final atlas is still TEXTURE_SIZE (resized at the end if needed).
    RENDER_RES = min(max(TEXTURE_SIZE, 512), 4096)
    pipeline.config.render_size = RENDER_RES
    pipeline.config.texture_size = TEXTURE_SIZE
    pipeline.render.set_default_render_resolution(RENDER_RES)
    pipeline.render.set_default_texture_resolution(TEXTURE_SIZE)

    pipeline.config.bake_exp = 4
    pipeline.render.bake_angle_thres = 85
    pipeline.render.bake_unreliable_kernel_size = 2

    import cv2 as _cv2
    from hy3dgen.texgen.differentiable_renderer.mesh_processor import meshVerticeInpaint as _mvi_orig

    def _fill_tri(tex, mask, uv0, uv1, uv2, c0, c1, c2, W, H):
        """Fill a triangle in UV space with barycentric-interpolated colors."""
        x0, y0 = float(uv0[0]) * (W - 1), (1.0 - float(uv0[1])) * (H - 1)
        x1, y1 = float(uv1[0]) * (W - 1), (1.0 - float(uv1[1])) * (H - 1)
        x2, y2 = float(uv2[0]) * (W - 1), (1.0 - float(uv2[1])) * (H - 1)
        min_x, max_x = max(0, int(min(x0, x1, x2))), min(W - 1, int(max(x0, x1, x2)))
        min_y, max_y = max(0, int(min(y0, y1, y2))), min(H - 1, int(max(y0, y1, y2)))
        if min_x > max_x or min_y > max_y:
            return
        xs = np.arange(min_x, max_x + 1, dtype=np.float32)
        ys = np.arange(min_y, max_y + 1, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys, indexing="xy")
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-10:
            return
        W0 = ((y1 - y2) * (XX - x2) + (x2 - x1) * (YY - y2)) / denom
        W1 = ((y2 - y0) * (XX - x2) + (x0 - x2) * (YY - y2)) / denom
        W2 = 1.0 - W0 - W1
        inside = (W0 >= -0.001) & (W1 >= -0.001) & (W2 >= -0.001)
        if not inside.any():
            return
        color = W0[..., None] * c0 + W1[..., None] * c1 + W2[..., None] * c2
        color = np.clip(color, 0, 1)
        fill_mask = inside & (mask[min_y:max_y + 1, min_x:max_x + 1] == 0)
        if not fill_mask.any():
            return
        tex[min_y:max_y + 1, min_x:max_x + 1][fill_mask] = color[fill_mask]
        mask[min_y:max_y + 1, min_x:max_x + 1][fill_mask] = 255

    def _patched_mvis(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
        H, W, C = texture.shape
        V = vtx_pos.shape[0]
        vtx_mask = np.zeros(V, dtype=np.float32)
        vtx_color = [np.zeros(C, dtype=np.float32) for _ in range(V)]
        uncolored = []
        G = [[] for _ in range(V)]
        for i in range(uv_idx.shape[0]):
            for k in range(3):
                uv_i = uv_idx[i, k]
                v_i = pos_idx[i, k]
                u = int(round(vtx_uv[uv_i, 0] * (W - 1)))
                v = int(round((1.0 - vtx_uv[uv_i, 1]) * (H - 1)))
                if mask[v, u] > 0:
                    vtx_mask[v_i] = 1.0
                    vtx_color[v_i] = texture[v, u]
                else:
                    uncolored.append(v_i)
                G[pos_idx[i, k]].append(pos_idx[i, (k + 1) % 3])

        smooth_count = 8
        prev = 0
        while smooth_count > 0:
            nc = 0
            for v_i in uncolored:
                v0 = vtx_pos[v_i]
                acc = []
                for n_i in G[v_i]:
                    if vtx_mask[n_i] > 0:
                        d = max(np.sqrt(np.sum((v0 - vtx_pos[n_i])**2)), 1e-4)
                        acc.append((vtx_color[n_i], 1.0 / (d * d)))
                if not acc:
                    nc += 1
                    continue
                wsum = sum(a for _, a in acc)
                base = sum(c * a for c, a in acc) / wsum
                _thresh = 0.15
                kept = [(c, w) for c, w in acc
                        if np.sqrt(np.sum((c - base) ** 2)) <= _thresh]
                if not kept:
                    nc += 1
                    continue
                wsum = sum(w for _, w in kept)
                vtx_color[v_i] = sum(c * w for c, w in kept) / wsum
                vtx_mask[v_i] = 1.0
            if prev == nc:
                smooth_count -= 1
            else:
                smooth_count += 1
            prev = nc

        new_tex = texture.copy()
        new_mask = mask.copy()
        _fallback = np.zeros(C, dtype=np.float32)
        _fc = 0
        for v_i in range(V):
            if vtx_mask[v_i] > 0:
                _fallback += vtx_color[v_i]
                _fc += 1
        if _fc > 0:
            _fallback /= _fc
        for i in range(uv_idx.shape[0]):
            v0_i, v1_i, v2_i = pos_idx[i, 0], pos_idx[i, 1], pos_idx[i, 2]
            if not (vtx_mask[v0_i] > 0 or vtx_mask[v1_i] > 0 or vtx_mask[v2_i] > 0):
                continue
            uv0 = vtx_uv[uv_idx[i, 0]]
            uv1 = vtx_uv[uv_idx[i, 1]]
            uv2 = vtx_uv[uv_idx[i, 2]]
            c0 = vtx_color[v0_i] if vtx_mask[v0_i] > 0 else _fallback
            c1 = vtx_color[v1_i] if vtx_mask[v1_i] > 0 else _fallback
            c2 = vtx_color[v2_i] if vtx_mask[v2_i] > 0 else _fallback
            _fill_tri(new_tex, new_mask, uv0, uv1, uv2, c0, c1, c2, W, H)

        return new_tex, new_mask

    def _patched_mvi(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
        return _patched_mvis(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)

    def _patched_uv_inpaint(self, texture, mask, _radius=10):
        if isinstance(texture, np.ndarray):
            tex_np = texture
        elif isinstance(texture, Image.Image):
            tex_np = np.array(texture) / 255.0
        else:
            tex_np = texture.cpu().numpy()
        # Pixels the bake actually painted. Everything else was never visible
        # from any camera and must be synthesized.
        _unpainted = (np.asarray(mask) <= 0)
        vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()
        tex_np, mask = _patched_mvi(tex_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
        img = (tex_np * 255).clip(0, 255).astype(np.uint8)
        # Pass 1: fill residual pixel gaps (TELEA preserves texture better
        # than NS for natural imagery).
        hole = 255 - mask
        if hole.sum() > 0:
            img = _cv2.inpaint(img, hole, _radius, _cv2.INPAINT_TELEA)
        # Pass 2: catch remaining seam cracks (Canny-detected) with small radius.
        # Restrict to pixels inside the original hole so we don't inpaint over
        # already-baked detail (grid lines, color boundaries, etc.).
        hole2 = _cv2.Canny(img, 10, 50)
        hole2 = _cv2.dilate(hole2, None, iterations=3)
        hole2 = hole2 & (255 - mask)
        if hole2.sum() > 0:
            img = _cv2.inpaint(img, hole2, max(2, _radius // 3), _cv2.INPAINT_TELEA)
        # Detail restore: formerly-unpainted regions come out as smooth blur
        # from the propagation above. Copy high-frequency detail sampled from
        # the nearest painted texel (with jitter to avoid blocky stamping),
        # scaled by the local texture energy of the source area — flat skin
        # stays clean, busy fabric gets matching grain.
        if _unpainted.any():
            from scipy.ndimage import distance_transform_edt
            _painted = ~_unpainted
            _dist, _inds = distance_transform_edt(_painted, return_indices=True)
            _flt = img.astype(np.float32)
            _detail = _flt - _cv2.GaussianBlur(img, (0, 0), 3).astype(np.float32)
            _energy = _cv2.GaussianBlur(np.abs(_detail), (0, 0), 8)
            _rng = np.random.default_rng(7)
            _ny = np.clip(_inds[0][_unpainted] + _rng.integers(-6, 7, size=_unpainted.sum()),
                          0, _flt.shape[0] - 1)
            _nx = np.clip(_inds[1][_unpainted] + _rng.integers(-6, 7, size=_unpainted.sum()),
                          0, _flt.shape[1] - 1)
            _gain = np.clip(_energy[_inds[0][_unpainted], _inds[1][_unpainted]] / 255.0,
                            0.0, 0.6).astype(np.float32)
            _flt[_unpainted] = np.clip(
                _flt[_unpainted] + _detail[_ny, _nx] * _gain * 0.6, 0, 255)
            img = _flt.astype(np.uint8)
        # Feather the boundary ring between painted and unpainted regions so
        # the transition reads as one continuous surface.
        _bw = 3
        _k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (2 * _bw + 1,) * 2)
        _band = _cv2.dilate(_unpainted.astype(np.uint8), _k) & (~_unpainted).astype(np.uint8)
        if _band.any():
            _flt = img.astype(np.float32)
            _blur = _cv2.GaussianBlur(img, (0, 0), 2.0).astype(np.float32)
            _flt[_band > 0] = 0.55 * _flt[_band > 0] + 0.45 * _blur[_band > 0]
            img = _flt.astype(np.uint8)
        return img  # uint8 [0,255] for pipeline.texture_inpaint which does /255

    import types
    pipeline.render.uv_inpaint = types.MethodType(_patched_uv_inpaint, pipeline.render)
    _orig_fast_bake = pipeline.render.fast_bake_texture
    def _patched_fast_bake(self, textures, cos_maps):
        channel = textures[0].shape[-1]
        tex_merge = torch.zeros(self.texture_size + (channel,), device=self.device)
        trust = torch.zeros(self.texture_size + (1,), device=self.device)
        for t, c in zip(textures, cos_maps):
            view_sum = (c > 0).sum()
            if view_sum > 0:
                painted = ((c > 0) * (trust > 0)).sum()
                if painted.float() / view_sum.float() > 0.999:
                    continue
            tex_merge += t * c
            trust += c
        tex_merge = tex_merge / torch.clamp(trust, min=1E-8)
        return tex_merge, trust > 1E-8
    pipeline.render.fast_bake_texture = types.MethodType(_patched_fast_bake, pipeline.render)

    report(32, "Paint models loaded", "starting texture pass")

    import gc

    # Debug: dump split views to output dir for inspection
    _debug_dir = os.path.join(os.path.dirname(output_path) or ".", "debug_textures")
    os.makedirs(_debug_dir, exist_ok=True)
    _view_names = ["front", "left", "back", "right", "top", "bottom"]
    for _i, (_v, _n) in enumerate(zip(cond_views, _view_names)):
        _v.save(os.path.join(_debug_dir, f"01_split_{_i}_{_n}.png"))

    # ── Stage 1: Delight conditioning views (optional) ───────────────────
    # Delight re-renders the subject under canonical lighting; OFF by default
    # because it washes out real colors/detail. When off we keep the raw
    # reference images as-is. Deform mode always skips it (model not loaded).
    if delight == "on" and texture_method != "deform":
        report(40, "Delighting reference views", f"removing shadows/highlights from {len(cond_views)} view(s)")
        cond_views = [pipeline.recenter_image(v.convert("RGB")) for v in cond_views]
        cond_views = [pipeline.models["delight_model"](v) for v in cond_views]
        for _i, (_v, _n) in enumerate(zip(cond_views, _view_names)):
            _v.save(os.path.join(_debug_dir, f"02_delight_{_i}_{_n}.png"))
        del pipeline.models["delight_model"]
        gc.collect()
        torch.cuda.empty_cache()
        report(46, "Delight done", "freed ~1 GB")
    else:
        report(40, "Skipping delight", "using raw reference views")
        for _i, (_v, _n) in enumerate(zip(cond_views, _view_names)):
            _v.save(os.path.join(_debug_dir, f"02_delight_{_i}_{_n}.png"))
        report(46, "Delight skipped", "")

    # ── Stage 2: UV-unwrap and load mesh into renderer ───────────────────
    report(48, "UV-unwrapping mesh", "preparing for baking")
    # xatlas unwrap: higher max_cost merges tiny single-face islands into
    # fewer, larger charts so the atlas has less black space to inpaint.
    # max_cost scales with face count — high-poly meshes need much more
    # aggressive merging to avoid hundreds of scattered micro-charts.
    _nfaces = len(mesh.faces)
    _mc = max(8.0, min(64.0, _nfaces / 200.0))
    mesh = mesh_uv_wrap(mesh, atlas_size=TEXTURE_SIZE, padding=2,
                        max_cost=_mc, max_iterations=3, rotate_charts=True)
    _uvs = getattr(mesh, "metadata", {}).get("uv_stats")
    if _uvs:
        print(json.dumps({"type": "log", "message":
            f"[texture] UV unwrap: {_uvs['charts']} charts, "
            f"{_uvs['utilization'] * 100:.1f}% packed, {_uvs['width']}x{_uvs['height']}"
            f" (max_cost={_mc:.0f}, {_nfaces} faces)"}),
            flush=True)
    pipeline.render.load_mesh(mesh)

    # ── Stage 3: Render normal/position maps from 6 cameras ──────────────
    elevs = pipeline.config.candidate_camera_elevs
    azims = pipeline.config.candidate_camera_azims
    view_weights = pipeline.config.candidate_view_weights

    report(52, "Rendering normal/position maps", "6 camera views")
    normal_maps = pipeline.render_normal_multiview(elevs, azims, use_abs_coor=True)
    position_maps = pipeline.render_position_multiview(elevs, azims)

    # ── Stage 4: Generate texture views ──────────────────────────────────
    if texture_method == "deform":
        report(58, "Deforming reference views", "warping views to mesh silhouettes")
        if len(cond_views) < 6:
            print(json.dumps({"type": "log", "message":
                f"[texture] deform mode has {len(cond_views)}/6 views — bake will cover "
                f"only the first {len(cond_views)} camera angles; top/bottom get inpainted"}), flush=True)
        multiviews = _deform_multiview(cond_views, position_maps, RENDER_RES)
        del cond_views, normal_maps, position_maps
        gc.collect()
        torch.cuda.empty_cache()
        report(72, "Views deformed to mesh", "baking next")
    else:
        # ── Stage 4a: Multiview diffusion ─────────────────────────────────
        # The multiview UNet (~3.5 GB) is the biggest CPU RAM consumer. After
        # diffusion we free it so only the bake/inpaint stage remains.
        camera_info = [
            (((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[elev]
            + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev]
            for azim, elev in zip(azims, elevs)
        ]

        report(58, "Multiview diffusion", "generating texture views")

        # Let the multiview model use its NATIVE reference conditioning
        # (ref_scale=[0.0, 1.0] by default). The model knows how to synthesize the
        # unknown views from the provided reference(s): views that have a reference
        # are rendered from it at full detail, and views without one are freely
        # synthesized. We don't override ref_scale — manually forcing the adherence
        # dial fights the model's own generation and produced chroma artifacts on
        # unreferenced views. The only lever we keep is how many references to feed
        # (reference_images), which tells the model which views have ground truth.
        _mv_model = pipeline.models["multiview_model"]
        _refs = list(cond_views)
        while len(_refs) < 4:
            _refs.append(cond_views[0])

        # Report per-step progress by wrapping the scheduler's step() method.
        # Map the chosen number of diffusion steps onto the 58->72 progress band so
        # the UI shows live movement instead of a stall.
        _MV_TOTAL_STEPS = TEXTURE_DIFFUSION_STEPS
        _mv_sched = _mv_model.pipeline.scheduler
        _mv_orig_step = _mv_sched.step
        _mv_counter = {"i": 0}

        def _mv_patched_step(*a, **kw):
            _out = _mv_orig_step(*a, **kw)
            _mv_counter["i"] += 1
            _i = _mv_counter["i"]
            _pct = 58 + int(14 * _i / max(1, _MV_TOTAL_STEPS))
            report(min(_pct, 72), "Multiview diffusion", f"step {_i}/{_MV_TOTAL_STEPS}")
            return _out

        _mv_sched.step = _mv_patched_step
        try:
            multiviews = _mv_model(
                _refs, normal_maps + position_maps, camera_info,
                num_inference_steps=TEXTURE_DIFFUSION_STEPS,
            )
        finally:
            _mv_sched.step = _mv_orig_step

        if texture_method == "hybrid":
            # Warp the original full-res references to match the diffusion
            # output's geometry (dense optical flow, applied at full res).
            report(68, "Hybrid: warping refs to diffusion geometry", "optical flow")
            multiviews = _hybrid_flow_multiview(
                cond_views, multiviews, position_maps, RENDER_RES)
        else:
            # Resize generated views to the render resolution for baking.
            multiviews = [v.resize((RENDER_RES, RENDER_RES)) for v in multiviews]

        # Free the multiview model + intermediates (~4.5 GB CPU RAM released).
        del cond_views, normal_maps, position_maps
        del pipeline.models["multiview_model"]
        gc.collect()
        torch.cuda.empty_cache()
        report(72, "Diffusion done, freed ~4.5 GB", "")

    # Debug: dump generated/back-projected multiviews
    _mv_names = ["front", "left", "back", "right", "top", "bottom"]
    for _i, (_v, _n) in enumerate(zip(multiviews, _mv_names)):
        _v.save(os.path.join(_debug_dir, f"03_multiview_{_i}_{_n}.png"))

    # Debug: quantify per-view diversity.
    try:
        import numpy as _np
        _mv_arrs = [_np.array(m.convert("RGB")).astype(float) for m in multiviews]
        _diffs = []
        for a in range(len(_mv_arrs)):
            for b in range(a + 1, len(_mv_arrs)):
                _diffs.append(_np.abs(_mv_arrs[a] - _mv_arrs[b]).mean())
        _means = [_np.array(m.convert("RGB")).reshape(-1, 3).mean(0) for m in multiviews]
        _avg_diff = sum(_diffs) / len(_diffs) if _diffs else 0
        _log = ["[multiview diversity] avg pairwise RGB diff = %.1f" % _avg_diff]
        for i, n in enumerate(_mv_names):
            _log.append("  %s meanRGB = %s" % (n, _means[i].round(1).tolist()))
        with open(os.path.join(_debug_dir, "06_diversity_report.txt"), "w") as _fh:
            _fh.write("\n".join(_log))
        print("\n".join(_log))
    except Exception as _e:
        print("[diversity report skipped] %s" % _e)

    # ── Normalize view brightness ─────────────────────────────────────────
    # DISABLED (tint testing): per-view channel matching was fading the tint
    # markers out of the multiviews. Set _NORMALIZE_VIEWS to True to re-enable.
    _NORMALIZE_VIEWS = False
    _norm_log = "[texture] "
    if _NORMALIZE_VIEWS:
        try:
            import numpy as _np_norm
            _mv_arrs = [_np_norm.array(v.convert("RGB")).astype(float) for v in multiviews]
            # Per-channel RGB mean matching (simple, robust, no cv2 dependency)
            _global_rgb_means = _np_norm.mean([a.mean(axis=(0, 1)) for a in _mv_arrs], axis=0)
            for _i in range(len(_mv_arrs)):
                _local_means = _mv_arrs[_i].mean(axis=(0, 1))
                _scales = _global_rgb_means / _np_norm.clip(_local_means, 1, None)
                for _c in range(3):
                    if 0.5 < _scales[_c] < 2.0:
                        _mv_arrs[_i][:, :, _c] = _np_norm.clip(_mv_arrs[_i][:, :, _c] * _scales[_c], 0, 255)
                multiviews[_i] = Image.fromarray(_mv_arrs[_i].astype(_np_norm.uint8))
            _norm_log += "per-channel RGB"
            # Also try LAB L-channel matching for extra smoothness
            try:
                import cv2 as _cv2_norm
                _mv_lab = [_cv2_norm.cvtColor(a.astype(_np_norm.uint8), _cv2_norm.COLOR_RGB2LAB).astype(float) for a in _mv_arrs]
                _global_l = _np_norm.mean([l[:, :, 0].mean() for l in _mv_lab])
                for _i in range(len(_mv_lab)):
                    _l_mean = _mv_lab[_i][:, :, 0].mean()
                    if _l_mean > 1:
                        _scale = _global_l / _l_mean
                        _mv_lab[_i][:, :, 0] = _np_norm.clip(_mv_lab[_i][:, :, 0] * _scale, 0, 255)
                    multiviews[_i] = Image.fromarray(_cv2_norm.cvtColor(_mv_lab[_i].astype(_np_norm.uint8), _cv2_norm.COLOR_LAB2RGB))
                _norm_log += " + LAB L"
            except Exception:
                _norm_log += " (no LAB)"
            _norm_log += " matched"
        except Exception as _norm_err:
            _norm_log = f"[texture] brightness normalization skipped: {_norm_err}"
    else:
        _norm_log += "normalization disabled"
    # Save views for debug comparison
    for _i, (_v, _n) in enumerate(zip(multiviews, _mv_names)):
        _v.save(os.path.join(_debug_dir, f"03b_normalized_{_i}_{_n}.png"))
    print(json.dumps({"type": "log", "message": _norm_log}), flush=True)

    # ── Stage 5: Bake textures + inpaint (runs on GPU) ───────────────────
    report(75, "Baking texture atlas", "merging projected views")
    texture, mask = pipeline.bake_from_multiview(
        multiviews, elevs, azims, view_weights,
        method=pipeline.config.merge_method,
    )
    _tex_np = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    _tex_h, _tex_w = texture.shape[:2]
    # Ensure the baked texture atlas matches the requested resolution.
    # The pipeline may internally downscale; if so, resize using PIL.
    if _tex_h != TEXTURE_SIZE or _tex_w != TEXTURE_SIZE:
        _msg = f"Baked texture is {_tex_w}x{_tex_h}, requested {TEXTURE_SIZE}x{TEXTURE_SIZE} — resizing"
        print(json.dumps({"type": "log", "message": _msg}), flush=True)
        _tex_pil = Image.fromarray(_tex_np).resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.LANCZOS)
        texture = torch.from_numpy(np.array(_tex_pil).astype(np.float32) / 255.0).to(texture.device)

    mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    _mh, _mw = mask.shape[:2]
    if _mh != TEXTURE_SIZE or _mw != TEXTURE_SIZE:
        _pil = Image.fromarray(mask_np).resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.NEAREST)
        mask = torch.from_numpy(np.array(_pil).astype(np.float32) / 255.0).to(mask.device).unsqueeze(-1)
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

    # Debug: dump baked texture and mask
    _tex_np = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if _tex_np.ndim == 3 and _tex_np.shape[2] in (3, 4):
        Image.fromarray(_tex_np).save(os.path.join(_debug_dir, "04_baked_texture.png"))
    else:
        print(f"[debug] unexpected texture shape {_tex_np.shape}, skipping debug dump")
    Image.fromarray(mask_np).save(os.path.join(_debug_dir, "05_bake_mask.png"))

    # Inpaint is mandatory (fills UV gaps that would otherwise leave the mesh
    # black). Always run it.
    report(88, "Inpainting UV seams", "filling gaps")
    texture = pipeline.texture_inpaint(texture, mask_np)
    _final = (texture.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if _final.ndim == 3 and _final.shape[2] in (3, 4):
        Image.fromarray(_final).save(os.path.join(_debug_dir, "06_inpainted_texture.png"))
    pipeline.render.set_texture(texture)
    textured_mesh = pipeline.render.save_mesh()

    report(96, "Saving textured mesh", "exporting GLB")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    textured_mesh.export(output_path)
    report(100, "Done", "texture complete")
    print(json.dumps({"type": "done", "output_path": output_path}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    _raw = sys.argv[1] if len(sys.argv) > 1 else "{}"
    # Accept either an inline JSON string (host passes this) or a path to a
    # .json file (handy for local testing).
    if os.path.isfile(_raw):
        with open(_raw, "r", encoding="utf-8") as _f:
            args = json.load(_f)
    else:
        args = json.loads(_raw)
    try:
        setup_paths(args)
        texture_mesh(args)
    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup_cuda()
