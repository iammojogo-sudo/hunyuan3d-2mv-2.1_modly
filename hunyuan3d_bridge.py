import os
import sys
import json
import shutil
import threading
import time
import trimesh
import torch
from pathlib import Path
from PIL import Image

# Safety checks for CUDA/Ampere features (will be silently skipped on macOS / CPU)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

# Watchdog: shared timestamp updated by report(), checked by a daemon thread.
# If no progress for STALL_TIMEOUT_MINUTES, the watchdog kills the process.
_last_activity = time.time()
_stall_timeout_minutes = 45


def _touch_activity():
    global _last_activity
    _last_activity = time.time()


def _watchdog_loop():
    while True:
        time.sleep(60)
        elapsed = time.time() - _last_activity
        if elapsed > _stall_timeout_minutes * 60:
            msg = f"No progress for {_stall_timeout_minutes} min — aborting"
            print(json.dumps({"type": "error", "message": msg}), flush=True)
            os._exit(1)


def _start_watchdog():
    t = threading.Thread(target=_watchdog_loop, daemon=True)
    t.start()


def setup_paths(args):
    hy3dgen_path = args.get("hy3dgen_path")
    if hy3dgen_path and os.path.isdir(hy3dgen_path):
        sys.path.insert(0, hy3dgen_path)


def remove_background(img):
    """Remove background from an image using rembg (CPU only). Returns RGBA."""
    if not hasattr(remove_background, "_session"):
        try:
            from rembg import new_session
            remove_background._session = new_session(providers=["CPUExecutionProvider"])
        except Exception as e:
            print(json.dumps({"type": "log",
                "message": f"rembg model failed to load: {e}"}), flush=True)
            remove_background._session = None
    if remove_background._session is None:
        return img.convert("RGBA") if img.mode != "RGBA" else img

    rgba = img.convert("RGBA") if img.mode != "RGBA" else img
    import numpy as _np
    _alpha = _np.array(rgba)[:, :, 3]
    if _alpha.min() < 128:
        return rgba

    from rembg import remove
    rgb = rgba.convert("RGB")
    try:
        return remove(rgb, session=remove_background._session, bgcolor=[255, 255, 255, 0])
    except Exception as e:
        print(json.dumps({"type": "log",
            "message": f"rembg inference failed: {e}"}), flush=True)
        return rgba


def report(pct, step, subtext=""):
    pct = int(round(pct))
    _touch_activity()
    print(json.dumps({"type": "progress", "pct": pct, "step": step, "subtext": subtext}), flush=True)
    if subtext:
        print(f"progress {pct}% | {step} — {subtext}", flush=True)
    else:
        print(f"progress {pct}% | {step}", flush=True)


def render_orthographic_view(mesh, azimuth_deg=0, elevation_deg=0, resolution=512):
    """Render a trimesh mesh from a given azimuth/elevation."""
    import numpy as np
    from trimesh.ray.ray_triangle import RayMeshIntersector
    inter = RayMeshIntersector(mesh)
    azim_rad = np.radians(azimuth_deg)
    elev_rad = np.radians(elevation_deg)
    cos_e = np.cos(elev_rad)
    fwd = np.array([
        -np.sin(azim_rad) * cos_e,
        -np.sin(elev_rad),
        -np.cos(azim_rad) * cos_e,
    ])
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-10:
        return Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    fwd /= fwd_norm

    center = mesh.bounds.mean(axis=0)
    extent = np.ptp(mesh.bounds, axis=0).max()
    if extent < 1e-10:
        return Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    nr = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if nr < 1e-10 else right / nr
    up = np.cross(fwd, right)
    up /= np.linalg.norm(up)

    cam_pos = center - fwd * extent * 2.0

    half = extent * 0.6
    xs = np.linspace(-half, half, resolution)
    ys = np.linspace(-half, half, resolution)
    X, Y = np.meshgrid(xs, ys)

    origins = cam_pos + X[:, :, np.newaxis] * right + Y[:, :, np.newaxis] * up
    directions = np.broadcast_to(fwd, origins.shape).copy()

    locations, idx_ray, idx_tri = inter.intersects_location(
        ray_origins=origins.reshape(-1, 3),
        ray_directions=directions.reshape(-1, 3), multiple_hits=False)

    if len(locations) == 0:
        return Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))

    face_normals = np.asarray(mesh.face_normals)
    if len(face_normals) == 0:
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
        fn_norm[fn_norm == 0] = 1
        face_normals = fn / fn_norm

    hit_normals = face_normals[idx_tri]
    light_dir = np.array([-0.3, 0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)

    ndotl = np.sum(hit_normals * light_dir, axis=1)
    brightness = np.clip(ndotl * 0.6 + 0.5, 0.0, 1.0)

    img = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    py, px = np.unravel_index(idx_ray, (resolution, resolution))
    r = np.clip(brightness * 160 + 95, 0, 255).astype(np.uint8)
    g = np.clip(brightness * 140 + 115, 0, 255).astype(np.uint8)
    b = np.clip(brightness * 130 + 125, 0, 255).astype(np.uint8)
    img[py, px, 0] = r
    img[py, px, 1] = g
    img[py, px, 2] = b
    img[py, px, 3] = 255

    return Image.fromarray(img, "RGBA")


def _repair_mesh(mesh):
    """Remove degenerate faces and marching-cubes bridge artifacts."""
    import numpy as np
    try:
        areas = mesh.area_faces
        median = float(np.median(areas)) if len(areas) > 0 else 1.0
        threshold = max(1e-12, median * 1e-6)
        valid = areas > threshold
        if not valid.all():
            n = valid.shape[0] - valid.sum()
            mesh.update_faces(valid)
            print(json.dumps({"type": "log", "message": f"Removed {n} degenerate faces"}), flush=True)
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()

        verts = mesh.vertices
        faces = mesh.faces
        if len(faces) > 0:
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            e01 = np.linalg.norm(v1 - v0, axis=1)
            e12 = np.linalg.norm(v2 - v1, axis=1)
            e20 = np.linalg.norm(v0 - v2, axis=1)
            longest = np.maximum(e01, np.maximum(e12, e20))
            med_edge = float(np.median(longest)) if len(longest) > 0 else 1.0
            bridge_thresh = max(med_edge * 10.0, 0.5)
            bridge_mask = longest > bridge_thresh
            if bridge_mask.any():
                n_bridges = int(bridge_mask.sum())
                mesh.update_faces(~bridge_mask)
                mesh.remove_unreferenced_vertices()
                print(json.dumps({"type": "log",
                    "message": f"Removed {n_bridges} bridge faces (longest edge > {bridge_thresh:.3f})"}), flush=True)

        if len(mesh.faces) > 0:
            import trimesh as _trimesh
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                face_counts = np.array([len(c.faces) for c in components])
                largest = int(face_counts.max())
                keep_mask = face_counts >= max(10, largest * 0.01)
                n_dropped = int((~keep_mask).sum())
                if n_dropped > 0:
                    kept = [c for c, k in zip(components, keep_mask) if k]
                    mesh = _trimesh.util.concatenate(kept)
                    print(json.dumps({"type": "log",
                        "message": f"Dropped {n_dropped} small orphan component(s)"}), flush=True)

    except Exception as e:
        print(json.dumps({"type": "log", "message": f"[warn] mesh repair skipped: {e}"}), flush=True)
    return mesh


def _compute_vertex_colors(mesh, views, output_size=512):
    """Assign per-vertex colors by projecting the mesh onto cardinal views."""
    import numpy as np

    view_imgs = {}
    for name in ("front", "left", "back", "right", "front_left", "top", "bottom"):
        v = views.get(name)
        if v is None:
            continue
        if isinstance(v, str):
            img = Image.open(v)
        elif hasattr(v, "convert"):
            img = v
        else:
            continue
        if img.mode == "RGBA":
            white_bg = Image.new("RGB", img.size, (255, 255, 255))
            white_bg.paste(img, mask=img.getchannel("A"))
            img = white_bg
        else:
            img = img.convert("RGB")
        if img.size != (output_size, output_size):
            img = img.resize((output_size, output_size), Image.LANCZOS)
        view_imgs[name] = np.asarray(img, dtype=np.float64)

    if not view_imgs:
        return mesh

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)

    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    extent = (bmax - bmin).max()
    if extent < 1e-8:
        extent = 1.0
    half = extent * 0.55

    cameras = {
        "front": {
            "view":   np.array([ 0., 0., 1.]),
            "right":  np.array([ 1., 0., 0.]),
            "up":     np.array([ 0., 1., 0.]),
        },
        "back": {
            "view":   np.array([ 0., 0.,-1.]),
            "right":  np.array([-1., 0., 0.]),
            "up":     np.array([ 0., 1., 0.]),
        },
        "left": {
            "view":   np.array([-1., 0., 0.]),
            "right":  np.array([ 0., 0., 1.]),
            "up":     np.array([ 0., 1., 0.]),
        },
        "right": {
            "view":   np.array([ 1., 0., 0.]),
            "right":  np.array([ 0., 0.,-1.]),
            "up":     np.array([ 0., 1., 0.]),
        },
        "front_left": {
            "view":   np.array([-0.707, 0., 0.707]),
            "right":  np.array([ 0.707, 0., 0.707]),
            "up":     np.array([ 0., 1., 0.]),
        },
        "top": {
            "view":   np.array([ 0., 1., 0.]),
            "right":  np.array([ 1., 0., 0.]),
            "up":     np.array([ 0., 0.,-1.]),
        },
        "bottom": {
            "view":   np.array([ 0.,-1., 0.]),
            "right":  np.array([ 1., 0., 0.]),
            "up":     np.array([ 0., 0., 1.]),
        },
    }

    face_colors = np.zeros((len(faces), 3), dtype=np.float64)
    face_wsum   = np.zeros(len(faces), dtype=np.float64)

    for cam_name, cam in cameras.items():
        img = view_imgs.get(cam_name)
        if img is None:
            continue

        w = np.dot(face_normals, cam["view"])
        w = np.maximum(w, 0.0)
        if w.max() < 1e-8:
            continue

        centres = verts[faces].mean(axis=1)
        u = np.dot(centres, cam["right"]) / half * 0.5 + 0.5
        v = 1.0 - (np.dot(centres, cam["up"]) / half * 0.5 + 0.5)

        u_px = np.clip((u * output_size).astype(np.int32), 0, output_size - 1)
        v_px = np.clip((v * output_size).astype(np.int32), 0, output_size - 1)

        colors = img[v_px, u_px]
        face_colors += colors * w[:, None]
        face_wsum   += w

    valid_faces = face_wsum > 1e-8
    face_colors[valid_faces] /= face_wsum[valid_faces, None]
    face_colors = np.clip(face_colors, 0, 255)

    vert_colors = np.zeros((len(verts), 3), dtype=np.float64)
    vert_wsum   = np.zeros(len(verts), dtype=np.float64)
    face_areas  = mesh.area_faces

    for fi, (f_verts, area) in enumerate(zip(faces, face_areas)):
        if face_wsum[fi] < 1e-8:
            continue
        c = face_colors[fi] * area
        for vi in f_verts:
            vert_colors[vi] += c
            vert_wsum[vi]   += area

    valid_verts = vert_wsum > 1e-8
    vert_colors[valid_verts] /= vert_wsum[valid_verts, None]
    vert_colors = np.clip(vert_colors, 0, 255).astype(np.uint8)

    vc = np.zeros((len(verts), 4), dtype=np.uint8)
    vc[:, :3] = vert_colors
    vc[:, 3] = 255
    mesh.visual.vertex_colors = vc

    print(json.dumps({
        "type": "log",
        "message": (
            f"Vertex colours computed: {len(cameras)} cameras, "
            f"{(face_wsum > 1e-8).sum()}/{len(faces)} faces coloured"
        ),
    }), flush=True)
    return mesh


def _resolve_hf_cache(repo_id, subfolder):
    """Resolve repo_id/subfolder to a LOCAL directory."""
    if os.path.isdir(repo_id):
        local_path = os.path.join(repo_id, subfolder) if subfolder else repo_id
        if os.path.isdir(local_path):
            os.environ["HY3DGEN_MODELS"] = ""
            return local_path, "", ""
        if os.path.isdir(repo_id) and any(Path(repo_id).iterdir()):
            os.environ["HY3DGEN_MODELS"] = ""
            return repo_id, "", ""
        raise RuntimeError(
            f"Local model dir '{repo_id}' exists but is missing subfolder "
            f"'{subfolder}'. Re-run the extension's setup.py to fetch weights."
        )
    local_base = os.path.expanduser("~/.cache/hy3dgen")
    local_path = os.path.join(local_base, repo_id, subfolder)
    if os.path.exists(local_path):
        os.environ["HY3DGEN_MODELS"] = ""
        return local_path, "", ""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError(
            f"Model '{repo_id}/{subfolder}' not found locally and offline mode "
            f"is enabled. Re-run the extension's setup.py to fetch weights."
        )
    from huggingface_hub import snapshot_download
    cached = snapshot_download(repo_id=repo_id, allow_patterns=[f"{subfolder}/*"])
    src = os.path.join(cached, subfolder)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copytree(src, local_path, dirs_exist_ok=True)
    os.environ["HY3DGEN_MODELS"] = ""
    return local_path, "", ""


def _bridge_first_load(args):
    """First-load bridge bootstrap."""
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
            "node_id": "generate",
            "siblings": siblings,
        })
    except Exception as e:
        print(json.dumps({"type": "log",
            "message": f"[bridge] first-load bootstrap skipped: {e}"}), flush=True)


def generate_mesh(args):
    """Shape model only: load views, run shape pipeline, save untextured mesh."""
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    _bridge_first_load(args)

    output_path = args.get("output_path", "output.glb")
    model_path = args.get("model_path", "tencent/Hunyuan3D-2mv")
    subfolder = args.get("subfolder", "hunyuan3d-dit-v2-mv-turbo")
    num_inference_steps = args.get("num_inference_steps", 5)
    octree_resolution = args.get("octree_resolution", 380)
    guidance_scale = float(args.get("guidance_scale", 5.0))
    dual_guidance_scale = float(args.get("dual_guidance_scale", 10.5))

    tiled_path = args.get("tiled_path")
    if not tiled_path or not os.path.exists(tiled_path):
        print(json.dumps({"type": "error", "message": "No input image (tiled_path) found"}), flush=True)
        return

    input_mode = str(args.get("input_mode", "tiled")).lower()
    _src_rgba = Image.open(tiled_path).convert("RGBA")

    if input_mode == "single":
        _raw_views = {"front": _src_rgba}
        print(json.dumps({"type": "log", "message": "Input mode: single — using whole image as front view"}), flush=True)
    else:
        try:
            _n = int(args.get("shape_views", 6))
        except (TypeError, ValueError):
            _n = 6
        _n = max(1, min(6, _n))

        w, h = _src_rgba.size
        _aspect = w / h if h > 0 else 1.0

        def _detect_grid(width, height, n_views):
            if n_views <= 1:
                return 1, 1
            candidates = [(2, 2), (2, 3), (3, 2)]
            best, best_err = (2, 2), float("inf")
            img_aspect = width / height if height > 0 else 1.0
            for nc, nr in candidates:
                expected = nc / nr
                err = abs(img_aspect - expected)
                if err < best_err:
                    best, best_err = (nc, nr), err
            return best

        _ncols, _nrows = _detect_grid(w, h, _n)
        _cw, _ch = w // _ncols, h // _nrows
        print(json.dumps({"type": "log", "message":
            f"Splitting tiled image {w}×{h} (aspect {_aspect:.2f}) "
            f"as {_ncols}×{_nrows} grid, {_cw}×{_ch} per cell"}), flush=True)

        try:
            import copy as _copy
            _dbg = _copy.deepcopy(_src_rgba)
            _draw = __import__("PIL").ImageDraw.Draw(_dbg)
            for _ri in range(_nrows + 1):
                _y = _ri * _ch
                _draw.line([(0, _y), (w, _y)], fill=(255, 0, 0, 180), width=2)
            for _ci in range(_ncols + 1):
                _x = _ci * _cw
                _draw.line([(_x, 0), (_x, h)], fill=(255, 0, 0, 180), width=2)
            _dbg_path = os.path.join(
                args.get("view_dir") or os.path.dirname(tiled_path),
                "_debug_grid_overlay.png")
            _dbg.save(_dbg_path)
            print(json.dumps({"type": "log", "message":
                f"Saved grid overlay debug image: {_dbg_path}"}), flush=True)
        except Exception as _e:
            print(json.dumps({"type": "log", "message":
                f"Grid overlay debug save failed: {_e}"}), flush=True)

        if _n == 1:
            _raw_views = {"front": _src_rgba}
        else:
            _VIEW_ORDER_2COL = ["front", "left", "back", "right", "top", "bottom"]
            _VIEW_ORDER_3COL = ["front", "left", "right", "back", "top", "bottom"]
            _view_order = _VIEW_ORDER_3COL if _ncols == 3 else _VIEW_ORDER_2COL
            _raw_views = {}
            for _idx in range(min(_n, _ncols * _nrows)):
                _ri = _idx // _ncols
                _ci = _idx % _ncols
                _cell = _src_rgba.crop((_ci * _cw, _ri * _ch,
                                        (_ci + 1) * _cw, (_ri + 1) * _ch))
                _name = _view_order[_idx] if _idx < len(_view_order) else f"view{_idx}"
                _raw_views[_name] = _cell

    view_dir = args.get("view_dir") or os.path.dirname(output_path)
    shape_views = {}
    for name, img in _raw_views.items():
        img = remove_background(img)
        img.save(os.path.join(view_dir, f"view_{name}.png"))
        shape_views[name] = img

    if not shape_views:
        print(json.dumps({"type": "error", "message": "No input views found"}), flush=True)
        return
    print(json.dumps({"type": "log", "message": f"Using {len(shape_views)} views for shape: {list(shape_views.keys())}"}), flush=True)

    report(10, f"Loading shape model ({subfolder})…")
    real_path, real_sub, _ = _resolve_hf_cache(model_path, subfolder)
    report(13, "Resolved model cache")
    cleanup_cuda()
    
    report(15, "Loading pipeline weights…")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        real_path, subfolder=real_sub, variant="fp16", device="cpu",
        local_files_only=True,
    )

    # Dynamic device allocation: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    report(18, f"Moving VAE to {device}…")
    shape_pipeline.vae.to(device)
    cleanup_cuda()

    report(20, f"Moving DiT model to {device}…")
    shape_pipeline.model.to(device)
    cleanup_cuda()

    report(22, f"Moving conditioner to {device}…")
    shape_pipeline.conditioner.to(device)
    shape_pipeline.device = device

    report(23, "Patching view indices…")
    if hasattr(shape_pipeline.image_processor, 'view2idx'):
        _v2i = dict(shape_pipeline.image_processor.view2idx)
        if 'top' not in _v2i and 'bottom' not in _v2i:
            _v2i['top'] = 4
            _v2i['bottom'] = 5
            shape_pipeline.image_processor.view2idx = _v2i
            print(json.dumps({"type": "log",
                "message": "[shape] patched image_processor.view2idx for top/bottom views"}), flush=True)

    report(24, "Enabling optimizations…")
    if hasattr(shape_pipeline, 'enable_flashvdm'):
        try:
            shape_pipeline.enable_flashvdm()
        except Exception:
            pass
    try:
        shape_pipeline.enable_attention_slicing()
    except Exception:
        pass
    try:
        shape_pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    if len(shape_views) > 0:
        encoder = shape_pipeline.conditioner.main_image_encoder
        n = len(shape_views)
        if hasattr(encoder, 'view_num') and isinstance(encoder.view_num, int) and encoder.view_num < n:
            import numpy as np
            from hy3dgen.shapegen.models.conditioner import get_1d_sincos_pos_embed_from_grid
            encoder.view_num = n
            pos = np.arange(n, dtype=np.float32)
            emb = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(encoder.model.config.hidden_size, pos)).float()
            emb = emb.unsqueeze(1).repeat(1, encoder.num_patches, 1)
            encoder.view_embed = emb.unsqueeze(0).to(device=encoder.model.device, dtype=encoder.model.dtype)

    report(25, "Shape model loaded — generating mesh")

    _has_triton = False
    try:
        import triton
        _has_triton = True
    except Exception:
        _has_triton = False

    if _has_triton and device.type == "cuda":
        try:
            shape_pipeline.compile()
            report(26, "Compiled shape model (torch.compile)")
        except Exception as e:
            report(26, "Shape compile skipped: %s" % e)
    else:
        report(26, "Shape compile skipped: Triton/CUDA not active")

    def shape_callback(step_idx, t, outputs):
        pct = 30 + int(55 * (step_idx + 1) / max(1, num_inference_steps))
        report(pct, f"Generating mesh ({step_idx + 1}/{num_inference_steps})")
        if step_idx % 4 == 0:
            cleanup_cuda()

    report(30, f"Generating mesh (1/{num_inference_steps})…")
    latents = shape_pipeline(
        image=shape_views, num_inference_steps=num_inference_steps,
        octree_resolution=octree_resolution, guidance_scale=guidance_scale,
        dual_guidance_scale=dual_guidance_scale, num_chunks=20000,
        generator=torch.manual_seed(42), output_type="latent",
        callback=shape_callback, callback_steps=1,
    )
    report(88, "Decoding volume (VAE decode)")
    latents = 1. / shape_pipeline.vae.scale_factor * latents
    latents = shape_pipeline.vae(latents)
    cleanup_cuda()

    report(90, "Decoding volume (volume grid)")
    mesh_outputs = shape_pipeline.vae.latents2mesh(
        latents, bounds=1.01, mc_level=0.0,
        num_chunks=20000, octree_resolution=octree_resolution,
        enable_pbar=True,
    )
    cleanup_cuda()

    report(92, "Converting to mesh")
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    mesh = export_to_trimesh(mesh_outputs)
    if isinstance(mesh, list):
        mesh = mesh[0]

    report(94, "Repairing mesh")
    mesh = _repair_mesh(mesh)

    report(96, "Cleaning up GPU memory…")
    shape_pipeline = shape_pipeline.to("cpu")
    del shape_pipeline
    import gc; gc.collect()
    cleanup_cuda()

    report(98, "Exporting .glb…")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mesh.export(output_path)

    print(json.dumps({"type": "done", "output_path": output_path}), flush=True)
    os._exit(0)


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    elif torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    try:
        setup_paths(args)
        generate_mesh(args)
    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup_cuda()
