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

# Ampere (RTX 3050 etc.) speedups for the diffusion matmuls — free ~10-20% win.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# Quality-neutral attention speedups on Ampere (RTX 3050 etc.).
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


# Shape-generation tunables (6 GB friendly defaults). Mesh resolution / view
# count / guidance are read from node params via `args` inside generate_mesh().


def setup_paths(args):
    hy3dgen_path = args.get("hy3dgen_path")
    if hy3dgen_path and os.path.isdir(hy3dgen_path):
        sys.path.insert(0, hy3dgen_path)


def remove_background(img):
    """Remove background from an image using rembg (CPU only). Returns RGBA."""
    # Lazily create a persistent ONNX session (reused across calls).
    # Force CPU — the bridge uses CUDA for PyTorch and onnxruntime's
    # CUDA provider often conflicts with the active CUDA context.
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

    # If the image already has a meaningful alpha channel (transparent pixels),
    # it was already background-removed by the generator.  Running rembg a
    # second time converts RGBA→RGB→RGBA, creating a new alpha matte that
    # doesn't perfectly align with the original edges — producing dark noise
    # outlines around the subject.  Skip to avoid double-rembg artifacts.
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
    # Coerce to int to match the progress_cb(percent: int, …) contract.
    pct = int(round(pct))
    _touch_activity()
    # JSON line drives the structured % bar…
    print(json.dumps({"type": "progress", "pct": pct, "step": step, "subtext": subtext}), flush=True)
    # …and a plaintext mirror goes to the live console/log. The host forwards
    # non-JSON stdout/stderr lines in realtime, so every stage/step is visible
    # there even if the structured bar coalesces. Starts with 'p' so the JSON
    # reader in generator.py doesn't mistake it for a broken JSON fragment.
    if subtext:
        print(f"progress {pct}% | {step} — {subtext}", flush=True)
    else:
        print(f"progress {pct}% | {step}", flush=True)


def render_orthographic_view(mesh, azimuth_deg=0, elevation_deg=0, resolution=512):
    """Render a trimesh mesh from a given azimuth/elevation.

    Args:
        mesh: trimesh.Trimesh mesh
        azimuth_deg: azimuth in degrees (0=front, 90=right, 180=back, -90=left)
        elevation_deg: elevation in degrees (+90=top, -90=bottom)
        resolution: output resolution in pixels (square)

    Returns:
        PIL.Image in RGBA mode
    """
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
    """Remove degenerate faces and marching-cubes bridge artifacts.

    1. Removes degenerate (near-zero-area) faces.
    2. Removes "bridge" faces — faces whose longest edge is >> median, which
       indicates marching cubes bridged across noise connecting separate parts
       (e.g. hand to body).  Uses a 10×-median threshold on longest edge.
    3. Splits into connected components and drops tiny orphan components.
    Preserves UV coordinates if present.
    """
    import numpy as np
    try:
        # ── Step 1: degenerate faces ──
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

        # ── Step 2: bridge faces (long edges from MC noise) ──
        verts = mesh.vertices
        faces = mesh.faces
        if len(faces) > 0:
            # Compute longest edge per face
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            e01 = np.linalg.norm(v1 - v0, axis=1)
            e12 = np.linalg.norm(v2 - v1, axis=1)
            e20 = np.linalg.norm(v0 - v2, axis=1)
            longest = np.maximum(e01, np.maximum(e12, e20))
            med_edge = float(np.median(longest)) if len(longest) > 0 else 1.0
            bridge_thresh = max(med_edge * 10.0, 0.5)  # 10× median or 0.5 units
            bridge_mask = longest > bridge_thresh
            if bridge_mask.any():
                n_bridges = int(bridge_mask.sum())
                mesh.update_faces(~bridge_mask)
                mesh.remove_unreferenced_vertices()
                print(json.dumps({"type": "log",
                    "message": f"Removed {n_bridges} bridge faces (longest edge > {bridge_thresh:.3f})"}), flush=True)

        # ── Step 3: drop tiny orphan components ──
        if len(mesh.faces) > 0:
            import trimesh as _trimesh
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                face_counts = np.array([len(c.faces) for c in components])
                largest = int(face_counts.max())
                # Keep components with ≥ 1% of the largest
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
    """Assign per-vertex colors by projecting the mesh onto 4 cardinal views.

    This is an alternative to the PBR pipeline — it bypasses UV unwrapping,
    diffusion, and texture baking entirely. Each face is colored by the camera
    whose view direction best matches its normal, then vertex colors are
    averaged from adjacent faces.

    Use as a diagnostic: if vertex colors look correct but UV textures don't,
    the problem is UV mapping. If both look wrong, the views or shape are off.
    """
    import numpy as np

    # Load view images as numpy RGB float64 arrays.  views values may be
    # PIL.Image objects (already loaded by remove_bg) or file-path strings.
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
        # Composite onto white if RGBA, otherwise convert transparent=black.
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

    # Bounding-box normalisation so the orthographic projection maps the whole
    # mesh into [0,1] regardless of absolute coordinates.
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    extent = (bmax - bmin).max()
    if extent < 1e-8:
        extent = 1.0
    half = extent * 0.55  # small pad — 5% margin to avoid edge clipping

    # --- Camera definitions (orthographic) ---
    # Each camera maps 3-D coordinates to a 2-D (u,v) ∈ [0,1].  The
    # view-axis, up-axis and right-axis form an orthonormal basis.
    # The convention matches Hunyuan3D-2mv / Era3D: Z is forward, Y is up,
    # X is right.
    #
    #   front       = camera looks from +Z (azimuth 0°)
    #   back        = camera looks from -Z (azimuth 180°)
    #   left        = camera looks from -X (azimuth 270° / Era3D "left")
    #   right       = camera looks from +X (azimuth 90°  / Era3D "right")
    #   front_right = camera looks from azimuth ~45°  (+X+Z diagonal)
    #   front_left  = camera looks from azimuth ~315° (-X+Z diagonal)

    # "view" = direction from object to camera (dot with face normal > 0 = visible).
    # "right" / "up" = image-plane axes; cross(right, up) must equal view.
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

        # Face weight: cosine between face normal and camera view direction
        w = np.dot(face_normals, cam["view"])
        w = np.maximum(w, 0.0)  # only front-facing faces get colour
        if w.max() < 1e-8:
            continue

        # Project face centres onto the camera's image plane.
        # V is flipped because screen-space Y goes up→down while
        # world-space Y goes bottom→top.  Without the flip the
        # top of the object samples from image bottom and vice-versa.
        centres = verts[faces].mean(axis=1)
        u = np.dot(centres, cam["right"]) / half * 0.5 + 0.5
        v = 1.0 - (np.dot(centres, cam["up"]) / half * 0.5 + 0.5)

        u_px = np.clip((u * output_size).astype(np.int32), 0, output_size - 1)
        v_px = np.clip((v * output_size).astype(np.int32), 0, output_size - 1)

        colors = img[v_px, u_px]  # (N_faces, 3)
        face_colors += colors * w[:, None]
        face_wsum   += w

    # Normalise face colours, then propagate to vertices (average over
    # adjacent faces) to get smooth per-vertex colours.
    valid_faces = face_wsum > 1e-8
    face_colors[valid_faces] /= face_wsum[valid_faces, None]
    face_colors = np.clip(face_colors, 0, 255)

    vert_colors = np.zeros((len(verts), 3), dtype=np.float64)
    vert_wsum   = np.zeros(len(verts), dtype=np.float64)
    face_areas  = mesh.area_faces  # weight by face area for smoother blending

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

    # Store as RGBA vertex colours on the mesh
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
    """Resolve repo_id/subfolder to a LOCAL directory, never touching the
    network. `repo_id` may be either a HF repo id (falls back to
    ~/.cache/hy3dgen) or an explicit local path (e.g. Modly's models/ dir).

    When an explicit local path is given and contains `subfolder`, it is used
    directly. This keeps a run fully offline — weights are read from Modly's
    models/ folder, which setup.py / the manifest download populates.
    """
    # Explicit local directory (Modly models/ dir): use it directly.
    if os.path.isdir(repo_id):
        local_path = os.path.join(repo_id, subfolder) if subfolder else repo_id
        if os.path.isdir(local_path):
            os.environ["HY3DGEN_MODELS"] = ""
            return local_path, "", ""
        if os.path.isdir(repo_id) and any(Path(repo_id).iterdir()):
            # repo_id already points at the subfolder contents
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
            "node_id": "generate",
            "siblings": siblings,
        })
    except Exception as e:
        print(json.dumps({"type": "log",
            "message": f"[bridge] first-load bootstrap skipped: {e}"}), flush=True)


def generate_mesh(args):
    """Shape model only: load views, run shape pipeline, save untextured mesh."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    _bridge_first_load(args)

    output_path = args.get("output_path", "output.glb")
    model_path = args.get("model_path", "tencent/Hunyuan3D-2mv")
    subfolder = args.get("subfolder", "hunyuan3d-dit-v2-mv-turbo")
    num_inference_steps = args.get("num_inference_steps", 5)
    octree_resolution = args.get("octree_resolution", 380)
    guidance_scale = float(args.get("guidance_scale", 5.0))
    dual_guidance_scale = float(args.get("dual_guidance_scale", 10.5))

    # Reference views for the shape model, handled EXACTLY like the texture
    # node (no background invention, original pixels preserved).
    # input_mode controls how the wired image is interpreted:
    #   "single" = whole image is the single subject view (no split)
    #   "tiled"  = grid split into views (2x2 for ≤4, 2x3 for 5-6)
    #   "folder" = same as tiled (auto-tile was done by generator.py)
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

        # Auto-detect grid layout from image aspect ratio instead of
        # hardcoding.  For a 2-col × N-row grid the aspect ≈ 2/N; for
        # 3-col × 2-row it ≈ 3/2.  We pick the layout whose expected
        # aspect is closest to what we see.
        def _detect_grid(width, height, n_views):
            """Return (ncols, nrows) best matching the image shape.

            Grid layout is determined purely from the image aspect ratio.
            n_views only controls how many cells to *extract* (reading order),
            not the grid layout itself.  This way a 6-view 3×2 tiled image is
            always sliced as 3×2 even when the user only requests 2 views.
            """
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

        # Save a debug overlay showing the grid on the source image
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
            # Build cell→name mapping based on detected grid.
            # For 2-col grids the reading order is:
            #   (0,0) (1,0)
            #   (0,1) (1,1)
            #   (0,2) (1,2)
            # Names follow the render_views convention:
            #   front, left, back, right, top, bottom
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

    # Ensure RGBA with transparent background (generator already ran rembg,
    # but remove_background is a safety net if that failed). Pass RGBA directly
    # to the pipeline so its built-in ImageProcessorV2.recenter() can properly
    # isolate the object via the alpha channel — this avoids the white-fringe
    # halo that would bloat the mesh silhouette.
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
    torch.cuda.empty_cache()
    report(15, "Loading pipeline weights…")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        real_path, subfolder=real_sub, variant="fp16", device="cpu",
        local_files_only=True,
    )
    report(18, "Moving VAE to GPU…")
    shape_pipeline.vae.to("cuda")
    torch.cuda.empty_cache()
    report(20, "Moving DiT model to GPU…")
    shape_pipeline.model.to("cuda")
    torch.cuda.empty_cache()
    report(22, "Moving conditioner to GPU…")
    shape_pipeline.conditioner.to("cuda")
    shape_pipeline.device = torch.device("cuda")

    # Monkey-patch the image processor's view2idx to accept top/bottom views.
    # The model's encoder dynamically adjusts to N views (handled below), but
    # the MVImageProcessorV2 dictionary only maps 4 view names by default.
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
    shape_pipeline.enable_flashvdm()
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
    if _has_triton:
        try:
            shape_pipeline.compile()
            report(26, "Compiled shape model (torch.compile)")
        except Exception as e:
            report(26, "Shape compile skipped: %s" % e)
    else:
        report(26, "Shape compile skipped: Triton not installed")

    def shape_callback(step_idx, t, outputs):
        pct = 30 + int(55 * (step_idx + 1) / max(1, num_inference_steps))
        report(pct, f"Generating mesh ({step_idx + 1}/{num_inference_steps})")
        if step_idx % 4 == 0:
            torch.cuda.empty_cache()

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
    torch.cuda.empty_cache()

    report(90, "Decoding volume (volume grid)")
    mesh_outputs = shape_pipeline.vae.latents2mesh(
        latents, bounds=1.01, mc_level=0.0,
        num_chunks=20000, octree_resolution=octree_resolution,
        enable_pbar=True,
    )
    torch.cuda.empty_cache()

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

    # Final done is reported by generator.py after rendering views (95-100%).
    print(json.dumps({"type": "done", "output_path": output_path}), flush=True)
    os._exit(0)


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
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
