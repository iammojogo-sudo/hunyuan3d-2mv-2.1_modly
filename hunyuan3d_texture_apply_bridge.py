"""PBR texture applicator bridge.

Takes an already-UV-unwrapped mesh (GLB) plus texture atlas image(s) and
exports a PBR GLB: albedo + normal + metallic/roughness, with optional real
vertex displacement. Pure trimesh + PIL + pygltflib + numpy: no GPU.

Inputs (JSON arg):
  mesh_path:              GLB that already has UV coordinates
  texture_path:           albedo atlas (PNG/JPG). Optional if texture_folder
                          holds a detectable albedo file.
  texture_folder:         folder scanned for maps by filename keyword
  normal_path:            tangent-space normal map (overrides folder scan)
  specular_path:          roughness map, or specular map (see rough_mode)
  rough_mode:             'auto' | 'roughness' | 'specular'
  displacement_path:      height map (overrides folder scan)
  displacement_strength:  fraction of bbox diagonal mapped to the full
                          black->white range; 0 = geometry untouched
  displacement_subdiv:    loop-subdivision iterations before displacement
  occlusion_from_height:  derive soft cavity occlusion into ORM R channel
  normal_scale:           glTF normalTexture scale
  normal_flip_y:          flip green channel (DirectX-style normal maps)
  metallic_factor:        uniform metallic factor (ORM B channel + factor)
  mask_red_metallic ... mask_cyan_rough:
                          per-palette-color metallic (0-1) and roughness
                          multiplier for painted maskmap.png regions
                          (white = default material)
  output_path:            where to write the PBR GLB
"""
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter


def _report(pct, step, subtext=""):
    print(json.dumps({
        "type": "progress", "pct": pct, "step": step, "subtext": subtext
    }), flush=True)


def _log(msg):
    print(json.dumps({"type": "log", "message": msg}), flush=True)


def _flatten_scene(scene):
    geoms = [g for g in scene.geometry.values() if hasattr(g, "faces") and len(getattr(g, "faces", []))]
    if not geoms:
        return None
    if len(geoms) == 1:
        return geoms[0]
    # Concatenate multi-geometry scenes into one mesh so the UVs+texture apply
    # consistently to the whole object.
    try:
        return trimesh.util.concatenate(geoms)
    except Exception:
        return geoms[0]


_ALBEDO_KEYS = ("texturemap", "albedo", "basecolor", "base_color", "base-color", "diffuse")
_NORMAL_KEYS = ("normalmap", "normal", "nrm", "nor_")
_ROUGH_KEYS = ("roughnessmap", "roughness", "rough")
_SPEC_KEYS = ("specularmap", "specular", "spec", "smoothness", "smooth")
_DISP_KEYS = ("displacementmap", "displacement", "displace", "height", "depth", "disp")
_MASK_KEYS = ("maskmap",)

# Paint palette for maskmap.png (white = default material).
_MASK_PALETTE = [
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)),
    ("magenta", (255, 0, 255)),
    ("cyan", (0, 255, 255)),
]
_MASK_DEFAULT = 6  # index for white/default
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _scan_folder(folder):
    """Detect maps in a folder by filename keyword. Returns {kind: path}."""
    found = {}
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return found
    imgs = [n for n in names if n.lower().endswith(_IMG_EXTS)]
    for n in imgs:
        low = n.lower()
        p = os.path.join(folder, n)
        if any(k in low for k in _MASK_KEYS):
            found.setdefault("mask", p)
        elif any(k in low for k in _NORMAL_KEYS):
            found.setdefault("normal", p)
        elif any(k in low for k in _ROUGH_KEYS) or any(k in low for k in _SPEC_KEYS):
            found.setdefault("rough", p)
        elif any(k in low for k in _DISP_KEYS):
            found.setdefault("displacement", p)
        elif any(k in low for k in _ALBEDO_KEYS):
            found.setdefault("albedo", p)
    if "albedo" not in found and len(imgs) == 1:
        # Single-image folder: treat it as the albedo atlas.
        found["albedo"] = os.path.join(folder, imgs[0])
    return found


def _sample_height_bilinear(h, uv):
    """Sample float heightmap h (H,W in 0..1) at uv coords (N,2)."""
    H, W = h.shape
    x = np.clip(uv[:, 0], 0.0, 1.0) * (W - 1)
    y = np.clip(1.0 - uv[:, 1], 0.0, 1.0) * (H - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, W - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (x - x0).astype(np.float64)
    fy = (y - y0).astype(np.float64)
    return (h[y0, x0] * (1 - fx) * (1 - fy) + h[y0, x1] * fx * (1 - fy) +
            h[y1, x0] * (1 - fx) * fy + h[y1, x1] * fx * fy)


def _apply_displacement(mesh, uv, disp_path, strength, subdiv):
    """Push vertices along normals by height sampled at their UVs."""
    img = Image.open(disp_path)
    h = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    if subdiv and int(subdiv) > 0:
        subdiv = int(subdiv)
        # Each loop-subdivision round ~4x verts; cap the estimate so a huge
        # mesh can't OOM-kill the process with no traceback.
        _cap = 2500000
        _est = len(mesh.vertices) * (4 ** subdiv)
        while subdiv > 0 and _est > _cap:
            subdiv -= 1
            _est = len(mesh.vertices) * (4 ** subdiv)
        if subdiv <= 0:
            _log(f"[texture_apply] subdivision skipped: mesh already has "
                 f"{len(mesh.vertices)} verts (cap {_cap}) — displacement runs "
                 "on the base mesh")
        else:
            try:
                sub = mesh.subdivide(iterations=subdiv)
                sub_uv = getattr(sub.visual, "uv", None)
                if sub_uv is not None and len(sub_uv) == len(sub.vertices):
                    mesh = sub
                    uv = np.asarray(sub_uv, dtype=np.float64)
                    _log(f"[texture_apply] subdivided x{subdiv}: {len(mesh.vertices)} verts")
                else:
                    _log("[texture_apply] subdivision dropped UVs — using base mesh")
            except Exception as e:
                _log(f"[texture_apply] subdivision skipped: {e}")
    d = _sample_height_bilinear(h, np.asarray(uv, dtype=np.float64))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    if diag <= 0:
        return mesh, uv
    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    except Exception:
        _log("[texture_apply] no vertex normals — displacement skipped")
        return mesh, uv
    mesh.vertices = verts + normals * ((d - 0.5) * float(strength) * diag)[:, None]
    _log(f"[texture_apply] displaced {len(verts)} verts (strength {strength})")
    return mesh, uv


def _mask_regions(mask_path, W, H):
    """Map each pixel to a palette index (0-5) or default (6, white).

    Painted maskmap.png is snapped to the nearest palette color, so
    anti-aliased brush edges still resolve cleanly. Resized NEAREST to
    preserve solid colors. Chunked int32 math to stay light at 4K.
    """
    m = Image.open(mask_path).convert("RGB").resize((W, H), Image.NEAREST)
    a = np.asarray(m, dtype=np.int32)
    pal = np.array([c for _, c in _MASK_PALETTE] + [(255, 255, 255)], dtype=np.int32)
    reg = np.full((H, W), _MASK_DEFAULT, dtype=np.int64)
    best = np.full((H, W), np.iinfo(np.int32).max, dtype=np.int64)
    for i in range(len(pal)):
        d = ((a - pal[i]) ** 2).sum(-1).astype(np.int64)
        take = d < best
        reg[take] = i
        best[take] = d[take]
        del d
    return reg


def _attach_pbr(temp_glb, output_path, normal_arr=None, rough_arr=None,
                occ_arr=None, metallic=0.0, normal_scale=1.0,
                metal_arr=None, rough_scale_arr=None):
    """Attach normal + ORM textures to every material in a GLB."""
    from pygltflib import GLTF2, BufferView, Image as GImage, Texture as GTexture
    from pygltflib import NormalMaterialTexture, TextureInfo
    import io as _io

    g = GLTF2().load(temp_glb)

    def _png_bytes(arr):
        buf = _io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def _embed_image(data, name):
        # pygltflib can't pack Image.blob into buffers on save, so append the
        # bytes to the BIN blob ourselves with a real BufferView (4B aligned).
        blob = bytearray(g.binary_blob() or b"")
        blob += b"\0" * (-len(blob) % 4)
        off = len(blob)
        blob += data
        g.set_binary_blob(bytes(blob))
        g.buffers[0].byteLength = len(blob)
        bv = BufferView(buffer=0, byteOffset=off, byteLength=len(data), name=name)
        g.bufferViews.append(bv)
        img = GImage()
        img.bufferView = len(g.bufferViews) - 1
        img.mimeType = "image/png"
        img.name = name
        g.images.append(img)
        return len(g.images) - 1

    if normal_arr is not None:
        _embed_image(_png_bytes(normal_arr), "normal")
        nt = GTexture()
        nt.source = len(g.images) - 1
        nt.name = "normal"
        g.textures.append(nt)
        normal_idx = len(g.textures) - 1
    else:
        normal_idx = None

    if rough_arr is not None:
        H, W = rough_arr.shape
        if occ_arr is not None and occ_arr.shape != (H, W):
            occ_arr = np.asarray(
                Image.fromarray((occ_arr * 255).astype(np.uint8)).resize(
                    (W, H), Image.BILINEAR), dtype=np.float64) / 255.0
        occ = occ_arr if occ_arr is not None else np.ones((H, W), dtype=np.float64)
        if rough_scale_arr is not None:
            rough_arr = np.clip(rough_arr * rough_scale_arr, 0, 1)
        metal = metal_arr if metal_arr is not None else np.full((H, W), float(metallic))
        orm = np.zeros((H, W, 3), dtype=np.uint8)
        orm[:, :, 0] = (np.clip(occ, 0, 1) * 255).astype(np.uint8)
        orm[:, :, 1] = (np.clip(rough_arr, 0, 1) * 255).astype(np.uint8)
        orm[:, :, 2] = (np.clip(metal, 0, 1) * 255).astype(np.uint8)
        _embed_image(_png_bytes(orm), "metallicRoughness")
        mt = GTexture()
        mt.source = len(g.images) - 1
        mt.name = "metallicRoughness"
        g.textures.append(mt)
        orm_idx = len(g.textures) - 1
    else:
        orm_idx = None

    for mat in g.materials:
        if normal_idx is not None:
            nm = NormalMaterialTexture()
            nm.index = normal_idx
            nm.scale = float(normal_scale)
            mat.normalTexture = nm
        if orm_idx is not None:
            ti = TextureInfo()
            ti.index = orm_idx
            mat.pbrMetallicRoughness.metallicRoughnessTexture = ti
            # Factors multiply the texture: when a per-pixel metal map carries
            # the values, keep the factor neutral so nothing double-applies.
            mat.pbrMetallicRoughness.metallicFactor = 1.0 if metal_arr is not None else float(metallic)
            mat.pbrMetallicRoughness.roughnessFactor = 1.0

    g.save(output_path)


def apply_texture(args):
    mesh_path = args.get("mesh_path", "")
    texture_path = args.get("texture_path", "")
    output_path = args.get("output_path", "output.glb")

    if not mesh_path or not os.path.exists(mesh_path):
        raise RuntimeError(f"mesh_path not found: {mesh_path}")

    # Mode toggle: 'albedo' maps the wired image only and ignores every map
    # setting; anything else runs the full PBR path.
    mode = str(args.get("mode", "pbr") or "pbr").lower()
    albedo_only = (mode == "albedo")

    # Resolve maps: explicit paths win, otherwise scan the folder.
    folder = "" if albedo_only else (args.get("texture_folder", "") or "")
    found = _scan_folder(folder) if folder and os.path.isdir(folder) else {}
    if folder and not os.path.isdir(folder):
        _log(f"[texture_apply] texture folder not found: {folder}")
    if (not texture_path or not os.path.exists(texture_path)) and found.get("albedo"):
        texture_path = found["albedo"]
    if not texture_path or not os.path.exists(texture_path):
        if albedo_only:
            raise RuntimeError(
                "texture_apply: albedo-only mode needs the wired image "
                "(or image_path) — no texture folder is read in this mode")
        raise RuntimeError(
            "texture_apply: no albedo texture provided (wire an image, set "
            "image_path, or point texture_folder at the maps)")

    def _pick(key, *names):
        for n in names:
            v = args.get(n, "")
            if v and os.path.exists(v):
                return v
        return found.get(key, "")

    normal_path = "" if albedo_only else _pick("normal", "normal_path")
    spec_path = "" if albedo_only else _pick("rough", "specular_path", "rough_path")
    disp_path = "" if albedo_only else _pick("displacement", "displacement_path")
    mask_path = "" if albedo_only else _pick("mask", "mask_path")

    def _f(key, default=0.0):
        try:
            return float(args.get(key, default) or 0)
        except (TypeError, ValueError):
            return float(default)

    _mask_metals = [_f(f"mask_{n}_metallic", 0.0) for n, _ in _MASK_PALETTE]
    _mask_roughs = [_f(f"mask_{n}_rough", 1.0) for n, _ in _MASK_PALETTE]
    _mask_active = mask_path and os.path.exists(mask_path) and any(
        m != 0.0 or r != 1.0 for m, r in zip(_mask_metals, _mask_roughs))

    rough_mode = str(args.get("rough_mode", "auto") or "auto").lower()
    disp_strength = float(args.get("displacement_strength", 0) or 0)
    disp_subdiv = int(args.get("displacement_subdiv", 0) or 0)
    occ_from_h = str(args.get("occlusion_from_height", "on") or "on").lower() == "on"
    normal_scale = float(args.get("normal_scale", 1.0) or 0)
    if normal_scale <= 0:
        normal_scale = 1.0
    normal_flip_y = str(args.get("normal_flip_y", "off") or "off").lower() == "on"
    metallic = float(args.get("metallic_factor", 0.0) or 0.0)

    _log(f"[texture_apply] mode={mode} albedo={os.path.basename(texture_path)}"
         + (f" normal={os.path.basename(normal_path)}" if normal_path else "")
         + (f" rough={os.path.basename(spec_path)}" if spec_path else "")
         + (f" disp={os.path.basename(disp_path)}" if disp_path else "")
         + (f" mask={os.path.basename(mask_path)}" if mask_path else ""))
    if disp_path and disp_strength > 0:
        _log(f"[texture_apply] displacement strength={disp_strength} subdiv={disp_subdiv}")

    _report(20, "Loading mesh", os.path.basename(mesh_path))
    loaded = trimesh.load(mesh_path, force=None)
    mesh = _flatten_scene(loaded) if isinstance(loaded, trimesh.Scene) else loaded
    if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise RuntimeError("Mesh has no usable geometry")
    _log(f"[texture_apply] mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} verts")

    # trimesh may return a Scene for GLB even with a single mesh; extract it.
    if isinstance(mesh, trimesh.Scene):
        mesh = _flatten_scene(mesh)
        if mesh is None:
            raise RuntimeError("Mesh has no usable geometry")

    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) == 0:
        raise RuntimeError(
            f"Mesh '{os.path.basename(mesh_path)}' has no UV coordinates — it "
            "must be UV-unwrapped before textures can be applied. Run the "
            "'Texture Mesh' node first."
        )
    uv = np.asarray(uv, dtype=np.float64)
    _log(f"[texture_apply] UVs OK: {len(uv)} coords")

    # Displacement first (real geometry, sampled at vertex UVs).
    if disp_path and os.path.exists(disp_path) and disp_strength > 0:
        _report(30, "Applying displacement", os.path.basename(disp_path))
        mesh, uv = _apply_displacement(mesh, uv, disp_path, disp_strength, disp_subdiv)
    elif disp_path and os.path.exists(disp_path):
        _log("[texture_apply] displacement map found but strength is 0 — "
             "geometry untouched (map still used for occlusion)")

    _report(50, "Loading albedo", os.path.basename(texture_path))
    tex = Image.open(texture_path).convert("RGB")
    _log(f"[texture_apply] albedo: {tex.size[0]}x{tex.size[1]}")

    # Normal map (optional).
    normal_arr = None
    if normal_path and os.path.exists(normal_path):
        _report(60, "Loading normal map", os.path.basename(normal_path))
        normal_arr = np.asarray(Image.open(normal_path).convert("RGB"))
        if normal_flip_y:
            normal_arr = normal_arr.copy()
            normal_arr[:, :, 1] = 255 - normal_arr[:, :, 1]
        _log(f"[texture_apply] normal: {normal_arr.shape[1]}x{normal_arr.shape[0]}"
             + (" (Y flipped)" if normal_flip_y else ""))
    elif folder:
        _log("[texture_apply] no normal map detected")

    # Roughness (optional): direct map, or inverted specular/smoothness map.
    rough_arr = None
    if spec_path and os.path.exists(spec_path):
        _report(65, "Loading roughness/specular", os.path.basename(spec_path))
        base = os.path.basename(spec_path).lower()
        if rough_mode == "roughness":
            is_rough = True
        elif rough_mode == "specular":
            is_rough = False
        else:
            is_rough = ("rough" in base) and ("spec" not in base)
            if not is_rough and "rough" not in base and "spec" not in base \
                    and "smooth" not in base:
                is_rough = True  # unknown name: assume roughness (no inversion)
        gray = np.asarray(Image.open(spec_path).convert("L"), dtype=np.float64) / 255.0
        rough_arr = gray if is_rough else (1.0 - gray)
        _log(f"[texture_apply] {'roughness' if is_rough else 'specular->roughness'}: "
             f"{gray.shape[1]}x{gray.shape[0]}")
    elif folder:
        _log("[texture_apply] no roughness/specular map detected")

    # Occlusion from height (optional): soft cavity term into ORM R.
    occ_arr = None
    if disp_path and os.path.exists(disp_path) and occ_from_h:
        try:
            h_img = Image.open(disp_path).convert("L")
            r = max(8, min(h_img.size) // 32)
            soft = h_img.filter(ImageFilter.GaussianBlur(radius=r))
            soft = np.asarray(soft, dtype=np.float64) / 255.0
            occ_arr = 0.65 + 0.35 * (soft - soft.min()) / max(soft.max() - soft.min(), 1e-6)
            _log("[texture_apply] occlusion derived from height map")
        except Exception as e:
            _log(f"[texture_apply] occlusion skipped: {e}")

    # Mask regions (optional): painted maskmap.png snaps to palette colors;
    # each color overrides metallic and scales roughness. Albedo untouched.
    metal_arr = None
    rough_scale_arr = None
    if mask_path and os.path.exists(mask_path):
        if rough_arr is None and _mask_active:
            _log("[texture_apply] maskmap needs a roughness map — using neutral "
                 "1.0 base so region scales still apply")
            rough_arr = np.ones((tex.size[1], tex.size[0]), dtype=np.float64)
        if rough_arr is not None:
            try:
                _report(68, "Applying mask regions", os.path.basename(mask_path))
                H, W = rough_arr.shape
                reg = _mask_regions(mask_path, W, H)
                metal_arr = np.full((H, W), float(metallic), dtype=np.float64)
                rough_scale_arr = np.ones((H, W), dtype=np.float64)
                _cov = []
                for _i, (_n, _c) in enumerate(_MASK_PALETTE):
                    _m = reg == _i
                    _frac = float(_m.mean() * 100)
                    if _frac > 0.01:
                        metal_arr[_m] = _mask_metals[_i]
                        rough_scale_arr[_m] = _mask_roughs[_i]
                        _cov.append(f"{_n} {_frac:.1f}% (metal {_mask_metals[_i]:g}, "
                                    f"rough x{_mask_roughs[_i]:g})")
                if _cov:
                    _log("[texture_apply] mask regions: " + "; ".join(_cov))
                else:
                    _log("[texture_apply] maskmap is all default (white) — no overrides")
                    metal_arr = None
                    rough_scale_arr = None
            except Exception as e:
                _log(f"[texture_apply] mask regions skipped: {e}")
                metal_arr = None
                rough_scale_arr = None
        elif folder:
            _log("[texture_apply] maskmap found but no roughness map — regions need a roughnessmap to take effect")
    elif folder:
        _log("[texture_apply] no maskmap detected")

    _report(70, "Applying textures to mesh", "mapping images onto UVs")
    material = trimesh.visual.texture.SimpleMaterial(
        image=tex, diffuse=(255, 255, 255))
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv, image=tex, material=material)

    _report(85, "Exporting PBR mesh", "writing GLB")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if normal_arr is not None or rough_arr is not None:
        import tempfile as _tf
        _fd, _tmp = _tf.mkstemp(suffix=".glb", prefix="hunyuan_pbr_base_")
        os.close(_fd)
        try:
            mesh.export(_tmp)
            _attach_pbr(_tmp, str(output_path),
                        normal_arr=normal_arr, rough_arr=rough_arr,
                        occ_arr=occ_arr, metallic=metallic,
                        normal_scale=normal_scale,
                        metal_arr=metal_arr,
                        rough_scale_arr=rough_scale_arr)
            attached = ([ "normal" ] if normal_arr is not None else []) + \
                       ([ "metallicRoughness" ] if rough_arr is not None else []) + \
                       ([ "mask-regions" ] if metal_arr is not None else [])
            _log(f"[texture_apply] PBR slots: albedo + {', '.join(attached)}")
        finally:
            try:
                os.remove(_tmp)
            except OSError:
                pass
    else:
        mesh.export(output_path)

    _report(100, "Done", "")
    print(json.dumps({"type": "done", "output_path": output_path}), flush=True)


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "{}"
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as f:
            args = json.load(f)
    else:
        args = json.loads(raw)
    try:
        apply_texture(args)
    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
