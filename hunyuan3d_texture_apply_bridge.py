"""Minimal texture applicator bridge.

Takes an already-UV-unwrapped mesh (GLB) plus a texture image (PNG/JPG — e.g.
an externally upscaled UV atlas) and maps the image onto the mesh's existing
UVs, exporting a new GLB. Pure trimesh + PIL: no GPU, no diffusion, no bake.

Inputs (JSON arg):
  mesh_path:     path to a GLB that already has UV coordinates
  texture_path:  path to a texture atlas image (PNG/JPG)
  output_path:   where to write the textured GLB
"""
import json
import os
import sys

import trimesh
from PIL import Image


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


def apply_texture(args):
    mesh_path = args.get("mesh_path", "")
    texture_path = args.get("texture_path", "")
    output_path = args.get("output_path", "output.glb")

    if not mesh_path or not os.path.exists(mesh_path):
        raise RuntimeError(f"mesh_path not found: {mesh_path}")
    if not texture_path or not os.path.exists(texture_path):
        raise RuntimeError(f"texture_path not found: {texture_path}")

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
        # Some GLBs store UVs under a mesh.visual.uv with wrong dtype/shape;
        # try to recover via the mesh's TextureVisuals directly.
        raise RuntimeError(
            "Mesh has no UV coordinates — it must be UV-unwrapped before a "
            "texture can be applied. Run the 'Texture Mesh' node first."
        )
    _report(50, "Loading texture", os.path.basename(texture_path))
    tex = Image.open(texture_path).convert("RGB")
    _log(f"[texture_apply] texture: {tex.size[0]}x{tex.size[1]}")

    _report(70, "Applying texture to mesh", "mapping image onto UVs")
    material = trimesh.visual.texture.SimpleMaterial(
        image=tex, diffuse=(255, 255, 255))
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv, image=tex, material=material)

    _report(90, "Exporting textured mesh", "writing GLB")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
