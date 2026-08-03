"""Test script: run the upstream Hunyuan3D-2.0 paint pipeline with ALL defaults.

This calls pipeline.__call__() directly — no custom UV unwrap, no quad remesh,
no view weight overrides, no median replacement. This is the baseline.

Usage:
    python test_default_pipeline.py --mesh path/to/mesh.glb --image path/to/image.png --output default_output.glb
"""

import argparse
import sys
import os

# Ensure the venv's hy3dgen is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages"))


def main():
    parser = argparse.ArgumentParser(description="Run upstream Hunyuan3D-2.0 paint pipeline (defaults only)")
    parser.add_argument("--mesh", required=True, help="Path to input GLB mesh")
    parser.add_argument("--image", required=True, help="Path to conditioning image (PNG/JPG)")
    parser.add_argument("--output", default="default_output.glb", help="Path for output GLB")
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2",
                        help="HuggingFace model path or local directory")
    parser.add_argument("--subfolder", default="hunyuan3d-paint-v2-0-turbo",
                        help="Model subfolder (hunyuan3d-paint-v2-0 or hunyuan3d-paint-v2-0-turbo)")
    args = parser.parse_args()

    if not os.path.exists(args.mesh):
        print(f"Error: mesh not found: {args.mesh}")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    import trimesh
    from PIL import Image
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    print(f"Loading mesh: {args.mesh}")
    mesh = trimesh.load(args.mesh)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if hasattr(g, "faces")]
        mesh = geoms[0] if geoms else None
    if mesh is None or not hasattr(mesh, "faces"):
        print("Error: mesh has no geometry")
        sys.exit(1)
    print(f"  {len(mesh.vertices)} verts / {len(mesh.faces)} faces")

    print(f"Loading conditioning image: {args.image}")
    image = Image.open(args.image)
    print(f"  {image.size[0]}x{image.size[1]} {image.mode}")

    print(f"Loading pipeline from: {args.model_path} ({args.subfolder})")
    pipeline = Hunyuan3DPaintPipeline.from_pretrained(args.model_path, subfolder=args.subfolder)

    # Print the default config so we can see exactly what's being used.
    cfg = pipeline.config
    print("\n=== Default Pipeline Config ===")
    print(f"  camera_azims:    {cfg.candidate_camera_azims}")
    print(f"  camera_elevs:    {cfg.candidate_camera_elevs}")
    print(f"  view_weights:    {cfg.candidate_view_weights}")
    print(f"  bake_exp:        {cfg.bake_exp}")
    print(f"  merge_method:    {cfg.merge_method}")
    print(f"  render_size:     {cfg.render_size}")
    print(f"  texture_size:    {cfg.texture_size}")
    print(f"  bake_angle_thres: {pipeline.render.bake_angle_thres}")
    print("================================\n")

    print("Running upstream pipeline.__call__() ...")
    textured_mesh = pipeline(mesh, image)

    print(f"Saving textured mesh: {args.output}")
    textured_mesh.export(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
