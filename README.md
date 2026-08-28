# Hunyuan3D-2mv — Modly Extension (v2.1)

Generate textured 3D meshes from images using Tencent's Hunyuan3D-2mv. Three nodes:

| Node | What it does |
|------|-------------|
| **Generate 3D Mesh** | Creates a 3D shape (untextured GLB mesh) from 1–6 photos of an object |
| **Texture Mesh** | Paints a texture onto an existing mesh using reference photos |
| **Apply Texture** | Maps an existing UV atlas (e.g. externally super-resolved) onto a UV'd mesh — no diffusion |

### What's new in v2.1
- **Self-contained install** — `setup.py` now installs *every* runtime dependency (torch + hy3dgen + scipy + scikit-image + onnxruntime + rembg + trimesh + …) into the isolated venv, including a `pip install .` packaging mode for GitHub users. No manual dependency steps.
- **Weights stay in Modly** — all model weights are still downloaded through the extension's nodes in Modly's **Extensions → model** view (per-node `hf_repo` / download check). Nothing model-related ships in the repo.
- **First-load bridge** — new `hunyuan3d_bootstrap.py` runs on first load after install/download and bridges anything living *outside* the extension dir:
  - weights Modly placed in any node model dir / HF hub cache are hardlinked into the layout the pipeline expects (instant, zero extra disk, fully offline afterwards),
  - the venv's `custom_rasterizer` CUDA kernel is re-installed from the bundled build if missing/stale,
  - a state file (`.bridge_state.json`, gitignored) makes every later run a no-op.
- **Bridges stay authoritative** — both generation bridges (shape + texture) re-run the (idempotent) first-load bridge right before loading models, so any change since the last run is patched into out-of-extension files automatically on the first generate.

### What was new in v2.0
- **Automatic background removal** — rembg runs on every input image before processing. No toggle needed.
- **Image Folder mode** — point to a folder of photos; filenames (`front`, `left`, `back`, `right`) determine grid placement. Auto-tiles and saves `folder_tiled.png` for reuse.
- **Hybrid Reference Mode** — directly uses your reference images for front/left/back/right views and only synthesizes top/bottom. Requires orthographic/isometric references.
- **Delight toggle** — turn off to skip the delight model (~1.5 GB VRAM saved) and keep original colors.
- **Progress tracking** — multiview diffusion step progress shown in the node status.
- **Debug views** — per-view split outputs saved to a `views_split` folder for troubleshooting.
- **Texture white-background normalization** — reference images for the texture multiview diffusion are always composited onto white.
- **Adjustable diffusion steps** — both mesh generation and texture multiview diffusion expose step counts as numeric parameters (1-60 for mesh, 5-60 for texture).

---

## Updating / Installing from GitHub

This repo is the canonical source: `https://github.com/iammojogo-sudo/hunyuan3d-2mv-2.1_modly`

1. **Add the extension in Modly's Extensions tab** (paste the repo URL) — Modly downloads it and re-runs `setup.py`, which installs every dependency (torch, hy3dgen, rembg, …) into the isolated venv. No git CLI or manual dependency steps are needed.
2. **Download the model weights** in the Extensions → model view (per-node **Download** button) — ~15 GB total from public Hugging Face repos.
3. **Model weights do not need re-downloading on updates** — they already live in Modly's `models/` folder; the first-load bridge links them into place automatically.

For a manual install:

```
pip install https://github.com/iammojogo-sudo/hunyuan3d-2mv-2.1_modly/archive/refs/heads/main.zip
```

install_requires covers every runtime dependency (use the CUDA torch wheel separately on GPU machines).

## Background Removal (rembg)

All input images automatically have their background removed via rembg before being fed to the mesh or texture pipeline. This happens for every input mode:

- **Single / Tiled mode:** rembg runs on the wired image before splitting or forwarding.
- **Folder mode:** rembg runs on each individual file in the folder before padding/tiling. Intermediate files (`rembg_*.png`, `padded_*.png`) are saved in the source folder.

Background removal uses a dedicated Python venv and the ONNX Runtime runs on CPU to avoid CUDA conflicts with PyTorch.

---

## Image Input Modes

Both nodes accept images in three ways. Select via the **Image Input Mode** dropdown in the node settings:

### Single Image — `input_mode: single`
Wire in **one photo** of your subject. The whole image is used as a single front view. Best for when you only have one angle of the object.

### Tiled Image — `input_mode: tiled`
Wire in a **pre-made 2×2 grid** image (front top-left, left top-right, back bottom-left, right bottom-right). The bridge auto-splits it into 1–4 views. This is the original/default mode.

### Image Folder — `input_mode: folder`
Put your photos in a folder, paste the folder path into the **Image Folder** field (or use the folder picker). The extension auto-tiles up to 4 images into a 2×2 grid and saves the composite (`folder_tiled.png`) to the run folder for reuse.

**Folder naming guide:** The extension reads filenames to place them in the correct grid position:
| Filename contains | Grid position |
|---|---|
| `front` | Top-left |
| `left` | Top-right |
| `back` | Bottom-left |
| `right` | Bottom-right |

If filenames don't contain any of these keywords, they fill remaining slots alphabetically.

---

## Step-by-Step: Generate 3D Mesh

### Method A: Wire a single photo (easiest)
1. Drop an image node and wire it into **Generate 3D Mesh**
2. Set **Image Input Mode** → `Single Image`
3. Set **Input Views** → `1 view (front)`
4. Click Generate — background removed automatically

### Method B: Wire a tiled 2×2 image
1. Create a 2×2 grid image (front/left/back/right in reading order)
2. Wire it into **Generate 3D Mesh**
3. Set **Image Input Mode** → `Tiled Image`
4. Set **Input Views** → how many views to use (1–4)
5. Click Generate

### Method C: Use a folder of images
1. Place 1–4 photos in a folder (name them with `front`, `left`, `back`, `right` in the filename)
2. Set **Image Input Mode** → `Image Folder`
3. Paste the folder path into **Image Folder** or use the folder picker
4. Set **Input Views** → how many views to use
5. Click Generate — the composite tile (`folder_tiled.png`) is saved in the run folder for later use

### Key Parameters

| Parameter | What it does |
|-----------|-------------|
| **Quality Steps** | Number of shape-generation diffusion steps (1-60). 5-10 = turbo fast, 30 = standard high quality |
| **Mesh Resolution** | Lower = coarser mesh, less VRAM. 128–256 for 6GB cards, 380+ for 12GB+ |
| **Dual Guidance** | On = best quality but 3× slower. Off = faster, slight quality loss |
| **Input Views** | How many of the 4 views to actually feed the model |
| **Image Input Mode** | Single / Tiled / Folder — how to interpret the input |

---

## Step-by-Step: Texture Mesh

1. **Generate 3D Mesh** first, or wire in your own GLB mesh
2. Wire the mesh output into **Texture Mesh**
3. Wire reference image(s) into the second input
4. Set **Image Input Mode**:
   - `Single Image` — one reference photo
   - `Tiled Image` — a 2×2 tile with up to 4 reference views
   - `Image Folder` — folder of reference photos (auto-tiled)
5. Set **Reference Images** to how many views to use for conditioning
6. Set **Reference Mode**:
   - `Diffusion` — multiview model generates all 6 views (smoother on spheres/organic shapes)
   - `Hybrid` — directly uses your reference images for front/left/back/right and only synthesizes top/bottom (sharper alignment on boxes/hard-surface objects). **Only use Hybrid with orthographic/isometric reference views. Do not use with perspective photos or single-image inputs — choose Diffusion instead.**
7. Set **Delight**:
   - `On` — runs the delight model to normalize lighting (can wash out colors, but may look more synthetic)
   - `Off` — skips the delight model, saves ~1.5 GB VRAM, keeps original image colors
8. Set **Texture Diffusion Steps** (5-60, default 30). Lower = faster/rougher; higher = slower/potentially sharper
9. Click Generate

### Key Parameters

| Parameter | What it does |
|-----------|-------------|
| **Texture Resolution** | Higher = sharper but more VRAM. 1024 is the sweet spot |
| **Decimate Faces** | Reduces mesh face count before UV unwrap. Lower = faster, less VRAM |
| **Reference Images** | 1–4 views to condition the texture generation |
| **Reference Mode** | Diffusion = model generates every view; Hybrid = use references for known views (requires orthographic/isometric refs — not for single/perspective photos) |
| **Delight** | Off = keep real colors (saves ~1.5 GB VRAM). On = normalize lighting |
| **Texture Diffusion Steps** | Number of multiview diffusion steps for texture generation (5-60). 5 = fast/rough, 30 = default, 60 = slow/sharp |
| **Image Input Mode** | Single / Tiled / Folder |

---

## Requirements

- **VRAM:** 6 GB minimum, 8 GB+ recommended
- **GPU:** NVIDIA with CUDA
- **Disk:** ~15 GB free for model weights
- **Python:** 3.11 (installed automatically into the extension venv)
- **Dependencies:** all installed by `setup.py` (torch, hy3dgen, diffusers, transformers, rembg, scipy, scikit-image, onnxruntime, trimesh, …)

---

## Troubleshooting

### "No input image (tiled_path) found"
- Make sure you wired an image into the node
- If using **Image Folder** mode, verify the folder path is correct

### Texture bake fails / black mesh
- Wire at least one image to the Texture Mesh node
- Try **Reference Images** = 1 first, then increase
- Check that background was cleanly removed (remrg runs automatically)

### CUDA out of memory
- Lower **Mesh Resolution** to 128 or 64
- Use **Quality Steps** = 5 (Turbo)
- Turn off **Dual Guidance**
- Set **Texture Resolution** to 512 or lower
- Set **Delight** to Off (saves ~1.5 GB VRAM)

### "No CUDA GPUs are available"
Kill lingering Python processes in Task Manager and restart Modly. If persistent, your GPU driver may need a reboot.

### Images placed in wrong grid position
In **Image Folder** mode, name your files with `front`, `left`, `back`, `right` in the filename (e.g., `myobject_front.png`). The extension reads these keywords to place them correctly.
