# roop-unleashed

A deepfake face-swap application for images and videos with an easy-to-use Gradio web UI.
Supports NVIDIA (CUDA / TensorRT), AMD (DirectML / ROCm), Apple Silicon, and CPU.

This repository contains both the Pinokio launcher scripts and the full application code.

---

## Installation — Option 1: Pinokio (Recommended)

[Pinokio](https://pinokio.computer) automates the entire install in one click.

1. Open Pinokio and click **Discover** (or paste the repo URL directly).
2. Paste the repo URL:
   ```
   https://github.com/Adutchguy/roop-unleashed-wip.git
   ```
3. Click **Download**, then **Install**.
4. Once installed, click **Start** to launch the web UI.

Pinokio will automatically detect your GPU and install the correct PyTorch and ONNX Runtime variant.

---

## Installation — Option 2: Manual (GitHub)

### Prerequisites

| Requirement | Notes |
|---|---|
| Python **3.10** | Other versions may break onnxruntime |
| Git | For cloning |
| ffmpeg | Required for video processing — [download](https://ffmpeg.org/download.html) |
| CUDA 12.8 Toolkit | NVIDIA GPU users only |
| TensorRT | Optional — fastest NVIDIA inference |

### Step 1 — Clone the repository

```bash
git clone https://github.com/Adutchguy/roop-unleashed-wip.git
cd roop-unleashed-wip
```

### Step 2 — Create a virtual environment

```bash
cd app
python -m venv env
```

Activate it:

```bash
# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install PyTorch + ONNX Runtime

Choose the command set that matches your hardware:

#### NVIDIA GPU (CUDA 12.8)
```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
pip install onnxruntime-gpu==1.19.0
```

#### AMD GPU — Windows (DirectML)
```bash
pip install torch torch-directml torchvision torchaudio --force-reinstall
pip install onnxruntime-directml
```

#### AMD GPU — Linux (ROCm 6.3)
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.3 --force-reinstall --no-deps
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/onnxruntime_rocm-1.19.0-cp310-cp310-linux_x86_64.whl
```

#### Apple Silicon (M1 / M2 / M3)
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
pip install onnxruntime-silicon==1.16.3
```

#### CPU only
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps
pip install onnxruntime==1.17.1
```

### Step 5 — (Optional) TensorRT acceleration — NVIDIA only

TensorRT provides the fastest inference on NVIDIA GPUs. Requires CUDA 12.x.

```bash
pip install tensorrt-cu12
```

Then open the Settings tab in the UI and change **Provider** to `tensorrt`.

> **Note:** On first use with TensorRT, each ONNX model is compiled to a TRT engine. This takes
> several minutes per model but is cached in `app/models/trt_cache/` — subsequent starts are instant.

### Step 6 — Run

```bash
python run.py
```

The Gradio UI opens at [http://127.0.0.1:7860](http://127.0.0.1:7860).

Models are downloaded automatically on first launch via the InsightFace model downloader.
Additional enhancement models (GFPGAN, GPEN, CodeFormer, etc.) can be downloaded from the
**Settings** tab inside the app.

---

## Usage

1. **Source Images / Facesets** — Upload one or more face images (or `.fsz` faceset files).
2. **Target File(s)** — Upload the image or video to apply the swap to.
3. Select a source face from the gallery and a target face (or use **All faces** mode).
4. Optionally enable **Face swap frames** preview to see the result before processing.
5. Click **▶ Start**.

---

## Features

### Face Swap Modes

| Mode | Description |
|---|---|
| First found | Swaps the first detected face in each frame |
| All faces | Swaps every detected face using the selected source |
| All input faces | Maps each input face to each detected target face in order |
| Selected face | Swaps only the specific target face chosen in the gallery |
| All female / All male | Swaps all faces matching the detected gender |

### Enhancers (post-processing)

Applied after the swap to sharpen and restore quality, especially around the eyes and mouth.

| Enhancer | Notes |
|---|---|
| GPEN | Fast, good general quality (default) |
| GFPGAN | Good for natural skin tone |
| CodeFormer | Strong detail recovery, slower |
| DMDNet | Sharpest output, best for high-res targets |
| RestoreFormer++ | High fidelity, slower |

### Masking

- **DFL XSeg** — Neural mask that follows the face outline, excluding hair and background.
- **Clip2Seg** — Text-guided mask (e.g. `hair,hands`) to exclude specific regions.
- **Manual canvas mask** — Draw include/exclude regions directly on the face in the preview. Supports per-frame masks for fine-grained control on video.
- **Mask offsets** — Sliders for top/bottom/left/right crop and edge-blend amount.

### Face Detection & Stability

**Detection score threshold** (Settings tab) — Lowers or raises the confidence required for a face to be detected. Reducing this below the default 0.5 can recover faces the detector would otherwise miss, reducing flicker in video output.

**Temporal ROI hint** — When the detector misses a face in a frame, the last known bounding box is used to crop and re-detect in that region, preventing stutters from brief detection failures mid-video.

**No-face action** — Controls what happens when no face is found in a frame:

| Option | Behaviour |
|---|---|
| Use untouched original frame | Output the original frame unmodified |
| Retry rotated | Try rotating the frame 90° each way before giving up |
| Skip frame | Drop the frame (may cause video length mismatch) |
| Skip frame if no similar face | Skip only when no face matches the target embedding |
| Use last swapped | Reuse the last successfully swapped frame (up to 15 frames) |

### Mouth Controls

**Restore original mouth** — Composites the target's original mouth region back over the swap result. Useful when the swap distorts lips or teeth.

**Inner Mouth Blend** — Independently blends the original target's teeth and tongue back over the swap output within the inner lip boundary only. More surgical than "Restore original mouth" — the outer lips stay swapped, only the inner mouth opening is blended. Controlled by a 0–1 slider (0 = off, 1 = full original inner mouth).

### Expression Warp

Upload an **Expression Reference** image (any photo showing a face with the desired expression — happy, sad, neutral, surprised, etc.) and set the **Expression Warp Strength** slider to warp the swapped face toward that expression.

The warp uses the insightface 106-point 2D landmarks already present in the pipeline — no additional model downloads required. A Thin Plate Spline (TPS) interpolation shifts expression-relevant landmarks (mouth, eyebrows, eye openings) while anchoring the face boundary so the face shape does not drift.

- **Strength 0** — no effect (default)
- **Strength 0.5** — partial expression blend
- **Strength 1.0** — full warp toward the reference expression

The expression reference is processed once per batch, so there is no per-frame overhead beyond the warp itself.

### 3D Pose Features (Experimental)

**3D source pose matching** — Warps the source face crop to approximate the target's head pose before ArcFace embedding extraction. Improves swap quality on profile and angled faces by reducing the pose mismatch seen by the swap model.

**Multi-angle source bank** — When a `.fsz` faceset contains multiple images of the same person at different angles, automatically selects the image whose pose best matches each target frame.

### Subsample Pixel Boost

Internally upscales the aligned face before the swap model runs, then downsamples after, effectively running the swap at higher resolution.

| Setting | Internal resolution | Notes |
|---|---|---|
| 128 px | 128 × 128 | Model native, fastest |
| 256 px | 256 × 256 | Good balance (default) |
| 512 px | 512 × 512 | Highest quality, slower |

### Other Options

- **VR mode** — Processes side-by-side VR video frames.
- **Auto-rotate faces** — Detects and corrects sideways faces before swapping.
- **Skip audio** — Strip audio from video output.
- **Keep frames** — Retain extracted frames after video assembly.
- **Output method** — File (default), Virtual Camera, or Both.
- **Video swapping method** — In-Memory processing (fast, higher RAM) or Extract Frames (safer for large videos).

---

## Settings

The Settings tab exposes all persistent options including provider, enhancer defaults, output format, face detection threshold, and mouth blend values. Settings are saved to a YAML config file and restored on next launch.

---

## Updating

### Pinokio
Click **Update** in the sidebar.

### Manual
```bash
git pull
pip install -r app/requirements.txt
```

---

## Resetting / Reinstalling

### Pinokio
Click **Reset** in the sidebar — this removes `app/` and lets you reinstall from scratch.

### Manual
Delete `app/env/` and recreate the virtual environment (Steps 2–4 above).

---

## Troubleshooting

**Uploads or preview appear to hang on first TensorRT use**
TRT is compiling ONNX models to engine files on first run. Let it complete — it will finish.
After the first run, engines are cached at `app/models/trt_cache/` and load instantly.

**Garbled / corrupt output with TensorRT**
Delete `app/models/trt_cache/` entirely and restart so engines recompile in FP32.

**`onnxruntime_providers_cuda.dll` Error 126 on Windows**
Use `onnxruntime-gpu==1.19.0` exactly. Newer versions have DLL dependency issues on Windows.

**ffmpeg not found**
Install ffmpeg and ensure it is on your system PATH. [Download here](https://ffmpeg.org/download.html).

**Video flickers or face disappears on some frames**
Lower the **Detection Score Threshold** in Settings (try 0.35–0.4). The temporal ROI hint will also retry detection in the last known face region automatically.

**Video upload broken after server restart**
Gradio temp files from the previous session cause an asyncio event loop mismatch. Restart the
app fully (stop and start again from Pinokio).

---

## Project Structure

```
roop-unleashed-wip/
├── app/                    # Application code
│   ├── roop/               # Core processing
│   │   ├── ProcessMgr.py       # Main frame-processing pipeline
│   │   ├── ProcessOptions.py   # Per-run configuration
│   │   ├── expression_reenact.py  # TPS expression warp module
│   │   ├── face_3d_recon.py    # Pose estimation and source crop warping
│   │   ├── face_frontalize.py  # Target frontalization
│   │   ├── face_util.py        # Face detection / alignment helpers
│   │   └── processors/         # Swap, mask, and enhance processor plugins
│   ├── ui/                 # Gradio web UI
│   ├── models/             # Downloaded model weights (gitignored)
│   ├── requirements.txt    # Python dependencies
│   └── run.py              # Entry point
├── install.js              # Pinokio install script
├── start.js                # Pinokio start script
├── update.js               # Pinokio update script
├── reset.js                # Pinokio reset script
├── torch.js                # Cross-platform PyTorch installer
├── pinokio.js              # Pinokio UI definition
└── README.md               # This file
```

---

## License

See [app/LICENSE](app/LICENSE).
