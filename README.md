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

### Key settings

| Setting | Description |
|---|---|
| Provider | `cuda` (default), `tensorrt` (fastest NVIDIA), `cpu` |
| Enhancer | Post-processing: GPEN, GFPGAN, CodeFormer, DMDNet, RestoreFormer++ |
| Restore original mouth | Composites the target's original mouth back over the swap |
| Video swapping method | **In-Memory** (fast, more RAM) or **Extract Frames** (large videos) |
| Subsample upscale | Internal face resolution: 128 → 256 → 512 px |

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

**Video upload broken after server restart**
Gradio temp files from the previous session cause an asyncio event loop mismatch. Restart the
app fully (stop and start again from Pinokio).

---

## Project Structure

```
roop-unleashed-wip/
├── app/                    # Application code
│   ├── roop/               # Core processing (face swap, processors, utilities)
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
