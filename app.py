# =======================================================
#  ComfyUI + FLUX on Modal – V11.1 CLEAN (2025)
#  - Removed heavy cairo-related deps (svglib, python-magic, pycocotools)
#  - Do NOT run pip install -r requirements.txt to avoid pycairo builds
#  - Preinstall essential Python deps so ComfyUI Manager can add nodes later
# =======================================================

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
import requests

import modal
from huggingface_hub import hf_hub_download, login as hf_login

# -----------------------
# Basic config
# -----------------------
app = modal.App("comfyui-2025")

DATA_ROOT = "/data/comfy"
BASE = Path(DATA_ROOT, "ComfyUI")

GPU = "L4"  # change if you want different GPU type

vol = modal.Volume.from_name("comfyui-vol", create_if_missing=True)

# -----------------------
# Image: minimal but with needed libs
# -----------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        # system tools + ffmpeg + libs for PyAV
        "git",
        "wget",
        "unzip",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        # PyAV build/runtime libs
        "libavformat-dev",
        "libavcodec-dev",
        "libavutil-dev",
        "libswscale-dev",
        "libswresample-dev",
        "libavfilter-dev",
        # pkg-config & build-essential (helps some Python wheels)
        "pkg-config",
        "build-essential",
        "cmake",
    )
    .run_commands(
        "python3 -m pip install --upgrade pip wheel setuptools",
        # CUDA wheel selection: adjust index-url if different GPU/CUDA
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        # core tooling
        "huggingface_hub[hf_transfer]",
        "requests",
        "tqdm",
        "einops",
        "numpy",
        "pillow",
        "psutil",
        "safetensors",

        # video
        "av",

        # image processing
        "opencv-python",
        "scipy",
        "kornia",

        # comfy frontend essentials (avoid heavy optional deps)
        "alembic",
        "pydantic-settings",
        "comfyui-embedded-docs",
        "comfyui-workflow-templates",

        # node / runtime helpers
        "packaging",
        "aiohttp",
        "lmdb",
        "pytorch_lightning",
        "torchsde",        # flow samplers & KJNodes
        "piexif",          # Impact-pack / image metadata

        # face / onnx
        "insightface",
        "onnxruntime-gpu",

        # segmentation / utilities
        "segment-anything",
        "spandrel",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# -----------------------
# Helpers
# -----------------------
def run(cmd: str, cwd: str | None = None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def hf_download(subdir: str, filename: str, repo: str):
    """Download a file from HuggingFace with token auth."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing (Modal secret huggingface-secret must export HF_TOKEN).")
    hf_login(token)
    dest = BASE / "models" / subdir
    dest.mkdir(parents=True, exist_ok=True)
    print(f"⬇️ HF: {repo}/{filename}")
    tmp = hf_hub_download(
        repo_id=repo,
        filename=filename,
        token=token,
        local_dir="/tmp",
        local_dir_use_symlinks=False,
    )
    shutil.move(tmp, dest / filename)


def civitai_download(model_id: str, filename: str, subdir: str = "loras"):
    token = os.getenv("CIVITAI_TOKEN")
    if not token:
        raise RuntimeError("CIVITAI_TOKEN missing (Modal secret civitai-token must export CIVITAI_TOKEN).")
    url = f"https://civitai.com/api/v1/model-versions/{model_id}/download"
    print(f"⬇️ Civitai: {model_id} -> {filename}")
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if r.status_code != 200:
        raise RuntimeError(f"Civitai download failed: {r.status_code} {r.text}")
    dest = BASE / "models" / subdir
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / filename, "wb") as f:
        f.write(r.content)


# -----------------------
# Setup: clone ComfyUI, install minimal extras, download flux models
# Note: we DO NOT run `pip install -r requirements.txt` here to avoid heavy builds.
# -----------------------
@app.function(
    gpu=GPU,
    volumes={DATA_ROOT: vol},
    timeout=1800,
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("civitai-token"),
    ],
)
def setup():
    print("\n===== SETUP START =====\n")

    # authenticate HF
    hf_login(os.getenv("HF_TOKEN"))

    # clone or update ComfyUI
    if not (BASE / "main.py").exists():
        print("📥 Cloning ComfyUI...")
        BASE.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone https://github.com/comfyanonymous/ComfyUI {BASE}")
    else:
        print("🔄 Updating ComfyUI...")
        run("git pull --ff-only", cwd=BASE)

    # OPTIONAL: Install a few local extras from the repo if present
    # (We avoid running requirements.txt to prevent heavy builds)
    local_reqs = BASE / "requirements.txt"
    if local_reqs.exists():
        print("⚠️ Found requirements.txt in repo but skipping full install to avoid heavy builds.")
        print("If you need extras, install them manually or update the image pip_install list.")

    # Install buffalo_l for face utilities (light-weight download)
    print("\n📦 Installing buffalo_l model...")
    face_dir = Path(DATA_ROOT, ".insightface", "models")
    face_dir.mkdir(parents=True, exist_ok=True)
    if not (face_dir / "buffalo_l").exists():
        zipf = face_dir / "buffalo_l.zip"
        run(
            f"wget -q -O {zipf} "
            "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        )
        with zipfile.ZipFile(zipf) as z:
            z.extractall(face_dir)
        zipf.unlink()

    # Download FLUX files (gated repos require HF token with access)
    print("\n📦 Downloading FLUX models (if token has access)...")
    flux_list = [
        ("checkpoints", "flux1-dev-fp8.safetensors", "camenduru/FLUX.1-dev"),
        ("vae/FLUX", "ae.safetensors", "black-forest-labs/FLUX.1-dev"),
        ("clip/FLUX", "clip_l.safetensors", "black-forest-labs/FLUX.1-dev"),
        ("clip/FLUX", "text_encoder.t5xxl.fp8_e4m3fn.safetensors", "black-forest-labs/FLUX.1-dev"),
    ]
    for sub, fn, repo in flux_list:
        try:
            hf_download(sub, fn, repo)
        except Exception as e:
            print(f"⚠️ Skipped {fn} — {e}")

    # commit volume
    vol.commit()
    print("\n===== SETUP DONE =====\n")


# -----------------------
# Launch ComfyUI
# -----------------------
@app.function(
    gpu=GPU,
    volumes={DATA_ROOT: vol},
    timeout=86400,
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("civitai-token"),
    ],
)
def launch():
    print("\n🔥 Launching ComfyUI...\n")
    os.chdir(BASE)

    # run main; logs are streamed to Modal (and to your modal run)
    proc = subprocess.Popen(
        ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        print(line, end="")

    proc.wait()
    print(f"\n⚠️ ComfyUI exited with code {proc.returncode}")
