import os
import subprocess
from pathlib import Path
import threading

import modal
from modal import asgi_app

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# -------------------
# CONFIG
# -------------------
APP_NAME = "comfyui-2025"
DATA_ROOT = "/data/comfy"
BASE = "/data/comfy/ComfyUI"

GPU = "A100-40GB"   # versi kompatibel universal

# -------------------
# IMAGE
# -------------------
image = (
    modal.Image.debian_slim()
    .apt_install(
        [
            "git",
            "ffmpeg",
            "libgl1",
            "libglib2.0-0",
        ]
    )
    .pip_install(
        [
            "fastapi",
            "uvicorn",
            "aiohttp",
            "alembic",
            "einops",
            "huggingface_hub[hf_transfer]",
            "insightface",
            "kornia",
            "lmdb",
            "numpy",
            "onnxruntime-gpu",
            "opencv-python",
            "packaging",
            "piexif",
            "pillow",
            "psutil",
            "pydantic-settings",
            "pytorch_lightning",
            "requests",
            "safetensors",
            "scipy",
            "segment-anything",
            "spandrel",
            "torchsde",
            "tqdm",
            "av",
            "comfyui-embedded-docs",
            "comfyui-workflow-templates",
        ]
    )
)

vol = modal.Volume.from_name("comfy-volume", create_if_missing=True)
app = modal.App(APP_NAME)

# -------------------
# ASGI FastAPI
# -------------------
web = FastAPI()

@web.get("/")
def home():
    return RedirectResponse("/comfy")

@web.get("/status")
def status():
    return {"status": "running"}

# -------------------
# Setup (clone ComfyUI + models)
# -------------------
@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA_ROOT: vol},
    timeout=10000,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def setup():
    import huggingface_hub as hf

    print("📥 Setup ComfyUI...")
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.chdir(DATA_ROOT)

    # Clone ComfyUI
    if not Path(BASE).exists():
        subprocess.run(["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git"], check=True)

    # Download FLUX models
    print("📥 Downloading FLUX models...")
    hf.login(token=os.environ.get("HF_TOKEN"))

    flux_files = [
        "ae.safetensors",
        "flux1-dev.safetensors",
        "flux_text_encoder/model.safetensors",
        "flux_text_encoder/config.json",
    ]

    for f in flux_files:
        dst = os.path.join(BASE, "models/flux", f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        hf.hf_hub_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            filename=f,
            local_dir=os.path.dirname(dst),
            local_dir_use_symlinks=False,
        )

    vol.commit()
    print("✅ Setup complete!")

# -------------------
# Launch (ASGI endpoint + background ComfyUI)
# -------------------
@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA_ROOT: vol},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@asgi_app()   # <--- WAJIB versi MODAL kamu
def launch():
    print("🔥 Launching ComfyUI...")

    os.chdir(BASE)

    # Jalankan ComfyUI sebagai thread background
    def start_backend():
        subprocess.Popen(
            ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    t = threading.Thread(target=start_backend, daemon=True)
    t.start()

    print("🚀 Backend berjalan di port 8188")
    print("👉 Akses UI via /comfy")

    # return ASGI app
    return web
