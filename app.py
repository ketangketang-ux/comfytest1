# ===========================
#  ComfyUI on Modal – CLEAN
#  FASTAPI + ASGI Version V15
# ===========================

import os
import subprocess
from pathlib import Path
import modal
from modal import asgi_app
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# ----------------------
# CONFIG
# ----------------------
APP_NAME = "comfyui-2025"
BASE = "/data/comfy/ComfyUI"
DATA_ROOT = "/data/comfy"

GPU = modal.gpu.A100()

# ----------------------
# BUILD IMAGE
# ----------------------
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
            # FastAPI + Uvicorn (WAJIB)
            "fastapi",
            "uvicorn",

            # Core deps required by Comfy
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

# ----------------------
# VOLUME
# ----------------------
vol = modal.Volume.from_name("comfy-volume", create_if_missing=True)

app = modal.App(APP_NAME)

# ----------------------
# ASGI WEB APP
# ----------------------
web = FastAPI()

@web.get("/")
def index():
    # alihkan user ke frontend ComfyUI
    return RedirectResponse("/comfy")

@web.get("/status")
def status():
    return {"status": "running"}

# ----------------------
# SETUP: Clone ComfyUI + install model FLUX
# ----------------------
@app.function(
    image=image,
    gpu=GPU,
    timeout=6000,
    volumes={DATA_ROOT: vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("civitai-token"),
    ],
)
def setup():
    import huggingface_hub as hf

    print("\n📥 Setup ComfyUI...")

    os.makedirs(DATA_ROOT, exist_ok=True)
    os.chdir(DATA_ROOT)

    # Clone ComfyUI
    if not Path(BASE).exists():
        subprocess.run(
            ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git"],
            check=True,
        )

    # Apply symlink fix
    models_dir = "/data/models"
    os.makedirs(models_dir, exist_ok=True)

    # Download FLUX models
    print("📥 Downloading FLUX models...")
    hf.login(token=os.environ.get("HF_TOKEN"))

    for fn in [
        "ae.safetensors",
        "flux1-dev.safetensors",
        "flux_text_encoder/model.safetensors",
        "flux_text_encoder/config.json",
    ]:
        dst = f"{BASE}/models/flux/{fn}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        hf.hf_hub_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            filename=fn,
            local_dir=os.path.dirname(dst),
            local_dir_use_symlinks=False,
        )

    vol.commit()
    print("✅ Setup selesai!")

# ----------------------
# LAUNCH → ASGI WEB FUNCTION + backend ComfyUI
# ----------------------
@app.function(
    image=image,
    gpu=GPU,
    timeout=86400,
    volumes={DATA_ROOT: vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("civitai-token"),
    ],
)
@asgi_app(web)
def launch():
    print("\n🔥 Launching ComfyUI backend...\n")

    os.chdir(BASE)

    # Start Comfy backend
    subprocess.Popen(
        ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("🚀 ComfyUI backend berjalan di dalam container.")
    print("👉 Akses UI di /comfy dari URL Modal!")

    # Return ASGI web app
    return web
