import os
import subprocess
import modal
from huggingface_hub import snapshot_download

# ============================================================
#  IMAGE — fix fastapi, uvicorn, huggingface_hub, comfy deps
# ============================================================
image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "uvicorn",
        "huggingface_hub[hf_transfer]",
        "requests",
        "tqdm",
        "safetensors",
        "numpy",
        "pillow"
    )
)

# ============================================================
#  APP CONFIG
# ============================================================
app = modal.App("comfyui-2025", image=image)

GPU = "A100-40GB"
VOL = modal.Volume.from_name("comfy-vol", create_if_missing=True)

DATA = "/data/comfy"
COMFY = f"{DATA}/ComfyUI"
MODEL_DIR = f"{COMFY}/models"
CHECKPOINTS = f"{MODEL_DIR}/checkpoints"


# ============================================================
#  UTIL
# ============================================================
def run(cmd, cwd=None):
    print(f"▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


# ============================================================
#  AUTO-DOWNLOAD BASE MODEL (FLUX .safetensors)
# ============================================================
def ensure_models():
    """Download base model jika belum ada"""
    os.makedirs(CHECKPOINTS, exist_ok=True)

    flux_ckpt = f"{CHECKPOINTS}/flux-dev.safetensors"

    if not os.path.exists(flux_ckpt):
        print("📥 Downloading FLUX checkpoint (.safetensors)...")

        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev.safetensors",
            local_dir=CHECKPOINTS,
            local_dir_use_symlinks=False,
            token=os.environ["HF_TOKEN"]
        )

        print(f"✔ FLUX checkpoint saved to: {flux_ckpt}")
    else:
        print(f"✔ FLUX checkpoint sudah ada: {flux_ckpt}")


# ============================================================
#  SETUP — INSTALL COMFYUI
# ============================================================
@app.function(
    timeout=3600,
    volumes={DATA: VOL},
)
def setup():
    print("🚀 Setup ComfyUI mulai...\n")

    os.makedirs(DATA, exist_ok=True)

    if not os.path.exists(COMFY):
        print("📥 Clone ComfyUI...")
        run(f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY}")
    else:
        print("✔ ComfyUI sudah ada, skip clone.")

    print("\n📦 Install requirements...")
    run("python3 -m pip install --upgrade pip", cwd=COMFY)
    run("python3 -m pip install -r requirements.txt", cwd=COMFY)

    print("\n🎉 SETUP SELESAI\n")
    return "Setup OK"


# ============================================================
#  LAUNCH — ASGI
# ============================================================
@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={DATA: VOL},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.asgi_app()
def launch():
    from fastapi import FastAPI
    import threading

    api = FastAPI()

    @api.get("/")
    def home():
        return {
            "status": "running",
            "info": "ComfyUI berjalan di port 8188",
            "docs": "/docs"
        }

    def start_comfy():
        print("🔥 Menjalankan ComfyUI...")

        # 🔥 Auto-download FLUX sebelum ComfyUI jalan
        ensure_models()

        os.chdir(COMFY)
        run("python3 main.py --listen 0.0.0.0 --port 8188")

    threading.Thread(target=start_comfy, daemon=True).start()

    return api
