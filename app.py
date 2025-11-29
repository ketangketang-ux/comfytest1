import os
import subprocess
import modal
from huggingface_hub import hf_hub_download

# ============================================================
#  APP & CONFIG
# ============================================================
app = modal.App("comfyui-2025")

GPU = "A100-40GB"

VOL = modal.Volume.from_name("comfy-vol", create_if_missing=True)

DATA_ROOT = "/data/comfy"
COMFY_DIR = f"{DATA_ROOT}/ComfyUI"
MODEL_DIR = f"{COMFY_DIR}/models"

# ============================================================
#  UTILITIES
# ============================================================
def run(cmd, cwd=None):
    print(f"▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


# ============================================================
#  SETUP — CLONE + INSTALL + DOWNLOAD FLUX MODELS
# ============================================================
@app.function(
    timeout=3600,
    volumes={DATA_ROOT: VOL},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("civitai-token"),
    ],
)
def setup():
    print("🚀 Setup ComfyUI mulai...\n")

    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # -----------------------
    # Clone ComfyUI
    # -----------------------
    if not os.path.exists(COMFY_DIR):
        print("📥 Clone ComfyUI...")
        run(f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY_DIR}")
    else:
        print("✔ ComfyUI sudah ada, skip clone.")

    # -----------------------
    # Install deps
    # -----------------------
    print("\n📦 Install dependencies...")
    run("python3 -m pip install --upgrade pip", cwd=COMFY_DIR)
    run("python3 -m pip install -r requirements.txt", cwd=COMFY_DIR)

    # Minimal essentials
    run("python3 -m pip install safetensors pillow numpy tqdm requests")

    # -----------------------
    # DOWNLOAD FLUX MODELS (NEW STRUCTURE)
    # -----------------------
    print("\n📥 Downloading FLUX models...")

    FLUX_REPO = "black-forest-labs/FLUX.1-dev"
    FLUX_FILES = [
        "ae.safetensors",
        "flux1-dev.safetensors",
        "flux1.txt",
        "config.json",
    ]

    FLUX_PATH = f"{MODEL_DIR}/flux"
    os.makedirs(FLUX_PATH, exist_ok=True)

    for f in FLUX_FILES:
        print(f"⬇️ Download: {f}")
        hf_hub_download(
            repo_id=FLUX_REPO,
            filename=f,
            local_dir=FLUX_PATH,
            token=os.environ.get("HF_TOKEN"),
        )

    print("\n🎉 SETUP SELESAI — MODEL TERPASANG\n")
    return "Setup OK"


# ============================================================
#  LAUNCH — RUN COMFYUI SERVER
# ============================================================
@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={DATA_ROOT: VOL},
)
@modal.asgi_app()
def launch():
    from fastapi import FastAPI
    import uvicorn

    api = FastAPI()

    @api.get("/")
    def root():
        return {"msg": "ComfyUI running via Modal"}

    # Jalankan ComfyUI
    def start_comfy():
        print("🔥 Menjalankan ComfyUI...")
        os.chdir(COMFY_DIR)
        run("python3 main.py --listen 0.0.0.0 --port 8188")

    # Jalankan background process
    import threading
    threading.Thread(target=start_comfy, daemon=True).start()

    return api
