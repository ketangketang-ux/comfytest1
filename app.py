import os
import subprocess
import modal

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

# ============================================================
#  UTIL
# ============================================================
def run(cmd, cwd=None):
    print(f"▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


# ============================================================
#  SETUP — CLEAN INSTALL TANPA DOWNLOAD MODEL
#  (download model = lewat models.py)
# ============================================================
@app.function(
    timeout=3600,
    volumes={DATA: VOL},
)
def setup():
    print("🚀 Setup ComfyUI mulai...\n")

    os.makedirs(DATA, exist_ok=True)

    # Clone repo
    if not os.path.exists(COMFY):
        print("📥 Clone ComfyUI...")
        run(f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY}")
    else:
        print("✔ ComfyUI sudah ada, skip clone.")

    # Install deps
    print("\n📦 Install requirements...")
    run("python3 -m pip install --upgrade pip", cwd=COMFY)
    run("python3 -m pip install -r requirements.txt", cwd=COMFY)

    print("\n🎉 SETUP SELESAI (Tidak download model — gunakan models.py)\n")
    return "Setup OK"


# ============================================================
#  LAUNCH — ASGI (RUN FASTAPI + START COMFYUI MAIN)
# ============================================================
@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={DATA: VOL},
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

    # ------------------------------
    # Start ComfyUI in background
    # ------------------------------
    def start_comfy():
        print("🔥 Menjalankan ComfyUI...")
        os.chdir(COMFY)
        run("python3 main.py --listen 0.0.0.0 --port 8188")

    threading.Thread(target=start_comfy, daemon=True).start()

    return api
