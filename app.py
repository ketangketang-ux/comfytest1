import os
import subprocess
import modal
from huggingface_hub import snapshot_download

app = modal.App("comfyui-simple")

GPU = "A100-40GB"
VOL = modal.Volume.from_name("comfy-vol", create_if_missing=True)

DATA = "/data/comfy"
COMFY = f"{DATA}/ComfyUI"
MODEL_DIR = f"{COMFY}/models"
CHECKPOINTS = f"{MODEL_DIR}/checkpoints"


# -----------------------------------------------------
# Util
# -----------------------------------------------------
def run(cmd, cwd=None):
    print("▶", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


# -----------------------------------------------------
# Auto-download model
# -----------------------------------------------------
def ensure_models():
    os.makedirs(CHECKPOINTS, exist_ok=True)
    ckpt = f"{CHECKPOINTS}/flux-dev.safetensors"

    if not os.path.exists(ckpt):
        print("📥 Downloading FLUX checkpoint...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev.safetensors",
            local_dir=CHECKPOINTS,
            token=os.environ["HF_TOKEN"],
            local_dir_use_symlinks=False,
        )
        print("✔ Done FLUX")
    else:
        print("✔ FLUX model exists")


# -----------------------------------------------------
# Setup (install ComfyUI)
# -----------------------------------------------------
@app.function(timeout=3600, volumes={DATA: VOL})
def setup():
    os.makedirs(DATA, exist_ok=True)

    if not os.path.exists(COMFY):
        run(f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY}")
    else:
        print("✔ ComfyUI already exists")

    run("pip install --upgrade pip", cwd=COMFY)
    run("pip install -r requirements.txt", cwd=COMFY)

    print("✔ Setup done")


# -----------------------------------------------------
# Backend ComfyUI (RUN WEB SERVER)
# -----------------------------------------------------
@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={DATA: VOL},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_endpoint()
def launch():
    ensure_models()

    os.chdir(COMFY)
    print("🔥 Starting ComfyUI...")
    run("python3 main.py --listen 0.0.0.0 --port 8188")
