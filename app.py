import os
import subprocess
import requests
import modal
from huggingface_hub import snapshot_download

# =============================================================================
# IMAGE — fix: include FastAPI & Uvicorn (Modal update requires this)
# =============================================================================
image = (
    modal.Image.debian_slim()
        .pip_install(
            "requests",
            "huggingface_hub",
            "safetensors",
            "fastapi[standard]",
            "uvicorn"
        )
)

app = modal.App("comfyui-simple")

GPU = "A100-40GB"
VOL = modal.Volume.from_name("comfy-vol", create_if_missing=True)

DATA = "/data/comfy"
COMFY = f"{DATA}/ComfyUI"
MODEL_DIR = f"{COMFY}/models"
CHECKPOINTS = f"{MODEL_DIR}/checkpoints"


def run(cmd, cwd=None):
    print("▶", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


# =============================================================================
# 1) DOWNLOAD BASEMODEL (Civitai)
# =============================================================================
@app.function(
    image=image,
    timeout=1200,
    volumes={"/data": VOL},
    secrets=[modal.Secret.from_name("civitai-token")]
)
def download_basemodel():

    os.makedirs(CHECKPOINTS, exist_ok=True)

    token = os.environ.get("CIVITAI_TOKEN")
    if not token:
        raise Exception("❌ Secret 'civitai-token' harus punya KEY 'CIVITAI_TOKEN'!")

    url = "https://civitai.com/api/download/models/2285644?type=Model&format=SafeTensor&size=pruned&fp=fp16"

    dst = f"{CHECKPOINTS}/basemodel_fp16.safetensors"

    print(f"⬇️ Downloading base model dari Civitai ke {dst} ...")

    headers = {"Authorization": f"Bearer {token}"}

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("🎉 DONE! Base model tersimpan di volume.")


# =============================================================================
# 2) SETUP COMFYUI
# =============================================================================
@app.function(
    image=image,
    timeout=3600,
    volumes={DATA: VOL},
)
def setup():
    os.makedirs(DATA, exist_ok=True)

    if not os.path.exists(COMFY):
        run(f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY}")
    else:
        print("✔ ComfyUI repo sudah ada")

    run("pip install --upgrade pip", cwd=COMFY)
    run("pip install -r requirements.txt", cwd=COMFY)

    print("✔ Setup Selesai.")


# =============================================================================
# 3) LAUNCH COMFYUI (web endpoint)
# =============================================================================
@app.function(
    image=image,
    gpu=GPU,
    timeout=86400,
    volumes={DATA: VOL},
    secrets=[modal.Secret.from_name("civitai-token")],
)
@modal.web_endpoint()   # boleh pakai @modal.fastapi_endpoint() juga
def launch():
    print("📦 Checking model...")
    download_basemodel.call()

    print("🔥 Starting ComfyUI...")
    os.chdir(COMFY)
    run("python3 main.py --listen 0.0.0.0 --port 8188")
