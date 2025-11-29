import os
import subprocess
import threading
import modal
from huggingface_hub import snapshot_download
from fastapi import FastAPI
from fastapi.responses import Response
import httpx
import asyncio

app = modal.App("comfyui-server")

GPU = "A100-40GB"
VOL = modal.Volume.from_name("comfy-vol", create_if_missing=True)

DATA = "/data/comfy"
COMFY = f"{DATA}/ComfyUI"
MODEL_DIR = f"{COMFY}/models"
CHECKPOINTS = f"{MODEL_DIR}/checkpoints"


# -----------------------
# UTIL
# -----------------------
def run(cmd, cwd=None):
    print("▶", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def ensure_models():
    """Auto-download FLUX checkpoint jika belum ada"""
    os.makedirs(CHECKPOINTS, exist_ok=True)
    dst = f"{CHECKPOINTS}/flux-dev.safetensors"

    if not os.path.exists(dst):
        print("📥 Downloading FLUX checkpoint...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev.safetensors",
            local_dir=CHECKPOINTS,
            token=os.environ["HF_TOKEN"],
            local_dir_use_symlinks=False
        )
        print("✔ FLUX checkpoint selesai")
    else:
        print("✔ Model sudah ada")


# -----------------------
# SETUP
# -----------------------
@app.function(timeout=3600, volumes={DATA: VOL})
def setup():
    os.makedirs(DATA, exist_ok=True)

    if not os.path.exists(COMFY):
        run(f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY}")
    else:
        print("✔ Repo ComfyUI sudah ada")

    run("pip install --upgrade pip", cwd=COMFY)
    run("pip install -r requirements.txt", cwd=COMFY)

    print("✔ Setup selesai")


# -----------------------
# BACKEND SERVER (Modal → ComfyUI passthrough)
# -----------------------
@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={DATA: VOL},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.asgi_app()
def launch():
    api = FastAPI()

    # Start comfy in background thread
    def start_comfy():
        ensure_models()
        os.chdir(COMFY)
        run("python3 main.py --listen 0.0.0.0 --port 8188")

    threading.Thread(target=start_comfy, daemon=True).start()

    # Forward ALL requests to ComfyUI backend
    client = httpx.AsyncClient(base_url="http://127.0.0.1:8188")

    @api.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request):
        url = f"http://127.0.0.1:8188/{path}"
        method = request.method

        if method == "GET":
            resp = await client.get(url)
        else:
            body = await request.body()
            resp = await client.post(url, content=body, headers=request.headers)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp.headers
        )

    return api
