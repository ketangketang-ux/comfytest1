# =======================================================
#  ComfyUI on Modal – V8 FINAL (HF TOKEN + CIVITAI TOKEN)
# =======================================================
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
import requests

import modal
from huggingface_hub import hf_hub_download, login as hf_login

# -------------------------------------------------------
# App Config
# -------------------------------------------------------
app = modal.App("comfyui-2025")

DATA_ROOT = "/data/comfy"
BASE = Path(DATA_ROOT, "ComfyUI")
GPU = "L4"

vol = modal.Volume.from_name("comfyui-vol", create_if_missing=True)

# -------------------------------------------------------
# Image Base
# -------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "unzip",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
    )
    .run_commands(
        "python3 -m pip install --upgrade pip wheel setuptools",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        # core
        "huggingface_hub[hf_transfer]",
        "requests",
        "tqdm",
        "einops",
        "numpy",
        "pillow",
        "psutil",
        "safetensors",

        # image libs
        "opencv-python",
        "scipy",

        # required by nodes
        "aiohttp",
        "packaging",
        "lmdb",
        "pytorch_lightning",

        # face + onnx
        "insightface",
        "onnxruntime-gpu",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def hf_download(subdir, filename, repo):
    """Download HF models using secret token."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from Modal secrets!")

    # login first
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

def civitai_download(model_id, filename, subdir="checkpoints"):
    """Download from Civitai using secret token."""
    token = os.getenv("CIVITAI_TOKEN")
    if not token:
        raise RuntimeError("CIVITAI_TOKEN missing!")

    url = f"https://civitai.com/api/v1/model-versions/{model_id}/download"

    print(f"⬇️ Civitai: {model_id} → {filename}")

    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers)

    if r.status_code != 200:
        raise RuntimeError(f"Civitai download failed: {r.text}")

    dest = BASE / "models" / subdir
    dest.mkdir(parents=True, exist_ok=True)

    with open(dest / filename, "wb") as f:
        f.write(r.content)

# -------------------------------------------------------
# SETUP
# -------------------------------------------------------
@app.function(
    gpu=GPU,
    volumes={DATA_ROOT: vol},
    timeout=1800,
    image=image,
    secrets=[
        modal.Secret.from_name("hf_secret"),
        modal.Secret.from_name("civitai_secret")
    ],
)
def setup():
    print("\n📦 SETUP START\n")

    # HuggingFace login
    hf_login(os.getenv("HF_TOKEN"))

    # Clone Comfy
    if not (BASE / "main.py").exists():
        print("📥 Clone ComfyUI...")
        BASE.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone https://github.com/comfyanonymous/ComfyUI {BASE}")
    else:
        print("🔄 Updating ComfyUI...")
        run("git pull --ff-only", cwd=BASE)

    # Custom Nodes
    print("\n📦 Installing custom nodes...\n")
    nodes = {
        "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager.git",
        "rgthree-comfy": "https://github.com/rgthree/rgthree-comfy.git",
        "Impact-Pack": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "ReActor": "https://github.com/Gourieff/ComfyUI-ReActor.git",
        "SUPIR": "https://github.com/cubiq/ComfyUI-SUPIR.git",
        "InsightFace": "https://github.com/cubiq/ComfyUI-InsightFace.git",
        "Essentials": "https://github.com/cubiq/ComfyUI_essentials.git",
        "IPAdapter+": "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
        "KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
    }

    for name, repo in nodes.items():
        dst = BASE / "custom_nodes" / name
        if dst.exists():
            shutil.rmtree(dst)
        print(f"🔧 Node: {name}")
        try:
            run(f"git clone --depth 1 {repo} {dst}")
        except:
            print(f"⚠️ FAIL install: {name}")

    # InsightFace buffalo_l
    print("\n📦 Installing buffalo_l...")
    face_dir = Path(DATA_ROOT, ".insightface", "models")
    face_dir.mkdir(parents=True, exist_ok=True)

    if not (face_dir / "buffalo_l").exists():
        zipf = face_dir / "buffalo_l.zip"
        run(f"wget -q -O {zipf} https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
        with zipfile.ZipFile(zipf) as z:
            z.extractall(face_dir)
        zipf.unlink()

    # ---------------------------------------------------
    # Download Flux Models (via HF)
    # ---------------------------------------------------
    print("\n📦 Downloading FLUX models...\n")

    models = [
        ("checkpoints", "flux1-dev-fp8.safetensors", "camenduru/FLUX.1-dev"),
        ("vae/FLUX", "ae.safetensors", "black-forest-labs/FLUX.1-dev"),
        ("clip/FLUX", "clip_l.safetensors", "black-forest-labs/FLUX.1-dev"),
        ("clip/FLUX", "text_encoder.t5xxl.fp8_e4m3fn.safetensors", "black-forest-labs/FLUX.1-dev"),
    ]

    for sub, fn, repo in models:
        print(f"Downloading {fn}...")
        hf_download(sub, fn, repo)

    vol.commit()
    print("\n✅ SETUP DONE\n")

# -------------------------------------------------------
# LAUNCH (LOG STREAM)
# -------------------------------------------------------
@app.function(
    gpu=GPU,
    volumes={DATA_ROOT: vol},
    timeout=86400,
    image=image,
    secrets=[
        modal.Secret.from_name("hf_secret"),
        modal.Secret.from_name("civitai_secret")
    ],
)
def launch():
    print("\n🔥 Starting ComfyUI...\n")
    os.chdir(BASE)

    proc = subprocess.Popen(
        ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in proc.stdout:
        print(line, end="")

    proc.wait()
    print(f"\n⚠️ ComfyUI exited with code {proc.returncode}")
