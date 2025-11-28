# =======================================================
#  ComfyUI + FLUX on Modal – V11 CLEAN (2025)
#  WITHOUT custom nodes, BUT with full Python deps
#  → So ComfyUI Manager can install nodes without errors
# =======================================================

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
import requests

import modal
from huggingface_hub import hf_hub_download, login as hf_login

# =======================================================
# CONFIG
# =======================================================
app = modal.App("comfyui-2025")

DATA_ROOT = "/data/comfy"
BASE = Path(DATA_ROOT, "ComfyUI")

GPU = "L4"

vol = modal.Volume.from_name("comfyui-vol", create_if_missing=True)

# =======================================================
# BASE IMAGE – with ALL deps required by ComfyUI & nodes
# =======================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "unzip",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",

        # PyAV deps
        "libavformat-dev",
        "libavcodec-dev",
        "libavutil-dev",
        "libswscale-dev",
        "libswresample-dev",
        "libavfilter-dev",
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

        # comfy frontend deps (2025)
        "alembic",
        "pydantic-settings",
        "comfyui-embedded-docs",
        "comfyui-workflow-templates",

        # image
        "opencv-python",
        "scipy",

        # video
        "av",

        # node deps global
        "packaging",
        "aiohttp",
        "lmdb",
        "pytorch_lightning",
        "torchsde",
        "python-magic",
        "svglib",
        "pycocotools",

        # face
        "insightface",
        "onnxruntime-gpu",

        # upscale / extras
        "spandrel",
        "kornia",
        "piexif",

        # React/Seg
        "segment-anything",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# =======================================================
# HELPERS
# =======================================================
def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def hf_download(subdir, filename, repo):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing!")

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

def civitai_download(model_id, filename, subdir="loras"):
    token = os.getenv("CIVITAI_TOKEN")
    if not token:
        raise RuntimeError("CIVITAI_TOKEN missing!")

    url = f"https://civitai.com/api/v1/model-versions/{model_id}/download"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if r.status_code != 200:
        raise RuntimeError(r.text)

    dest = BASE / "models" / subdir
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / filename, "wb") as f:
        f.write(r.content)

# =======================================================
# SETUP – installs ComfyUI + dependencies + Flux models
# =======================================================
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

    print("\n===== SETUP =====\n")

    # HF auth
    hf_login(os.getenv("HF_TOKEN"))

    # Clone/update ComfyUI
    if not (BASE / "main.py").exists():
        print("📥 Cloning ComfyUI...")
        BASE.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone https://github.com/comfyanonymous/ComfyUI {BASE}")
    else:
        print("🔄 Updating ComfyUI...")
        run("git pull --ff-only", cwd=BASE)

    # Install requirements of ComfyUI (2025)
    print("\n📦 Installing ComfyUI requirements.txt ...\n")
    run("pip install -r requirements.txt", cwd=BASE)

    # Install buffalo_l for ReActor/face nodes (optional but useful)
    print("\n📦 Installing buffalo_l...")
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

    # Download FLUX base models
    print("\n📦 Downloading FLUX models...\n")
    flux = [
        ("checkpoints", "flux1-dev-fp8.safetensors", "camenduru/FLUX.1-dev"),
        ("vae/FLUX", "ae.safetensors", "black-forest-labs/FLUX.1-dev"),
        ("clip/FLUX", "clip_l.safetensors", "black-forest-labs/FLUX.1-dev"),
        ("clip/FLUX", "text_encoder.t5xxl.fp8_e4m3fn.safetensors", "black-forest-labs/FLUX.1-dev"),
    ]
    for sub, file, repo in flux:
        hf_download(sub, file, repo)

    vol.commit()

    print("\n===== SETUP DONE =====\n")

# =======================================================
# LAUNCH – ComfyUI server
# =======================================================
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

    print("\n🔥 Starting ComfyUI...\n")

    os.chdir(BASE)

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
    print(f"\n⚠️ ComfyUI exited: {proc.returncode}")
