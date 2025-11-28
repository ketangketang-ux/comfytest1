# =======================================================
#  ComfyUI on Modal – FINAL VERSION (Modal V2 Compatible)
# =======================================================
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import modal
from huggingface_hub import hf_hub_download

# -------------------------------------------------------
# App Config
# -------------------------------------------------------
app = modal.App("comfyui-2025")
DATA_ROOT = "/data/comfy"
BASE = Path(DATA_ROOT, "ComfyUI")
GPU = "L4"

vol = modal.Volume.from_name("comfyui-vol", create_if_missing=True)

# -------------------------------------------------------
# Image with ALL dependencies preinstalled
# -------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "unzip",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    )
    .run_commands(
        "python3 -m pip install --upgrade pip wheel setuptools",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "insightface",
        "onnxruntime-gpu",
        "tqdm",
        "requests"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def hf_get(subdir, filename, repo):
    dest = BASE / "models" / subdir
    dest.mkdir(parents=True, exist_ok=True)

    tmp = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir="/tmp",
        local_dir_use_symlinks=False,
    )
    shutil.move(tmp, dest / filename)

# -------------------------------------------------------
# SETUP Function
# -------------------------------------------------------
@app.function(gpu=GPU, volumes={DATA_ROOT: vol}, timeout=900, image=image)
def setup():
    # Clone ComfyUI if not installed
    if not (BASE / "main.py").exists():
        BASE.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone https://github.com/comfyanonymous/ComfyUI {BASE}")

    run("git pull --ff-only", cwd=BASE)

    # ComfyUI Manager
    mgr = BASE / "custom_nodes" / "ComfyUI-Manager"
    if mgr.exists():
        run("git pull --ff-only", cwd=mgr)
    else:
        run(f"git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager", cwd=BASE)

    # Custom Nodes
    nodes = {
        "rgthree-comfy": "https://github.com/rgthree/rgthree-comfy.git",
        "comfyui-impact-pack": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "ComfyUI-ReActor": "https://github.com/Gourieff/ComfyUI-ReActor.git",
        "ComfyUI-SUPIR": "https://github.com/cubiq/ComfyUI-SUPIR.git",
        "ComfyUI-InsightFace": "https://github.com/cubiq/ComfyUI-InsightFace.git",
        "ComfyUI_essentials": "https://github.com/cubiq/ComfyUI_essentials.git",
        "ComfyUI_IPAdapter_plus": "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
        "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
    }

    for name, repo in nodes.items():
        dst = BASE / "custom_nodes" / name
        if dst.exists(): shutil.rmtree(dst)
        try:
            run(f"git clone --depth 1 {repo} {dst}")
        except:
            print(f"[WARN] gagal clone node: {name}")

    # InsightFace buffalo_l
    face_dir = Path(DATA_ROOT, ".insightface", "models")
    face_dir.mkdir(parents=True, exist_ok=True)

    if not (face_dir / "buffalo_l").exists():
        zipf = face_dir / "buffalo_l.zip"
        run(f"wget -q -O {zipf} https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
        with zipfile.ZipFile(zipf) as z:
            z.extractall(face_dir)
        zipf.unlink()

    # FLUX Models
    models = [
        ("checkpoints", "flux1-dev-fp8.safetensors", "camenduru/FLUX.1-dev"),
        ("vae/FLUX", "ae.safetensors", "comfyanonymous/flux_vae"),
        ("clip/FLUX", "t5xxl_fp8_e4m3fn.safetensors", "comfyanonymous/flux_text_encoders"),
        ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders"),
    ]

    for sub, fn, repo in models:
        if not (BASE / "models" / sub / fn).exists():
            hf_get(sub, fn, repo)

    vol.commit()
    print("✅ SETUP COMPLETED. READY TO LAUNCH.")


# -------------------------------------------------------
# LAUNCH Function (Run ComfyUI server)
# -------------------------------------------------------
@app.function(gpu=GPU, volumes={DATA_ROOT: vol}, timeout=86400, image=image)
def launch():
    os.chdir(BASE)
    print("🔥 Starting ComfyUI on port 8188 ...")
    run("python3 main.py --listen 0.0.0.0 --port 8188")
