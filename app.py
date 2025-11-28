# =======================================================
#  ComfyUI on Modal – V6 FINAL (2025) - Fully Stable
#  Compatible with Modal CLI 1.2.1
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
# Image with all stable dependencies
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
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # CORE
        "huggingface_hub[hf_transfer]",
        "requests",
        "tqdm",
        "einops",
        "numpy",
        "pillow",
        "psutil",
        "safetensors",

        # IMAGE + SCIENTIFIC
        "opencv-python",
        "scipy",

        # REQUIRED BY MANAGER
        "aiohttp",
        "packaging",
        "lmdb",

        # REQUIRED FOR SUPIR / lightning nodes
        "pytorch_lightning",

        # INSIGHTFACE + ONNX
        "insightface",
        "onnxruntime-gpu",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# -------------------------------------------------------
# Helper Functions
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
# SETUP
# -------------------------------------------------------
@app.function(gpu=GPU, volumes={DATA_ROOT: vol}, timeout=1200, image=image)
def setup():
    print("\n📦 STARTING SETUP...\n")

    # Clone Comfy
    if not (BASE / "main.py").exists():
        print("📥 Cloning ComfyUI...")
        BASE.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone https://github.com/comfyanonymous/ComfyUI {BASE}")
    else:
        print("🔄 Updating ComfyUI...")
    run("git pull --ff-only", cwd=BASE)

    # Custom Nodes
    print("\n📦 Installing Custom Nodes...\n")
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
        print(f"🔧 Installing Node: {name}")
        try:
            run(f"git clone --depth 1 {repo} {dst}")
        except:
            print(f"⚠️ Failed installing node: {name}")

    # InsightFace buffalo_l
    print("\n📦 Installing buffalo_l face model...\n")
    face_dir = Path(DATA_ROOT, ".insightface", "models")
    face_dir.mkdir(parents=True, exist_ok=True)

    if not (face_dir / "buffalo_l").exists():
        zipf = face_dir / "buffalo_l.zip"
        run(f"wget -q -O {zipf} https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
        with zipfile.ZipFile(zipf) as z:
            z.extractall(face_dir)
        zipf.unlink()

    # MODELS: FLUX (repo baru: Black Forest Labs)
    print("\n📦 Downloading FLUX models...\n")

    models = [
        # FLUX main checkpoint
        ("checkpoints", "flux1-dev-fp8.safetensors", "camenduru/FLUX.1-dev"),

        # FLUX VAE
        ("vae/FLUX", "ae.safetensors", "black-forest-labs/FLUX.1-dev"),

        # FLUX CLIP + T5 encoders
        ("clip/FLUX", "clip_l.safetensors", "black-forest-labs/FLUX.1-dev"),
        ("clip/FLUX", "text_encoder.t5xxl.fp8_e4m3fn.safetensors", "black-forest-labs/FLUX.1-dev"),
    ]

    for sub, fn, repo in models:
        path = BASE / "models" / sub / fn
        if not path.exists():
            print(f"⬇️ Downloading {fn}...")
            hf_get(sub, fn, repo)
        else:
            print(f"✔ {fn} already exists.")

    vol.commit()
    print("\n✅ SETUP COMPLETED\n")

# -------------------------------------------------------
# LAUNCH (WITH LIVE LOG STREAM)
# -------------------------------------------------------
@app.function(gpu=GPU, volumes={DATA_ROOT: vol}, timeout=86400, image=image)
def launch():
    print("\n🔥 Starting ComfyUI server...\n")
    os.chdir(BASE)

    proc = subprocess.Popen(
        ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # STREAM LOGS LIVE
    for line in proc.stdout:
        print(line, end="")

    proc.wait()
    print(f"\n⚠️ ComfyUI exited with code {proc.returncode}\n")
