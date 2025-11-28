# ==========================
# ComfyUI on Modal (FINAL 2025)
# ==========================
import modal, subprocess, os, shutil, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

DATA_ROOT = "/data/comfy"
BASE = Path(DATA_ROOT, "ComfyUI")
GPU = os.getenv("MODAL_GPU", "L4")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "unzip", "ffmpeg",
        "libgl1-mesa-glx", "libglib2.0-0"
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "insightface",
        "onnxruntime-gpu",
        "requests",
        "tqdm"
    )
)

vol = modal.Volume.from_name("comfyui-vol", create_if_missing=True)
app = modal.App(name="comfyui-2025", image=image)


def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def hf_get(subdir, filename, repo, subfolder=None):
    dest = BASE / "models" / subdir
    dest.mkdir(parents=True, exist_ok=True)
    out = hf_hub_download(
        repo_id=repo,
        filename=filename,
        subfolder=subfolder,
        local_dir="/tmp",
        local_dir_use_symlinks=False,
    )
    shutil.move(out, dest / filename)


# ===========================
# SETUP FUNCTION
# ===========================
@app.function(gpu=GPU, volumes={DATA_ROOT: vol}, timeout=900)
def setup():
    # clone comfyui
    if not (BASE / "main.py").exists():
        BASE.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone https://github.com/comfyanonymous/ComfyUI {BASE}")
    run("git pull --ff-only", cwd=BASE)

    # comfyui manager
    mgr = BASE / "custom_nodes" / "ComfyUI-Manager"
    if mgr.exists():
        run("git pull --ff-only", cwd=mgr)
    else:
        run("git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager", cwd=BASE)

    # custom nodes (clean reinstall)
    nodes = {
        "rgthree-comfy":          "https://github.com/rgthree/rgthree-comfy.git",
        "comfyui-impact-pack":    "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "ComfyUI-ReActor":        "https://github.com/Gourieff/ComfyUI-ReActor.git",
        "ComfyUI-SUPIR":          "https://github.com/cubiq/ComfyUI-SUPIR.git",
        "ComfyUI-InsightFace":    "https://github.com/cubiq/ComfyUI-InsightFace.git",
        "ComfyUI_essentials":     "https://github.com/cubiq/ComfyUI_essentials.git",
        "ComfyUI_IPAdapter_plus": "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
        "ComfyUI-KJNodes":        "https://github.com/kijai/ComfyUI-KJNodes.git",
    }

    for name, url in nodes.items():
        dst = BASE / "custom_nodes" / name
        if dst.exists(): shutil.rmtree(dst)
        try:
            run(f"git clone --depth 1 {url} {dst}")
        except:
            print(f"⚠ Failed clone: {name}")

    # insightface model
    inf = Path(DATA_ROOT, ".insightface", "models")
    inf.mkdir(parents=True, exist_ok=True)
    if not (inf / "buffalo_l").exists():
        zipf = inf / "buffalo_l.zip"
        run(f"wget -q -O {zipf} https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
        with zipfile.ZipFile(zipf) as z: z.extractall(inf)
        zipf.unlink()

    # flux models
    models = [
        ("checkpoints", "flux1-dev-fp8.safetensors", "camenduru/FLUX.1-dev"),
        ("vae/FLUX",    "ae.safetensors", "comfyanonymous/flux_vae"),
        ("clip/FLUX",   "t5xxl_fp8_e4m3fn.safetensors", "comfyanonymous/flux_text_encoders"),
        ("clip/FLUX",   "clip_l.safetensors", "comfyanonymous/flux_text_encoders"),
    ]
    for sub, fn, repo in models:
        if not (BASE / "models" / sub / fn).exists():
            hf_get(sub, fn, repo)

    vol.commit()
    print("✅ Setup selesai")


# ===========================
# SERVER RUNNER
# ===========================
@app.function(gpu=GPU, volumes={DATA_ROOT: vol}, timeout=86400)
def launch():
    os.chdir(BASE)
    print("🔥 Starting ComfyUI server...")
    run("python3 main.py --listen 0.0.0.0 --port 8188")
