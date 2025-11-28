import modal, subprocess, os, shutil, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

app = modal.App("comfyui-2025")

DATA_ROOT = "/data/comfy"
BASE = Path(DATA_ROOT, "ComfyUI")
GPU = "L4"

vol = modal.Volume.from_name("comfy-vol", create_if_missing=True)

def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def hf_get(subdir, fname, repo):
    dest = BASE / "models" / subdir
    dest.mkdir(parents=True, exist_ok=True)
    out = hf_hub_download(repo_id=repo, filename=fname, local_dir="/tmp", local_dir_use_symlinks=False)
    shutil.move(out, dest / fname)

# ------------------ SETUP ------------------
@app.function(volumes={DATA_ROOT: vol}, gpu=GPU, timeout=900)
def setup():
    if not (BASE / "main.py").exists():
        BASE.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone https://github.com/comfyanonymous/ComfyUI {BASE}")
    run("git pull --ff-only", cwd=BASE)

    nodes = {
        "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager.git",
        "comfyui-impact-pack": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "rgthree-comfy": "https://github.com/rgthree/rgthree-comfy.git",
    }

    for name, url in nodes.items():
        dst = BASE / "custom_nodes" / name
        if dst.exists(): shutil.rmtree(dst)
        run(f"git clone --depth 1 {url} {dst}")

    vol.commit()
    print("SETUP DONE")

# ------------------ LAUNCH ------------------
@app.function(volumes={DATA_ROOT: vol}, gpu=GPU, timeout=86400)
def launch():
    os.chdir(BASE)
    print("🔥 Starting ComfyUI...")
    run("python3 main.py --listen 0.0.0.0 --port 8188")
