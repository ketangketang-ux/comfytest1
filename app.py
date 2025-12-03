# app.py — modal ComfyUI backend (fixed)
import os
import subprocess
import shutil
from typing import Optional
from huggingface_hub import hf_hub_download
import modal

# -------------------------
# Config — Ubah sesuai kebutuhan
# -------------------------
VOLUME_NAME = "comfyui-app"   # Modal volume name
# mount to a directory that is empty in container and safe — DO NOT use "/" or "/data"
DATA_ROOT = "/work/data_comfy"      # <-- safe mount point
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"   # optional local default copy if exists in image

# -------------------------
# Helpers
# -------------------------
def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False):
    name = node_repo.split("/")[-1]
    dest = os.path.join("/root/comfy/ComfyUI", "custom_nodes", name)  # only used if copying default comfy
    cmd = f"git clone https://github.com/{node_repo} {dest}"
    if recursive:
        cmd += " --recursive"
    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"
    return cmd

def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None, token_env: str = "HF_TOKEN"):
    """
    Download a single file from HF repo_id into MODELS_DIR/subdir.
    Uses huggingface_hub.hf_hub_download with token if provided in env var token_env.
    """
    os.makedirs(TMP_DL, exist_ok=True)
    hf_token = os.environ.get(token_env) or None

    try:
        out = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=TMP_DL,
            use_auth_token=hf_token,
            repo_type="model",
        )
    except Exception as e:
        raise RuntimeError(f"hf_hub_download failed for {repo_id}/{filename}: {e}")

    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))
    print(f"Downloaded {filename} -> {os.path.join(target, filename)}")


# -------------------------
# Build image (clean)
# -------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "unzip", "ffmpeg")
    .pip_install(
        "huggingface_hub[cli,requests]==0.28.1",
        "comfy-cli",
        "huggingface_hub",
        "requests",
        "onnxruntime",
        "insightface"
    )
    .run_commands([
        "pip install --upgrade pip || true",
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# -------------------------
# Modal volume and app
# -------------------------
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(name="comfyui-fixed", image=image)

# include secrets here — ensure secret names exist in Modal dashboard
# use names WITHOUT hyphen as env var keys when possible (e.g., HF_TOKEN)
@app.function(
    image=image,
    volumes={DATA_ROOT: vol},
    timeout=3600,
    cpu=2,
    memory=8192,
    gpu=os.environ.get("MODAL_GPU_TYPE", None),
    secrets=[
        modal.Secret.from_name("civitai-token"),
        modal.Secret.from_name("HF_TOKEN"),
    ],
)
@modal.web_server(port=8000, startup_timeout=300)
def ui():
    """
    Entrypoint for ComfyUI backend in Modal.
    Expects secrets injected as env vars:
      - CIVITAI_TOKEN  (if present) -> from secret 'civitai-token' (Modal will provide env var named like the secret key)
      - HF_TOKEN       -> from secret 'HF_TOKEN'
    """
    # Ensure mount point exists and is safe
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[ui] env HF_TOKEN present?:", bool(os.environ.get("HF_TOKEN")))
    print("[ui] env civitai present?:", bool(os.environ.get("civitai-token") or os.environ.get("CIVITAI_TOKEN")))

    # If first run, try to copy bundled ComfyUI from image (optional)
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("[ui] First run - prepping ComfyUI directory in volume...")
        if os.path.exists(DEFAULT_COMFY_DIR):
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_BASE}", shell=True, check=False)
        else:
            # create skeleton directories
            os.makedirs(DATA_BASE, exist_ok=True)

    # Try to update repo in DATA_BASE if it's a git repo
    try:
        if os.path.exists(os.path.join(DATA_BASE, ".git")):
            print("[ui] Pulling latest ComfyUI in volume...")
            subprocess.run("git -C {} config pull.ff only".format(DATA_BASE), shell=True, check=False)
            subprocess.run("git -C {} pull --ff-only".format(DATA_BASE), shell=True, check=False)
        else:
            print("[ui] No git repo in DATA_BASE, skipping git pull.")
    except Exception as e:
        print("[ui] git update error:", e)

    # Write a relaxed manager config (only if ComfyUI-Manager reads it)
    cfg_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.ini"), "w") as f:
        f.write("[default]\nnetwork_mode = private\nsecurity_level = weak\nlog_to_file = false\n")

    # Model download list (example) — adjust to your needs
    model_tasks = [
        ("checkpoints", "juggernautXL_juggXIByRundiffusion.safetensors", "camenduru/FLUX.1-dev"),
        # add more (subdir, filename, repo_id) as needed
    ]

    # download models if missing
    for sub, fn, repo in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            try:
                print(f"[ui] Downloading {fn} from {repo} ...")
                hf_download(sub, fn, repo, None, token_env="HF_TOKEN")
            except Exception as e:
                print("[ui] Model download failed:", e)

    # Launch Comfy (assumes comfy-cli entrypoint installed)
    os.environ["COMFY_DIR"] = DATA_BASE
    launch_cmd = [
        "comfy", "launch", "--",
        "--listen", "0.0.0.0",
        "--port", "8000",
        "--front-end-version", "Comfy-Org/ComfyUI_frontend@latest"
    ]
    print("[ui] Launching ComfyUI with command:", " ".join(launch_cmd))
    subprocess.Popen(launch_cmd, cwd=DATA_BASE, env=os.environ.copy())

    return {"status": "ok", "url": "http://0.0.0.0:8000"}

# End of app.py
