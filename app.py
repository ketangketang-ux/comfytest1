import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download
import modal

# Paths
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"

# Helpers
def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False):
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_comfy_DIR, "custom_nodes", name)
    cmd = f"git clone https://github.com/{node_repo} {dest}"
    if recursive:
        cmd += " --recursive"
    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"
    return cmd

def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    out = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        local_dir=TMP_DL
    )
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))


# =====================================================================
# BUILD IMAGE
# =====================================================================

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        "comfy --skip-prompt install --nvidia"
    ])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

# Built-in comfy nodes
image = image.run_commands([
    "comfy node install "
    "rgthree-comfy "
    "comfyui-impact-pack "
    "comfyui-impact-subpack "
    "ComfyUI-YOLO "
    "comfyui-inspire-pack "
    "comfyui_ipadapter_plus "
    "wlsh_nodes "
    "ComfyUI_Comfyroll_CustomNodes "
    "comfyui_essentials "
    "ComfyUI-GGUF "
    "comfyui-manager-civitai-extension"
])

# Git-based nodes
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("welltop-cn/ComfyUI-TeaCache", {'install_reqs': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
    ("crystian/ComfyUI-Crystools", {'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# InsightFace NOT installed here (runtime only)

# Runtime model downloads
model_tasks = [
    ("unet/FLUX", "flux1-dev-Q8_0.gguf", "city96/FLUX.1-dev-gguf", None),
    ("clip/FLUX", "t5-v1_1-xxl-encoder-Q8_0.gguf", "city96/t5-v1_1-xxl-encoder-gguf", None),
    ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders", None),
    ("checkpoints", "flux1-dev-fp8-all-in-one.safetensors", "camenduru/FLUX.1-dev", None),
    ("loras", "mjV6.safetensors", "strangerzonehf/Flux-Midjourney-Mix2-LoRA", None),
    ("vae/FLUX", "ae.safetensors", "ffxvs/vae-flux", None),
]

# InsightFace Model packages
model_tasks += [
    ("insightface", "det_10g.onnx", "ltdrdata/insightface_models", "antelopev2"),
    ("insightface", "2d106det.onnx", "ltdrdata/insightface_models", "antelopev2"),
    ("insightface", "genderage.onnx", "ltdrdata/insightface_models", "antelopev2"),
    ("insightface", "scrfd_2.5g_bnkps.onnx", "ltdrdata/insightface_models", "antelopev2"),
    ("insightface", "w600k_r50.onnx", "ltdrdata/insightface_models", "antelopev2"),
]

extra_cmds = [
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {MODELS_DIR}/upscale_models",
]

# =====================================================================
# APP
# =====================================================================

vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)

app = modal.App(
    name="comfyui",
    image=image
)

@app.function(
    max_containers=1,
    scaledown_window=600,
    timeout=1800,
    gpu=os.environ.get("MODAL_GPU_TYPE", "L4-24GB"),
    volumes={DATA_ROOT: vol},
    secrets=[modal.Secret.from_name("civitai-token")]
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=300)
def ui():

    # first-time install
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("First run detected. Copying ComfyUI to volume...")

        os.makedirs(DATA_ROOT, exist_ok=True)

        if os.path.exists(DEFAULT_COMFY_DIR):
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True, check=True)
        else:
            os.makedirs(DATA_BASE, exist_ok=True)

    # Fix & update backend
    print("Updating ComfyUI backend...")
    os.chdir(DATA_BASE)

    try:
        result = subprocess.run("git symbolic-ref HEAD", shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            subprocess.run("git checkout -B main origin/main", shell=True, check=True)

        subprocess.run("git config pull.ff only", shell=True, check=True)
        subprocess.run("git pull --ff-only", shell=True, check=True)
    except Exception as e:
        print("Backend update error:", e)


    # Update Manager
    manager_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager")
    if os.path.exists(manager_dir):
        try:
            os.chdir(manager_dir)
            subprocess.run("git config pull.ff only", shell=True, check=True)
            subprocess.run("git pull --ff-only", shell=True, check=True)
        except Exception as e:
            print("Manager update error:", e)
    else:
        subprocess.run("comfy node install ComfyUI-Manager", shell=True)

    # pip upgrade
    subprocess.run("pip install --no-cache-dir --upgrade pip", shell=True)

    # comfy-cli upgrade
    subprocess.run("pip install --no-cache-dir --upgrade comfy-cli", shell=True)

    # Frontend update
    req = os.path.join(DATA_BASE, "requirements.txt")
    if os.path.exists(req):
        subprocess.run(f"/usr/local/bin/python -m pip install -r {req}", shell=True)

    # Manager config (for UI only)
    cfg_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.ini"), "w") as f:
        f.write(
            "[default]\n"
            "network_mode = private\n"
            "security_level = weak\n"
            "log_to_file = false\n"
            "allow_node_updates = true\n"
            "allow_node_install = true\n"
        )

    # ================================
    # 🔥 Patch Manager security (core fix)
    # ================================
    try:
        manager_cfg_py = os.path.join(manager_dir, "config.py")
        if os.path.exists(manager_cfg_py):
            subprocess.run(
                "sed -i \"s/security_level *= *['\\\"]*.*['\\\"]/security_level = 'weak'/\" config.py",
                shell=True
            )
            print("Manager security level forced to 'weak'")
    except Exception as e:
        print("Security patch failed:", e)

    # signature bypass
    try:
        manager_main = os.path.join(manager_dir, "manager.py")
        if os.path.exists(manager_main):
            subprocess.run(
                "sed -i 's/if not verify_signature/if False and verify_signature/' manager.py",
                shell=True
            )
    except Exception as e:
        print("Manager signature patch failed:", e)

    # Make dirs
    for d in [CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL, os.path.join(MODELS_DIR, "insightface")]:
        os.makedirs(d, exist_ok=True)

    # InsightFace Node Install (runtime safe)
    ins_face = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-InsightFace")
    if not os.path.exists(ins_face):
        try:
            print("Installing InsightFace...")
            subprocess.run(
                f"git clone https://github.com/ltdrdata/ComfyUI-InsightFace {ins_face}",
                shell=True, check=True
            )
            req = os.path.join(ins_face, "requirements.txt")
            if os.path.exists(req):
                subprocess.run(f"pip install -r {req}", shell=True)
        except Exception as e:
            print("InsightFace install error:", e)

    # Download models
    for sub, fn, repo, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            try:
                hf_download(sub, fn, repo, subf)
            except Exception as e:
                print("Model download failed:", e)

    # Extra downloads
    for cmd in extra_cmds:
        subprocess.run(cmd, shell=True)

    # Start ComfyUI
    os.environ["COMFY_DIR"] = DATA_BASE

    print("Launching ComfyUI...")
    subprocess.Popen(
        [
            "comfy", "launch", "--",
            "--listen", "0.0.0.0",
            "--port", "8000",
            "--front-end-version", "Comfy-Org/ComfyUI_frontend@latest"
        ],
        cwd=DATA_BASE,
        env=os.environ.copy()
    )
