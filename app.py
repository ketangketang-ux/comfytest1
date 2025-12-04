import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download
import modal
import threading
import time

# ------------------------
# Paths
# ------------------------
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"

# ------------------------
# Helpers
# ------------------------
def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False):
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes", name)
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
# BUILD IMAGE (minimal, avoid installing InsightFace here)
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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Built-in Comfy nodes (install what the build can safely do)
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

# Git-based nodes (installed in build where allowed)
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("welltop-cn/ComfyUI-TeaCache", {'install_reqs': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
    ("crystian/ComfyUI-Crystools", {'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])


# =====================================================================
# Runtime model tasks (HuggingFace)
# =====================================================================

model_tasks = [
    ("unet/FLUX", "flux1-dev-Q8_0.gguf", "city96/FLUX.1-dev-gguf", None),
    ("clip/FLUX", "t5-v1_1-xxl-encoder-Q8_0.gguf", "city96/t5-v1_1-xxl-encoder-gguf", None),
    ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders", None),
    ("checkpoints", "flux1-dev-fp8-all-in-one.safetensors", "camenduru/FLUX.1-dev", None),
    ("loras", "mjV6.safetensors", "strangerzonehf/Flux-Midjourney-Mix2-LoRA", None),
    ("vae/FLUX", "ae.safetensors", "ffxvs/vae-flux", None),
]

# InsightFace ONNX models (download via HF)
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
app = modal.App(name="comfyui", image=image)

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

    # ---------------------------------------------------------
    # FIRST RUN: copy default ComfyUI into the persistent volume
    # ---------------------------------------------------------
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("First run: copying ComfyUI to volume...")
        os.makedirs(DATA_ROOT, exist_ok=True)
        if os.path.exists(DEFAULT_COMFY_DIR):
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True)
        else:
            os.makedirs(DATA_BASE, exist_ok=True)

    # ---------------------------------------------------------
    # SAFE GIT SYNC (handles missing .git / origin/main issues)
    # ---------------------------------------------------------
    print("Updating ComfyUI backend safely...")
    os.chdir(DATA_BASE)

    if not os.path.exists(os.path.join(DATA_BASE, ".git")):
        print("No .git found → initializing git and adding origin...")
        subprocess.run("git init", shell=True)
        subprocess.run("git remote add origin https://github.com/comfyanonymous/ComfyUI.git", shell=True)

    subprocess.run("git fetch origin --tags --prune", shell=True)

    branch_check = subprocess.run(
        "git ls-remote --heads origin main",
        shell=True, capture_output=True, text=True
    )
    branch = "main" if branch_check.stdout.strip() else "master"
    print(f"Using branch: {branch}")

    subprocess.run(f"git checkout -B {branch}", shell=True)
    subprocess.run(f"git pull origin {branch}", shell=True)

    # ---------------------------------------------------------
    # Manager update / install
    # ---------------------------------------------------------
    manager_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager")
    if os.path.exists(manager_dir):
        try:
            os.chdir(manager_dir)
            subprocess.run("git fetch origin", shell=True, check=False)
            subprocess.run("git pull", shell=True, check=False)
        except Exception as e:
            print("Manager update error:", e)
    else:
        subprocess.run("comfy node install ComfyUI-Manager", shell=True, check=False)

    # ---------------------------------------------------------
    # pip + comfy-cli upgrade
    # ---------------------------------------------------------
    subprocess.run("pip install --no-cache-dir --upgrade pip", shell=True, check=False)
    subprocess.run("pip install --no-cache-dir --upgrade comfy-cli", shell=True, check=False)

    # ---------------------------------------------------------
    # Manager user config (UI-level)
    # ---------------------------------------------------------
    cfg_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.ini"), "w") as f:
        f.write(
            "[default]\n"
            "security_level = weak\n"
            "network_mode = private\n"
            "allow_node_install = true\n"
            "allow_node_updates = true\n"
            "log_to_file = false\n"
        )

    # ---------------------------------------------------------
    # BRUTE-FORCE Manager patch (applies to many versions)
    # ---------------------------------------------------------
    print("Applying Manager brute-force patches...")
    if os.path.exists(manager_dir):
        for root, dirs, files in os.walk(manager_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                full = os.path.join(root, file)
                try:
                    # Force security_level = 'weak'
                    subprocess.run(
                        f"sed -i \"s/security_level *= *['\\\"]*[a-zA-Z0-9_]*['\\\"]/security_level = 'weak'/g\" \"{full}\"",
                        shell=True, check=False
                    )
                    # Replace 'strict' -> 'weak'
                    subprocess.run(
                        f"sed -i \"s/'strict'/'weak'/g\" \"{full}\"",
                        shell=True, check=False
                    )
                    # Disable signature checks (multiple patterns)
                    subprocess.run(
                        f"sed -i \"s/verify_signature\\s*(/False and verify_signature(/g\" \"{full}\"",
                        shell=True, check=False
                    )
                    subprocess.run(
                        f"sed -i \"s/if not verify_signature/if False and verify_signature/g\" \"{full}\"",
                        shell=True, check=False
                    )
                except Exception as e:
                    print("Patch error for", full, e)
    else:
        print("Manager directory not present; skipping brute-force patch.")

    print("Manager patches done.")

    # ---------------------------------------------------------
    # Ensure model dirs exist and download models
    # ---------------------------------------------------------
    os.makedirs(os.path.join(MODELS_DIR, "insightface"), exist_ok=True)
    for sub, fn, repo, subf in model_tasks:
        dst = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(dst):
            try:
                print(f"Downloading model {fn} into {sub} ...")
                hf_download(sub, fn, repo, subf)
            except Exception as e:
                print("Model download failed:", e)

    # Extra downloads (safe)
    for cmd in extra_cmds:
        try:
            subprocess.run(cmd, shell=True, check=False)
        except Exception as e:
            print("Extra download failed:", e)

    # ---------------------------------------------------------
    # LAUNCH ComfyUI (single launch, avoid restarts)
    # ---------------------------------------------------------
    print("Launching ComfyUI (single process)...")
    subprocess.Popen(
        [
            "comfy", "launch", "--",
            "--listen", "0.0.0.0",
            "--port", "8000",
            "--front-end-version", "Comfy-Org/ComfyUI_frontend@latest"
        ],
        cwd=DATA_BASE
    )

    # ---------------------------------------------------------
    # DELAYED InsightFace install (post-boot) - NO RESTART
    # ---------------------------------------------------------
    def install_insightface_postboot():
        time.sleep(10)  # give ComfyUI time to initialize
        ins = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-InsightFace")

        # Clone if missing
        if not os.path.exists(ins):
            try:
                print("Cloning ComfyUI-InsightFace...")
                subprocess.run(
                    f"git clone https://github.com/ltdrdata/ComfyUI-InsightFace {ins}",
                    shell=True, check=False
                )
            except Exception as e:
                print("InsightFace git clone error:", e)
        else:
            print("InsightFace node already present.")

        # Install *minimal* safe dependencies (CPU ONNX runtime + headless OpenCV)
        try:
            print("Installing minimal InsightFace dependencies (onnxruntime CPU, opencv-headless, scikit-image, numpy)...")
            subprocess.run(
                "pip install --no-cache-dir onnxruntime opencv-python-headless scikit-image numpy==1.26.4",
                shell=True, check=False
            )
        except Exception as e:
            print("Dependency install failed:", e)

        # Brute patch all .py files in the cloned InsightFace node:
        # - comment out any 'import mxnet'
        # - try to force ONNX InferenceSession to use CPUExecutionProvider where applicable
        try:
            if os.path.exists(ins):
                for root, dirs, files in os.walk(ins):
                    for file in files:
                        if not file.endswith(".py"):
                            continue
                        full = os.path.join(root, file)
                        try:
                            # comment out mxnet imports
                            subprocess.run(
                                f"sed -i \"s/^\\s*import mxnet/# import mxnet (disabled)/g\" \"{full}\"",
                                shell=True, check=False
                            )
                            subprocess.run(
                                f"sed -i \"s/^\\s*from mxnet import /# from mxnet import (disabled)/g\" \"{full}\"",
                                shell=True, check=False
                            )

                            # If onnxruntime used, prefer CPUExecutionProvider
                            # Replace common InferenceSession instantiation patterns
                            subprocess.run(
                                f"sed -i \"s/InferenceSession(\\([^)]*\\))/InferenceSession(\\1, providers=['CPUExecutionProvider'])/g\" \"{full}\"",
                                shell=True, check=False
                            )

                            # Replace providers=None -> providers=['CPUExecutionProvider']
                            subprocess.run(
                                f"sed -i \"s/providers\\s*=\\s*None/providers=['CPUExecutionProvider']/g\" \"{full}\"",
                                shell=True, check=False
                            )
                        except Exception as e:
                            print("InsightFace file patch error:", full, e)
                print("InsightFace node patched to avoid MXNet and prefer CPU ONNX.")
            else:
                print("InsightFace folder not found for patching.")
        except Exception as e:
            print("InsightFace patching exception:", e)

        # Trigger Comfy's node rescan (safe, non-blocking)
        try:
            print("Triggering Comfy node rescan...")
            subprocess.run("comfy node rescan || true", shell=True, check=False)
        except Exception as e:
            print("Node rescan failed:", e)

        print("InsightFace install/post-boot tasks completed.")

    threading.Thread(target=install_insightface_postboot, daemon=True).start()

    print("Startup complete. Background tasks launched (InsightFace install + Manager patches).")

