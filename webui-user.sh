#!/bin/bash
#########################################################
# Uncomment and change the variables below to your need:#
#########################################################

# Install directory without trailing slash
#install_dir="/home/$(whoami)"

# Name of the subdirectory
#clone_dir="stable-diffusion-webui"

# Commandline arguments for webui.py, for example: export COMMANDLINE_ARGS="--medvram --opt-split-attention"
# finetuned-no-lora
# export COMMANDLINE_ARGS="--allow-code --xformers  --skip-install --api --no-gradio-queue --skip-torch-cuda-test --enable-insecure-extension-access --listen --device-id 1"
# finetuned-with-lora
export COMMANDLINE_ARGS="--allow-code --xformers --skip-install --api --skip-torch-cuda-test --share --disable-tls-verify --listen --port 8302"

# python3 executable
python_cmd="/mnt/homes/minghao/anaconda3/envs/ldm/bin/python"
use_venv=false
VIRTUAL_ENV="/mnt/homes/minghao/anaconda3/envs/ldm"

# export ALL_PROXY="https://100.99.98.110:7890"
# export HTTPS_PROXY="https://100.99.98.110:7890"
# export HTTP_PROXY="https://100.99.98.110:7890"
# export all_proxy="https://100.99.98.110:7890"
# export https_proxy="https://100.99.98.110:7890"
# export http_proxy="https://100.99.98.110:7890"
# export no_proxy="127.0.0.1, .devops.com, localhost, local, .local, 172.28.0.0/16"
# export NO_PROXY="127.0.0.1, .devops.com, localhost, local, .local, 172.28.0.0/16"

# git executable
export GIT="git"

# python3 venv without trailing slash (defaults to ${w}/${clone_dir}/venv)
venv_dir="venv"

# script to launch to start the app
#export LAUNCH_SCRIPT="launch.py"

# install command for torch
#export TORCH_COMMAND="pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"

# Requirements file to use for stable-diffusion-webui
#export REQS_FILE="requirements_versions.txt"

# Fixed git repos
#export K_DIFFUSION_PACKAGE=""
#export GFPGAN_PACKAGE=""

# Fixed git commits
#export STABLE_DIFFUSION_COMMIT_HASH=""
#export CODEFORMER_COMMIT_HASH=""
#export BLIP_COMMIT_HASH=""

# Uncomment to enable accelerated launch
# export ACCELERATE="True"

# Uncomment to disable TCMalloc
# export NO_TCMALLOC="True"

# pytorch and numpy-relate coeffs
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

###########################################
