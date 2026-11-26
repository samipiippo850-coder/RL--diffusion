#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/path_measure_preparation
PY="$BASE/env38/bin/python"
W="$BASE/wheels38"
cd "$BASE"
echo '2ce38a6e4ea7c4b7f5baa51e65243a5f687f6e19ab7915ba5b2a431105f50bbe  wheels38/torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl' | sha256sum -c -
"$PY" -m pip install --no-index --find-links "$W" 'numpy==1.23.5' 'Pillow==7.2.0' 'scipy==1.10.1' 'einops==0.6.1' 'typed-argument-parser==1.10.1' 'tqdm' 'matplotlib==3.7.5' 'h5py==3.11.0' 'Cython<3' 'pyglet==1.5.0' 'cloudpickle==1.6.0' 'glfw==2.7.0' 'imageio==2.9.0' 'cffi==1.15.1' 'fasteners==0.15' 'GitPython<3.2' filelock 'sympy<1.14' 'networkx<3.2' jinja2
"$PY" -m pip install --no-index --no-deps "$W/torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl"
"$PY" -c 'import torch; print("torch",torch.__version__,"cuda",torch.cuda.is_available()); assert torch.cuda.is_available(); print((torch.ones(2, device="cuda")+1).cpu())'
"$PY" -m pip install --no-index --no-deps --no-build-isolation "$W/gym-0.18.0.tar.gz"
mkdir -p "$BASE/mujoco"
unzip -oq "$W/mujoco200_linux.zip" -d "$BASE/mujoco"
echo 'MuJoCo files unpacked; compile and D4RL checks remain.'
