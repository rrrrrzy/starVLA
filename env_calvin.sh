export BASE=/inspire/qb-ilm2/project/26summer-camp-10/public/ten
export REPO=$BASE/starVLA

export PATH=$BASE/.local/bin:$PATH
export UV_CACHE_DIR=$BASE/.cache/uv
export UV_PYTHON_INSTALL_DIR=$BASE/.uv-python
export PIP_CACHE_DIR=$BASE/.cache/pip

# 内部 PyPI 源：给 pip 用
export PIP_INDEX_URL=http://nexus.sii.shaipower.online/repository/pypi/simple/
export PIP_TRUSTED_HOST=nexus.sii.shaipower.online

# 内部 PyPI 源：给 uv 用
export UV_DEFAULT_INDEX=http://nexus.sii.shaipower.online/repository/pypi/simple/
export UV_INSECURE_HOST=nexus.sii.shaipower.online

# 防止 GPU 节点系统 Python / 系统 torch 污染 Calvin venv
unset PYTHONPATH
unset LD_PRELOAD
export PYTHONNOUSERSITE=1

# 清理可能来自系统 Python3.12 / StarVLA torch 的动态库路径
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(printf "%s" "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -vE '/usr/local/lib/python3\.12|dist-packages/torch|site-packages/torch|\.venvs/starvla' | paste -sd: -)"
fi

# 激活 Calvin 环境
source $BASE/.venvs/calvin-py38/bin/activate

# 显式加入源码路径：StarVLA 的 deployment 模块 + CALVIN 源码
export PYTHONPATH=$REPO:$BASE/calvin/calvin_models:$BASE/calvin/calvin_env:$BASE/calvin/calvin_env/tacto
cd $REPO

# CALVIN / PyBullet EGL rendering on GPU nodes
unset DISPLAY
export PYOPENGL_PLATFORM=egl
export EGL_VISIBLE_DEVICES=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
