#!/usr/bin/env zsh
# Source this file from repository launchers. It locates a user-space Conda
# installation, creates the project environment if needed, and exposes
# cnn_python / cnn_pip wrappers through `conda run` (no shell activation and no
# administrator privileges required).

if [[ -n "${CNN_CONDA_HELPER_LOADED:-}" ]]; then
  return 0
fi
export CNN_CONDA_HELPER_LOADED=1

CNN_ENV_NAME="${CNN_ENV_NAME:-prior-templates-cnns}"
DEVICE="${DEVICE:-cuda}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

_find_conda() {
  local candidate
  for candidate in conda.exe conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  # Git-Bash/MSYS/Cygwin: convert the native Windows LOCALAPPDATA location.
  if [[ -n "${LOCALAPPDATA:-}" ]] && command -v cygpath >/dev/null 2>&1; then
    for candidate in \
      "$(cygpath -u "$LOCALAPPDATA")/miniconda3/Scripts/conda.exe" \
      "$(cygpath -u "$LOCALAPPDATA")/anaconda3/Scripts/conda.exe"; do
      if [[ -x "$candidate" ]]; then
        print -r -- "$candidate"
        return 0
      fi
    done
  fi

  # POSIX user installs.
  for candidate in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "$HOME/.miniconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      print -r -- "$candidate"
      return 0
    fi
  done

  return 1
}

CONDA_EXE="${CONDA_EXE:-$(_find_conda || true)}"
if [[ -z "$CONDA_EXE" ]]; then
  print -u2 "Could not find Conda. This workflow intentionally does not require admin rights."
  print -u2 "Install Miniconda for the current user, or set CONDA_EXE to your conda executable."
  return 2
fi

cnn_python() {
  "$CONDA_EXE" run --no-capture-output -n "$CNN_ENV_NAME" python "$@"
}

cnn_pip() {
  "$CONDA_EXE" run --no-capture-output -n "$CNN_ENV_NAME" python -m pip "$@"
}

# Create a named environment under the user's Conda installation. A user-space
# Miniconda install stores this below the user's profile and needs no elevation.
if ! "$CONDA_EXE" run -n "$CNN_ENV_NAME" python -c "import sys; assert sys.version_info[:2] == (3, 11)" >/dev/null 2>&1; then
  print "Creating user Conda environment: $CNN_ENV_NAME"
  "$CONDA_EXE" create -y -n "$CNN_ENV_NAME" python=3.11 pip
fi

# Install only project dependencies. CuPy is not required by this repository.
if ! cnn_python - <<'PY' >/dev/null 2>&1
import numpy, pandas, scipy, matplotlib, PIL, psutil
PY
then
  print "Installing project Python dependencies into Conda environment: $CNN_ENV_NAME"
  cnn_pip install --upgrade pip
  cnn_pip install \
    'numpy>=1.26,<3' \
    'pandas>=2.1,<4' \
    'scipy>=1.11,<2' \
    'matplotlib>=3.8,<4' \
    'Pillow>=10,<13' \
    'psutil>=5.9,<8'
fi

# Install/check PyTorch separately so a CPU wheel is not silently accepted when
# the experiment requests CUDA. The NVIDIA driver may be newer than the wheel's
# CUDA runtime; that is expected and supported by the driver compatibility path.
if [[ "$DEVICE" == "cuda" ]]; then
  if ! cnn_python - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
  then
    print "Installing CUDA-enabled PyTorch into $CNN_ENV_NAME from: $TORCH_INDEX_URL"
    cnn_pip install --upgrade torch --index-url "$TORCH_INDEX_URL"
  fi
else
  if ! cnn_python -c 'import torch' >/dev/null 2>&1; then
    print "Installing CPU-capable PyTorch into $CNN_ENV_NAME"
    cnn_pip install --upgrade torch
  fi
fi

cnn_python - "$DEVICE" <<'PY'
import sys, torch
requested=sys.argv[1]
print('Conda Python:', sys.executable)
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if requested == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDA was requested but PyTorch cannot access the GPU.')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY
