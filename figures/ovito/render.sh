#!/bin/bash
# Headless OVITO 3.17 in a venv at ~/.venvs/ovito: libOpenGL from ~/.venvs/ovito/extra, Vulkan from lavapipe.
# Usage (from the project root): bash figures/ovito/render.sh <script.py> [args]
L=$(dirname "$(find "$HOME/.venvs/ovito/extra" -name 'libOpenGL.so.0' | head -1)")
exec env -u DISPLAY -u WAYLAND_DISPLAY LD_LIBRARY_PATH="$L" QT_QPA_PLATFORM=ovitoheadless \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  "$HOME/.venvs/ovito/bin/python" "$@"
