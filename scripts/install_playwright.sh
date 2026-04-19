#!/usr/bin/env bash
set -euo pipefail

# Install Playwright (Chromium) for a given Python/conda environment.
# Usage: bash install_playwright.sh [PYTHON]
# Default PYTHON: python
# Supported: Ubuntu 22.04/24.04, macOS (arm64/amd64)
#
# Examples:
#   bash install_playwright.sh
#   bash install_playwright.sh /path/to/conda/envs/agentos/bin/python

PYTHON="${1:-python}"

echo "Python: ${PYTHON}"

# Detect OS
OS="$(uname -s)"
case "$OS" in
  Linux)
    if ! grep -qi ubuntu /etc/os-release 2>/dev/null; then
      echo "Warning: non-Ubuntu Linux detected. Proceeding with Ubuntu path."
    fi
    PLATFORM="ubuntu"
    ;;
  Darwin)
    PLATFORM="macos"
    ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Step 1: Install playwright Python package if not already present
# ---------------------------------------------------------------------------
echo "[1/3] Installing playwright Python package..."
"${PYTHON}" -m pip install --upgrade playwright

# ---------------------------------------------------------------------------
# Step 2: Install Chromium browser binary via playwright
# ---------------------------------------------------------------------------
echo "[2/3] Installing Chromium via playwright..."
"${PYTHON}" -m playwright install chromium

# ---------------------------------------------------------------------------
# Step 3: Install Chromium system dependencies (Ubuntu only)
# ---------------------------------------------------------------------------
if [[ "$PLATFORM" == "ubuntu" ]]; then
  echo "[3/3] Installing Chromium system dependencies..."
  "${PYTHON}" -m playwright install-deps chromium
else
  echo "[3/3] Skipping system dependencies (not required on macOS)."
fi

echo
echo "Done."
CHROMIUM_PATH=$("${PYTHON}" -c "
import subprocess, sys, pathlib

result = subprocess.run(
    [sys.executable, '-m', 'playwright', 'show-path', 'chromium'],
    capture_output=True, text=True
)
browser_dir = result.stdout.strip()

import platform
if platform.system() == 'Darwin':
    # macOS: binary is inside the .app bundle
    candidates = [
        'chromium-mac/chrome-mac/Chromium.app/Contents/MacOS/Chromium',
        'chromium-mac-arm64/chrome-mac/Chromium.app/Contents/MacOS/Chromium',
    ]
    base = pathlib.Path(browser_dir).parent
    for rel in candidates:
        p = base / rel
        if p.exists():
            print(p)
            break
    else:
        for p in base.rglob('Chromium'):
            if p.is_file():
                print(p)
                break
else:
    # Linux
    base = pathlib.Path(browser_dir).parent
    binary = base / 'chrome-linux64' / 'chrome'
    if binary.exists():
        print(binary)
    else:
        for p in base.rglob('chrome'):
            if p.is_file():
                print(p)
                break
" 2>/dev/null || true)

if [[ -n "${CHROMIUM_PATH}" ]]; then
  echo "Chromium binary: ${CHROMIUM_PATH}"
else
  echo "Chromium binary: (run '${PYTHON} -m playwright show-path chromium' to locate)"
fi
