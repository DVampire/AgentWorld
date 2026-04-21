#!/usr/bin/env bash
set -euo pipefail

# Install opencode and generate opencode.json with secrets from Vault.
# Usage: bash install_opencode.sh [INSTALL_PREFIX]
# Default INSTALL_PREFIX: /usr/local
# Supported: Ubuntu 22.04/24.04, macOS (arm64/amd64)

INSTALL_PREFIX="${1:-/usr/local}"
OPENCODE_DIR="${INSTALL_PREFIX}/opencode"
OPENCODE_CONFIG="${OPENCODE_DIR}/opencode.json"
OPENCODE_BIN_DIR="${INSTALL_PREFIX}/opencode/bin"

echo "Install prefix: ${INSTALL_PREFIX}"
mkdir -p "${OPENCODE_DIR}" "${OPENCODE_BIN_DIR}"

# Use sudo only when INSTALL_PREFIX is not writable by the current user
if [[ -w "${INSTALL_PREFIX}" ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

# Detect OS
OS="$(uname -s)"
case "$OS" in
  Linux)
    if ! grep -qi ubuntu /etc/os-release 2>/dev/null; then
      echo "Warning: non-Ubuntu Linux detected. Proceeding with Ubuntu (apt) path."
    fi
    PLATFORM="ubuntu"
    SHELL_PROFILE="${HOME}/.bashrc"
    ;;
  Darwin)
    PLATFORM="macos"
    SHELL_PROFILE="${HOME}/.zshrc"
    ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Step 1: Install opencode binary
# ---------------------------------------------------------------------------
echo "[1/3] Installing opencode..."
curl -fsSL https://opencode.ai/install | bash

# The official installer always puts the binary in ~/.opencode/bin.
# Move it to the requested INSTALL_PREFIX.
${SUDO} mkdir -p "${OPENCODE_BIN_DIR}"
${SUDO} mv "${HOME}/.opencode/bin/opencode" "${OPENCODE_BIN_DIR}/opencode"

# Export so opencode is usable in subsequent steps
export PATH="${OPENCODE_BIN_DIR}:${PATH}"
export OPENCODE_CONFIG_PATH="${OPENCODE_CONFIG}"

# Remove existing opencode entries before re-adding
sed -i '' '/# opencode/d' "${SHELL_PROFILE}"
sed -i '' '/export OPENCODE_CONFIG=/d' "${SHELL_PROFILE}"
sed -i '' "\|export PATH=${OPENCODE_BIN_DIR}|d" "${SHELL_PROFILE}"

{
  echo ''
  echo '# opencode'
  echo "export OPENCODE_CONFIG=${OPENCODE_CONFIG}"
  echo "export PATH=${OPENCODE_BIN_DIR}:\$PATH"
} >> "${SHELL_PROFILE}"

# ---------------------------------------------------------------------------
# Step 2: Fetch secrets from Vault
# ---------------------------------------------------------------------------
echo "[2/3] Fetching secrets from Vault..."

: "${VAULT_ADDR:?VAULT_ADDR is not set}"
: "${VAULT_TOKEN:?VAULT_TOKEN is not set}"
: "${SECRET_ENGINE_PATH:=${VAULT_SECRET_PATH:-cubbyhole/env}}"

vault_get() {
  vault kv get -field="$1" "${SECRET_ENGINE_PATH}"
}

OPENROUTER_API_BASE="$(vault_get OPENROUTER_API_BASE)"
OPENROUTER_API_KEY="$(vault_get OPENROUTER_API_KEY)"
OPENAI_API_BASE="$(vault_get OPENAI_API_BASE)"
OPENAI_API_KEY="$(vault_get OPENAI_API_KEY)"
NEWAPI_API_BASE="$(vault_get NEWAPI_API_BASE)"
NEWAPI_API_KEY="$(vault_get NEWAPI_API_KEY)"

# ---------------------------------------------------------------------------
# Step 3: Write opencode.json
# ---------------------------------------------------------------------------
echo "[3/3] Writing ${OPENCODE_CONFIG}..."

${SUDO} tee "${OPENCODE_CONFIG}" > /dev/null <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "permission": {
    "bash": "allow",
    "edit": "allow",
    "write": "allow"
  },
  "provider": {
    "openrouter": {
      "options": {
        "baseURL": "${OPENROUTER_API_BASE}",
        "apiKey": "${OPENROUTER_API_KEY}"
      },
      "models": {
        "anthropic/claude-opus-4.6": {}
      }
    },

    "newapi": {
      "options": {
        "baseURL": "${NEWAPI_API_BASE}",
        "apiKey": "${NEWAPI_API_KEY}"
      },
      "models": {
        "claude-opus-4-6": {}
      }
    },

    "openai": {
      "options": {
        "baseURL": "${OPENAI_API_BASE}",
        "apiKey": "${OPENAI_API_KEY}"
      },
      "models": {
        "gpt-5.4": {},
        "gpt-5.4-pro": {}
      }
    }
  },
  "model": "openrouter/anthropic/claude-opus-4.6"
}
EOF

echo
echo "Done."
echo "Config: ${OPENCODE_CONFIG}"
echo "Binary: ${OPENCODE_BIN_DIR}/opencode"
echo
echo "Tip:"
echo "  opencode run 'hello world'"
