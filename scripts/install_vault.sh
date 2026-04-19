#!/usr/bin/env bash
set -euo pipefail

# Cross-platform Vault installer:
# - Ubuntu: installs build deps via apt, uses dpkg for arch detection
# - macOS:  installs build deps via Homebrew, uses uname for arch detection
# - Both:   installs Go + Vault from source into INSTALL_PREFIX
#
# Tested targets: Ubuntu 22.04/24.04, macOS (arm64/amd64)
# Supported arch: amd64/arm64

INSTALL_PREFIX="${1:-/usr/local}"
VAULT_TAG="${2:-v1.18.3}"
VAULT_DATA_DIR="${INSTALL_PREFIX}/vault/data"
VAULT_CONFIG_DIR="${INSTALL_PREFIX}/vault/config"
VAULT_BIN_DIR="${INSTALL_PREFIX}/vault/bin"
VAULT_LOG="${INSTALL_PREFIX}/vault/vault.log"

echo "Install prefix: ${INSTALL_PREFIX}"
mkdir -p "${INSTALL_PREFIX}"

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
    ;;
  Darwin)
    PLATFORM="macos"
    ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

echo "[1/8] Checking architecture..."
if [[ "$PLATFORM" == "ubuntu" ]]; then
  ARCH="$(dpkg --print-architecture)"
  case "$ARCH" in
    amd64) GOARCH="amd64" ;;
    arm64) GOARCH="arm64" ;;
    *)
      echo "Unsupported architecture: $ARCH"
      exit 1
      ;;
  esac
  GO_OS="linux"
else
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64)  GOARCH="amd64" ;;
    arm64)   GOARCH="arm64" ;;
    *)
      echo "Unsupported architecture: $ARCH"
      exit 1
      ;;
  esac
  GO_OS="darwin"
fi

echo "[2/8] Installing dependencies..."
if [[ "$PLATFORM" == "ubuntu" ]]; then
  sudo apt-get update
  sudo apt-get install -y \
    curl \
    wget \
    git \
    make \
    unzip \
    jq \
    build-essential \
    ca-certificates

  echo "[2b/8] Installing nvm and Node.js 20..."
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  export NVM_DIR="${HOME}/.nvm"
  # shellcheck source=/dev/null
  source "${NVM_DIR}/nvm.sh"
  nvm install 20
  nvm use 20
  npm install -g pnpm yarn
else
  if ! command -v brew &>/dev/null; then
    echo "Homebrew not found. Install it from https://brew.sh first."
    exit 1
  fi
  brew install curl wget git make unzip jq
  # Install nvm if not present
  if ! command -v nvm &>/dev/null && [[ ! -s "${HOME}/.nvm/nvm.sh" ]]; then
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
  fi
  export NVM_DIR="${HOME}/.nvm"
  # shellcheck source=/dev/null
  source "${NVM_DIR}/nvm.sh"
  nvm install 20
  nvm use 20
  npm install -g pnpm yarn
fi

echo "[3/8] Using Go 1.23.8 (required by Vault ${VAULT_TAG})..."
GO_VERSION="go1.23.8"

GO_TARBALL="${GO_VERSION}.${GO_OS}-${GOARCH}.tar.gz"
GO_URL="https://go.dev/dl/${GO_TARBALL}"

echo "Latest stable Go: ${GO_VERSION}"
echo "Downloading: ${GO_URL}"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

cd "${TMP_DIR}"
curl -fL --progress-bar -o "${GO_TARBALL}" "${GO_URL}"

echo "[4/8] Installing Go to ${INSTALL_PREFIX}/go ..."
${SUDO} rm -rf "${INSTALL_PREFIX}/go"
${SUDO} tar -C "${INSTALL_PREFIX}" -xzf "${GO_TARBALL}"

# Detect shell profile
if [[ "$PLATFORM" == "macos" ]]; then
  SHELL_PROFILE="${HOME}/.zshrc"
else
  SHELL_PROFILE="${HOME}/.bashrc"
fi

# Remove existing Go entries before re-adding
sed -i '' '/# Go/d' "${SHELL_PROFILE}"
sed -i '' '/export GOPATH=/d' "${SHELL_PROFILE}"
sed -i '' '/export PATH=.*go\/bin/d' "${SHELL_PROFILE}"
sed -i '' '/export PATH=.*gopath\/bin/d' "${SHELL_PROFILE}"

{
  echo ''
  echo '# Go'
  echo "export PATH=\$PATH:${INSTALL_PREFIX}/go/bin"
  echo "export GOPATH=${INSTALL_PREFIX}/gopath"
  echo "export PATH=\$PATH:${INSTALL_PREFIX}/gopath/bin"
} >> "${SHELL_PROFILE}"

export PATH="$PATH:${INSTALL_PREFIX}/go/bin"
export GOPATH="${INSTALL_PREFIX}/gopath"
export PATH="$PATH:${GOPATH}/bin"

# Remove existing Vault entries before re-adding
sed -i '' '/# Vault/d' "${SHELL_PROFILE}"
sed -i '' '/export VAULT_ADDR=/d' "${SHELL_PROFILE}"
sed -i '' "\|export PATH=${VAULT_BIN_DIR}|d" "${SHELL_PROFILE}"

{
  echo ''
  echo '# Vault'
  echo "export PATH=${VAULT_BIN_DIR}:\$PATH"
  echo "export VAULT_ADDR='http://127.0.0.1:8200'"
} >> "${SHELL_PROFILE}"

echo "[5/8] Verifying Go installation..."
go version
go env GOPATH >/dev/null

echo "[6/8] Cloning Vault source..."
mkdir -p "${GOPATH}/src/hashicorp"
cd "${GOPATH}/src/hashicorp"

if [[ -d vault ]]; then
  echo "Vault source already exists, updating..."
  cd vault
  git fetch --tags --prune
else
  git clone https://github.com/hashicorp/vault.git
  cd vault
fi

echo "Checking out tag ${VAULT_TAG}..."
git checkout "${VAULT_TAG}"

echo "[7/8] Building Vault from source with UI..."
# Ensure Node 20 is active for UI build
export NVM_DIR="${HOME}/.nvm"
# shellcheck source=/dev/null
source "${NVM_DIR}/nvm.sh"
nvm use 20
make bootstrap
export NODE_OPTIONS="--max-old-space-size=4096"
make static-dist
make dev-ui

if [[ ! -f "${GOPATH}/src/hashicorp/vault/bin/vault" ]]; then
  echo "Build finished but binary not found at expected path."
  exit 1
fi

echo "[8/8] Installing Vault binary to ${VAULT_BIN_DIR}..."
${SUDO} mkdir -p "${VAULT_BIN_DIR}"
${SUDO} install -m 0755 "${GOPATH}/src/hashicorp/vault/bin/vault" "${VAULT_BIN_DIR}/vault"

# setcap is skipped: container environments typically deny file capabilities,
# which causes the binary to be unexecutable. mlock is disabled in vault.hcl instead.

echo "[+] Generating Vault production config..."
${SUDO} mkdir -p "${VAULT_DATA_DIR}"
${SUDO} mkdir -p "${VAULT_CONFIG_DIR}"

${SUDO} tee "${VAULT_CONFIG_DIR}/vault.hcl" > /dev/null <<EOF
ui            = true
disable_mlock = true

storage "file" {
  path = "${VAULT_DATA_DIR}"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = true
}

api_addr = "http://127.0.0.1:8200"
EOF

echo "[+] Starting Vault in background..."
${SUDO} mkdir -p "$(dirname "${VAULT_LOG}")"
nohup "${VAULT_BIN_DIR}/vault" server -config="${VAULT_CONFIG_DIR}/vault.hcl" > "${VAULT_LOG}" 2>&1 &
echo "Vault PID: $!"
echo "Logs: ${VAULT_LOG}"

echo
echo "Done."
echo "Go version:    $(go version)"
echo "Vault version: $(vault --version)"
echo
echo "First-time setup:"
echo "  export VAULT_ADDR='http://127.0.0.1:8200'"
echo "  vault operator init"
echo "  vault operator unseal  # run 3 times with different keys"
