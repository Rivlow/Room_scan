#!/usr/bin/env bash
# Installe, dans WSL Ubuntu 24.04, ce qu'il faut pour compiler RoomScan avec xtool :
#   paquets système (usbmuxd, libimobiledevice, FUSE pour l'AppImage), Swift via swiftly, xtool.
# Usage (depuis Windows) :
#   wsl -d Ubuntu-24.04 -u root -- bash setup_wsl.sh system     # paquets apt (root)
#   wsl -d Ubuntu-24.04 -- bash setup_wsl.sh user               # Swift + xtool (utilisateur)
set -euo pipefail

step="${1:-}"
case "$step" in
system)
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y curl ca-certificates gnupg2 git zip unzip xz-utils \
        usbmuxd libimobiledevice-utils libfuse2t64 \
        binutils libc6-dev libcurl4-openssl-dev libedit2 libgcc-13-dev libncurses-dev libpython3-dev \
        libsqlite3-0 libstdc++-13-dev libxml2-dev libz3-dev pkg-config tzdata zlib1g-dev
    echo "== paquets système OK"
    ;;
user)
    cd "$HOME"   # swiftly écrit un .swift-version dans le dossier courant
    arch="$(uname -m)"
    if ! command -v swiftly >/dev/null 2>&1 && [ ! -x "$HOME/.local/share/swiftly/bin/swiftly" ]; then
        tmp="$(mktemp -d)"
        curl -fsSL "https://download.swift.org/swiftly/linux/swiftly-${arch}.tar.gz" -o "$tmp/swiftly.tar.gz"
        tar -xzf "$tmp/swiftly.tar.gz" -C "$tmp"
        "$tmp/swiftly" init --assume-yes --skip-install --quiet-shell-followup
        rm -rf "$tmp"
    fi
    # shellcheck disable=SC1091
    . "${SWIFTLY_HOME_DIR:-$HOME/.local/share/swiftly}/env.sh"
    swiftly install --use --assume-yes latest
    swiftly link
    hash -r
    swift --version

    mkdir -p "$HOME/.local/bin"
    curl -fL "https://github.com/xtool-org/xtool/releases/latest/download/xtool-${arch}.AppImage" \
        -o "$HOME/.local/bin/xtool"
    chmod +x "$HOME/.local/bin/xtool"
    grep -q 'HOME/.local/bin' "$HOME/.bashrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    "$HOME/.local/bin/xtool" --help | head -n 3
    echo "== Swift + xtool OK.  Reste : 'xtool setup' (connexion Apple + chemin de Xcode.xip)"
    ;;
*)
    echo "usage: $0 system|user" >&2; exit 2 ;;
esac
