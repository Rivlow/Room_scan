#!/usr/bin/env bash
# Copie les sources dans le système de fichiers Linux (build bien plus rapide que sur /mnt/c) puis lance xtool.
# Usage (depuis Windows) :
#   wsl -d Ubuntu-24.04 -- bash setup-chemin/build_wsl.sh            # compile seulement
#   wsl -d Ubuntu-24.04 -- bash setup-chemin/build_wsl.sh run        # compile et installe sur l'iPhone
set -uo pipefail

src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dst="$HOME/roomscan"

. "$HOME/.local/share/swiftly/env.sh"
export PATH="$HOME/.local/bin:$PATH"
# usbipd bloque xtool (message usbmux de 65536 octets, xtool issue #19) : on passe par le service Apple d'iTunes sous
# Windows, joint à travers le portproxy 27015 de l'interface WSL, via usbmux_relay.py (voir ce fichier).
gateway="$(ip route list default | awk '{print $3; exit}')"
if [ "${1:-build}" = run ] && ! timeout 3 bash -c "exec 3<>/dev/tcp/$gateway/27015" 2>/dev/null; then
    cat >&2 <<EOF
Le portproxy Windows $gateway:27015 (service Apple d'iTunes) ne répond pas : xtool attendrait l'iPhone sans fin.
Après un redémarrage de Windows, la redirection n'est pas montée tant que le service IP Helper n'est pas relancé.
Dans un PowerShell administrateur :   Restart-Service iphlpsvc -Force
puis vérifier :   netsh interface portproxy show all   (172.30.x.1:27015 -> 127.0.0.1:27015, adresse = passerelle WSL)
EOF
    exit 3
fi
python3 -I "$src/usbmux_relay.py" "$gateway" &
relay=$!
trap 'kill "$relay" 2>/dev/null' EXIT
sleep 0.5
export USBMUXD_SOCKET_ADDRESS=127.0.0.1:27016

mkdir -p "$dst"
rm -rf "$dst/RoomScan"
cp -r "$src/RoomScan" "$src/Package.swift" "$src/xtool.yml" "$dst/"
cd "$dst"

case "${1:-build}" in
    build) xtool dev build ;;
    run)   xtool dev run --usb ;;
    *)     echo "usage: $0 [build|run]" >&2; exit 2 ;;
esac
