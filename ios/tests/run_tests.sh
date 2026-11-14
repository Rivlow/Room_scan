#!/usr/bin/env bash
# Tests du code Swift pur de l'application (sans ARKit), compilés et lancés sous Linux.
# Depuis Windows : wsl -d Ubuntu-24.04 -- bash Room_scan/ios/tests/run_tests.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HOME/.local/share/swiftly/env.sh"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
# main.swift : les instructions de haut niveau du test ne sont permises que dans ce fichier
cp "$here/coverage_tests.swift" "$out/main.swift"
swiftc -O "$here/../RoomScan/Sources/CoverageMap.swift" "$out/main.swift" -o "$out/coverage_tests"
"$out/coverage_tests"
