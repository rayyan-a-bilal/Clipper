#!/usr/bin/env bash
# clipper font installer — MANUAL RUN ONLY (never invoked by tests).
#
# Downloads caption fonts into <repo>/assets/fonts/. Every font is
# SKIPped if already present (idempotent). Prints the resolved target
# path of each font.
#
# Usage:  bash scripts/install_fonts.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FONTS_DIR="${FONTS_DIR:-${REPO_ROOT}/assets/fonts}"
mkdir -p "${FONTS_DIR}"

# --- M7a: Montserrat ExtraBold (ASS Fontname "Montserrat ExtraBold") ----
# Source: Montserrat (SIL OFL) canonical TTF — JulietaUla/Montserrat.
MONTserrat="https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf"
MONT_TARGET="${FONTS_DIR}/Montserrat-ExtraBold.ttf"

install_one() {
  local url="$1" target="$2"
  if [[ -f "${target}" && -s "${target}" ]]; then
    echo "[install_fonts] skip (already present): ${target}"
  else
    echo "[install_fonts] downloading ${url}"
    local tmp
    tmp="$(mktemp "${target}.XXXXXX")"
    # shellcheck disable=SC2064
    trap "rm -f '${tmp}'" RETURN
    curl -fSL --retry 3 --retry-delay 2 -o "${tmp}" "${url}"
    mv "${tmp}" "${target}"
  fi
  echo "[install_fonts] target: ${target}"
}

install_one "${MONTserrat}" "${MONT_TARGET}"

echo "[install_fonts] done."
