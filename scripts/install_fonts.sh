#!/usr/bin/env bash
# clipper font installer — MANUAL RUN ONLY (never invoked by tests).
#
# Downloads caption fonts into <repo>/assets/fonts/. Every font is
# SKIPped if already present (idempotent). Prints the resolved target
# path of each font.
#
# M9c (docs/spec/m9.md m9-3, explicitly authorizing this change): after
# installing, rebuilds the font FAMILY inventory assets/fonts/inventory.json
# via fontconfig's fc-scan (libass matches the INTERNAL family name, not the
# filename stem). fc-scan comes from fontconfig (already present for libass)
# — NO new pip dependency. fc-scan unavailable/empty -> filename-stem
# fallback flagged fallback_used=true + a LOUD warning. Idempotent: every
# run rebuilds the inventory from the font files on disk. This script
# remains the ONLY font add path — NO auto-download of anything else.
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

# --- M9c (m9-3): font FAMILY inventory (fc-scan, stem fallback flagged) --
#
# Single source of truth is Python: clipper.adapters.assets.build_font_inventory
# (CPU-only, unit-tested with a simulated missing fc-scan). The bash
# fallback below is only for a host without a usable Python+pydantic and
# also flags fallback_used=true on every stem-derived family.

INVENTORY_FILE="${FONTS_DIR}/inventory.json"

build_inventory_bash() {
  local f base stem family fb first_entry=1
  local entries=""
  local file_list
  file_list="$(find "${FONTS_DIR}" -maxdepth 1 -type f \( -name '*.ttf' -o -name '*.otf' -o -name '*.ttc' \) | LC_ALL=C sort)"
  if command -v fc-scan >/dev/null 2>&1; then :; else
    echo "WARNING: [install_fonts] fc-scan is UNAVAILABLE — deriving families from FILENAME STEMS (fallback_used=true in inventory.json)" >&2
  fi
  while IFS= read -r f; do
    [[ -z "${f}" ]] && continue
    base="$(basename "${f}")"
    stem="${base%.*}"
    family=""
    if command -v fc-scan >/dev/null 2>&1; then
      family="$(fc-scan --format '%{family}\n' "${f}" 2>/dev/null | head -n1 | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || true)"
    fi
    if [[ -n "${family}" ]] && [[ "${family}" == *,* ]]; then
      # fc-scan lists several names (family-name + full name); keep the most
      # specific (longest; last on ties) — same rule as the Python builder (M9c-J1)
      local best="" n
      local -a names
      IFS=',' read -ra names <<< "${family}"
      for n in "${names[@]}"; do
        n="$(printf '%s' "${n}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -z "${n}" ]] && continue
        if (( ${#n} >= ${#best} )); then best="${n}"; fi
      done
      family="${best}"
    fi
    fb="false"
    if [[ -z "${family}" ]]; then
      family="${stem}"
      fb="true"
      echo "WARNING: [install_fonts] no family from fc-scan for ${base} — using filename stem '${stem}' (fallback_used=true)" >&2
    fi
    if [[ "${first_entry}" -eq 1 ]]; then first_entry=0; else entries="${entries},"; fi
    entries="${entries}
    {
      \"file\": \"${base}\",
      \"family\": \"${family//\\/\\\\}\",
      \"stem\": \"${stem//\\/\\\\}\",
      \"fallback_used\": ${fb}
    }"
  done <<< "${file_list}"
  printf '{\n  "schema_version": 1,\n  "fonts": [%s\n  ]\n}\n' "${entries}"
}

echo "[install_fonts] building font inventory (m9-3, fc-scan): ${INVENTORY_FILE}"

# pick the first python (venv preferred if active) that can run the builder
pick_python() {
  local cand
  for cand in python python3; do
    if command -v "${cand}" >/dev/null 2>&1 && \
       PYTHONPATH="${REPO_ROOT}/src" "${cand}" -c 'import pydantic, clipper.adapters.assets' >/dev/null 2>&1; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

PY="$(pick_python || true)"

if [[ -n "${PY}" ]]; then
  if INVENTORY_JSON="$(
    PYTHONPATH="${REPO_ROOT}/src" "${PY}" - "${FONTS_DIR}" <<'PYEOF'
import sys
from clipper.adapters.assets import build_font_inventory

print(build_font_inventory(sys.argv[1]).model_dump_json(indent=2))
PYEOF
  )"; then
    printf '%s\n' "${INVENTORY_JSON}" > "${INVENTORY_FILE}"
    echo "[install_fonts] inventory rebuilt: ${INVENTORY_FILE}"
  else
    echo "WARNING: [install_fonts] python inventory builder failed — using the bash fallback" >&2
    build_inventory_bash > "${INVENTORY_FILE}"
    echo "[install_fonts] inventory rebuilt (bash fallback): ${INVENTORY_FILE}"
  fi
else
  echo "WARNING: [install_fonts] usable python+pydantic not found — using the bash inventory fallback" >&2
  build_inventory_bash > "${INVENTORY_FILE}"
  echo "[install_fonts] inventory rebuilt (bash fallback): ${INVENTORY_FILE}"
fi

echo "[install_fonts] done."
