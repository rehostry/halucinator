#!/usr/bin/env bash
# Run the Falcon verification the way it is meant to be run.
#
# This exists because the *documented* path has been the broken one four times
# in this effort: the extension jar, `space:` in the config loader, the Ghidra
# CLI dropping `space` and `emulate`, and the config attaching the wrong
# peripheral model. Each was invisible because verification went through ad-hoc
# Python that built its own regions. A script that is the documented path, and
# is itself run, is the check on that.
#
# Prerequisites, all reported rather than assumed:
#   GHIDRA_INSTALL_DIR   a Ghidra install with ghidra-falcon under
#                        Ghidra/Processors/Falcon, jar included
#   FALCON_FIRMWARE_DIR  a linux-firmware `nvidia/` tree (not vendored)
#   NOUVEAU_GR_DIR       optional; nouveau's nvkm/engine/gr, for the
#                        cross-check against its context tables
#
# Usage:  test/falcon_fecs/verify.sh [--fast] [extra pytest args...]
#   --fast  skip the two slow files (ctxctl boots the microcode per test and
#           corpus sweeps 28 images), leaving roughly a minute of tests
set -uo pipefail

cd "$(dirname "$0")/../.."
fail=0

need() {
    if [ -z "${!1:-}" ]; then
        echo "MISSING: \$$1 -- $2" >&2
        fail=1
    else
        echo "  \$$1 = ${!1}"
    fi
}

echo "Prerequisites:"
need GHIDRA_INSTALL_DIR "a Ghidra install carrying the Falcon processor module"
need FALCON_FIRMWARE_DIR "a linux-firmware nvidia/ tree; see README.md"
[ -n "${NOUVEAU_GR_DIR:-}" ] \
    && echo "  \$NOUVEAU_GR_DIR = $NOUVEAU_GR_DIR" \
    || echo "  \$NOUVEAU_GR_DIR unset -- the nouveau cross-check will skip"
[ "$fail" = 0 ] || exit 2

mod="$GHIDRA_INSTALL_DIR/Ghidra/Processors/Falcon"
for f in "$mod/data/languages/falcon_fuc5.sla" "$mod/lib/Falcon.jar"; do
    if [ -f "$f" ]; then
        echo "  found $(basename "$f")"
    else
        echo "MISSING: $f" >&2
        echo "  install ghidra-falcon's built extension, not just data/languages:" >&2
        echo "  the pspec names a Java instruction state modifier, and without" >&2
        echo "  the jar Ghidra cannot construct an emulator at all." >&2
        exit 2
    fi
done

FILES=(
    test/pytest/backends/test_falcon_conformance.py
    test/pytest/backends/test_falcon_engine.py
    test/pytest/backends/test_falcon_irq.py
    test/pytest/test_hal_config.py
)
if [ "${1:-}" = "--fast" ]; then
    shift
    echo
    echo "Fast set only; skipping test_falcon_ctxctl.py and test_falcon_corpus.py."
else
    FILES+=(
        test/pytest/backends/test_falcon_ctxctl.py
        test/pytest/backends/test_falcon_corpus.py
    )
fi

echo
echo "Running ${#FILES[@]} files..."
PYTHONPATH="src:test/pytest/helpers${PYTHONPATH:+:$PYTHONPATH}" \
    python -m pytest "${FILES[@]}" -p no:timeout "$@"
