#!/bin/bash
# 11_run_flow_case.sh - Run OpenFOAM with multi-height probes in WSL2
#
# Copies cfd_twin_v1/case/ to /tmp/virtual_lab_v1/case/,
# overlays the modified probes file (4 heights), runs blockMesh + pimpleFoam,
# copies postProcessing back to virtual_lab_v1/flow_inputs/.
#
# Usage (from Windows):
#   wsl bash '/mnt/c/.../virtual_lab_v1/scripts/11_run_flow_case.sh' [endTime]
#
# Optional argument: endTime override (e.g., 10 for test run)

set -e

# Source OpenFOAM environment
source /usr/lib/openfoam/openfoam2506/etc/bashrc

# Locate directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VL_ROOT="$(dirname "$SCRIPT_DIR")"
CFD_CASE="$(dirname "$VL_ROOT")/cfd_twin_v1/case"
OVERLAY_DIR="$VL_ROOT/flow_inputs/case_overlay"
WSL_CASE="/tmp/virtual_lab_v1/case"

echo "============================================================"
echo "Virtual Lab v1 - Multi-Height Probe CFD Run"
echo "============================================================"
echo "Source case:  $CFD_CASE"
echo "Probe overlay: $OVERLAY_DIR"
echo "WSL workdir:  $WSL_CASE"
echo ""

# Check prerequisites
if [ ! -d "$CFD_CASE" ]; then
    echo "ERROR: cfd_twin_v1/case not found at $CFD_CASE"
    exit 1
fi

if [ ! -f "$OVERLAY_DIR/system/probes" ]; then
    echo "ERROR: Overlay probes file not found. Run 10_add_probes.py first."
    exit 1
fi

# Copy case to WSL-native filesystem
echo "--- Copying case to WSL filesystem ---"
rm -rf "$WSL_CASE"
mkdir -p "$(dirname "$WSL_CASE")"
cp -r "$CFD_CASE" "$WSL_CASE"
echo "Done."
echo ""

# Overlay multi-height probes
echo "--- Overlaying multi-height probes ---"
cp "$OVERLAY_DIR/system/probes" "$WSL_CASE/system/probes"
echo "Probes file replaced with 4-height version."
echo ""

cd "$WSL_CASE"

# Optional endTime override
if [ -n "$1" ]; then
    echo "Overriding endTime to $1 s"
    foamDictionary system/controlDict -entry endTime -set "$1"
    echo ""
fi

# Step 1: Generate mesh
echo "--- blockMesh ---"
blockMesh 2>&1 | tail -5
echo ""

# Step 2: Check mesh quality
echo "--- checkMesh ---"
checkMesh 2>&1 | tail -10
echo ""

# Step 3: Run solver
echo "--- pimpleFoam ---"
echo "Starting at $(date)"
pimpleFoam 2>&1 | tail -20
echo ""
echo "Finished at $(date)"

# Step 4: Attempt wallShearStress (non-blocking if unavailable)
echo ""
echo "--- wallShearStress (optional) ---"
if command -v postProcess &> /dev/null; then
    postProcess -func wallShearStress 2>&1 | tail -5 || \
        echo "wallShearStress not available or failed -- skipping (not blocking)"
else
    echo "postProcess not available -- skipping wallShearStress"
fi

# Step 5: Copy probe results back to Windows
echo ""
echo "--- Copying results back ---"
DEST="$VL_ROOT/flow_inputs"
if [ -d "$WSL_CASE/postProcessing" ]; then
    mkdir -p "$DEST/postProcessing"
    cp -r "$WSL_CASE/postProcessing/"* "$DEST/postProcessing/"
    echo "Probe data copied to: $DEST/postProcessing/"
else
    echo "WARNING: No postProcessing directory found!"
fi

echo ""
echo "============================================================"
echo "MULTI-HEIGHT PROBE RUN COMPLETE"
echo "============================================================"
