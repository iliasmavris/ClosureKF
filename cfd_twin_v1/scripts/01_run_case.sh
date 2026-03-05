#!/bin/bash
# 01_run_case.sh - Run OpenFOAM case in WSL2
#
# OpenFOAM cannot handle paths with spaces, so the case is copied to
# /tmp/cfd_twin_v1/case for meshing/solving, then probe results are
# copied back to the Windows source tree.
#
# Usage (from Windows):
#   wsl bash '/mnt/c/.../cfd_twin_v1/scripts/01_run_case.sh' [endTime]
#
# Optional argument: endTime override (e.g., 10 for test run)

set -e

# Source OpenFOAM environment
source /usr/lib/openfoam/openfoam2506/etc/bashrc

# Locate the case on the Windows side
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIN_CASE="$(dirname "$SCRIPT_DIR")/case"
WSL_CASE="/tmp/cfd_twin_v1/case"

echo "============================================================"
echo "OpenFOAM CFD Twin v1 - Case Runner"
echo "============================================================"
echo "Windows case: $WIN_CASE"
echo "WSL workdir:  $WSL_CASE"
echo ""

# Copy case to WSL-native filesystem (avoids space-in-path issue)
echo "--- Copying case to WSL filesystem ---"
rm -rf "$WSL_CASE"
mkdir -p "$(dirname "$WSL_CASE")"
cp -r "$WIN_CASE" "$WSL_CASE"
echo "Done."
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

# Step 4: Copy probe results back to Windows
echo ""
echo "--- Copying results back ---"
if [ -d "$WSL_CASE/postProcessing" ]; then
    mkdir -p "$WIN_CASE/postProcessing"
    cp -r "$WSL_CASE/postProcessing/"* "$WIN_CASE/postProcessing/"
    echo "Probe data copied to Windows."
else
    echo "WARNING: No postProcessing directory found!"
fi

echo ""
echo "============================================================"
echo "CASE RUN COMPLETE"
echo "============================================================"
