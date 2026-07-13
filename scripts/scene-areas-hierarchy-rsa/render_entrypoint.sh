#!/bin/bash
set -e

# Copy the mounted scripts/ into the container's own filesystem so Quarto's
# project cache (.quarto/, SQLite/Deno KV-backed) doesn't hit Windows
# bind-mount I/O issues. data/ has no such cache writes anymore (the RDM and
# permutation pickles were removed), so a symlink back to the mounted folder
# is enough for the atlas downloads/candidate_metrics.tsv it still writes.
cp -r /workbench/scripts/. /build/scripts/
ln -s /workbench/data /build/data

cd /build/scripts/scene-areas-hierarchy-rsa
quarto render scene-areas-hierarchy-rsa.qmd --output-dir "/Nole-Nexus Website/Workbench"
