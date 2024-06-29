#!/bin/bash
set -eu

here=$(dirname "$0")
py="${here}/../venv/bin/python"

"$py" "${here}/../vendor/TripoSR/run.py" "$1" --output-dir "$2"
"$py" "${here}/postproc_tripo.py" "$2"/0/mesh.obj "$2/mesh.obj"

