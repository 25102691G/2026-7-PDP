#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export XDG_CACHE_HOME="$PWD/tooling/cache"
export TECTONIC_CACHE_DIR="$PWD/tooling/cache/tectonic"
mkdir -p "$XDG_CACHE_HOME" "$TECTONIC_CACHE_DIR" build
./tooling/tectonic --keep-logs --keep-intermediates --outdir build main.tex > build/compile_console.log 2>&1
cp build/main.pdf build/dissertation_initial_draft.pdf
printf '%s\n' 'Compiled build/dissertation_initial_draft.pdf'
