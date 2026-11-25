#!/bin/bash
# Generate start poses and run the expert on each requested yard in a CPU container.
R=${FLAMBE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
R=$(realpath -m -- "$R")
T=$(realpath -m -- "${FLAMBE_TERRA:-$R/terra}")
YARDS=$(realpath -m -- "${FLAMBE_YARDS:-$R/synth_yards}")
YARDS_FALLBACK=$(realpath -m -- "${FLAMBE_YARDS_FALLBACK:-$R/flambe_yards}")
GATE2=$(realpath -m -- "${FLAMBE_GATE2:-$R/research/02_experiments/gate2}")
# Docker mount targets must be absolute paths inside the container.
CONTAINER_ROOT=${FLAMBE_CONTAINER_ROOT:-/flambe}
for Y in "$@"; do
  O=$GATE2/$Y; mkdir -p "$O"
  echo "[g2x] $(date +%m-%d\ %H:%M) $Y start"
  docker run --rm --name "g2x_$Y" --cpus 16 -e "YARD=$Y" -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -v "$R/common:$CONTAINER_ROOT/common:ro" -v "$R/expert:$CONTAINER_ROOT/expert:ro" \
    -v "$YARDS_FALLBACK:$CONTAINER_ROOT/flambe_yards:ro" -v "$T:$CONTAINER_ROOT/terra" \
    -v "$YARDS:$CONTAINER_ROOT/synth_yards:ro" -v "$O:$CONTAINER_ROOT/out" -w "$CONTAINER_ROOT/common" \
    -e "FLAMBE_OUTPUT=$CONTAINER_ROOT/out" -e "FLAMBE_ROOT=$CONTAINER_ROOT" -e "FLAMBE_TERRA=$CONTAINER_ROOT/terra" \
    -e "FLAMBE_YARDS=$CONTAINER_ROOT/synth_yards" -e "FLAMBE_YARDS_FALLBACK=$CONTAINER_ROOT/flambe_yards" \
    -e "PYTHONPATH=$CONTAINER_ROOT/common:$CONTAINER_ROOT/expert:${FLAMBE_SOUSVIDE_SRC:-$CONTAINER_ROOT/terra/SousVide_official/src}:$CONTAINER_ROOT/terra/FiGS-Standalone/src" figs-sv:latest \
    bash -c 'python3 "$1/lead_expert_yard.py" draw "$2/starts.json" && python3 "$1/lead_expert_yard.py" fly "$2/starts.json" "$2/expert.json" 16' \
    bash "$CONTAINER_ROOT/common" "$CONTAINER_ROOT/out" > "$O/expert.log" 2>&1
  echo "[g2x] $(date +%m-%d\ %H:%M) $Y done: $(grep -E '\[0b\] all:' "$O/expert.log" | tail -1)"
done
