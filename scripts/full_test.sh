#!/usr/bin/env bash
# Full verification: lint, import layering, fast suite, slow integration, and the oracle tests
# on a default-profile dataset (generated if absent). This is what runs before a push.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== ruff";        uv run ruff check src tests || exit 1
echo "== format";      uv run ruff format --check src tests || exit 1
echo "== layers";      uv run lint-imports || exit 1
echo "== fast suite";  uv run pytest tests -q -m "not slow" -p no:warnings || exit 1

# the oracle must run on data from the CURRENT generator: a 5,000-vehicle dataset whose
# recorded generator_version matches world/params.yaml, else generate one
GEN=$(uv run python -c "import yaml;print(yaml.safe_load(open('src/motorq_de/world/params.yaml'))['generator_version'])")
DS=""
for d in data/synthetic/*/; do
  m=$(uv run python -c "import json;p=json.load(open('${d}params.json'));print(p['population']['n_vehicles'], p.get('generator_version'))" 2>/dev/null)
  [ "$m" = "5000 $GEN" ] && DS="${d%/}"
done
if [ -z "$DS" ]; then
  echo "== generating default dataset (generator $GEN)"
  uv run mde world generate --profile default --seed 42 >/dev/null || exit 1
  DS=$(ls -td data/synthetic/*/ | head -1); DS="${DS%/}"
fi
echo "== oracle on $DS"
MDE_ORACLE_DATASET="$DS" uv run pytest tests/oracle -q -p no:warnings || exit 1
echo "== slow integration"
uv run pytest tests -q -m slow -p no:warnings || exit 1
echo "ALL GREEN"
