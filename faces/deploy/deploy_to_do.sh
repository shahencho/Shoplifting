#!/usr/bin/env bash
# Deploys the face watchlist demo. Runs ON the shared droplet (do-deploy), inside ~/Shoplifting-faces, as pm2 app
# "faces-demo". Copied from theft_demo/deploy/deploy_to_do.sh. Steady state, from the laptop:
#   git push origin faces-demo
#   ssh do-deploy "bash ~/Shoplifting-faces/faces/deploy/deploy_to_do.sh"
# First-time machine setup: faces/deploy/setup_droplet.sh (see faces/deploy/README.md).
#
# Editing this script is safe: the whole file is parsed before main runs, and after the pull the
# script re-runs its new copy, so there's no bootstrap step.
# SKIP_TESTS=1 skips the faces/tests gate (emergencies only).
set -euo pipefail

BRANCH="${BRANCH:-faces-demo}"
PY=faces/.venv/bin/python
HEALTH_URL="http://127.0.0.1:8012/"

main() {
  cd "$(dirname "$0")/../.."    # repo root

  if [ "${1:-}" != "--pulled" ]; then
    echo "==> Fetching origin/$BRANCH"
    git fetch origin "$BRANCH"
    # Tracked files only. Gitignored files (faces/.env, state/, data/, models/, outputs/, .venv/) are untouched.
    # Never `git clean` here: it would delete the watchlist, the recordings and the linked Telegram chats.
    git checkout -q -f -B "$BRANCH" "origin/$BRANCH"
    echo "    at $(git log -1 --format='%h %s')"
    exec bash faces/deploy/deploy_to_do.sh --pulled
  fi

  echo "==> Checking faces/.env"
  if [ ! -s faces/.env ]; then
    echo "FATAL: faces/.env is missing or empty. Create it first (faces/deploy/README.md)."
    exit 1
  fi
  if ! grep -Eq "^DEMO_PASSWORD=.+" faces/.env; then
    echo "FATAL: DEMO_PASSWORD is empty in faces/.env (required here: the dashboard is on the internet)."
    exit 1
  fi
  tok=$(grep -E '^TELEGRAM_BOT_TOKEN=' faces/.env | tail -1 | cut -d= -f2- || true)
  if [ -z "$tok" ]; then
    echo "    WARNING: TELEGRAM_BOT_TOKEN is empty: alerts are only logged until the faces bot's token is set"
  else
    for other in "$HOME/Shoplifting/fire/.env" "$HOME/Shoplifting-theft/theft_demo/.env"; do
      if [ -f "$other" ] && grep -qxF "TELEGRAM_BOT_TOKEN=$tok" "$other"; then
        echo "FATAL: TELEGRAM_BOT_TOKEN is the bot of $other: both apps would poll it (409 Conflict). Use the faces bot."
        exit 1
      fi
    done
  fi

  echo "==> Python venv"
  [ -x "$PY" ] || python3 -m venv faces/.venv
  $PY -m pip install -q --upgrade pip
  # system ffmpeg is installed by setup_droplet.sh, so the bundled one (imageio-ffmpeg) isn't needed here
  grep -vE '^imageio-ffmpeg' faces/requirements.txt > /tmp/faces-demo-requirements.txt
  $PY -m pip install -q -r /tmp/faces-demo-requirements.txt

  echo "==> Face model (buffalo_l, ~280 MB, downloaded once into faces/models)"
  $PY -W ignore - <<'EOF'
from faces.detector import FaceModel, load_config
m = FaceModel(load_config())
print(f"    {m.name} ready")
EOF

  echo "==> Recordings (faces/data/archive + data/index, not in git)"
  $PY - <<'EOF'
from faces.archive import recordings
from faces.index import index_path
recs = recordings()
missing = [r["id"] for r in recs if not index_path(r["id"]).exists()]
print(f"    {len(recs) - len(missing)} searchable" + (f"; not indexed: {', '.join(missing)} "
      "(scp the index from the laptop, or they are indexed only when uploaded through the dashboard)" if missing else ""))
EOF

  if [ "${SKIP_TESTS:-0}" != "1" ]; then
    echo "==> Tests (faces/tests)"
    $PY -m pytest -q faces/tests
  fi

  echo "==> Memory before restart (shared box)"
  free -m | head -2

  echo "==> Restarting faces-demo (pm2)"
  pm2 startOrRestart faces/deploy/ecosystem.config.js --update-env
  pm2 save > /dev/null        # comes back after a reboot, like the other apps

  echo "==> Verifying health"
  for attempt in $(seq 1 30); do
    if curl -fsS --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
      echo "Deploy OK: dashboard answered (attempt $attempt)."
      exit 0
    fi
    sleep 1
  done
  echo "Deploy FAILED: no answer on $HEALTH_URL within 30 s. Check: pm2 logs faces-demo --lines 100"
  exit 1
}

main "$@"
exit
