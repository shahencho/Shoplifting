#!/usr/bin/env bash
# Deploys the fire demo. Runs ON the shared droplet (do-deploy), inside ~/Shoplifting, as pm2 app "fire-demo".
# Steady state, from the laptop:
#   git push origin fire-demo
#   ssh do-deploy "bash ~/Shoplifting/fire/deploy/deploy_to_do.sh"
# First-time machine setup: fire/deploy/setup_droplet.sh (see fire/deploy/README.md).
#
# Editing this script is safe: the whole file is parsed before main runs, and after the pull the
# script re-runs its new copy, so there's no bootstrap step.
# SKIP_TESTS=1 skips the fire/tests gate (emergencies only).
set -euo pipefail

BRANCH="${BRANCH:-fire-demo}"
PY=fire/.venv/bin/python
HEALTH_URL="http://127.0.0.1:8010/"

main() {
  cd "$(dirname "$0")/../.."    # repo root

  if [ "${1:-}" != "--pulled" ]; then
    echo "==> Fetching origin/$BRANCH"
    git fetch origin "$BRANCH"
    # Tracked files only. Gitignored files (fire/.env, state/, models/, outputs/, .venv/) are untouched.
    # Never `git clean` here: it would delete the linked Telegram chats and saved camera.
    git checkout -q -f -B "$BRANCH" "origin/$BRANCH"
    echo "    at $(git log -1 --format='%h %s')"
    exec bash fire/deploy/deploy_to_do.sh --pulled
  fi

  echo "==> Checking fire/.env"
  if [ ! -s fire/.env ]; then
    echo "FATAL: fire/.env is missing or empty. Create it by hand first (fire/.env.example, fire/deploy/README.md)."
    exit 1
  fi
  for key in VLM_API_KEY TELEGRAM_BOT_TOKEN DEMO_PASSWORD; do
    if ! grep -Eq "^${key}=.+" fire/.env; then
      echo "FATAL: $key is empty in fire/.env (DEMO_PASSWORD is required here: the dashboard is on the internet)."
      exit 1
    fi
  done

  echo "==> Python venv"
  [ -x "$PY" ] || python3 -m venv fire/.venv
  $PY -m pip install -q --upgrade pip
  # CPU-only torch first. Otherwise ultralytics pulls the CUDA build (~2.5 GB, no GPU here).
  $PY -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  $PY -m pip install -q -r fire/requirements.txt

  echo "==> Model weights"
  $PY fire/models/download.py

  echo "==> Clip encoder check"
  $PY - <<'EOF'
import os, tempfile, cv2
p = os.path.join(tempfile.mkdtemp(), "t.mp4")
ok = cv2.VideoWriter(p, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"avc1"), 10, (64, 64)).isOpened()
print("    H.264 (avc1): OK" if ok else
      "    WARNING: no H.264 encoder in OpenCV. Clips fall back to mp4v; Telegram / browsers may not play them inline.")
EOF

  if [ "${SKIP_TESTS:-0}" != "1" ]; then
    echo "==> Tests (fire/tests)"
    $PY -m pytest -q fire/tests
  fi

  echo "==> Restarting fire-demo (pm2)"
  pm2 startOrRestart fire/deploy/ecosystem.config.js --update-env
  pm2 save > /dev/null        # comes back after a reboot, like the other apps

  echo "==> Verifying health"
  # Torch import + YOLO load take several seconds, so poll instead of one sleep.
  for attempt in $(seq 1 45); do
    if curl -fsS --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
      echo "Deploy OK: dashboard answered (attempt $attempt)."
      exit 0
    fi
    sleep 1
  done
  echo "Deploy FAILED: no answer on $HEALTH_URL within 45 s. Check: pm2 logs fire-demo --lines 100"
  exit 1
}

main "$@"
exit
