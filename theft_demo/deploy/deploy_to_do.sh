#!/usr/bin/env bash
# Deploys the theft demo. Runs ON the shared droplet (do-deploy), inside ~/Shoplifting-theft, as pm2 app "theft-demo".
# Steady state, from the laptop:
#   git push origin theft-demo
#   ssh do-deploy "bash ~/Shoplifting-theft/theft_demo/deploy/deploy_to_do.sh"
# First-time machine setup: theft_demo/deploy/setup_droplet.sh (see theft_demo/deploy/README.md).
#
# Editing this script is safe: the whole file is parsed before main runs, and after the pull the
# script re-runs its new copy, so there's no bootstrap step.
# SKIP_TESTS=1 skips the theft_demo/tests gate (emergencies only).
set -euo pipefail

BRANCH="${BRANCH:-theft-demo}"
PY=theft_demo/.venv/bin/python
HEALTH_URL="http://127.0.0.1:8011/"

main() {
  cd "$(dirname "$0")/../.."    # repo root

  if [ "${1:-}" != "--pulled" ]; then
    echo "==> Fetching origin/$BRANCH"
    git fetch origin "$BRANCH"
    # Tracked files only. Gitignored files (theft_demo/.env, state/, data/, outputs/, .venv/) are untouched.
    # Never `git clean` here: it would delete the linked Telegram chats and the recordings.
    git checkout -q -f -B "$BRANCH" "origin/$BRANCH"
    echo "    at $(git log -1 --format='%h %s')"
    exec bash theft_demo/deploy/deploy_to_do.sh --pulled
  fi

  echo "==> Checking theft_demo/.env"
  if [ ! -s theft_demo/.env ]; then
    echo "FATAL: theft_demo/.env is missing or empty. Create it first (theft_demo/deploy/README.md)."
    exit 1
  fi
  for key in VLM_API_KEY TELEGRAM_BOT_TOKEN DEMO_PASSWORD; do
    if ! grep -Eq "^${key}=.+" theft_demo/.env; then
      echo "FATAL: $key is empty in theft_demo/.env (DEMO_PASSWORD is required here: the dashboard is on the internet)."
      exit 1
    fi
  done
  echo "    Qwen: $(grep -E '^THEFT_QWEN=' theft_demo/.env | tail -1 | cut -d= -f2- || true)  (THEFT_QWEN in theft_demo/.env, default on)"

  echo "==> Python venv"
  [ -x "$PY" ] || python3 -m venv theft_demo/.venv
  $PY -m pip install -q --upgrade pip
  # YOLO is precomputed on the laptop (data/tracks), so no ultralytics / torch here; yt-dlp isn't needed either.
  grep -vE '^(ultralytics|yt-dlp)' theft_demo/requirements.txt > /tmp/theft-demo-requirements.txt
  $PY -m pip install -q -r /tmp/theft-demo-requirements.txt

  echo "==> Recordings (theft_demo/data, not in git)"
  $PY - <<'EOF'
from theft_demo.precompute import track_path
from theft_demo.tools.get_videos import DATA, videos
missing = [v["id"] for v in videos() if not (DATA / f"{v['id']}.mp4").exists() or not track_path(v["id"]).exists()]
print(f"    {len(videos()) - len(missing)} playable" + (f"; missing video or tracks: {', '.join(missing)} "
      "(scp from the laptop, see theft_demo/deploy/README.md)" if missing else ""))
EOF

  if [ "${SKIP_TESTS:-0}" != "1" ]; then
    echo "==> Tests (theft_demo/tests)"
    $PY -m pytest -q theft_demo/tests
  fi

  echo "==> Restarting theft-demo (pm2)"
  pm2 startOrRestart theft_demo/deploy/ecosystem.config.js --update-env
  pm2 save > /dev/null        # comes back after a reboot, like the other apps

  echo "==> Verifying health"
  for attempt in $(seq 1 30); do
    if curl -fsS --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
      echo "Deploy OK: dashboard answered (attempt $attempt)."
      exit 0
    fi
    sleep 1
  done
  echo "Deploy FAILED: no answer on $HEALTH_URL within 30 s. Check: pm2 logs theft-demo --lines 100"
  exit 1
}

main "$@"
exit
