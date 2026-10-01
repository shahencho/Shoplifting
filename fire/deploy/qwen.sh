#!/usr/bin/env bash
# Turns Qwen verification on or off on the droplet. Runs ON the droplet, from the laptop:
#   ssh do-deploy "bash ~/Shoplifting/fire/deploy/qwen.sh on"     # or: off / status
# The setting is FIRE_QWEN in fire/.env, which deploys never touch, so it stays until changed here.
# on/off restarts fire-demo: a playing video stops and waits for Play in the dashboard again.
set -euo pipefail

cd "$(dirname "$0")/../.."    # repo root
ENV=fire/.env

current() { grep -E '^FIRE_QWEN=' "$ENV" | tail -1 | cut -d= -f2- || true; }

case "${1:-status}" in
  on|off)
    if [ "$1" = on ] && ! grep -Eq '^VLM_API_KEY=.+' "$ENV"; then
      echo "FATAL: VLM_API_KEY is empty in $ENV: Qwen can't run without it."
      exit 1
    fi
    if grep -Eq '^FIRE_QWEN=' "$ENV"; then
      sed -i "s/^FIRE_QWEN=.*/FIRE_QWEN=$1/" "$ENV"
    else
      [ -z "$(tail -c1 "$ENV")" ] || echo >> "$ENV"
      echo "FIRE_QWEN=$1" >> "$ENV"
    fi
    pm2 restart fire-demo > /dev/null
    echo "Qwen: $1 (fire-demo restarted; press Play again for a video file)"
    ;;
  status)
    echo "Qwen: $(current || echo on)"
    pm2 logs fire-demo --lines 200 --nostream 2>/dev/null | grep -E 'Qwen: ' | tail -1 || true
    ;;
  *)
    echo "usage: qwen.sh on|off|status"
    exit 2
    ;;
esac
