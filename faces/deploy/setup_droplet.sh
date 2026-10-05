#!/usr/bin/env bash
# One-time setup of the face watchlist demo on the SHARED droplet (do-deploy, next to fire-demo and theft-demo).
# Safe to re-run. Run from the laptop (repo root):
#   ssh do-deploy "FACES_DOMAIN=faces.alarmius.duckdns.org bash -s" < faces/deploy/setup_droplet.sh
# (DuckDNS resolves every name under alarmius.duckdns.org to the droplet: nothing to add at duckdns.org.)
#
# Additive only: its own clone ~/Shoplifting-faces (branch faces-demo; fire and theft have their own checkouts),
# one nginx site + its certbot certificate and a daily clean-up. Does not create faces/.env or start the app:
# see faces/deploy/README.md. Copied from theft_demo/deploy/setup_droplet.sh.
set -euo pipefail

main() {
  REPO_URL="${REPO_URL:-https://github.com/shahencho/Shoplifting.git}"
  BRANCH="${BRANCH:-faces-demo}"
  APP_DIR="$HOME/Shoplifting-faces"
  FACES_DOMAIN="${FACES_DOMAIN:?set FACES_DOMAIN, e.g. FACES_DOMAIN=faces.alarmius.duckdns.org}"
  SITE=/etc/nginx/sites-available/faces-demo

  echo "==> Packages (only what is missing)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -yq --no-upgrade python3-venv libglib2.0-0 ffmpeg     # ffmpeg: small H.264 clips, phone videos

  echo "==> Repo $APP_DIR ($BRANCH)"
  if [ ! -d "$APP_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
  fi

  echo "==> nginx site $FACES_DOMAIN -> 127.0.0.1:8012"
  if [ ! -f "$SITE" ]; then      # certbot edits the site later; never overwrite it on a re-run
    sed "s/FACES_DOMAIN/$FACES_DOMAIN/" "$APP_DIR/faces/deploy/nginx-faces-demo.conf" > "$SITE"
    ln -sf "$SITE" /etc/nginx/sites-enabled/faces-demo
  fi
  nginx -t
  systemctl reload nginx

  echo "==> HTTPS certificate (certbot, like the other sites)"
  if certbot certificates 2>/dev/null | grep -q "Domains: $FACES_DOMAIN"; then
    echo "    already have one"
  else
    my_ip=$(curl -fsS --max-time 5 https://api.ipify.org)
    dns_ip=$(getent hosts "$FACES_DOMAIN" | awk '{print $1}' | head -1)
    if [ "$dns_ip" != "$my_ip" ]; then
      echo "!! $FACES_DOMAIN resolves to '${dns_ip:-nothing}', not $my_ip. Fix it at duckdns.org, then re-run."
      exit 2
    fi
    certbot --nginx -d "$FACES_DOMAIN" --non-interactive --redirect
  fi

  echo "==> Daily clean-up: evidence older than 14 days (config evidence.retention_days)"
  cat > /etc/cron.daily/faces-demo-cleanup <<EOF
#!/bin/sh
for d in search replay faces_cache uploads; do
  find $APP_DIR/faces/outputs/\$d -mindepth 1 -maxdepth 1 -mtime +14 -exec rm -rf {} + 2>/dev/null
done
exit 0
EOF
  chmod 755 /etc/cron.daily/faces-demo-cleanup

  echo
  echo "Setup done. Dashboard (after the first deploy): https://$FACES_DOMAIN"
  echo "Next: create $APP_DIR/faces/.env, copy the recordings + index up, then: bash $APP_DIR/faces/deploy/deploy_to_do.sh"
}

main "$@"
exit
