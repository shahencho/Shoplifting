#!/usr/bin/env bash
# One-time setup of the theft demo on the SHARED droplet (do-deploy, next to fire-demo). Safe to re-run.
# Run from the laptop (repo root):
#   ssh do-deploy "THEFT_DOMAIN=theft.alarmius.duckdns.org bash -s" < theft_demo/deploy/setup_droplet.sh
# (DuckDNS resolves every name under alarmius.duckdns.org to the droplet: nothing to add at duckdns.org.)
#
# Additive only: its own clone ~/Shoplifting-theft (branch theft-demo; the fire deploy force-checks out
# fire-demo in ~/Shoplifting, so the two never share a checkout), one nginx site + its certbot certificate
# and a daily clean-up. Does not create theft_demo/.env or start the app: see theft_demo/deploy/README.md.
set -euo pipefail

main() {
  REPO_URL="${REPO_URL:-https://github.com/shahencho/Shoplifting.git}"
  BRANCH="${BRANCH:-theft-demo}"
  APP_DIR="$HOME/Shoplifting-theft"
  THEFT_DOMAIN="${THEFT_DOMAIN:?set THEFT_DOMAIN, e.g. THEFT_DOMAIN=theft.alarmius.duckdns.org}"
  SITE=/etc/nginx/sites-available/theft-demo

  echo "==> Packages (only what is missing)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -yq --no-upgrade python3-venv libglib2.0-0 ffmpeg

  echo "==> Repo $APP_DIR ($BRANCH)"
  if [ ! -d "$APP_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
  fi

  echo "==> nginx site $THEFT_DOMAIN -> 127.0.0.1:8011"
  if [ ! -f "$SITE" ]; then      # certbot edits the site later; never overwrite it on a re-run
    sed "s/THEFT_DOMAIN/$THEFT_DOMAIN/" "$APP_DIR/theft_demo/deploy/nginx-theft-demo.conf" > "$SITE"
    ln -sf "$SITE" /etc/nginx/sites-enabled/theft-demo
  fi
  nginx -t
  systemctl reload nginx

  echo "==> HTTPS certificate (certbot, like the other sites)"
  if certbot certificates 2>/dev/null | grep -q "Domains: $THEFT_DOMAIN"; then
    echo "    already have one"
  else
    my_ip=$(curl -fsS --max-time 5 https://api.ipify.org)
    dns_ip=$(getent hosts "$THEFT_DOMAIN" | awk '{print $1}' | head -1)
    if [ "$dns_ip" != "$my_ip" ]; then
      echo "!! $THEFT_DOMAIN resolves to '${dns_ip:-nothing}', not $my_ip. Fix it at duckdns.org, then re-run."
      exit 2
    fi
    certbot --nginx -d "$THEFT_DOMAIN" --non-interactive --redirect
  fi

  echo "==> Daily clean-up: run folders older than 14 days"
  cat > /etc/cron.daily/theft-demo-cleanup <<EOF
#!/bin/sh
find $APP_DIR/theft_demo/outputs/live -mindepth 1 -maxdepth 1 -type d -mtime +14 -exec rm -rf {} + 2>/dev/null
exit 0
EOF
  chmod 755 /etc/cron.daily/theft-demo-cleanup

  echo
  echo "Setup done. Dashboard (after the first deploy): https://$THEFT_DOMAIN"
  echo "Next: create $APP_DIR/theft_demo/.env, copy the recordings up, then: bash $APP_DIR/theft_demo/deploy/deploy_to_do.sh"
}

main "$@"
exit
