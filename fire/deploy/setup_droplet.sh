#!/usr/bin/env bash
# One-time setup of the fire demo on the SHARED droplet (do-deploy, next to bookalyzer & co). Safe to re-run.
# Run from the laptop (repo root), after the DuckDNS name points to the droplet:
#   ssh do-deploy "FIRE_DOMAIN=<name>.duckdns.org bash -s" < fire/deploy/setup_droplet.sh
#
# Additive only: installs 3 packages, clones the repo to ~/Shoplifting, adds one nginx site + its certbot
# certificate and a daily clean-up. Does NOT upgrade packages, change the timezone, touch the firewall,
# or restart other apps. Does not create fire/.env or start the app: see fire/deploy/README.md.
set -euo pipefail

main() {
  REPO_URL="${REPO_URL:-https://github.com/shahencho/Shoplifting.git}"   # public repo: no key needed
  BRANCH="${BRANCH:-fire-demo}"
  APP_DIR="$HOME/Shoplifting"
  FIRE_DOMAIN="${FIRE_DOMAIN:?set FIRE_DOMAIN, e.g. FIRE_DOMAIN=firedemo.duckdns.org}"
  SITE=/etc/nginx/sites-available/fire-demo

  echo "==> Packages (only what is missing)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -yq --no-upgrade python3-venv libglib2.0-0 ffmpeg

  echo "==> Repo $APP_DIR ($BRANCH)"
  if [ ! -d "$APP_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
  fi

  echo "==> nginx site $FIRE_DOMAIN -> 127.0.0.1:8010"
  if [ ! -f "$SITE" ]; then      # certbot edits the site later; never overwrite it on a re-run
    sed "s/FIRE_DOMAIN/$FIRE_DOMAIN/" "$APP_DIR/fire/deploy/nginx-fire-demo.conf" > "$SITE"
    ln -sf "$SITE" /etc/nginx/sites-enabled/fire-demo
  fi
  nginx -t
  systemctl reload nginx

  echo "==> HTTPS certificate (certbot, like the other sites)"
  if certbot certificates 2>/dev/null | grep -q "Domains: $FIRE_DOMAIN"; then
    echo "    already have one"
  else
    my_ip=$(curl -fsS --max-time 5 https://api.ipify.org)
    dns_ip=$(getent hosts "$FIRE_DOMAIN" | awk '{print $1}' | head -1)
    if [ "$dns_ip" != "$my_ip" ]; then
      echo "!! $FIRE_DOMAIN resolves to '${dns_ip:-nothing}', not $my_ip. Fix it at duckdns.org, then re-run."
      exit 2
    fi
    certbot --nginx -d "$FIRE_DOMAIN" --non-interactive --redirect
  fi

  echo "==> Daily clean-up: event folders older than 14 days (config evidence.retention_days)"
  cat > /etc/cron.daily/fire-demo-cleanup <<EOF
#!/bin/sh
find $APP_DIR/fire/outputs/live -mindepth 3 -maxdepth 3 -path '*/events/E*' -type d -mtime +14 -exec rm -rf {} + 2>/dev/null
exit 0
EOF
  chmod 755 /etc/cron.daily/fire-demo-cleanup

  echo
  echo "Setup done. Dashboard (after the first deploy): https://$FIRE_DOMAIN"
  echo "Next: create $APP_DIR/fire/.env, then: bash $APP_DIR/fire/deploy/deploy_to_do.sh"
}

main "$@"
exit
