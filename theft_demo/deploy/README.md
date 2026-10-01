# Theft demo on the DigitalOcean droplet

Runs on the shared droplet (`ssh do-deploy`, 139.59.136.124) next to the fire demo, the same way: root, pm2,
nginx + certbot. Dashboard: **https://theft.alarmius.duckdns.org**

```
laptop ──git push──► GitHub (theft-demo) ◄──git fetch── droplet ~/Shoplifting-theft
                                                        ├─ pm2: theft-demo  (python -m theft_demo.run_live, 127.0.0.1:8011)
browser / phone ──HTTPS──► nginx :443 (certbot) ────────┘
droplet ──► OpenRouter (Qwen) · Telegram (its own bot)                       (all outbound)
```

- Its own checkout `~/Shoplifting-theft`: the fire deploy force-checks out `fire-demo` in `~/Shoplifting`.
- Its own Telegram bot: Telegram lets only one program poll a bot, and the fire demo runs all the time.
  Don't run the theft demo on the laptop with the same token while the droplet one runs ("409 Conflict").
- No YOLO on the droplet: the recordings and their YOLO tracks are copied up from the laptop.
- `theft.alarmius.duckdns.org` needs nothing at duckdns.org: every name under `alarmius` resolves to the droplet.

| File | What |
|---|---|
| `setup_droplet.sh` | once: clone, nginx site + certificate, daily clean-up. Additive only |
| `deploy_to_do.sh` | every deploy: pull, venv (no torch), recordings check, tests, pm2 restart, health check |
| `ecosystem.config.js` | pm2 app `theft-demo` (port 8011, Yerevan time for this process only, 800 MB cap) |
| `nginx-theft-demo.conf` | nginx site template (domain filled in by setup) |

## 1. One-time setup

1. Push `theft-demo` to GitHub.
2. Setup (from the laptop, repo root):
   ```bash
   ssh do-deploy "THEFT_DOMAIN=theft.alarmius.duckdns.org bash -s" < theft_demo/deploy/setup_droplet.sh
   ```
3. Secrets: `~/Shoplifting-theft/theft_demo/.env` (chmod 600): `VLM_API_URL`, `VLM_API_KEY`, `TELEGRAM_BOT_TOKEN`
   (the theft bot; empty = alerts only logged; the fire bot's token is refused), `DEMO_USER`, `DEMO_PASSWORD`
   (strong: it's on the internet), `THEFT_QWEN` (`on` / `off`). After changing it: `pm2 restart theft-demo --update-env`.
4. Recordings + YOLO tracks:
   ```bash
   ssh do-deploy "mkdir -p Shoplifting-theft/theft_demo/data/tracks"
   scp theft_demo/data/*.mp4 do-deploy:Shoplifting-theft/theft_demo/data/
   scp theft_demo/data/tracks/*.json.gz do-deploy:Shoplifting-theft/theft_demo/data/tracks/
   ```
5. First deploy: `ssh do-deploy "bash ~/Shoplifting-theft/theft_demo/deploy/deploy_to_do.sh"`
6. Open the dashboard → log in → Camera (choose a recording) → Alerts (scan the QR, Start) → Live → Play.

## Every deploy after that

```bash
git push origin theft-demo
ssh do-deploy "bash ~/Shoplifting-theft/theft_demo/deploy/deploy_to_do.sh"
```

Not touched by deploys: `theft_demo/.env`, `theft_demo/state/` (chosen recording, linked chats), `data/`, `outputs/`.

| Task | Command |
|---|---|
| Live log | `ssh do-deploy "pm2 logs theft-demo"` |
| Restart / stop | `ssh do-deploy "pm2 restart theft-demo"` / `pm2 stop theft-demo` |
| Get run evidence | `scp -r do-deploy:Shoplifting-theft/theft_demo/outputs/live ./theft_demo/outputs/from_droplet` |
