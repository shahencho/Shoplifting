# Face watchlist demo on the DigitalOcean droplet

Runs on the shared droplet (`ssh do-deploy`) next to the fire and theft demos, the same way: root, pm2, nginx +
certbot. Dashboard: **https://faces.alarmius.duckdns.org**

```
laptop ──git push──► GitHub (faces-demo) ◄──git fetch── droplet ~/Shoplifting-faces
                                                        ├─ pm2: faces-demo  (python -m faces.run_live, 127.0.0.1:8012)
browser / phone ──HTTPS──► nginx :443 (certbot) ────────┘   uploads up to 200 MB
droplet ──► Telegram (its own bot) · GitHub (face model, once) · YouTube (links; often blocked)   (all outbound)
```

- Its own checkout `~/Shoplifting-faces` and its own Telegram bot (Telegram lets one program poll a bot; the deploy
  refuses the fire or theft bot's token).
- The archive (recordings) and its index are copied up from the laptop, where indexing is faster. Recordings added
  through the dashboard are indexed on the droplet in the background.
- **Shared box: check memory first.** The face model needs ~300 MB; indexing peaks at ~620 MB, ~1.1 GB when a photo
  is enrolled at the same time (pm2 cap 1100 MB). Run `ssh do-deploy "free -m; pm2 ls"` before the first deploy.

| File | What |
|---|---|
| `setup_droplet.sh` | once: ffmpeg, clone, nginx site + certificate, daily clean-up. Additive only |
| `deploy_to_do.sh` | every deploy: pull, venv, face model, archive check, tests, pm2 restart, health check |
| `ecosystem.config.js` | pm2 app `faces-demo` (port 8012, Yerevan time for this process only, 1100 MB cap) |
| `nginx-faces-demo.conf` | nginx site template (domain filled in by setup; 200 MB uploads) |

## 1. One-time setup

1. Push `faces-demo` to GitHub.
2. Setup (from the laptop, repo root):
   ```bash
   ssh do-deploy "FACES_DOMAIN=faces.alarmius.duckdns.org bash -s" < faces/deploy/setup_droplet.sh
   ```
3. Secrets: `~/Shoplifting-faces/faces/.env` (chmod 600): `TELEGRAM_BOT_TOKEN` (the faces bot; empty = alerts only
   logged), `DEMO_USER`, `DEMO_PASSWORD` (strong: it's on the internet). After changing it:
   `pm2 restart faces-demo --update-env`.
4. Archive + index:
   ```bash
   ssh do-deploy "mkdir -p Shoplifting-faces/faces/data/archive Shoplifting-faces/faces/data/index"
   scp faces/data/archive/*.mp4 faces/data/archive/*.json do-deploy:Shoplifting-faces/faces/data/archive/
   scp faces/data/index/*.npz do-deploy:Shoplifting-faces/faces/data/index/
   ```
   (`*.json` = titles of recordings uploaded through the laptop dashboard, if any.)
5. First deploy: `ssh do-deploy "bash ~/Shoplifting-faces/faces/deploy/deploy_to_do.sh"`
6. Open the dashboard → log in → Alerts (scan the QR, Start) → Watchlist → add a photo.

## Every deploy after that

```bash
git push origin faces-demo
ssh do-deploy "bash ~/Shoplifting-faces/faces/deploy/deploy_to_do.sh"
```

Not touched by deploys: `faces/.env`, `faces/state/` (linked chats, sessions), `faces/data/` (watchlist, archive,
index), `faces/models/`, `faces/outputs/`.

| Task | Command |
|---|---|
| Live log | `ssh do-deploy "pm2 logs faces-demo"` |
| Restart / stop | `ssh do-deploy "pm2 restart faces-demo"` / `pm2 stop faces-demo` |
| Memory | `ssh do-deploy "free -m; pm2 ls"` |
| Get evidence | `scp -r do-deploy:Shoplifting-faces/faces/outputs/search ./faces/outputs/from_droplet` |
