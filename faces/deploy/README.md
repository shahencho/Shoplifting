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
4. The two demo samples (recordings + index + photo; the repo is public, so none of it is in git). Only recordings
   where search works: the overhead CCTV ones (bar, retail) stay on the laptop, every face in them is too small.
   ```bash
   ssh do-deploy "mkdir -p Shoplifting-faces/faces/data/archive Shoplifting-faces/faces/data/index Shoplifting-faces/faces/data/originals"
   scp faces/data/archive/Egy6I9XB5u8.mp4 faces/data/archive/up_1005_195521_link.* do-deploy:Shoplifting-faces/faces/data/archive/
   scp faces/data/index/Egy6I9XB5u8.npz faces/data/index/up_1005_195521_link.npz do-deploy:Shoplifting-faces/faces/data/index/
   scp faces/data/originals/shahen.jpg do-deploy:Shoplifting-faces/faces/data/originals/
   ```
   (`up_1005_195521_link` = VIDEO0050; its `.json` holds the title.)
5. First deploy: `ssh do-deploy "bash ~/Shoplifting-faces/faces/deploy/deploy_to_do.sh"`
6. Watchlist samples, once: "Woman at the till" (clicked in the grocery store at 0:06, found at 0:51, 1:09, 2:32,
   2:43) and "Shahen" (the photo, found in VIDEO0050 at 0:02 and 0:14). Skips a sample already on the list.
   ```bash
   ssh do-deploy "cd ~/Shoplifting-faces && faces/.venv/bin/python -m faces.tools.seed_demo && pm2 restart faces-demo"
   ```
7. Open the dashboard → log in → Alerts (scan the QR, Start) → Watchlist: both samples with their matches.

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
