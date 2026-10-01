# Fire demo on the DigitalOcean droplet (setup B, plan §6)

Runs on the **existing shared droplet** (`ssh do-deploy`, 139.59.136.124) next to bookalyzer and the other apps,
the same way they run: root, `~/Shoplifting`, pm2, nginx + certbot, a DuckDNS name.

```
laptop ──git push──► GitHub (fire-demo) ◄──git fetch── droplet ~/Shoplifting
                                                       ├─ pm2: fire-demo  (python -m fire.run_live, 127.0.0.1:8010)
browser / phone ──HTTPS──► nginx :443 (certbot) ───────┘
droplet ──► camera (RTSP / HLS / YouTube) · OpenRouter (Qwen) · Telegram      (all outbound)
```

| File | What |
|---|---|
| `setup_droplet.sh` | once: 3 packages, clone, nginx site + certificate, daily clean-up. Additive only |
| `deploy_to_do.sh` | every deploy: pull, venv + CPU torch, weights, tests, pm2 restart, health check |
| `ecosystem.config.js` | pm2 app `fire-demo` (port 8010, Yerevan time for this process only, 1.5 GB memory cap) |
| `nginx-fire-demo.conf` | nginx site template (domain filled in by setup) |

Checked 01.10.2026: 2 vCPU, 3.8 GB RAM (~2.1 GB free), 61 GB disk free, Ubuntu 24.04, Python 3.12, pm2 starts on boot.
Port 8000 is used by other nginx sites, so fire uses **8010**.

## 1. One-time setup

1. **DuckDNS** (free): log in at duckdns.org → add a subdomain (ours: `alarmius`) → current ip `139.59.136.124`.
2. **Push** `fire-demo` to GitHub (the droplet clones it from there).
3. **Setup** (from the laptop, repo root):
   ```bash
   ssh do-deploy "FIRE_DOMAIN=alarmius.duckdns.org bash -s" < fire/deploy/setup_droplet.sh
   ```
4. **Secrets** on the droplet (never in git):
   ```bash
   ssh do-deploy
   cp ~/Shoplifting/fire/.env.example ~/Shoplifting/fire/.env && chmod 600 ~/Shoplifting/fire/.env
   nano ~/Shoplifting/fire/.env      # VLM_API_KEY, TELEGRAM_BOT_TOKEN, DEMO_USER, DEMO_PASSWORD (strong)
   ```
5. **First deploy** (slow once: torch + ultralytics download):
   ```bash
   ssh do-deploy "bash ~/Shoplifting/fire/deploy/deploy_to_do.sh"
   ```
6. Open `https://alarmius.duckdns.org` → log in → setup screens: camera, detections, Telegram QR.

## 2. Every deploy after that

```bash
git push origin fire-demo
ssh do-deploy "bash ~/Shoplifting/fire/deploy/deploy_to_do.sh"
```

Not touched by deploys: `fire/.env`, `fire/state/` (saved camera, linked Telegram chats), `fire/models/`, `fire/outputs/`.

## Day to day

| Task | Command |
|---|---|
| Live log | `ssh do-deploy "pm2 logs fire-demo"` |
| Restart / stop | `ssh do-deploy "pm2 restart fire-demo"` / `pm2 stop fire-demo` |
| Copy a test video up | `scp fire/data/BJ9ng9L1CA0.mp4 do-deploy:Shoplifting/fire/data/` |
| Get event evidence | `scp -r do-deploy:Shoplifting/fire/outputs/live ./fire/outputs/from_droplet` |
| Stream check from the droplet | `ssh do-deploy "cd Shoplifting && fire/.venv/bin/python -m fire.tools.stream_check"` |

Event folders older than 14 days are deleted daily (`/etc/cron.daily/fire-demo-cleanup`).

## Current test mode

- **Qwen is off** (`--no-qwen` in `ecosystem.config.js`): YOLO + the 3 s filter alone decide. Every event goes to
  Telegram as "Possible fire (not verified)" with photo, clip and a timing line
  (`⏱ YOLO <first seen> → alert <trigger> (+s) · sent +s`). Remove `--no-qwen` and deploy to turn Qwen back on.
- **Early note** (`alerts.early_note` in `config.yaml`): at 60% of checks over 3 s a silent
  "🟡 Suspicious smoke/fire, checking…" with photo; the alert or "✅ all clear" (after 15 s) follows as a reply.
- **Video files never play by themselves.** After a start, deploy or camera save the dashboard shows "Video ready";
  it plays only on **Play / Replay from the start**. Live cameras start at once.
- **AV1 videos** (most YouTube downloads) don't decode here: convert them first:
  `ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p -an out.mp4`.

## Watch out

- **Shared box.** Other apps run here. fire is capped at 1.5 GB by pm2; check `pm2 ls` after the first deploy.
- **One bot, one process.** Telegram allows only one `getUpdates` poller per bot token. If the laptop and the droplet
  run at the same time with the same `TELEGRAM_BOT_TOKEN`, both get 409 errors and linking breaks. Stop the laptop run,
  or give the droplet its own bot. Linked chats live in `fire/state/telegram.json` per machine: re-link via the QR on
  the droplet (or `scp` that file up).
- **The repo is public.** Everything committed is visible to anyone. Secrets, camera URLs, videos and outputs are
  gitignored; keep it that way.
- **YouTube from a datacenter IP** is often blocked ("Sign in to confirm you're not a bot"). The Orbeli test stream
  may work on the laptop and fail on the droplet. RTSP / HLS sources are not affected. Check with `stream_check` first.
- **Clip codec.** The deploy prints `H.264 (avc1): OK` or a warning. With the warning, clips are saved as mp4v and may
  not play inline in Telegram or the browser; the fix is a system-ffmpeg transcode in `fire/evidence.py`.
- **Camera reachability.** A camera on the client's LAN is not reachable from Frankfurt unless the client gives a
  route (port forward, VPN, cloud re-stream). That's why setup A (on site) is the default (plan §6).
- **Cross-border data** (plan §9): frames go to OpenRouter, photos/clips to Telegram. Get the client's written OK first.
