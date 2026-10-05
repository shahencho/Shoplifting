// pm2 process for the face watchlist demo on the shared droplet (next to fire-demo, theft-demo & co).
// Started / restarted by faces/deploy/deploy_to_do.sh. Logs: pm2 logs faces-demo
const path = require("path");

module.exports = {
  apps: [{
    name: "faces-demo",
    cwd: path.resolve(__dirname, "../.."),
    script: "faces/.venv/bin/python",
    args: "-m faces.run_live --host 127.0.0.1 --port 8012",
    interpreter: "none",
    env: {
      PYTHONUNBUFFERED: "1",
      TZ: "Asia/Yerevan",          // alert times; only this process, the server stays on UTC
    },
    // face model ~300 MB; indexing an upload peaks ~620 MB, ~1.1 GB if a photo is enrolled at the same time.
    // Shared 4 GB box: check `free -m` and `pm2 ls` first (faces/deploy/README.md)
    max_memory_restart: "1100M",
    restart_delay: 5000,
  }],
};
