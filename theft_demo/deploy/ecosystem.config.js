// pm2 process for the theft demo on the shared droplet (next to fire-demo, bookalyzer & co).
// Started / restarted by theft_demo/deploy/deploy_to_do.sh. Logs: pm2 logs theft-demo
// The recording comes from the dashboard (theft_demo/state/settings.json) and plays only on Play.
const path = require("path");

module.exports = {
  apps: [{
    name: "theft-demo",
    cwd: path.resolve(__dirname, "../.."),
    script: "theft_demo/.venv/bin/python",
    // Qwen on/off: THEFT_QWEN in theft_demo/.env
    args: "-m theft_demo.run_live --host 127.0.0.1 --port 8011",
    interpreter: "none",
    env: {
      PYTHONUNBUFFERED: "1",
      TZ: "Asia/Yerevan",          // event and alert times; only this process, the server stays on UTC
    },
    max_memory_restart: "800M",    // YOLO is precomputed (no torch here); shared 4 GB box
    restart_delay: 5000,
  }],
};
