// pm2 process for the fire demo on the shared droplet (next to bookalyzer & co).
// Started / restarted by fire/deploy/deploy_to_do.sh. Logs: pm2 logs fire-demo
// The camera comes from the dashboard setup (fire/state/settings.json), so no --source here.
const path = require("path");

module.exports = {
  apps: [{
    name: "fire-demo",
    cwd: path.resolve(__dirname, "../.."),
    script: "fire/.venv/bin/python",
    // Qwen on/off: FIRE_QWEN in fire/.env, switched with fire/deploy/qwen.sh (survives deploys)
    args: "-m fire.run_live --host 127.0.0.1 --port 8010",
    interpreter: "none",
    env: {
      PYTHONUNBUFFERED: "1",
      TZ: "Asia/Yerevan",          // event and alert times; only this process, the server stays on UTC
    },
    max_memory_restart: "1500M",   // shared 4 GB box: never starve the other apps
    restart_delay: 5000,
  }],
};
