const path = require("path");
const { spawn } = require("child_process");

const port = process.env.PORT;

if (!port) {
  console.error("PORT must be set before starting the production static server.");
  process.exit(1);
}

const serveBin = require.resolve("serve/build/main.js");
const configPath = path.resolve(__dirname, "..", "serve.json");
const buildPath = path.resolve(__dirname, "..", "build");

const child = spawn(
  process.execPath,
  [serveBin, "-c", configPath, buildPath, "-l", `tcp://0.0.0.0:${port}`],
  { stdio: "inherit" },
);

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code ?? 1);
});

child.on("error", (error) => {
  console.error("Unable to start the production static server:", error);
  process.exit(1);
});
