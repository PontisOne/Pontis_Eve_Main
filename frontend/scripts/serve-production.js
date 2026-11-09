const fs = require("fs");
const http = require("http");
const path = require("path");
const { promisify } = require("util");
const compression = require("compression");
const serveHandler = require("serve-handler");

const compress = promisify(compression());
const configPath = path.resolve(__dirname, "..", "serve.json");
const buildPath = path.resolve(__dirname, "..", "build");

function isReservedPath(requestUrl) {
  let pathname;
  try {
    pathname = decodeURIComponent(new URL(requestUrl, "http://localhost").pathname);
  } catch {
    return false;
  }

  return ["/api", "/health"].some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
  );
}

function createProductionServer() {
  const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
  config.public = buildPath;
  config.etag = true;

  return http.createServer(async (request, response) => {
    if (isReservedPath(request.url)) {
      response.writeHead(404, {
        "Cache-Control": "no-store",
        "Content-Type": "application/json; charset=utf-8",
      });
      response.end(JSON.stringify({ error: "Not found" }));
      return;
    }

    try {
      await compress(request, response);
      await serveHandler(request, response, config);
    } catch (error) {
      console.error("Unable to serve production request:", error);
      if (response.headersSent) {
        response.destroy(error);
      } else {
        response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
        response.end("Internal Server Error");
      }
    }
  });
}

function startProductionServer() {
  const port = Number(process.env.PORT);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    console.error("PORT must be set to a valid TCP port before starting the production static server.");
    process.exitCode = 1;
    return;
  }

  const server = createProductionServer();
  server.on("error", (error) => {
    console.error("Unable to start the production static server:", error);
    process.exitCode = 1;
  });
  server.listen(port, "0.0.0.0", () => {
    console.log(`Production static server listening on port ${port}.`);
  });
}

if (require.main === module) {
  startProductionServer();
}

module.exports = { createProductionServer, isReservedPath };
