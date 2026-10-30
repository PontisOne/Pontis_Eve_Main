const path = require("path");
const { defineConfig, transformWithEsbuild } = require("vite");
const react = require("@vitejs/plugin-react");

const HEALTH_CHECK_ENABLED = "true";

function isHealthCheckEnabled(env = process.env) {
  return env.ENABLE_HEALTH_CHECK === HEALTH_CHECK_ENABLED;
}

function healthCheckPlugin(enabled) {
  return {
    name: "development-health-check",
    apply: "serve",
    configureServer(server) {
      if (!enabled) {
        return;
      }

      const respond = (status, body) => (_request, response) => {
        response.statusCode = status;
        response.setHeader("Content-Type", "application/json");
        response.end(JSON.stringify(body));
      };

      const routes = new Map([
        ["/health", respond(200, { status: "ok" })],
        ["/health/live", respond(200, { status: "ok" })],
        ["/health/ready", respond(200, { status: "ok", ready: true })],
      ]);

      server.middlewares.use((request, response, next) => {
        const route = routes.get(request.url.split("?", 1)[0]);
        if (route) route(request, response, next);
        else next();
      });
    },
  };
}

module.exports = defineConfig({
  plugins: [
    healthCheckPlugin(isHealthCheckEnabled()),
    {
      name: "jsx-in-js",
      enforce: "pre",
      async transform(code, id) {
        if (id.endsWith(".js") && id.includes("/src/")) {
          return transformWithEsbuild(code, id, { loader: "jsx" });
        }
      },
    },
    react(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  optimizeDeps: {
    esbuildOptions: {
      loader: {
        ".js": "jsx",
      },
    },
  },
  build: {
    outDir: "build",
    sourcemap: false,
  },
  server: {
    headers: {
      "Cross-Origin-Resource-Policy": "same-origin",
    },
    watch: {
      ignored: [
        "**/node_modules/**",
        "**/.git/**",
        "**/build/**",
        "**/dist/**",
        "**/coverage/**",
        "**/public/**",
      ],
    },
  },
});
