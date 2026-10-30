// @vitest-environment node

import { describe, expect, it } from "vitest";

const viteConfig = require("./vite.config");

function resolveConfig(env = {}) {
  return viteConfig({ mode: "development", command: "serve", ssrBuild: false, ...env });
}

function getHealthPlugin(config) {
  return config.plugins.find((plugin) => plugin.name === "development-health-check");
}

function createServer() {
  const middleware = [];
  return {
    middlewares: {
      use(route, handler) {
        middleware.push(typeof route === "function" ? route : handler);
      },
    },
    middleware,
  };
}

describe("Vite development safeguards", () => {
  it("adds CORP only to the Vite development server configuration", () => {
    const config = resolveConfig();

    expect(config.server.headers).toEqual({
      "Cross-Origin-Resource-Policy": "same-origin",
    });
    expect(config.build).toEqual({ outDir: "build", sourcemap: false });
  });

  it.each([undefined, "false", "1", "TRUE"])(
    "does not register health routes when ENABLE_HEALTH_CHECK is %s",
    (value) => {
      const original = process.env.ENABLE_HEALTH_CHECK;
      if (value === undefined) {
        delete process.env.ENABLE_HEALTH_CHECK;
      } else {
        process.env.ENABLE_HEALTH_CHECK = value;
      }

      try {
        const server = createServer();
        getHealthPlugin(resolveConfig()).configureServer(server);
        expect(server.middleware).toHaveLength(0);
      } finally {
        if (original === undefined) delete process.env.ENABLE_HEALTH_CHECK;
        else process.env.ENABLE_HEALTH_CHECK = original;
      }
    },
  );

  it("registers minimal health routes only when explicitly enabled", async () => {
    const original = process.env.ENABLE_HEALTH_CHECK;
    process.env.ENABLE_HEALTH_CHECK = "true";

    try {
      const server = createServer();
      getHealthPlugin(resolveConfig()).configureServer(server);

      expect(server.middleware).toHaveLength(1);

      for (const [route, expectedStatus] of [
        ["/health", 200],
        ["/health/live", 200],
        ["/health/ready", 200],
      ]) {
        const response = {
          headers: {},
          setHeader(name, value) {
            this.headers[name] = value;
          },
          end(body) {
            this.body = JSON.parse(body);
          },
        };
        server.middleware[0]({ url: route }, response, () => {});
        expect(response.statusCode).toBe(expectedStatus);
        expect(response.headers["Content-Type"]).toBe("application/json");
        expect(response.body).not.toHaveProperty("stack");
        expect(response.body).not.toHaveProperty("environment");
        expect(response.body).not.toHaveProperty("memory");
        expect(response.body).not.toHaveProperty("nodeVersion");
        expect(response.body).not.toHaveProperty("filesystem");
      }

      let passedThrough = false;
      server.middleware[0]({ url: "/health/errors" }, {}, () => {
        passedThrough = true;
      });
      expect(passedThrough).toBe(true);
    } finally {
      if (original === undefined) delete process.env.ENABLE_HEALTH_CHECK;
      else process.env.ENABLE_HEALTH_CHECK = original;
    }
  });
});
