// @vitest-environment node

import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const config = JSON.parse(
  fs.readFileSync(path.resolve(process.cwd(), "serve.json"), "utf8"),
);
const require = createRequire(import.meta.url);
const { isReservedPath } = require("../scripts/serve-production");

describe("production static hosting configuration", () => {
  it("rewrites known and extensionless SPA routes", () => {
    expect(config.rewrites).toEqual([
      { source: "/", destination: "/index.html" },
      { source: "/onboarding", destination: "/index.html" },
      { source: "/dashboard", destination: "/index.html" },
      { source: "/resume-editor", destination: "/index.html" },
      { source: "/:path([^.]+)", destination: "/index.html" },
    ]);
  });

  it.each(["/api", "/api/health", "/health", "/health/live", "/%61pi/status"])(
    "reserves %s for its owning service",
    (requestUrl) => {
      expect(isReservedPath(requestUrl)).toBe(true);
    },
  );

  it.each(["/", "/not-a-route", "/healthcheck", "/apiary", "/assets/missing.js"])(
    "does not reserve %s",
    (requestUrl) => {
      expect(isReservedPath(requestUrl)).toBe(false);
    },
  );

  it("keeps HTML uncached and fingerprinted assets immutable", () => {
    expect(config.headers).toEqual(
      expect.arrayContaining([
        {
          source: "index.html",
          headers: [
            { key: "Cache-Control", value: "no-store, max-age=0, must-revalidate" },
          ],
        },
        {
          source: "assets/**/*-????????.@(js|css)",
          headers: [
            { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
          ],
        },
      ]),
    );
  });
});
