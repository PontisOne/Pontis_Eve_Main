// @vitest-environment node

import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";

const config = JSON.parse(
  fs.readFileSync(path.resolve(process.cwd(), "serve.json"), "utf8"),
);

describe("production static hosting configuration", () => {
  it("rewrites only the supported SPA routes", () => {
    expect(config.rewrites).toEqual([
      { source: "/", destination: "/index.html" },
      { source: "/onboarding", destination: "/index.html" },
      { source: "/dashboard", destination: "/index.html" },
      { source: "/resume-editor", destination: "/index.html" },
    ]);
  });

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
