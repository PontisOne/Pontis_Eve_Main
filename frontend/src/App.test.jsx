import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import RouteErrorBoundary from "./components/RouteErrorBoundary";

function BrokenRoute() {
  throw new Error("chunk load failed");
}

describe("RouteErrorBoundary", () => {
  it("preserves the loading fallback while a route is pending", () => {
    const PendingRoute = React.lazy(() => new Promise(() => {}));

    render(
      <RouteErrorBoundary>
        <React.Suspense fallback={<main aria-busy="true">Loading page…</main>}>
          <PendingRoute />
        </React.Suspense>
      </RouteErrorBoundary>,
    );

    expect(screen.getByText("Loading page…")).toBeTruthy();
  });

  it("renders its children normally", () => {
    render(
      <RouteErrorBoundary>
        <p>Loaded route</p>
      </RouteErrorBoundary>,
    );

    expect(screen.getByText("Loaded route")).toBeTruthy();
  });

  it("shows an accessible recovery action for a failed lazy route", () => {
    render(
      <RouteErrorBoundary>
        <BrokenRoute />
      </RouteErrorBoundary>,
    );

    expect(screen.getByRole("alert").textContent).toContain("This page could not be loaded");
    expect(screen.getByRole("button", { name: "Reload page" })).toBeTruthy();
  });
});
