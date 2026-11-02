import { QueryClient } from "@tanstack/react-query";
import queryClient, { queryClient as namedQueryClient } from "../queryClient";

describe("shared TanStack Query client", () => {
  it("exports one shared client instance", () => {
    expect(namedQueryClient).toBe(queryClient);
    expect(queryClient).toBeInstanceOf(QueryClient);
  });

  it("preserves existing query defaults", () => {
    const defaults = queryClient.getDefaultOptions();
    expect(defaults.queries.staleTime).toBe(60_000);
    expect(defaults.queries.refetchOnWindowFocus).toBe(false);
  });

  it("disables mutation retries by default", () => {
    expect(queryClient.getDefaultOptions().mutations.retry).toBe(false);
  });
});
