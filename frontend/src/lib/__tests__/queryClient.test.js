import { QueryClient } from "@tanstack/react-query";
import queryClient, { clearCandidateSessionCache, queryClient as namedQueryClient } from "../queryClient";

describe("shared TanStack Query client", () => {
  beforeEach(() => queryClient.clear());
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

  it("clears the whole shared cache for a candidate session boundary", () => {
    queryClient.setQueryData(["candidate-a", "profile"], { name: "Candidate A" });

    clearCandidateSessionCache();

    expect(queryClient.getQueryData(["candidate-a", "profile"])).toBeUndefined();
  });
});
