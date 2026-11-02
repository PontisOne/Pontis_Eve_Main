import { QueryClient } from "@tanstack/react-query";

// Keep one QueryClient instance for both the provider and session lifecycle
// code. This module intentionally has no app or auth imports to stay cycle-free.
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: false,
    },
  },
});

// The application currently has no public TanStack Query consumers, so clearing
// the shared client is the safest way to prevent candidate data crossing sessions.
export function clearCandidateSessionCache() {
  queryClient.clear();
}

export default queryClient;
