import { candidateKeys, isCandidateIdReady } from "../queryKeys";

describe("candidate query keys", () => {
  it("creates deterministic keys with candidate identity before the resource", () => {
    expect(candidateKeys.profile(" candidate-a ")).toEqual([
      "candidate",
      "candidate-a",
      "profile",
    ]);
    expect(candidateKeys.jobs("candidate-a", "tracked")).toEqual([
      "candidate",
      "candidate-a",
      "jobs",
      "tracked",
    ]);
  });

  it("isolates resources for different candidates", () => {
    expect(candidateKeys.profile("candidate-a")).not.toEqual(
      candidateKeys.profile("candidate-b"),
    );
    expect(candidateKeys.jobs("candidate-a")).not.toEqual(
      candidateKeys.jobs("candidate-b"),
    );
  });

  it("provides a shared identity guard convention", () => {
    expect(isCandidateIdReady("candidate-a")).toBe(true);
    expect(isCandidateIdReady("  ")).toBe(false);
    expect(isCandidateIdReady(null)).toBe(false);
  });
});
