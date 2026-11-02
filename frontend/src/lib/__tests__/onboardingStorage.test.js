import { isVoiceIntakeCompleteStatus, loadOnboardingState, saveOnboardingState } from "../onboardingStorage";

describe("isVoiceIntakeCompleteStatus", () => {
  it("treats the backend success variants as complete", () => {
    expect(isVoiceIntakeCompleteStatus("completed")).toBe(true);
    expect(isVoiceIntakeCompleteStatus("complete")).toBe(true);
    expect(isVoiceIntakeCompleteStatus("done")).toBe(true);
    expect(isVoiceIntakeCompleteStatus("finished")).toBe(true);
    expect(isVoiceIntakeCompleteStatus("duplicate")).toBe(true);
  });

  it("keeps in-progress and empty statuses incomplete", () => {
    expect(isVoiceIntakeCompleteStatus("in_progress")).toBe(false);
    expect(isVoiceIntakeCompleteStatus("partial")).toBe(false);
    expect(isVoiceIntakeCompleteStatus("")).toBe(false);
    expect(isVoiceIntakeCompleteStatus(null)).toBe(false);
  });
});

describe("Google registration state", () => {
  beforeEach(() => localStorage.clear());

  it("persists the short-lived registration credential for resume onboarding", () => {
    saveOnboardingState({
      authenticated: true,
      googleRegistrationToken: "registration-credential",
    });

    expect(loadOnboardingState().googleRegistrationToken).toBe("registration-credential");
  });
});

describe("validated onboarding storage", () => {
  beforeEach(() => localStorage.clear());

  it("loads a valid current payload and preserves opaque profile data", () => {
    localStorage.setItem("eve_onboarding_v1", JSON.stringify({
      step: 3,
      candidateId: "candidate-a",
      parsedProfile: { unexpected: ["backend", 42] },
    }));

    expect(loadOnboardingState()).toMatchObject({
      step: 3,
      candidateId: "candidate-a",
      parsedProfile: { unexpected: ["backend", 42] },
      voiceIntakeProgress: 0,
    });
  });

  it("defaults missing supported fields from an older payload", () => {
    localStorage.setItem("eve_onboarding_v1", JSON.stringify({ step: 2 }));

    expect(loadOnboardingState()).toMatchObject({
      step: 2,
      countryCode: "US",
      candidateToken: null,
      certsMeta: [],
    });
  });

  it.each([
    ["malformed JSON", "{"],
    ["schema-invalid values", JSON.stringify({ step: "not-a-number" })],
  ])("removes %s and returns safe defaults", (_label, raw) => {
    localStorage.setItem("eve_onboarding_v1", raw);

    expect(loadOnboardingState().candidateToken).toBeNull();
    expect(localStorage.getItem("eve_onboarding_v1")).toBeNull();
  });

  it("returns safe defaults when invalid storage cannot be removed", () => {
    localStorage.setItem("eve_onboarding_v1", JSON.stringify({ step: "invalid" }));
    const removeItem = vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new Error("storage unavailable");
    });

    expect(loadOnboardingState()).toMatchObject({ step: 1, candidateId: null });
    removeItem.mockRestore();
  });

  it("returns safe defaults when storage reads throw", () => {
    const getItem = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("storage unavailable");
    });

    expect(loadOnboardingState()).toMatchObject({ step: 1, candidateId: null });
    getItem.mockRestore();
  });
});
