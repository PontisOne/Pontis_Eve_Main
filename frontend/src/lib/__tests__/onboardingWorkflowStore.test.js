import {
  loadOnboardingWorkflow,
  saveOnboardingState,
} from "../onboardingStorage";
import {
  resetOnboardingWorkflowStore,
  useOnboardingWorkflowStore,
} from "../onboardingWorkflowStore";

describe("onboarding workflow store", () => {
  beforeEach(() => {
    localStorage.clear();
    useOnboardingWorkflowStore.setState(loadOnboardingWorkflow());
  });

  it("hydrates resumable workflow data without session fields", () => {
    saveOnboardingState({
      step: 3,
      parsedProfile: { name: "Candidate" },
      candidateToken: "secret",
      authenticated: true,
      activeTab: "jobs",
    });

    useOnboardingWorkflowStore.getState().hydrate();
    expect(useOnboardingWorkflowStore.getState()).toMatchObject({
      step: 3,
      parsedProfile: { name: "Candidate" },
    });
    expect(useOnboardingWorkflowStore.getState().candidateToken).toBeUndefined();
    expect(useOnboardingWorkflowStore.getState().activeTab).toBeUndefined();
  });

  it("persists step and voice progress through the validated boundary", () => {
    useOnboardingWorkflowStore.getState().setStep(4);
    useOnboardingWorkflowStore.getState().setVoiceProgress(2);

    expect(loadOnboardingWorkflow()).toMatchObject({ step: 4, voiceIntakeProgress: 2 });
    useOnboardingWorkflowStore.getState().setVoiceProgress(1);
    expect(loadOnboardingWorkflow().voiceIntakeProgress).toBe(2);
  });

  it("reset clears workflow data while preserving auth/session compatibility fields", () => {
    saveOnboardingState({
      step: 4,
      parsedProfile: { name: "Candidate" },
      candidateId: "candidate-1",
      candidateToken: "secret",
      authenticated: true,
      activeTab: "jobs",
      voiceIntakeProgress: 3,
    });
    useOnboardingWorkflowStore.getState().hydrate();

    resetOnboardingWorkflowStore();

    expect(loadOnboardingWorkflow()).toMatchObject({
      step: 1,
      parsedProfile: null,
      voiceIntakeProgress: 0,
    });
    expect(JSON.parse(localStorage.getItem("eve_onboarding_v1"))).toMatchObject({
      candidateId: "candidate-1",
      candidateToken: "secret",
      authenticated: true,
      activeTab: "jobs",
    });
  });

  it("recovers safely from malformed persisted storage", () => {
    localStorage.setItem("eve_onboarding_v1", "{");
    useOnboardingWorkflowStore.getState().hydrate();

    expect(useOnboardingWorkflowStore.getState()).toMatchObject({
      step: 1,
      countryCode: "US",
      parsedProfile: null,
      voiceIntakeProgress: 0,
    });
    expect(localStorage.getItem("eve_onboarding_v1")).toBeNull();
  });

  it("does not persist excluded fields from workflow updates", () => {
    useOnboardingWorkflowStore.getState().updateWorkflow({
      step: 2,
      candidateToken: "must-not-enter-store",
      authenticated: true,
      activeTab: "profile",
      transcription: true,
      muted: true,
      voiceElapsedMs: 9000,
      file: new File(["resume"], "resume.pdf"),
    });

    const persisted = JSON.parse(localStorage.getItem("eve_onboarding_v1"));
    expect(useOnboardingWorkflowStore.getState().candidateToken).toBeUndefined();
    expect(useOnboardingWorkflowStore.getState().activeTab).toBeUndefined();
    expect(persisted.candidateToken).toBeNull();
    expect(persisted.authenticated).toBe(false);
    expect(persisted.activeTab).toBeNull();
    expect(persisted).not.toHaveProperty("transcription");
    expect(persisted).not.toHaveProperty("muted");
    expect(persisted).not.toHaveProperty("voiceElapsedMs");
    expect(persisted).not.toHaveProperty("file");
    expect(loadOnboardingWorkflow()).not.toMatchObject({
      transcription: expect.anything(),
      muted: expect.anything(),
      voiceElapsedMs: expect.anything(),
    });
  });
});
