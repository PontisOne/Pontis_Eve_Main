import { create } from "zustand";
import {
  loadOnboardingWorkflow,
  saveOnboardingWorkflow,
} from "./onboardingStorage";

const initialWorkflow = loadOnboardingWorkflow();

export const useOnboardingWorkflowStore = create((set) => ({
  ...initialWorkflow,
  hydrate: () => set(loadOnboardingWorkflow()),
  updateWorkflow: (patch) => {
    const next = saveOnboardingWorkflow(patch);
    set(next);
    return next;
  },
  setStep: (step) => {
    const next = saveOnboardingWorkflow({ step });
    set(next);
  },
  setParsedProfile: (parsedProfile) => {
    const next = saveOnboardingWorkflow({ parsedProfile });
    set(next);
  },
  setVoiceProgress: (voiceIntakeProgress) => {
    const current = loadOnboardingWorkflow();
    if (voiceIntakeProgress <= (current.voiceIntakeProgress ?? 0)) return;
    const next = saveOnboardingWorkflow({ voiceIntakeProgress });
    set(next);
  },
  resetWorkflow: () => {
    const current = loadOnboardingWorkflow();
    const next = saveOnboardingWorkflow({
      ...current,
      step: 1,
      countryCode: "US",
      phoneDigits: "",
      resumeMeta: null,
      certsMeta: [],
      parsedProfile: null,
      voiceIntakeCompleted: false,
      voiceIntakeProgress: 0,
    });
    set(next);
  },
}));

export function resetOnboardingWorkflowStore() {
  useOnboardingWorkflowStore.getState().resetWorkflow();
}
