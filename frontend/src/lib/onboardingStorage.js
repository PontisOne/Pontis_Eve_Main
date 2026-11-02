import { z } from "zod";

// Simple, resilient localStorage wrapper for onboarding progress.
// File objects can't be serialized, so we only persist metadata + parsed profile.

const KEY = "eve_onboarding_v1";

const DEFAULT_STATE = {
  step: 1,
  countryCode: "US",
  phoneDigits: "",
  resumeMeta: null, // { name, size }
  certsMeta: [], // [{ name, size }]
  parsedProfile: null, // full backend response
  candidateId: null, // UUID from the Dashboard candidate record
  candidateToken: null, // signed backend-issued candidate session token
  googleRegistrationToken: null, // short-lived credential for atomic Google registration
  newlyOnboarded: false,
  voiceIntakeCompleted: false,
  voiceIntakeProgress: 0, // last answered question index (0 = not started)
  isOpenToMatches: true,
  authenticated: false,
  profile: null, // { name, email, picture, google_id }
};

const fileMetaSchema = z.object({
  name: z.string(),
  size: z.number().nonnegative(),
}).nullable();

const onboardingStateSchema = z.object({
  step: z.number().int().positive().optional(),
  countryCode: z.string().optional(),
  phoneDigits: z.string().optional(),
  resumeMeta: fileMetaSchema.optional(),
  certsMeta: z.array(fileMetaSchema.unwrap()).optional(),
  // Backend response shape is intentionally opaque at this browser boundary.
  parsedProfile: z.unknown().nullable().optional(),
  candidateId: z.string().nullable().optional(),
  candidateToken: z.string().nullable().optional(),
  googleRegistrationToken: z.string().nullable().optional(),
  newlyOnboarded: z.boolean().optional(),
  voiceIntakeCompleted: z.boolean().optional(),
  voiceIntakeProgress: z.number().int().nonnegative().optional(),
  isOpenToMatches: z.boolean().optional(),
  authenticated: z.boolean().optional(),
  profile: z.unknown().nullable().optional(),
  activeTab: z.string().nullable().optional(),
});

// Issue #14 workflow persistence is deliberately narrower than the legacy
// onboarding payload. Authentication/session and dashboard fields remain
// owned by the existing compatibility API below.
export const onboardingWorkflowSchema = z.object({
  step: z.number().int().positive().optional(),
  countryCode: z.string().optional(),
  phoneDigits: z.string().optional(),
  resumeMeta: fileMetaSchema.optional(),
  certsMeta: z.array(fileMetaSchema.unwrap()).optional(),
  parsedProfile: z.unknown().nullable().optional(),
  voiceIntakeCompleted: z.boolean().optional(),
  voiceIntakeProgress: z.number().int().nonnegative().optional(),
});

function safeDefaults() {
  return { ...DEFAULT_STATE, certsMeta: [] };
}

function workflowDefaults() {
  return {
    step: 1,
    countryCode: "US",
    phoneDigits: "",
    resumeMeta: null,
    certsMeta: [],
    parsedProfile: null,
    voiceIntakeCompleted: false,
    voiceIntakeProgress: 0,
  };
}

export function loadOnboardingState() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return safeDefaults();
    const parsed = JSON.parse(raw);
    const result = onboardingStateSchema.safeParse(parsed);
    if (!result.success) {
      try { localStorage.removeItem(KEY); } catch { /* storage is optional */ }
      return safeDefaults();
    }
    return { ...safeDefaults(), ...result.data };
  } catch {
    try { localStorage.removeItem(KEY); } catch { /* storage is optional */ }
    return safeDefaults();
  }
}

export function loadOnboardingWorkflow() {
  const state = loadOnboardingState();
  const result = onboardingWorkflowSchema.safeParse(state);
  return result.success ? { ...workflowDefaults(), ...result.data } : workflowDefaults();
}

export function saveOnboardingWorkflow(patch) {
  const current = loadOnboardingState();
  const workflow = onboardingWorkflowSchema.parse({
    ...loadOnboardingWorkflow(),
    ...patch,
  });
  saveOnboardingState({ ...current, ...workflow });
  return workflow;
}

export function saveOnboardingState(state) {
  try {
    // Only persist serializable slice
    const slim = {
      step: state.step,
      countryCode: state.countryCode,
      phoneDigits: state.phoneDigits,
      resumeMeta: state.resumeMeta,
      certsMeta: state.certsMeta,
      parsedProfile: state.parsedProfile,
      candidateId: state.candidateId ?? null,
      candidateToken: state.candidateToken ?? null,
      googleRegistrationToken: state.googleRegistrationToken ?? null,
      newlyOnboarded: state.newlyOnboarded ?? false,
      voiceIntakeCompleted: state.voiceIntakeCompleted ?? false,
      voiceIntakeProgress: state.voiceIntakeProgress ?? 0,
      isOpenToMatches: state.isOpenToMatches ?? true,
      authenticated: state.authenticated ?? false,
      profile: state.profile ?? null,
      activeTab: state.activeTab ?? null,
    };
    localStorage.setItem(KEY, JSON.stringify(slim));
  } catch {
    // Quota / private mode — ignore silently
  }
}

export function clearOnboardingState() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}

export function isVoiceIntakeCompleteStatus(status) {
  const normalized = String(status || "").toLowerCase();
  // Accept the backend's success variants so onboarding never falls through to dashboard.
  return ["completed", "complete", "done", "finished", "duplicate"].includes(normalized);
}

/**
 * When restoring, clamp the step to something we can actually resume:
 * - If no parsedProfile yet → user must re-upload their resume, so cap at 2.
 * - If parsedProfile exists → user can resume anywhere from 3 onward.
 */
export function resumableStep(state) {
  const step = state?.step || 1;
  if (state?.parsedProfile) return step;
  return Math.min(step, 2);
}
