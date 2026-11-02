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
  transcription: false,
  muted: false,
  voiceElapsedMs: 0, // how long the candidate stayed on the voice-intake step
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
  transcription: z.boolean().optional(),
  muted: z.boolean().optional(),
  voiceElapsedMs: z.number().nonnegative().optional(),
  voiceIntakeCompleted: z.boolean().optional(),
  voiceIntakeProgress: z.number().int().nonnegative().optional(),
  isOpenToMatches: z.boolean().optional(),
  authenticated: z.boolean().optional(),
  profile: z.unknown().nullable().optional(),
  activeTab: z.string().nullable().optional(),
});

function safeDefaults() {
  return { ...DEFAULT_STATE, certsMeta: [] };
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
      transcription: state.transcription,
      muted: state.muted,
      voiceElapsedMs: state.voiceElapsedMs,
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
