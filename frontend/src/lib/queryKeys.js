function candidateIdOrNull(candidateId) {
  if (typeof candidateId !== "string") return null;
  const normalized = candidateId.trim();
  return normalized || null;
}

export function isCandidateIdReady(candidateId) {
  return candidateIdOrNull(candidateId) !== null;
}

export const candidateKeys = {
  all: (candidateId) => ["candidate", candidateIdOrNull(candidateId)],
  profile: (candidateId) => ["candidate", candidateIdOrNull(candidateId), "profile"],
  jobs: (candidateId, variant = "list") => [
    "candidate",
    candidateIdOrNull(candidateId),
    "jobs",
    variant,
  ],
  billing: (candidateId) => ["candidate", candidateIdOrNull(candidateId), "billing"],
  documents: (candidateId) => ["candidate", candidateIdOrNull(candidateId), "documents"],
  notifications: (candidateId) => [
    "candidate",
    candidateIdOrNull(candidateId),
    "notifications",
  ],
  jobMatch: (candidateId, jobId) => [
    "candidate",
    candidateIdOrNull(candidateId),
    "jobMatch",
    jobId,
  ],
  resumeImprovement: (candidateId, jobId) => [
    "candidate",
    candidateIdOrNull(candidateId),
    "resumeImprovement",
    jobId,
  ],
};
