import axios from "axios";
import { API } from "./apiConfig";
import { hydrateProfileStrength } from "./profileStrength";
import { normalizeProfileForDisplay } from "./profileNormalization";

export function mapCandidateProfile(data, candidateId, isOpenToMatches = true) {
  const profileCandidateId = data?.candidate_id ?? data?.candidateId ?? candidateId ?? null;
  return hydrateProfileStrength(normalizeProfileForDisplay({
    ...data,
    candidate_id: profileCandidateId,
    candidateId: profileCandidateId,
    avatar: data?.photo_url ?? data?.avatar ?? null,
    isOpenToMatches,
    voice_intake_resume: data?.voice_intake_resume ?? null,
  }));
}

export async function fetchCandidateProfile(candidateId) {
  const { data } = await axios.get(`${API}/candidate/${candidateId}/profile`);
  return data;
}

export async function updateCandidateProfile({ candidateId, payload }) {
  const { data } = await axios.put(`${API}/candidate/${candidateId}/profile`, payload);
  return data;
}
