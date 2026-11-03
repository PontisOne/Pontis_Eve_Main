import axios from "axios";
import { API } from "./apiConfig";

export async function fetchCandidateJobs({ candidateId, opportunityType, requestMore = false }) {
  const response = await axios.get(`${API}/candidate/${candidateId}/jobs`, {
    params: {
      ...(requestMore ? { request_more: true } : {}),
      type: opportunityType === "intern" ? "intern" : "jobs",
    },
  });
  const jobs = Array.isArray(response.data) ? response.data : [];
  const headerTotal = Number(response.headers?.["x-total-matching-jobs"]);
  return {
    jobs,
    total: Number.isFinite(headerTotal) ? headerTotal : jobs.length,
  };
}

export function trackCandidateJob({ candidateId, jobId, tracked }) {
  return tracked
    ? axios.post(`${API}/candidate/${candidateId}/jobs/${jobId}/track`)
    : axios.delete(`${API}/candidate/${candidateId}/jobs/${jobId}/track`);
}

export function dismissCandidateJob({ candidateId, jobId, reason }) {
  return axios.post(
    `${API}/candidate/${candidateId}/jobs/${jobId}/dismiss`,
    reason ? { reason } : {},
  );
}

export function viewCandidateJob({ candidateId, jobId }) {
  return axios.post(`${API}/candidate/${candidateId}/jobs/${jobId}/view`);
}

export function applyCandidateJob({ candidateId, jobId }) {
  return axios.post(`${API}/candidate/${candidateId}/jobs/${jobId}/apply`);
}
