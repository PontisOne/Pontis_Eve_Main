import axios from "axios";
import {
  applyCandidateJob,
  dismissCandidateJob,
  fetchCandidateJobs,
  trackCandidateJob,
  viewCandidateJob,
} from "../candidateJobs";

vi.mock("axios", () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
  },
}));

describe("candidate job requests", () => {
  beforeEach(() => vi.clearAllMocks());

  it("preserves jobs and the server total from a candidate-scoped read", async () => {
    axios.get.mockResolvedValue({ data: [{ id: "job-1" }], headers: { "x-total-matching-jobs": "12" } });

    await expect(fetchCandidateJobs({ candidateId: "cand-1", opportunityType: "intern" }))
      .resolves.toEqual({ jobs: [{ id: "job-1" }], total: 12 });
    expect(axios.get).toHaveBeenCalledWith(expect.stringContaining("/candidate/cand-1/jobs"), {
      params: { type: "intern" },
    });
  });

  it("exposes action requests without retrying or hiding failures", async () => {
    const error = new Error("request failed");
    axios.post.mockRejectedValueOnce(error).mockResolvedValueOnce({});

    await expect(trackCandidateJob({ candidateId: "cand-1", jobId: "job-1", tracked: true })).rejects.toBe(error);
    await expect(dismissCandidateJob({ candidateId: "cand-1", jobId: "job-1", reason: "remote" })).resolves.toEqual({});
    await viewCandidateJob({ candidateId: "cand-1", jobId: "job-1" });
    await applyCandidateJob({ candidateId: "cand-1", jobId: "job-1" });
    expect(axios.post).toHaveBeenCalledTimes(4);
  });

  it("uses delete for untrack and includes pagination requests", async () => {
    axios.delete.mockResolvedValue({});
    axios.get.mockResolvedValue({ data: [], headers: {} });

    await trackCandidateJob({ candidateId: "cand-1", jobId: "job-1", tracked: false });
    await fetchCandidateJobs({ candidateId: "cand-1", opportunityType: "jobs", requestMore: true });
    expect(axios.delete).toHaveBeenCalledWith(expect.stringContaining("/candidate/cand-1/jobs/job-1/track"));
    expect(axios.get).toHaveBeenCalledWith(expect.any(String), { params: { request_more: true, type: "jobs" } });
  });
});
