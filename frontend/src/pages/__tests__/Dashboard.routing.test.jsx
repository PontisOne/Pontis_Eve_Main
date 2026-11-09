import React from "react";
import { act } from "react";
import axios from "axios";
import { MemoryRouter } from "react-router-dom";
import { render } from "@testing-library/react";

import Dashboard from "../Dashboard";
import { saveOnboardingState } from "../../lib/onboardingStorage";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let lastLivingProfileProps = null;
let mockVoiceIntakeOnComplete = null;

vi.mock("../../components/Sidebar", () => ({ default: (props) => (
  <div data-testid="sidebar">
    <button data-testid="sidebar-jobs" onClick={() => props.setActiveTab?.("jobs")}>Jobs</button>
    <button data-testid="sidebar-profile" onClick={() => props.setActiveTab?.("profile")}>Profile</button>
  </div>
)}));
let mockChatHubOnMicClick = null;
vi.mock("../../components/ChatHub", () => ({ default: (props) => (
  <div data-testid="chat-hub">
    <div data-testid="chat-feed">
      {(props.chats || []).map((chat) => (
        <div key={chat.id} data-testid={`chat-message-${chat.id}`}>
          {chat.content}
        </div>
      ))}
      {(props.quickActions || []).map((question) => (
        <button key={question} data-testid="chat-suggestion" onClick={() => props.onSuggestionClick?.(question)}>
          {question}
        </button>
      ))}
    </div>
    <input
      data-testid="chat-text-input"
      value={props.inputValue || ""}
      onChange={(e) => props.setInputValue?.(e.target.value)}
    />
    <button data-testid="chat-send-btn" onClick={(e) => props.onSend?.(e)}>
      send
    </button>
    {props.onMicClick && (
      <button data-testid="chat-mic-btn" onClick={props.onMicClick}>
        Mic
      </button>
    )}
  </div>
)}));
let mockVoiceIntakeCompletionResult = null;
vi.mock("../../components/LivingProfile", () => ({ __esModule: true, default: (props) => {
  lastLivingProfileProps = props;
  return (
    <div data-testid="living-profile" data-candidate-id={props.userProfile?.candidate_id || ""}>
      <button data-testid="refresh-profile-btn" onClick={() => props.onPhotoChange("new-photo-url")}>
        Refresh profile
      </button>
    </div>
  );
}, JobsTab: () => <div data-testid="jobs-deck" /> }));
vi.mock("../../components/SwipeJobCard", () => ({ default: () => <div data-testid="jobs-deck" /> }));
let capturedVoiceIntakeCandidateProfile = null;
vi.mock("../../components/onboarding/VoiceIntake", () => ({ default: (props) => {
  mockVoiceIntakeOnComplete = props.onComplete;
  capturedVoiceIntakeCandidateProfile = props.candidateProfile;
  return <div data-testid="voice-intake" />;
}}));
vi.mock("react-resizable-panels", () => ({
  PanelGroup: ({ children }) => <div>{children}</div>,
  Panel: ({ children }) => <div>{children}</div>,
  PanelResizeHandle: ({ children }) => <div>{children}</div>,
}));
vi.mock("sonner", () => ({
  Toaster: () => null,
  toast: {
    error: jest.fn(),
    success: jest.fn(),
  },
}));

const currentQuestion =
  "Are there specific industries or types of companies you'd prefer to work with in your next role?";

const inProgressResume = {
  status: "in_progress",
  has_open_question: true,
  current_question: currentQuestion,
  next_question: "",
};

function makeProfile(overrides = {}) {
  const profile = {
    photo_url: null,
    name: "Suram Test",
    email: "suram@example.com",
    phone: "5551234567",
    headline: "Product Manager",
    location: "New York, NY",
    bio: "",
    experience: [],
    education: [],
    keySkills: ["Product", "Strategy"],
    experience_years: 5,
    availability: "Immediate",
    preferred_roles: ["Product Manager"],
    certifications: [],
    additional_information: "",
    strengthPercent: 90,
    profile_strength_percent: 90,
    profile_strength_label: "Strong",
    voice_intake_resume: {
      status: "completed",
      has_open_question: false,
      current_question: "",
      next_question: "",
    },
    ...overrides,
  };
  if (profile.voice_intake_resume?.status === "in_progress" && profile.voice_intake_resume.current_question) {
    profile.missing_questions = [{ question: profile.voice_intake_resume.current_question, status: "OPEN" }];
  }
  return profile;
}

function mockDashboardRequests(profileResponses) {
  let profileCallCount = 0;

  axios.get.mockImplementation((url) => {
    if (url.includes("/profile")) {
      const index = Math.min(profileCallCount, profileResponses.length - 1);
      profileCallCount += 1;
      return Promise.resolve({ data: profileResponses[index] });
    }

    if (url.includes("/chat")) {
      return Promise.resolve({ data: { messages: [] } });
    }

    if (url.includes("/opportunities") || url.includes("/notifications")) {
      return Promise.resolve({ data: [] });
    }

    if (url.includes("/jobs")) {
      return Promise.resolve({ data: [{ id: "job-1", company: "Acme" }] });
    }

    if (url.includes("/documents")) {
      return Promise.resolve({ data: { resume: null, certificates: [] } });
    }

    return Promise.resolve({ data: {} });
  });
}

function renderDashboard() {
  return render(
    <MemoryRouter initialEntries={["/dashboard"]}>
      <Dashboard />
    </MemoryRouter>,
  );
}

async function waitForSelector(container, selector) {
  for (let i = 0; i < 20; i += 1) {
    await act(async () => { await Promise.resolve(); });
    const node = container.querySelector(selector);
    if (node) return node;
  }
  throw new Error(`Timed out waiting for ${selector}`);
}

async function waitForCondition(predicate) {
  for (let i = 0; i < 20; i += 1) {
    await act(async () => { await Promise.resolve(); });
    if (predicate()) return true;
  }
  throw new Error("Timed out waiting for condition");
}

async function mountVoiceIntakeFromChat(container) {
  const jobsSidebar = container.querySelector('[data-testid="sidebar-jobs"]');
  if (jobsSidebar) {
    act(() => { jobsSidebar.click(); });
  }
  const chatTab = container.querySelector('[data-testid="chat-tab"]');
  if (chatTab) {
    act(() => { chatTab.click(); });
  }
  const chatHub = await waitForSelector(container, '[data-testid="chat-hub"]');
  expect(chatHub).toBeTruthy();
  const mic = container.querySelector('[data-testid="chat-mic-btn"]');
  expect(mic).toBeTruthy();
  act(() => { mic.click(); });
  await waitForSelector(container, '[data-testid="voice-intake"]');
  expect(mockVoiceIntakeOnComplete).toEqual(expect.any(Function));
}

function setInputValue(input, value) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
  setter.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

describe("Dashboard voice intake routing", () => {
  let renderResult;

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    lastLivingProfileProps = null;
    mockVoiceIntakeCompletionResult = null;
    mockVoiceIntakeOnComplete = null;
    capturedVoiceIntakeCandidateProfile = null;
    mockChatHubOnMicClick = null;
    saveOnboardingState({
      candidateId: "cand-123",
      voiceIntakeCompleted: true,
      isOpenToMatches: true,
    });
  });

  afterEach(() => {
    renderResult?.unmount?.();
    renderResult = null;
  });

  it("shows Jobs for you by default when the backend profile is completed", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
    mockDashboardRequests([makeProfile()]);

    renderResult = renderDashboard();

    expect(await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
  });

  it("opens Chat with Eve and restores the current question when the backend profile is in progress", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    const chatHub = await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    expect(chatHub.textContent).toContain(currentQuestion);
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
  });

  it("does not let stale localStorage completed state override a backend in-progress resume", async () => {
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
    ]);

    saveOnboardingState({
      candidateId: "cand-123",
      voiceIntakeCompleted: true,
      isOpenToMatches: true,
      activeTab: "jobs",
    });

    renderResult = renderDashboard();

    const chatHub = await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    expect(chatHub.textContent).toContain(currentQuestion);
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });

  it("switches back to Jobs for you after a refreshed profile reports voice intake completed", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
      makeProfile({
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    const chatHub = await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    expect(chatHub.textContent).toContain(currentQuestion);

    const refreshBtn = renderResult.container.querySelector('[data-testid="refresh-profile-btn"]');
    expect(refreshBtn).toBeTruthy();

    act(() => {
      refreshBtn.click();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
  });

  it("routes an incomplete voice intake back to Chat with Eve", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
    mockVoiceIntakeCompletionResult = { status: "in_progress" };
    mockDashboardRequests([
      makeProfile({
        strengthPercent: 62,
        profile_strength_percent: 62,
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
      makeProfile({
        strengthPercent: 62,
        profile_strength_percent: 62,
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    await mountVoiceIntakeFromChat(renderResult.container);

    await act(async () => {
      mockVoiceIntakeOnComplete?.(mockVoiceIntakeCompletionResult);
      await Promise.resolve();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });

  it("routes a completed voice intake to Jobs for you", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
    mockVoiceIntakeCompletionResult = { status: "completed" };
    mockDashboardRequests([
      makeProfile({
        strengthPercent: 62,
        profile_strength_percent: 62,
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
      makeProfile({
        strengthPercent: 90,
        profile_strength_percent: 90,
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    await mountVoiceIntakeFromChat(renderResult.container);

    await act(async () => {
      mockVoiceIntakeOnComplete?.(mockVoiceIntakeCompletionResult);
      await Promise.resolve();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
  });

  it("preserves the backend candidate id on the profile object so photo uploads can use it", async () => {
    mockDashboardRequests([
      makeProfile({
        candidate_id: "cand-456",
      }),
    ]);

    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="living-profile"]');

    expect(lastLivingProfileProps).toBeTruthy();
    expect(lastLivingProfileProps.userProfile.candidate_id).toBe("cand-456");
    expect(lastLivingProfileProps.userProfile.candidateId).toBe("cand-456");
  });

  it("keeps Voice Intake additions in the dashboard profile after refresh", async () => {
    mockDashboardRequests([
      makeProfile({
        keySkills: ["Product", "Strategy", "Leadership"],
        certifications: ["AWS Certified Solutions Architect - Associate"],
        experience: [
          {
            id: "exp-voice",
            title: "Senior Product Manager",
            company: "VoiceCo",
            dates: "2024 — Present",
            description: "Led platform expansion.",
          },
        ],
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
    ]);

    renderResult = renderDashboard();
    await waitForCondition(
      () => Array.isArray(lastLivingProfileProps?.userProfile?.keySkills) &&
        lastLivingProfileProps.userProfile.keySkills.length > 0
    );
    expect(lastLivingProfileProps.userProfile.keySkills).toEqual(["Product", "Strategy", "Leadership"]);
    expect(lastLivingProfileProps.userProfile.certifications).toEqual(["AWS Certified Solutions Architect - Associate"]);
    expect(lastLivingProfileProps.userProfile.experience.map((exp) => exp.company)).toEqual(["VoiceCo"]);
  });
});

describe("Voice Intake completion/routing regression tests", () => {
  let renderResult;

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    lastLivingProfileProps = null;
    mockVoiceIntakeOnComplete = null;
    mockChatHubOnMicClick = null;
    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: false, isOpenToMatches: true, activeTab: "jobs" });
  });

  afterEach(() => {
    renderResult?.unmount?.();
    renderResult = null;
  });

  it("backend returns completed while localStorage says incomplete → Jobs for you", async () => {
    // localStorage says incomplete, backend says completed
    saveOnboardingState({
      candidateId: "cand-123",
      voiceIntakeCompleted: false,
      activeTab: "jobs",
    });
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    expect(await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
  });

  it("backend returns completed while voice_intake_resume says in_progress → Jobs for you", async () => {
    // First profile call returns in_progress, second (after refresh) returns completed
    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: false, activeTab: "jobs" });
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
      makeProfile({
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    // Initially shows chat (in_progress)
    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    // Trigger refresh (simulates VoiceIntake onComplete calling refreshProfile)
    const refreshBtn = renderResult.container.querySelector('[data-testid="refresh-profile-btn"]');
    act(() => { refreshBtn.click(); });

    // After refresh with completed status → Jobs for you
    expect(await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
  });

  it("backend returns in_progress → Chat with Eve", async () => {
    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: true });
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    const chatHub = await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    expect(chatHub.textContent).toContain(currentQuestion);
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });

  it("completed Voice Intake onComplete → refreshProfile → Jobs for you (never bypasses via local status)", async () => {
    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: false, activeTab: "jobs" });
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: inProgressResume }),
      makeProfile({
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    // Click Chat with Eve to show voice intake
    await mountVoiceIntakeFromChat(renderResult.container);

    // Simulate VoiceIntake calling onComplete with completed status
    await act(async () => {
      mockVoiceIntakeOnComplete?.({ status: "completed" });
      await Promise.resolve();
    });

    // After refreshProfile resolves with completed backend status → Jobs for you
    expect(await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
  });

  it("incomplete Voice Intake onComplete → refreshProfile → Chat with Eve", async () => {
    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: false });
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: inProgressResume }),
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    await mountVoiceIntakeFromChat(renderResult.container);

    await act(async () => {
      mockVoiceIntakeOnComplete?.({ status: "in_progress" });
      await Promise.resolve();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });

  it("call ended during/before Eve's introduction (no candidate speech) → Chat with Eve UI", async () => {
    // Candidate starts call, Eve begins speaking, candidate hangs up before saying anything
    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: false });
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: inProgressResume }),
      makeProfile({ voice_intake_resume: inProgressResume }),
    ]);

    renderResult = renderDashboard();

    await mountVoiceIntakeFromChat(renderResult.container);

    // Simulate VoiceIntake calling onComplete with no_interaction status
    // (triggered when call ends with no candidate speech)
    await act(async () => {
      mockVoiceIntakeOnComplete?.({ status: "no_interaction" });
      await Promise.resolve();
    });

    // Must show normal ChatHub with suggestion chips and input — never voice-intake screen
    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });

  it("no candidate speech save keeps the unanswered question available through Continue with Eve and mic resume", async () => {
    vi.useFakeTimers();
    const interruptedResume = {
      status: "in_progress",
      has_open_question: true,
      current_question: currentQuestion,
      next_question: currentQuestion,
      progress: 0,
      completed_turns: [],
      known_topics: [],
      missing_topics: ["background_experience"],
    };

    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: false });
    mockDashboardRequests([
      makeProfile({
        strengthPercent: 60,
        profile_strength_percent: 60,
        profile_strength_label: "Developing",
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
      makeProfile({
        strengthPercent: 60,
        profile_strength_percent: 60,
        profile_strength_label: "Developing",
        voice_intake_resume: interruptedResume,
      }),
    ]);

    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    act(() => {
      jest.advanceTimersByTime(900);
    });

    const popupChatBtn = await waitForSelector(renderResult.container, '[data-testid="weak-profile-chat-btn"]');
    act(() => {
      popupChatBtn.click();
    });

    await mountVoiceIntakeFromChat(renderResult.container);

    await act(async () => {
      mockVoiceIntakeOnComplete?.({ status: "no_interaction" });
      await Promise.resolve();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });

    await waitForSelector(renderResult.container, '[data-testid="voice-intake"]');
    expect(capturedVoiceIntakeCandidateProfile?.voice_intake_resume?.status).toBe("in_progress");
    expect(capturedVoiceIntakeCandidateProfile?.voice_intake_resume?.current_question).toBe(currentQuestion);
    jest.useRealTimers();
  });

  it("no candidate speech save keeps the unanswered question available through Maybe Later -> Jobs -> Chat", async () => {
    jest.useFakeTimers();
    const interruptedResume = {
      status: "in_progress",
      has_open_question: true,
      current_question: currentQuestion,
      next_question: currentQuestion,
      progress: 0,
      completed_turns: [],
      known_topics: [],
      missing_topics: ["background_experience"],
    };

    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: false });
    mockDashboardRequests([
      makeProfile({
        strengthPercent: 60,
        profile_strength_percent: 60,
        profile_strength_label: "Developing",
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
      makeProfile({
        strengthPercent: 60,
        profile_strength_percent: 60,
        profile_strength_label: "Developing",
        voice_intake_resume: interruptedResume,
      }),
    ]);

    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    act(() => {
      jest.advanceTimersByTime(900);
    });

    const dismissBtn = await waitForSelector(renderResult.container, '[data-testid="weak-profile-dismiss-btn"]');
    act(() => {
      dismissBtn.click();
    });

    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();

    await mountVoiceIntakeFromChat(renderResult.container);

    await act(async () => {
      mockVoiceIntakeOnComplete?.({ status: "no_interaction" });
      await Promise.resolve();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    await mountVoiceIntakeFromChat(renderResult.container);
    expect(capturedVoiceIntakeCandidateProfile?.voice_intake_resume?.status).toBe("in_progress");
    expect(capturedVoiceIntakeCandidateProfile?.voice_intake_resume?.current_question).toBe(currentQuestion);
    jest.useRealTimers();
  });

  it("call ended with 0 answers → normal Chat with Eve UI (no voice UI)", async () => {
    // Backend returns null voice_intake_resume (no data captured)
    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: false });
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: inProgressResume }),
      makeProfile({ voice_intake_resume: inProgressResume }),
    ]);

    renderResult = renderDashboard();

    await mountVoiceIntakeFromChat(renderResult.container);

    // Simulate call ended with no answers (empty/partial result)
    await act(async () => {
      mockVoiceIntakeOnComplete?.({ status: "partial" });
      await Promise.resolve();
    });

    // Must show ChatHub, not voice-intake, not jobs-deck
    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });

  it("call ended after partial answers → normal Chat with Eve UI (no voice UI)", async () => {
    // Backend returns in_progress but without has_open_question (partial capture)
    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: false });
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: inProgressResume }),
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    await mountVoiceIntakeFromChat(renderResult.container);

    await act(async () => {
      mockVoiceIntakeOnComplete?.({ status: "in_progress" });
      await Promise.resolve();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });

  it("completed Voice Intake → existing completed flow routes to Jobs for you", async () => {
    saveOnboardingState({ candidateId: "cand-123", voiceIntakeCompleted: false });
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: inProgressResume }),
      makeProfile({
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    await mountVoiceIntakeFromChat(renderResult.container);

    await act(async () => {
      mockVoiceIntakeOnComplete?.({ status: "completed" });
      await Promise.resolve();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
  });

  it("refreshing the dashboard after completed Voice Intake still shows Jobs for you", async () => {
    // Simulate a reload: localStorage has voiceIntakeCompleted=true, backend confirms completed
    saveOnboardingState({
      candidateId: "cand-123",
      voiceIntakeCompleted: true,
      activeTab: "jobs",
      isOpenToMatches: true,
    });
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);

    renderResult = renderDashboard();

    expect(await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
  });
});

describe("Dashboard mic button — voice intake from chat panel", () => {
  let renderResult;

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    lastLivingProfileProps = null;
    mockVoiceIntakeOnComplete = null;
    mockChatHubOnMicClick = null;
    saveOnboardingState({
      candidateId: "cand-123",
      voiceIntakeCompleted: false,
      isOpenToMatches: true,
      activeTab: "jobs",
    });
  });

  afterEach(() => {
    renderResult?.unmount?.();
    renderResult = null;
  });

  it("mic button is present in ChatHub when centerView is chat", async () => {
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    expect(renderResult.container.querySelector('[data-testid="chat-mic-btn"]')).toBeTruthy();
  });

  it("clicking mic button shows VoiceIntake inline and hides ChatHub", async () => {
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="voice-intake"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });

  it("after mic voice intake completes, stays in dashboard (no summary) and shows ChatHub", async () => {
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
      makeProfile({
        bio: "Updated bio from voice",
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
    ]);
    renderResult = renderDashboard();

    await mountVoiceIntakeFromChat(renderResult.container);

    await act(async () => {
      mockVoiceIntakeOnComplete?.({ status: "completed", profile: { bio: "Updated bio from voice" } });
      await Promise.resolve();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });

  it("profile panel refreshes after mic voice intake completes with profile_updates", async () => {
    mockDashboardRequests([
      makeProfile({
        keySkills: ["Product"],
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
      makeProfile({
        keySkills: ["Product", "Leadership", "Strategy"],
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
    ]);
    renderResult = renderDashboard();

    await mountVoiceIntakeFromChat(renderResult.container);

    await act(async () => {
      mockVoiceIntakeOnComplete?.({
        status: "completed",
        profile_updates: { keySkills: ["Product", "Leadership", "Strategy"] },
      });
      await Promise.resolve();
    });

    act(() => { renderResult.container.querySelector('[data-testid="sidebar-profile"]').click(); });

    await waitForCondition(
      () =>
        Array.isArray(lastLivingProfileProps?.userProfile?.keySkills) &&
        lastLivingProfileProps.userProfile.keySkills.length >= 3
    );
    expect(lastLivingProfileProps.userProfile.keySkills).toContain("Leadership");
  });

  it("partial intake → Dashboard → mic → VoiceIntake receives fresh voice_intake_resume so VAPI resumes from saved question", async () => {
    const partialResume = {
      status: "in_progress",
      has_open_question: true,
      current_question: "What are your key skills?",
      next_question: "",
      completed_turns: [
        { question: "Tell me about your background.", answer: "I build APIs." },
        { question: "What is your current role?", answer: "Backend Developer." },
      ],
      known_topics: ["background_experience", "current_role"],
      missing_topics: ["skills_technologies", "target_role"],
    };

    // First call: mount (in_progress → ChatHub). Second call: refreshProfile on mic click.
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: partialResume }),
      makeProfile({ voice_intake_resume: partialResume }),
    ]);

    renderResult = renderDashboard();
    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    capturedVoiceIntakeCandidateProfile = null;
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });

    await waitForSelector(renderResult.container, '[data-testid="voice-intake"]');
    await waitForCondition(() => capturedVoiceIntakeCandidateProfile !== null);

    const vir = capturedVoiceIntakeCandidateProfile?.voice_intake_resume;
    expect(vir).toBeTruthy();
    expect(vir.status).toBe("in_progress");
    expect(vir.current_question).toBe("What are your key skills?");
    expect(vir.completed_turns).toHaveLength(2);
    expect(vir.known_topics).toContain("background_experience");
    expect(vir.missing_topics).toContain("skills_technologies");
  });

  it("mic voice intake with partial result stays in ChatHub (not jobs, not voice)", async () => {
    mockDashboardRequests([
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
      makeProfile({
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: currentQuestion,
          next_question: "",
        },
      }),
    ]);
    renderResult = renderDashboard();

    await mountVoiceIntakeFromChat(renderResult.container);

    await act(async () => {
      mockVoiceIntakeOnComplete?.({ status: "partial" });
      await Promise.resolve();
    });

    act(() => { renderResult.container.querySelector('[data-testid="sidebar-profile"]').click(); });

    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });
});

describe("Mic button always visible in Chat with Eve", () => {
  let renderResult;

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    lastLivingProfileProps = null;
    mockVoiceIntakeOnComplete = null;
    capturedVoiceIntakeCandidateProfile = null;
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
  });

  afterEach(() => {
    renderResult?.unmount?.();
    renderResult = null;
  });

  it("incomplete intake → mic visible + resume works", async () => {
    const partialResume = {
      status: "in_progress",
      has_open_question: true,
      current_question: "What are your key skills?",
      next_question: "",
      completed_turns: [{ question: "Tell me about yourself.", answer: "I build APIs." }],
      known_topics: ["background_experience"],
      missing_topics: ["skills_technologies", "target_role"],
    };
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: partialResume }),
      makeProfile({ voice_intake_resume: partialResume }),
    ]);
    renderResult = renderDashboard();
    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    // Mic is visible
    expect(renderResult.container.querySelector('[data-testid="chat-mic-btn"]')).toBeTruthy();

    // Clicking mic resumes from persisted state
    capturedVoiceIntakeCandidateProfile = null;
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });
    await waitForSelector(renderResult.container, '[data-testid="voice-intake"]');
    await waitForCondition(() => capturedVoiceIntakeCandidateProfile !== null);

    const vir = capturedVoiceIntakeCandidateProfile?.voice_intake_resume;
    expect(vir?.status).toBe("in_progress");
    expect(vir?.current_question).toBe("What are your key skills?");
    expect(vir?.completed_turns).toHaveLength(1);
    expect(vir?.known_topics).toContain("background_experience");
  });

  it("completed intake with profile <75% → mic visible in ChatHub", async () => {
    mockDashboardRequests([
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);
    renderResult = renderDashboard();

    // Wait for profile to load (jobs-deck is the default view for completed intake)
    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');

    const chatBtn = Array.from(renderResult.container.querySelectorAll("button")).find(
      (b) => b.textContent?.includes("Chat with Eve")
    );
    act(() => { chatBtn.click(); });

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    expect(renderResult.container.querySelector('[data-testid="chat-mic-btn"]')).toBeTruthy();
  });

  it("completed intake with profile >=75% → mic visible in ChatHub", async () => {
    mockDashboardRequests([
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Strong",
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Strong",
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');

    const chatBtn = Array.from(renderResult.container.querySelectorAll("button")).find(
      (b) => b.textContent?.includes("Chat with Eve")
    );
    act(() => { chatBtn.click(); });

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    expect(renderResult.container.querySelector('[data-testid="chat-mic-btn"]')).toBeTruthy();
  });

  it("mic click on completed intake opens VoiceIntake with completed state, does not reset profile", async () => {
    const completedResume = {
      status: "completed",
      has_open_question: false,
      current_question: "",
      next_question: "",
    };
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: completedResume }),
      makeProfile({ voice_intake_resume: completedResume }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');

    const chatBtn = Array.from(renderResult.container.querySelectorAll("button")).find(
      (b) => b.textContent?.includes("Chat with Eve")
    );
    act(() => { chatBtn.click(); });

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    capturedVoiceIntakeCandidateProfile = null;
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });
    await waitForSelector(renderResult.container, '[data-testid="voice-intake"]');
    await waitForCondition(() => capturedVoiceIntakeCandidateProfile !== null);

    // VoiceIntake receives the completed state — not null, not reset
    const vir = capturedVoiceIntakeCandidateProfile?.voice_intake_resume;
    expect(vir?.status).toBe("completed");
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
  });
});

describe("Dashboard mic — returning candidate behavior", () => {
  let renderResult;

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    lastLivingProfileProps = null;
    mockVoiceIntakeOnComplete = null;
    capturedVoiceIntakeCandidateProfile = null;
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
  });

  afterEach(() => {
    renderResult?.unmount?.();
    renderResult = null;
  });

  it("completed returning candidate: mic click opens VoiceIntake with completed voice_intake_resume (welcome-back flow)", async () => {
    const completedResume = {
      status: "completed",
      has_open_question: false,
      current_question: "",
      next_question: "",
      completed_turns: [
        { question: "Tell me about yourself.", answer: "I build APIs." },
      ],
      known_topics: ["background_experience", "skills_technologies"],
      missing_topics: [],
    };
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: completedResume }),
      makeProfile({ voice_intake_resume: completedResume }),
    ]);
    renderResult = renderDashboard();

    // Completed intake → default view is jobs-deck
    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');

    // Navigate to Chat with Eve
    const chatBtn = Array.from(renderResult.container.querySelectorAll("button")).find(
      (b) => b.textContent?.includes("Chat with Eve")
    );
    act(() => { chatBtn.click(); });
    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    // Click mic
    capturedVoiceIntakeCandidateProfile = null;
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });
    await waitForSelector(renderResult.container, '[data-testid="voice-intake"]');
    await waitForCondition(() => capturedVoiceIntakeCandidateProfile !== null);

    // VoiceIntake receives the completed resume — not null, not reset
    const vir = capturedVoiceIntakeCandidateProfile?.voice_intake_resume;
    expect(vir?.status).toBe("completed");
    // No restart: completed_turns preserved
    expect(vir?.completed_turns).toHaveLength(1);
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
  });

  it("completed returning candidate: VoiceIntake receives completed status so buildVoiceIntakeAssistantOverrides sets welcome-back firstMessage", async () => {
    // This test verifies the profile passed to VoiceIntake has status=completed,
    // which triggers the welcome-back firstMessage branch in buildVoiceIntakeAssistantOverrides.
    const completedResume = {
      status: "completed",
      has_open_question: false,
      current_question: "",
      next_question: "",
    };
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: completedResume }),
      makeProfile({ voice_intake_resume: completedResume }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');
    const chatBtn = Array.from(renderResult.container.querySelectorAll("button")).find(
      (b) => b.textContent?.includes("Chat with Eve")
    );
    act(() => { chatBtn.click(); });
    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    capturedVoiceIntakeCandidateProfile = null;
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });
    await waitForSelector(renderResult.container, '[data-testid="voice-intake"]');
    await waitForCondition(() => capturedVoiceIntakeCandidateProfile !== null);

    // The profile passed to VoiceIntake has completed status — the component
    // will build a welcome-back firstMessage from this.
    expect(capturedVoiceIntakeCandidateProfile?.voice_intake_resume?.status).toBe("completed");
  });

  it("incomplete returning candidate: mic click passes exact voice_intake_resume state so VAPI resumes from saved question", async () => {
    const partialResume = {
      status: "in_progress",
      has_open_question: true,
      current_question: "What kind of role are you targeting?",
      next_question: "",
      completed_turns: [
        { question: "Tell me about your background.", answer: "I build APIs." },
        { question: "What are your key skills?", answer: "Python, FastAPI." },
      ],
      known_topics: ["background_experience", "skills_technologies"],
      missing_topics: ["target_role", "availability_location"],
    };
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: partialResume }),
      makeProfile({ voice_intake_resume: partialResume }),
    ]);
    renderResult = renderDashboard();

    // In-progress → default view is chat-hub
    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    capturedVoiceIntakeCandidateProfile = null;
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });
    await waitForSelector(renderResult.container, '[data-testid="voice-intake"]');
    await waitForCondition(() => capturedVoiceIntakeCandidateProfile !== null);

    const vir = capturedVoiceIntakeCandidateProfile?.voice_intake_resume;
    expect(vir?.status).toBe("in_progress");
    expect(vir?.current_question).toBe("What kind of role are you targeting?");
    expect(vir?.completed_turns).toHaveLength(2);
    expect(vir?.known_topics).toContain("background_experience");
    expect(vir?.missing_topics).toContain("target_role");
  });

  it("incomplete returning candidate: does not restart from beginning (current_question is preserved, not empty)", async () => {
    const partialResume = {
      status: "in_progress",
      has_open_question: true,
      current_question: "What is your availability?",
      next_question: "",
      completed_turns: [
        { question: "Tell me about yourself.", answer: "I am a developer." },
      ],
      known_topics: ["background_experience"],
      missing_topics: ["availability_location"],
    };
    mockDashboardRequests([
      makeProfile({ voice_intake_resume: partialResume }),
      makeProfile({ voice_intake_resume: partialResume }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    capturedVoiceIntakeCandidateProfile = null;
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });
    await waitForSelector(renderResult.container, '[data-testid="voice-intake"]');
    await waitForCondition(() => capturedVoiceIntakeCandidateProfile !== null);

    // current_question must not be empty — a non-empty value means VAPI will resume, not restart
    expect(capturedVoiceIntakeCandidateProfile?.voice_intake_resume?.current_question).toBe(
      "What is your availability?"
    );
    expect(capturedVoiceIntakeCandidateProfile?.voice_intake_resume?.current_question).not.toBe("");
  });
});

describe("Maybe Later → Chat with Eve regression tests", () => {
  let renderResult;

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    lastLivingProfileProps = null;
    mockVoiceIntakeOnComplete = null;
    capturedVoiceIntakeCandidateProfile = null;
    jest.useFakeTimers();
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
  });

  afterEach(() => {
    renderResult?.unmount?.();
    renderResult = null;
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it("Maybe Later → Chat with Eve tab shows ChatHub, not VoiceIntake", async () => {
    mockDashboardRequests([
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');

    // Navigate to Chat with Eve tab
    act(() => { renderResult.container.querySelector('[data-testid="chat-tab"]').click(); });

    // Must show ChatHub — never VoiceIntake
    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
    expect(renderResult.container.querySelector('[data-testid="jobs-deck"]')).toBeNull();
  });

  it("Maybe Later → Chat with Eve tab shows mic button in ChatHub", async () => {
    mockDashboardRequests([
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');
    act(() => { renderResult.container.querySelector('[data-testid="chat-tab"]').click(); });

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    expect(renderResult.container.querySelector('[data-testid="chat-mic-btn"]')).toBeTruthy();
  });

  it("Maybe Later → Chat with Eve → mic click launches VoiceIntake", async () => {
    mockDashboardRequests([
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');
    act(() => { renderResult.container.querySelector('[data-testid="chat-tab"]').click(); });

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    capturedVoiceIntakeCandidateProfile = null;
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="voice-intake"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeNull();
  });

  it("Maybe Later → mic → VoiceIntake completes → profile updates reflected in Profile section", async () => {
    mockDashboardRequests([
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        keySkills: ["JavaScript"],
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        keySkills: ["JavaScript"],
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Strong",
        keySkills: ["JavaScript", "React", "Node.js"],
        bio: "Experienced full-stack developer",
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');
    act(() => { renderResult.container.querySelector('[data-testid="chat-tab"]').click(); });

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });
    await waitForSelector(renderResult.container, '[data-testid="voice-intake"]');

    // VoiceIntake completes with new profile data
    await act(async () => {
      mockVoiceIntakeOnComplete?.({
        status: "completed",
        profile_updates: {
          keySkills: ["JavaScript", "React", "Node.js"],
          bio: "Experienced full-stack developer",
        },
      });
      await Promise.resolve();
    });

    act(() => { renderResult.container.querySelector('[data-testid="sidebar-profile"]').click(); });

    // Profile section must reflect the new data from voice intake
    await waitForCondition(
      () =>
        Array.isArray(lastLivingProfileProps?.userProfile?.keySkills) &&
        lastLivingProfileProps.userProfile.keySkills.length >= 3
    );
    expect(lastLivingProfileProps.userProfile.keySkills).toContain("React");
    expect(lastLivingProfileProps.userProfile.keySkills).toContain("Node.js");
  });

  it("Maybe Later → Chat tab → routing effect does not override to voice after profile refresh", async () => {
    // Profile has no voice_intake_resume — routing effect returns null, so it must not override
    mockDashboardRequests([
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Developing",
        voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
      }),
    ]);
    renderResult = renderDashboard();

    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');
    act(() => { renderResult.container.querySelector('[data-testid="chat-tab"]').click(); });

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    // Simulate a profile refresh
    const refreshBtn = renderResult.container.querySelector('[data-testid="refresh-profile-btn"]');
    act(() => { refreshBtn.click(); });
    await flush();

    // Must still show ChatHub — not voice-intake, not jobs-deck
    expect(renderResult.container.querySelector('[data-testid="chat-hub"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
  });

  async function flush() {
    await act(async () => { await Promise.resolve(); });
  }
});
