import React from "react";
import { act } from "react";
import axios from "axios";
import { render } from "@testing-library/react";

import Dashboard from "../Dashboard";
import { TestQueryProvider } from "../../testUtils";
import { saveOnboardingState } from "../../lib/onboardingStorage";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let mockVoiceIntakeOnComplete = null;
let lastLivingProfileProps = null;

vi.mock("react-router-dom", () => ({
  useNavigate: () => jest.fn(),
  useSearchParams: () => [new URLSearchParams(), jest.fn()],
}), { virtual: true });

vi.mock("../../components/Sidebar", () => ({ default: () => <div data-testid="sidebar" /> }));
vi.mock("../../components/SwipeJobCard", () => ({ default: () => <div data-testid="jobs-deck" /> }));
vi.mock("../../components/onboarding/VoiceIntake", () => ({ default: (props) => {
  mockVoiceIntakeOnComplete = props.onComplete;
  return <div data-testid="voice-intake" />;
}}));
vi.mock("../../components/ChatHub", () => ({ default: (props) => (
  <div data-testid="chat-hub">
    <div data-testid="chat-transcript">
      {(props.chats || []).map((chat) => (
        <div key={chat.id} data-sender={chat.sender}>{chat.content}</div>
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
    {(props.quickActions || []).map((question) => (
      <button key={question} data-testid="chat-suggestion" onClick={() => props.onSuggestionClick?.(question)}>
        {question}
      </button>
    ))}
    {props.onMicClick && (
      <button data-testid="chat-mic-btn" onClick={props.onMicClick}>
        mic
      </button>
    )}
  </div>
)}));
vi.mock("../../components/LivingProfile", () => ({ __esModule: true, default: (props) => {
  lastLivingProfileProps = props;
  return <div data-testid="living-profile" />;
}, JobsTab: () => <div data-testid="jobs-deck" /> }));
vi.mock("react-resizable-panels", () => ({
  PanelGroup: ({ children }) => <div>{children}</div>,
  Panel: ({ children }) => <div>{children}</div>,
  PanelResizeHandle: () => <div />,
}));
vi.mock("sonner", () => ({ Toaster: () => null, toast: { error: jest.fn(), success: jest.fn() } }));

function makeProfile(overrides = {}) {
  return {
    photo_url: null,
    name: "Test User",
    email: "test@example.com",
    phone: "",
    headline: "Engineer",
    location: "NYC",
    bio: "",
    experience: [],
    education: [],
    keySkills: ["Product", "Strategy"],
    experience_years: 2,
    availability: "",
    preferred_roles: [],
    certifications: [],
    additional_information: "",
    profile_strength_percent: 55,
    profile_strength_label: "Developing",
    voice_intake_resume: {
      status: "completed",
      has_open_question: false,
      current_question: "",
      next_question: "",
    },
    ...overrides,
  };
}

function mockRequests(profileResponses) {
  let profileCallCount = 0;
  axios.get.mockImplementation((url) => {
    if (url.includes("/profile")) {
      const index = Math.min(profileCallCount, profileResponses.length - 1);
      profileCallCount += 1;
      return Promise.resolve({ data: profileResponses[index] });
    }
    if (url.includes("/chat")) return Promise.resolve({ data: { messages: [] } });
    if (url.includes("/opportunities") || url.includes("/notifications")) return Promise.resolve({ data: [] });
    if (url.includes("/jobs")) return Promise.resolve({ data: [{ id: "job-1", company: "Acme" }] });
    if (url.includes("/documents")) return Promise.resolve({ data: { resume: null, certificates: [] } });
    return Promise.resolve({ data: {} });
  });
}

function renderDashboard() {
  return render(<TestQueryProvider><Dashboard /></TestQueryProvider>);
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

function setInputValue(input, value) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
  setter.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

describe("Dashboard chat flow regressions", () => {
  let renderResult;

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    mockVoiceIntakeOnComplete = null;
    lastLivingProfileProps = null;
    jest.useFakeTimers();
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true });
  });

  afterEach(() => {
    renderResult?.unmount?.();
    renderResult = null;
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it("Maybe Later -> Jobs for You -> Chat with Eve shows ChatHub and mic, not VoiceIntake", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
    mockRequests([
      makeProfile({
        profile_strength_percent: 55,
        profile_strength_label: "Developing",
        voice_intake_resume: {
          status: "in_progress",
          has_open_question: true,
          current_question: "What is your current role?",
          next_question: "",
        },
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
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();

    act(() => {
      Array.from(renderResult.container.querySelectorAll("button")).find((b) =>
        b.textContent?.includes("Chat with Eve")
      )?.click();
    });

    expect(await waitForSelector(renderResult.container, '[data-testid="chat-hub"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="chat-mic-btn"]')).toBeTruthy();
    expect(renderResult.container.querySelector('[data-testid="voice-intake"]')).toBeNull();
  });

  it("Chat and Voice intake updates both land in the right panel without duplicating existing skills", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
    mockRequests([
      makeProfile({
        profile_strength_percent: 90,
        keySkills: ["Product", "Strategy"],
        additional_information: "Enjoys building products",
      }),
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Strong",
        keySkills: ["Product", "Strategy", "React"],
        additional_information: "Enjoys building products",
      }),
      makeProfile({
        profile_strength_percent: 90,
        profile_strength_label: "Strong",
        keySkills: ["Product", "Strategy", "React", "Leadership"],
        additional_information: "Enjoys building products",
        voice_intake_resume: {
          status: "completed",
          has_open_question: false,
          current_question: "",
          next_question: "",
        },
      }),
    ]);

    axios.post.mockImplementation((url) => {
      if (url.includes("/chat")) {
        return Promise.resolve({
          data: {
            reply: "Absolutely, I added React.",
            profile_updates: {
              keySkills: ["Product", "React"],
              additional_information: "Enjoys building products",
            },
          },
        });
      }

      return Promise.resolve({ data: {} });
    });

    renderResult = renderDashboard();
    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');

    act(() => {
      Array.from(renderResult.container.querySelectorAll("button")).find((b) =>
        b.textContent?.includes("Chat with Eve")
      )?.click();
    });

    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    expect(renderResult.container.querySelector('[data-testid="chat-mic-btn"]')).toBeTruthy();

    act(() => {
      setInputValue(renderResult.container.querySelector('[data-testid="chat-text-input"]'), "Please add React to my profile");
    });
    act(() => {
      renderResult.container.querySelector('[data-testid="chat-send-btn"]').click();
    });

    await waitForCondition(
      () =>
        Array.isArray(lastLivingProfileProps?.userProfile?.keySkills) &&
        lastLivingProfileProps.userProfile.keySkills.includes("React")
    );
    expect(lastLivingProfileProps.userProfile.keySkills).toEqual(["Product", "Strategy", "React"]);
    expect(lastLivingProfileProps.userProfile.additional_information).toBe("Enjoys building products");

    act(() => {
      renderResult.container.querySelector('[data-testid="chat-mic-btn"]').click();
    });
    await waitForSelector(renderResult.container, '[data-testid="voice-intake"]');

    await act(async () => {
      mockVoiceIntakeOnComplete?.({
        status: "completed",
        profile_updates: {
          keySkills: ["Product", "Strategy", "React", "Leadership"],
          additional_information: "Enjoys building products",
        },
      });
      await Promise.resolve();
    });

    await waitForCondition(
      () =>
        Array.isArray(lastLivingProfileProps?.userProfile?.keySkills) &&
        lastLivingProfileProps.userProfile.keySkills.includes("Leadership")
    );
    expect(lastLivingProfileProps.userProfile.keySkills).toEqual([
      "Product",
      "Strategy",
      "React",
      "Leadership",
    ]);
  });

  it("persists a certification deletion and refreshes the right profile panel from backend data", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
    mockRequests([
      makeProfile({ profile_strength_percent: 90, certifications: ["AWS Certificate", "PMP"] }),
      makeProfile({ profile_strength_percent: 90, certifications: ["PMP"] }),
    ]);
    axios.post.mockResolvedValue({
      data: {
        reply: "AWS Certificate has been removed.",
        profile_updates: { profile_deletions: { certifications: ["AWS Certificate"] } },
      },
    });

    renderResult = renderDashboard();
    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');
    act(() => {
      Array.from(renderResult.container.querySelectorAll("button")).find((b) =>
        b.textContent?.includes("Chat with Eve")
      )?.click();
    });
    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    act(() => {
      setInputValue(renderResult.container.querySelector('[data-testid="chat-text-input"]'), "Remove AWS Certificate from Certifications");
      renderResult.container.querySelector('[data-testid="chat-send-btn"]').click();
    });

    await waitForCondition(() => lastLivingProfileProps?.userProfile?.certifications?.length === 1);
    expect(lastLivingProfileProps.userProfile.certifications).toEqual(["PMP"]);
    expect(axios.get.mock.calls.some(([url]) => url.includes("/candidate/cand-123/profile"))).toBe(true);
  });

  it("renders a fresh Eve reply for a new salary message instead of replaying the prior reply", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true, activeTab: "jobs" });
    mockRequests([
      makeProfile({ profile_strength_percent: 90 }),
      makeProfile({ profile_strength_percent: 90, salary_expectation: "7-10 LPA" }),
    ]);
    axios.post
      .mockResolvedValueOnce({ data: { reply: "AWS Certificate has been removed.", profile_updates: null } })
      .mockResolvedValueOnce({ data: { reply: "I've noted your expected salary of 7-10 LPA.", profile_updates: { salary_expectation: "7-10 LPA" } } });

    renderResult = renderDashboard();
    await waitForSelector(renderResult.container, '[data-testid="jobs-deck"]');
    act(() => {
      Array.from(renderResult.container.querySelectorAll("button")).find((b) => b.textContent?.includes("Chat with Eve"))?.click();
    });
    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');
    const input = renderResult.container.querySelector('[data-testid="chat-text-input"]');
    const send = renderResult.container.querySelector('[data-testid="chat-send-btn"]');

    act(() => { setInputValue(input, "Remove AWS Certificate from Certifications"); send.click(); });
    await waitForCondition(() => renderResult.container.textContent.includes("AWS Certificate has been removed."));
    act(() => { setInputValue(input, "I am expecting 7-10 LPA"); send.click(); });
    await waitForCondition(() => renderResult.container.textContent.includes("I've noted your expected salary of 7-10 LPA."));

    const secondRequest = axios.post.mock.calls[1][1];
    expect(secondRequest.messages.at(-1)).toEqual({ role: "user", content: "I am expecting 7-10 LPA" });
    expect(renderResult.container.querySelector('[data-testid="chat-transcript"]').textContent).toContain("7-10 LPA");
  });

  it("prefills an authoritative missing question without sending it to Eve", async () => {
    mockRequests([makeProfile({
      missing_questions: [{
        topic_id: "expected_salary",
        question: "What salary range are you targeting?",
        status: "NOT_ASKED",
      }],
    })]);
    renderResult = renderDashboard();
    await waitForSelector(renderResult.container, '[data-testid="chat-hub"]');

    act(() => {
      renderResult.container.querySelector('[data-testid="chat-suggestion"]').click();
    });

    expect(renderResult.container.querySelector('[data-testid="chat-text-input"]').value)
      .toBe("What salary range are you targeting: ");
    expect(axios.post).not.toHaveBeenCalled();
  });
});
