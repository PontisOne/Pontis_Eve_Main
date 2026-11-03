import React from "react";
import { act } from "react";
import ReactDOM from "react-dom/client";
import axios from "axios";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import Dashboard from "../Dashboard";
import { saveOnboardingState } from "../../lib/onboardingStorage";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let lastLivingProfileProps = null;

vi.mock("react-router-dom", () => ({
  useNavigate: () => jest.fn(),
  useSearchParams: () => [new URLSearchParams(), jest.fn()],
}), { virtual: true });
vi.mock("../../components/Sidebar", () => ({ default: () => <div data-testid="sidebar" /> }));
vi.mock("../../components/ChatHub", () => ({ default: () => <div data-testid="chat-hub" /> }));
vi.mock("../../components/LivingProfile", () => ({ default: (props) => {
  lastLivingProfileProps = props;
  return <div data-testid="living-profile" />;
}}));
vi.mock("../../components/SwipeJobCard", () => ({ default: () => <div data-testid="jobs-deck" /> }));
vi.mock("../../components/onboarding/VoiceIntake", () => ({ default: () => <div data-testid="voice-intake" /> }));
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
    keySkills: ["React"],
    experience_years: 3,
    availability: "",
    preferred_roles: [],
    certifications: [],
    additional_information: "",
    profile_strength_percent: 80,
    profile_strength_label: "Strong",
    voice_intake_resume: { status: "completed", has_open_question: false, current_question: "", next_question: "" },
    ...overrides,
  };
}

function mockRequests(profileResponses) {
  let callCount = 0;
  axios.get.mockImplementation((url) => {
    if (url.includes("/profile")) {
      const idx = Math.min(callCount, profileResponses.length - 1);
      callCount += 1;
      return Promise.resolve({ data: profileResponses[idx] });
    }
    if (url.includes("/chat")) return Promise.resolve({ data: { messages: [] } });
    if (url.includes("/opportunities") || url.includes("/notifications")) return Promise.resolve({ data: [] });
    if (url.includes("/jobs")) return Promise.resolve({ data: [] });
    if (url.includes("/documents")) return Promise.resolve({ data: { resume: null, certificates: [] } });
    return Promise.resolve({ data: {} });
  });
}

function renderDashboard() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = ReactDOM.createRoot(container);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  act(() => {
    root.render(
      <QueryClientProvider client={queryClient}>
        <Dashboard />
      </QueryClientProvider>
    );
  });
  return {
    container,
    unmount() { act(() => { root.unmount(); }); container.remove(); },
  };
}

async function flush(n = 5) {
  for (let i = 0; i < n; i++) {
    await act(async () => { await Promise.resolve(); });
  }
}

async function waitForCondition(pred, attempts = 20) {
  for (let i = 0; i < attempts; i++) {
    await flush(1);
    if (pred()) return;
  }
  throw new Error("Timed out waiting for condition");
}

describe("Dashboard toggle persistence", () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
    lastLivingProfileProps = null;
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true });
  });

  afterEach(() => {
    lastLivingProfileProps = null;
  });

  it("initialises isOpenToMatches from stored value (true)", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true });
    mockRequests([makeProfile()]);
    const view = renderDashboard();
    await waitForCondition(() => lastLivingProfileProps !== null);
    expect(lastLivingProfileProps.userProfile.isOpenToMatches).toBe(true);
    view.unmount();
  });

  it("initialises isOpenToMatches from stored value (false)", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: false });
    mockRequests([makeProfile()]);
    const view = renderDashboard();
    await waitForCondition(() => lastLivingProfileProps !== null);
    expect(lastLivingProfileProps.userProfile.isOpenToMatches).toBe(false);
    view.unmount();
  });

  it("manual toggle ON→OFF is preserved after refreshProfile()", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true });
    mockRequests([makeProfile(), makeProfile()]);
    const view = renderDashboard();
    await waitForCondition(() => lastLivingProfileProps !== null);

    // Simulate candidate toggling OFF
    act(() => { lastLivingProfileProps.onToggleOpenToMatches(); });
    await flush();
    expect(lastLivingProfileProps.userProfile.isOpenToMatches).toBe(false);

    // Simulate refreshProfile (photo change triggers it)
    act(() => { lastLivingProfileProps.onPhotoChange("new-url"); });
    await flush(10);

    expect(lastLivingProfileProps.userProfile.isOpenToMatches).toBe(false);
    view.unmount();
  });

  it("manual toggle OFF→ON is preserved after refreshProfile()", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: false });
    mockRequests([makeProfile(), makeProfile()]);
    const view = renderDashboard();
    await waitForCondition(() => lastLivingProfileProps !== null);

    // Simulate candidate toggling ON
    act(() => { lastLivingProfileProps.onToggleOpenToMatches(); });
    await flush();
    expect(lastLivingProfileProps.userProfile.isOpenToMatches).toBe(true);

    // Simulate refreshProfile
    act(() => { lastLivingProfileProps.onPhotoChange("new-url"); });
    await flush(10);

    expect(lastLivingProfileProps.userProfile.isOpenToMatches).toBe(true);
    view.unmount();
  });

  it("toggle state is preserved across multiple re-renders (prop changes)", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true });
    mockRequests([makeProfile(), makeProfile(), makeProfile()]);
    const view = renderDashboard();
    await waitForCondition(() => lastLivingProfileProps !== null);

    act(() => { lastLivingProfileProps.onToggleOpenToMatches(); });
    await flush();
    expect(lastLivingProfileProps.userProfile.isOpenToMatches).toBe(false);

    // Two more refreshes
    act(() => { lastLivingProfileProps.onPhotoChange("url-1"); });
    await flush(10);
    act(() => { lastLivingProfileProps.onPhotoChange("url-2"); });
    await flush(10);

    expect(lastLivingProfileProps.userProfile.isOpenToMatches).toBe(false);
    view.unmount();
  });

  it("profile API update does not reset a manually set OFF toggle", async () => {
    saveOnboardingState({ candidateId: "cand-123", isOpenToMatches: true });
    // Second profile response has different data (simulates a real API update)
    mockRequests([
      makeProfile({ headline: "Engineer" }),
      makeProfile({ headline: "Senior Engineer", keySkills: ["React", "Node"] }),
    ]);
    const view = renderDashboard();
    await waitForCondition(() => lastLivingProfileProps !== null);

    act(() => { lastLivingProfileProps.onToggleOpenToMatches(); });
    await flush();
    expect(lastLivingProfileProps.userProfile.isOpenToMatches).toBe(false);

    act(() => { lastLivingProfileProps.onPhotoChange("photo.jpg"); });
    await flush(10);

    // Profile data updated but toggle preserved
    expect(lastLivingProfileProps.userProfile.isOpenToMatches).toBe(false);
    view.unmount();
  });
});
