import React from "react";
import ReactDOM from "react-dom/client";
import { act } from "react";
import axios from "axios";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ProfileEditor } from "../LivingProfile";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock("react-router-dom", () => ({
  useNavigate: () => jest.fn(),
  useSearchParams: () => [new URLSearchParams(), jest.fn()],
}), { virtual: true });
jest.mock("../../components/onboarding/VoiceIntake", () => () => null);

const profile = {
  name: "Taylor", keySkills: [], certifications: ["AWS Certified Developer"],
  experience: [{ title: "Engineer", company: "Acme", dates: "2022 - Present", description: "Builds products" }],
  education: [{ degree: "B.Sc", institution: "State University", dates: "2018 - 2022" }],
};

function renderEditor() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = ReactDOM.createRoot(container);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  act(() => root.render(
    <QueryClientProvider client={queryClient}>
      <ProfileEditor profile={profile} candidateId="candidate-1" onClose={jest.fn()} onSaved={jest.fn()} />
    </QueryClientProvider>
  ));
  return { container, unmount: () => act(() => root.unmount()) };
}

describe("ProfileEditor certifications", () => {
  afterEach(() => jest.resetAllMocks());

  it("adds and removes individual certificate entries while preserving existing data", () => {
    const view = renderEditor();
    expect(view.container.querySelectorAll('input[aria-label="Certificate name"]')).toHaveLength(1);
    expect(view.container.querySelector('input[aria-label="Certificate name"]').value).toBe("AWS Certified Developer");

    const addCertificate = Array.from(view.container.querySelectorAll("button")).find((button) => button.textContent.includes("Add certificate"));
    act(() => addCertificate.click());
    act(() => addCertificate.click());
    expect(view.container.querySelectorAll('input[aria-label="Certificate name"]')).toHaveLength(3);

    act(() => view.container.querySelectorAll('button[aria-label="Remove certificate"]')[1].click());
    expect(view.container.querySelectorAll('input[aria-label="Certificate name"]')).toHaveLength(2);
    expect(view.container.querySelector('input[aria-label="Certificate name"]').value).toBe("AWS Certified Developer");
    view.unmount();
  });

  it("saves certificate names in the existing list-based API format", async () => {
    axios.put.mockResolvedValue({ data: {} });
    const view = renderEditor();
    const input = view.container.querySelector('input[aria-label="Certificate name"]');
    const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    act(() => {
      valueSetter.call(input, "PMP");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const saveButton = Array.from(view.container.querySelectorAll("button")).find((button) => button.textContent.includes("Save profile"));
    await act(async () => saveButton.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(axios.put).toHaveBeenCalledWith(expect.stringContaining("/candidate/candidate-1/profile"), expect.objectContaining({ certifications: ["PMP"] }));
    view.unmount();
  });

  it("rejects invalid experience years without calling the API", async () => {
    axios.put.mockResolvedValue({ data: {} });
    const view = renderEditor();
    const input = view.container.querySelector('input[name="experience_years"]');
    const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    act(() => {
      valueSetter.call(input, "-2");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const saveButton = Array.from(view.container.querySelectorAll("button")).find((button) => button.textContent.includes("Save profile"));
    await act(async () => saveButton.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(axios.put).not.toHaveBeenCalled();
    expect(view.container.textContent).toContain("Please check the highlighted profile fields");
    view.unmount();
  });
});
