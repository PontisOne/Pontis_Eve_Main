import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from "sonner";
import RouteErrorBoundary from "./components/RouteErrorBoundary";

const GoogleAuth = React.lazy(() => import("./pages/GoogleAuth"));
const Onboarding = React.lazy(() => import("./pages/Onboarding"));
const Dashboard = React.lazy(() => import("./pages/Dashboard"));
const ResumeEditor = React.lazy(() => import("./pages/ResumeEditor"));

function RouteLoadingFallback() {
  return (
    <main
      aria-busy="true"
      aria-live="polite"
      className="flex min-h-screen items-center justify-center"
    >
      <p>Loading page…</p>
    </main>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Toaster position="top-right" richColors closeButton />
      <RouteErrorBoundary>
        <React.Suspense fallback={<RouteLoadingFallback />}>
          <Routes>
            <Route path="/" element={<GoogleAuth />} />
            <Route path="/onboarding" element={<Onboarding />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/resume-editor" element={<ResumeEditor />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </React.Suspense>
      </RouteErrorBoundary>
    </BrowserRouter>
  );
}
