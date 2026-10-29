import React from "react";

export default class RouteErrorBoundary extends React.Component {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <main role="alert" className="flex min-h-screen items-center justify-center p-6">
        <div className="max-w-md text-center">
          <h1 className="text-xl font-semibold">This page could not be loaded</h1>
          <p className="mt-2">Please reload the page and try again.</p>
          <button type="button" className="mt-4 rounded-md px-4 py-2" onClick={this.handleReload}>
            Reload page
          </button>
        </div>
      </main>
    );
  }
}
