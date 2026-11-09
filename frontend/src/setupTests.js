const { webcrypto } = require("node:crypto");

// Vitest's jsdom environment does not consistently expose Web Crypto. Keep the
// browser-compatible implementation available to every test without changing
// application code or individual test files.
if (!globalThis.crypto?.subtle) {
  Object.defineProperty(globalThis, "crypto", {
    configurable: true,
    value: webcrypto,
  });
}

import { vi } from "vitest";

// React 19 requires this flag before any createRoot-rendered test work so
// asynchronous updates are recognized as part of the supported act() test
// environment, including tests that manage roots directly.
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

globalThis.jest = vi;

// CRA's Jest automock created mock functions for Axios methods. Vitest does
// not reproduce that shape for Axios's callable default export, so provide a
// single shared mock for the legacy Jest-style tests.
vi.mock("axios", () => {
  const axios = vi.fn();
  const methods = ["get", "post", "put", "patch", "delete", "head", "options", "request"];

  const createInterceptorManager = () => {
    const manager = {
      handlers: [],
      use(onFulfilled, onRejected) {
        manager.handlers.push({ fulfilled: onFulfilled, rejected: onRejected });
        return manager.handlers.length - 1;
      },
      eject(id) {
        if (manager.handlers[id]) manager.handlers[id] = null;
      },
      clear() {
        manager.handlers.length = 0;
      },
    };
    return manager;
  };

  for (const method of methods) {
    axios[method] = vi.fn();
  }

  axios.create = vi.fn(() => axios);
  axios.CancelToken = { source: vi.fn() };
  axios.isAxiosError = vi.fn(() => false);
  axios.interceptors = {
    request: createInterceptorManager(),
    response: createInterceptorManager(),
  };

  return { default: axios };
});

afterEach(() => {
  vi.clearAllMocks();
});

// Vitest's jsdom File implementation may lack Blob.arrayBuffer.
// FileReader preserves the file's actual bytes, so hashing remains real.
if (!globalThis.File.prototype.arrayBuffer) {
  Object.defineProperty(globalThis.File.prototype, "arrayBuffer", {
    configurable: true,
    value() {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.addEventListener("load", () => resolve(reader.result));
        reader.addEventListener("error", () => reject(reader.error));
        reader.readAsArrayBuffer(this);
      });
    },
  });
}

// daily-js probes canvas capabilities during module initialization. JSDOM does
// not implement a canvas context, but these tests do not exercise rendering.
if (globalThis.HTMLCanvasElement) {
  Object.defineProperty(globalThis.HTMLCanvasElement.prototype, "getContext", {
    configurable: true,
    value() {
      return {
        canvas: this,
        getImageData: () => ({ data: new Uint8ClampedArray(4) }),
        putImageData: () => {},
        createImageData: () => ({ data: new Uint8ClampedArray(4) }),
      };
    },
  });
}
