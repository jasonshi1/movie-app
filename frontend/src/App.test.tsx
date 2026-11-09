import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import App from "./App";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function stubHealth(body: unknown) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ json: () => Promise.resolve(body) }));
}

it("shows the pgvector version when the backend is healthy", async () => {
  stubHealth({ status: "ok", database: true, pgvector: "0.8.0" });

  render(<App />);

  expect(await screen.findByText("Connected to the database (pgvector 0.8.0).")).toBeTruthy();
});

it("says when the database isn't ready", async () => {
  stubHealth({ status: "unavailable", database: false, pgvector: null });

  render(<App />);

  expect(await screen.findByText("The backend is running, but the database isn't ready.")).toBeTruthy();
});

it("says when the backend can't be reached", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network error")));

  render(<App />);

  expect(await screen.findByText("The backend can't be reached.")).toBeTruthy();
});
