import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import App from "./App";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

const MATRIX = { movie_id: 1, title: "The Matrix", year: 1999, directors: ["The Wachowskis"] };

function pickOf(movieId: number, title: string) {
  return {
    film: {
      movie_id: movieId,
      title,
      year: 1982,
      genres: ["Sci-Fi"],
      directors: ["Ridley Scott"],
      runtime: 117,
      certification: "R",
      wikipedia_url: "https://en.wikipedia.org/wiki/Blade_Runner",
      where_to_watch_url: "https://www.justwatch.com/ca/search?q=Blade%20Runner",
    },
    reason: { similar_to: "The Matrix", themes: ["dystopia", "cyberpunk"] },
  };
}

function picksOf(...films: [number, string][]) {
  return { picks: films.map(([movieId, title]) => pickOf(movieId, title)) };
}

const THREE = picksOf([2, "Blade Runner"], [3, "Gattaca"], [4, "Alien"]);

type Reply = { status: number; body: unknown };
type PickBody = { liked: number[]; rejected: number[]; count: number; showing: number[] };

function stubApi(pickReplies: Reply[], chat = false) {
  const pickBodies: PickBody[] = [];
  const replies = [...pickReplies];
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string, init?: RequestInit) => {
      const reply = url.startsWith("/api/films/search")
        ? { status: 200, body: [MATRIX] }
        : url === "/api/config"
          ? { status: 200, body: { conversation: chat } }
          : replies.shift()!;
      if (init?.method === "POST") pickBodies.push(JSON.parse(init.body as string));
      return Promise.resolve({
        ok: reply.status < 400,
        status: reply.status,
        json: () => Promise.resolve(reply.body),
      });
    }),
  );
  return pickBodies;
}

async function addMatrixAsFavourite() {
  fireEvent.change(screen.getByLabelText("Films you love"), { target: { value: "matr" } });
  fireEvent.click(await screen.findByRole("button", { name: /The Matrix \(1999\)/ }));
}

function pickButton() {
  return screen.getByRole("button", { name: "Pick three films" });
}

function titles() {
  return screen.getAllByRole("heading", { level: 2 }).map((heading) => heading.textContent);
}

it("can't pick until there is a favourite", () => {
  stubApi([]);

  render(<App />);

  expect(pickButton()).toHaveProperty("disabled", true);
});

it("adds a searched film and shows three picks, each with its reason", async () => {
  const bodies = stubApi([{ status: 200, body: THREE }]);
  render(<App />);

  await addMatrixAsFavourite();
  fireEvent.click(pickButton());

  expect(await screen.findByRole("heading", { name: "Blade Runner (1982)" })).toBeTruthy();
  expect(titles()).toEqual(["Blade Runner (1982)", "Gattaca (1982)", "Alien (1982)"]);
  expect(screen.getAllByText(/Shared themes: dystopia, cyberpunk\./)).toHaveLength(3);
  expect(screen.getByText("Tonight · three to choose from")).toBeTruthy();
  expect(bodies).toEqual([{ liked: [1], rejected: [], count: 3, showing: [] }]);
});

it("replaces just the pick that's turned down and remembers it", async () => {
  const bodies = stubApi([
    { status: 200, body: THREE },
    { status: 200, body: picksOf([5, "Brazil"]) },
  ]);
  render(<App />);
  await addMatrixAsFavourite();
  fireEvent.click(pickButton());
  await screen.findByRole("heading", { name: "Blade Runner (1982)" });

  fireEvent.click(screen.getByRole("button", { name: "Not this one: Blade Runner" }));

  expect(await screen.findByRole("heading", { name: "Brazil (1982)" })).toBeTruthy();
  expect(titles()).toEqual(["Brazil (1982)", "Gattaca (1982)", "Alien (1982)"]);
  expect(bodies[1]).toEqual({ liked: [1], rejected: [2], count: 1, showing: [3, 4] });
});

it("drops a turned-down pick when there's nothing left to replace it", async () => {
  stubApi([{ status: 200, body: THREE }, { status: 404, body: { detail: "No film" } }]);
  render(<App />);
  await addMatrixAsFavourite();
  fireEvent.click(pickButton());
  await screen.findByRole("heading", { name: "Blade Runner (1982)" });

  fireEvent.click(screen.getByRole("button", { name: "Not this one: Gattaca" }));

  await waitFor(() => expect(titles()).toEqual(["Blade Runner (1982)", "Alien (1982)"]));
  expect(screen.getByText("Tonight · two to choose from")).toBeTruthy();
});

it("swaps all three for three others", async () => {
  const bodies = stubApi([
    { status: 200, body: THREE },
    { status: 200, body: picksOf([5, "Brazil"], [6, "Moon"], [7, "Solaris"]) },
  ]);
  render(<App />);
  await addMatrixAsFavourite();
  fireEvent.click(pickButton());
  await screen.findByRole("heading", { name: "Blade Runner (1982)" });

  fireEvent.click(screen.getByRole("button", { name: "Three others" }));

  expect(await screen.findByRole("heading", { name: "Moon (1982)" })).toBeTruthy();
  expect(titles()).toEqual(["Brazil (1982)", "Moon (1982)", "Solaris (1982)"]);
  expect(bodies[1]).toEqual({ liked: [1], rejected: [2, 3, 4], count: 3, showing: [] });
});

it("says when there are no films left", async () => {
  stubApi([{ status: 404, body: { detail: "No film is available to pick." } }]);
  render(<App />);
  await addMatrixAsFavourite();

  fireEvent.click(pickButton());

  expect(await screen.findByText("There are no more films to suggest.")).toBeTruthy();
});

it("shows an error when picking fails", async () => {
  stubApi([{ status: 503, body: { detail: "The database isn't available." } }]);
  render(<App />);
  await addMatrixAsFavourite();

  fireEvent.click(pickButton());

  expect(await screen.findByRole("alert")).toBeTruthy();
});

it("asks people to slow down when the server limits requests", async () => {
  stubApi([{ status: 429, body: { detail: "You're going a bit fast." } }]);
  render(<App />);
  await addMatrixAsFavourite();

  fireEvent.click(pickButton());

  expect((await screen.findByRole("alert")).textContent).toBe(
    "You're going a bit fast. Try again in a moment.",
  );
});

it("removes a favourite", async () => {
  stubApi([]);
  render(<App />);
  await addMatrixAsFavourite();

  fireEvent.click(screen.getByRole("button", { name: "Remove The Matrix" }));

  await waitFor(() => expect(pickButton()).toHaveProperty("disabled", true));
});

it("explains when chat with Claude isn't set up", async () => {
  stubApi([]);

  render(<App />);

  expect(await screen.findByRole("note")).toHaveProperty(
    "textContent",
    expect.stringContaining("isn't set up on this server"),
  );
  expect(screen.queryByRole("region", { name: "Chat with Claude" })).toBeNull();
});

it("shows the chat when the server has Claude set up", async () => {
  stubApi([], true);

  render(<App />);

  expect(await screen.findByRole("region", { name: "Chat with Claude" })).toBeTruthy();
});

it("remembers favourites after a reload", async () => {
  stubApi([]);
  render(<App />);
  await addMatrixAsFavourite();
  cleanup();

  render(<App />);

  expect(screen.getByRole("button", { name: "Remove The Matrix" })).toBeTruthy();
});
