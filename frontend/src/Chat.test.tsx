import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import Chat from "./Chat";
import type { ChatReply, Preferences } from "./types";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const FAVOURITES = [{ movie_id: 1, title: "The Matrix", year: 1999, directors: [] }];
const NONE: Preferences = { avoid_genres: [], max_runtime: null };

function reply(overrides: Partial<ChatReply> = {}): ChatReply {
  return {
    conversation_id: "c-1",
    reply: "Dark, clever and under two hours.",
    pick: {
      film: {
        movie_id: 2,
        title: "Blade Runner",
        year: 1982,
        genres: ["Sci-Fi"],
        directors: ["Ridley Scott"],
        runtime: 117,
        certification: "R",
        wikipedia_url: null,
        where_to_watch_url: "https://www.justwatch.com/ca/search?q=Blade%20Runner",
        poster_url: "https://upload.wikimedia.org/wikipedia/en/blade-runner.png",
        imdb_rating: 8.1,
        imdb_url: "https://www.imdb.com/title/tt0083658/",
        rotten_tomatoes: 89,
        rotten_tomatoes_url: "https://www.rottentomatoes.com/m/blade_runner",
      },
      reason: "Like The Matrix, it questions what's real.",
    },
    proposal: null,
    ...overrides,
  };
}

function stubChat(...responses: { status: number; body: unknown }[]) {
  const bodies: Record<string, unknown>[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn((_url: string, init: RequestInit) => {
      bodies.push(JSON.parse(init.body as string));
      const next = responses.shift()!;
      return Promise.resolve({
        ok: next.status < 400,
        status: next.status,
        json: () => Promise.resolve(next.body),
      });
    }),
  );
  return bodies;
}

function say(text: string) {
  fireEvent.change(screen.getByLabelText("Your message"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
}

it("sends the message with favourites and preferences, then shows Claude's pick", async () => {
  const bodies = stubChat({ status: 200, body: reply() });
  const preferences = { avoid_genres: ["Horror"], max_runtime: 120 };
  render(<Chat favourites={FAVOURITES} preferences={preferences} onPreferencesChange={() => {}} />);

  say("Something mind-bending");

  expect(await screen.findByRole("heading", { name: "Blade Runner (1982)" })).toBeTruthy();
  expect(screen.getByText("Like The Matrix, it questions what's real.")).toBeTruthy();
  const pick = screen.getByRole("article", { name: "Claude's pick" });
  expect(pick.querySelector("img")?.getAttribute("src")).toBe(
    "https://upload.wikimedia.org/wikipedia/en/blade-runner.png",
  );
  expect(screen.getByRole("link", { name: "IMDb rating 8.1 out of 10" })).toBeTruthy();
  expect(screen.getByRole("link", { name: "Rotten Tomatoes 89%" })).toBeTruthy();
  expect(bodies[0]).toEqual({
    conversation_id: null,
    message: "Something mind-bending",
    favourites: [1],
    preferences,
    reject_movie_id: null,
  });
});

it("turns down the latest pick in the same conversation, with the typed reason", async () => {
  const bodies = stubChat({ status: 200, body: reply() }, { status: 200, body: reply({ pick: null }) });
  render(<Chat favourites={FAVOURITES} preferences={NONE} onPreferencesChange={() => {}} />);
  say("Something dark");
  await screen.findByRole("heading", { name: "Blade Runner (1982)" });

  fireEvent.change(screen.getByLabelText("Your message"), { target: { value: "Too long" } });
  fireEvent.click(screen.getByRole("button", { name: "Not that one" }));

  await screen.findByText("Too long");
  expect(bodies[1]).toMatchObject({ conversation_id: "c-1", message: "Too long", reject_movie_id: 2 });
});

it("saves a proposed preference only when accepted", async () => {
  stubChat({
    status: 200,
    body: reply({
      pick: null,
      proposal: { kind: "avoid_genre", genre: "Horror", minutes: null, reason: "Turned down 3" },
    }),
  });
  const onPreferencesChange = vi.fn();
  render(<Chat favourites={FAVOURITES} preferences={NONE} onPreferencesChange={onPreferencesChange} />);
  say("No more horror");

  fireEvent.click(await screen.findByRole("button", { name: "Save it" }));

  expect(onPreferencesChange).toHaveBeenCalledWith({ avoid_genres: ["Horror"], max_runtime: null });
  expect(screen.getByText("Saved: Stop suggesting Horror films.")).toBeTruthy();
});

it("shows the server's explanation when chat fails", async () => {
  stubChat({ status: 503, body: { detail: "Chat with Claude isn't set up on this server." } });
  render(<Chat favourites={FAVOURITES} preferences={NONE} onPreferencesChange={() => {}} />);

  say("Hello");

  expect((await screen.findByRole("alert")).textContent).toBe(
    "Chat with Claude isn't set up on this server.",
  );
});
