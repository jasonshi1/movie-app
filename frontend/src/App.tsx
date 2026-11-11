import { useEffect, useState } from "react";

import Chat from "./Chat";
import FilmTitle from "./FilmTitle";
import { details, withYear } from "./format";
import { loadFavourites, loadPreferences, saveFavourites, savePreferences } from "./storage";
import type { FilmMatch, PickedFilm, Preferences } from "./types";

type Pick = {
  film: PickedFilm;
  reason: { similar_to: string; themes: string[] };
};

type PickStatus = "idle" | "picking" | "none-left" | "too-fast" | "error";

const SEARCH_DELAY_MS = 200;

const today = new Intl.DateTimeFormat(undefined, {
  weekday: "long",
  day: "numeric",
  month: "long",
}).format(new Date());

export default function App() {
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<FilmMatch[]>([]);
  const [favourites, setFavourites] = useState<FilmMatch[]>(loadFavourites);
  const [preferences, setPreferences] = useState<Preferences>(loadPreferences);
  const [rejected, setRejected] = useState<number[]>([]);
  const [pick, setPick] = useState<Pick | null>(null);
  const [status, setStatus] = useState<PickStatus>("idle");
  const [chatAvailable, setChatAvailable] = useState<boolean | null>(null);

  useEffect(() => saveFavourites(favourites), [favourites]);
  useEffect(() => savePreferences(preferences), [preferences]);

  useEffect(() => {
    fetch("/api/config")
      .then((response) => (response.ok ? (response.json() as Promise<{ conversation: boolean }>) : null))
      .then((config) => setChatAvailable(config?.conversation ?? false))
      .catch(() => setChatAvailable(false));
  }, []);

  useEffect(() => {
    const term = query.trim();
    if (!term) {
      setMatches([]);
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetch(`/api/films/search?q=${encodeURIComponent(term)}`, { signal: controller.signal })
        .then((response) => (response.ok ? (response.json() as Promise<FilmMatch[]>) : []))
        .then(setMatches)
        .catch((error: unknown) => {
          if (!(error instanceof DOMException && error.name === "AbortError")) setMatches([]);
        });
    }, SEARCH_DELAY_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  function addFavourite(film: FilmMatch) {
    setFavourites((current) => [...current, film]);
    setQuery("");
    setMatches([]);
    setPick(null);
    setStatus("idle");
  }

  function removeFavourite(movieId: number) {
    setFavourites((current) => current.filter((film) => film.movie_id !== movieId));
    setPick(null);
    setStatus("idle");
  }

  async function requestPick(rejectedIds: number[]) {
    setStatus("picking");
    try {
      const response = await fetch("/api/pick", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          liked: favourites.map((film) => film.movie_id),
          rejected: rejectedIds,
        }),
      });
      if (response.status === 404) {
        setPick(null);
        setStatus("none-left");
        return;
      }
      if (response.status === 429) {
        setStatus("too-fast");
        return;
      }
      if (!response.ok) throw new Error(`Pick failed with ${response.status}`);
      setPick((await response.json()) as Pick);
      setStatus("idle");
    } catch {
      setPick(null);
      setStatus("error");
    }
  }

  function rejectPick() {
    if (!pick) return;
    const next = [...rejected, pick.film.movie_id];
    setRejected(next);
    void requestPick(next);
  }

  const favouriteIds = new Set(favourites.map((film) => film.movie_id));
  const suggestions = matches.filter((film) => !favouriteIds.has(film.movie_id));

  return (
    <>
      <header className="masthead">
        <p className="dateline">
          <span>{today}</span>
          <span>Films up to 2014</span>
        </p>
        <h1>
          Tonight’s <em>Pick</em>
        </h1>
        <p className="intro">Name a few films you love, and get one film to watch tonight.</p>
      </header>

      <main>
        <section className="step">
          <label className="step-title" htmlFor="film-search">
            Films you love
          </label>
          <input
            id="film-search"
            type="search"
            autoComplete="off"
            placeholder="Start typing a title"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />

          {suggestions.length > 0 && (
            <ul className="suggestions" aria-label="Matching films">
              {suggestions.map((film) => (
                <li key={film.movie_id}>
                  <button type="button" onClick={() => addFavourite(film)}>
                    <span>{withYear(film)}</span>
                    {film.directors[0] && <small>{film.directors[0]}</small>}
                  </button>
                </li>
              ))}
            </ul>
          )}

          {favourites.length > 0 && (
            <ul className="tags" aria-label="Your favourites">
              {favourites.map((film) => (
                <li key={film.movie_id}>
                  {withYear(film)}
                  <button
                    type="button"
                    aria-label={`Remove ${film.title}`}
                    onClick={() => removeFavourite(film.movie_id)}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}

          <button
            className="primary"
            type="button"
            disabled={favourites.length === 0 || status === "picking"}
            onClick={() => void requestPick(rejected)}
          >
            {status === "picking" ? "Choosing…" : "Pick a film"}
          </button>

          {status === "none-left" && (
            <p className="notice" role="status">
              There are no more films to suggest.
            </p>
          )}
          {status === "error" && (
            <p className="notice" role="alert">
              Something went wrong. Please try again.
            </p>
          )}
          {status === "too-fast" && (
            <p className="notice" role="alert">
              You're going a bit fast. Try again in a moment.
            </p>
          )}
        </section>

        {pick && (
          <article className="pick" aria-label="Your pick">
            <p className="kicker">Tonight</p>
            <h2>
              <FilmTitle film={pick.film} />
            </h2>
            <p className="details">{details(pick.film)}</p>
            <p className="reason">
              Closest to <strong>{pick.reason.similar_to}</strong> among your favourites.
              {pick.reason.themes.length > 0 && ` Shared themes: ${pick.reason.themes.join(", ")}.`}
            </p>
            <div className="actions">
              <button type="button" disabled={status === "picking"} onClick={rejectPick}>
                Not that one
              </button>
              <a href={pick.film.where_to_watch_url} target="_blank" rel="noreferrer">
                Where to watch
              </a>
              {pick.film.wikipedia_url && (
                <a href={pick.film.wikipedia_url} target="_blank" rel="noreferrer">
                  Wikipedia
                </a>
              )}
            </div>
          </article>
        )}

        {chatAvailable && (
          <Chat favourites={favourites} preferences={preferences} onPreferencesChange={setPreferences} />
        )}
        {chatAvailable === false && (
          <p className="notice quiet" role="note">
            Chat with Claude isn't set up on this server yet. You can still pick from your favourites
            above.
          </p>
        )}
      </main>

      <footer>
        Film data from <a href="https://grouplens.org/datasets/movielens/">MovieLens</a> and the{" "}
        <a href="https://grouplens.org/datasets/movielens/tag-genome-2021/">Tag Genome</a> by
        GroupLens (CC BY-NC 3.0), <a href="https://www.wikidata.org/">Wikidata</a> and{" "}
        <a href="https://en.wikipedia.org/">Wikipedia</a> (CC BY-SA). Free and non-commercial. The
        catalog covers films up to 2014.
      </footer>
    </>
  );
}
