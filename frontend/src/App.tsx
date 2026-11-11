import { useEffect, useState } from "react";

import Chat from "./Chat";
import FilmTitle from "./FilmTitle";
import { details, withYear } from "./format";
import { loadFavourites, loadPreferences, saveFavourites, savePreferences } from "./storage";
import { useTheme } from "./theme";
import type { FilmMatch, PickedFilm, Preferences } from "./types";

type Pick = {
  film: PickedFilm;
  reason: { similar_to: string; themes: string[] };
  // Released in the catalog's last three years.
  recent: boolean;
};

type PickStatus = "idle" | "picking" | "none-left" | "too-fast" | "error";

const SEARCH_DELAY_MS = 200;
const PICK_COUNT = 3;
// The last pick is a recent release. The ranker learned from MovieLens ratings, which end in
// October 2023, so it would rarely choose a newer film otherwise.
const RECENT_PICKS = 1;
const COUNT_WORDS = ["none", "one", "two", "three", "four", "five"];

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
  const [picks, setPicks] = useState<Pick[]>([]);
  // The film whose replacement is being fetched.
  const [replacing, setReplacing] = useState<number | null>(null);
  const [status, setStatus] = useState<PickStatus>("idle");
  const [chatAvailable, setChatAvailable] = useState<boolean | null>(null);
  const [newestYear, setNewestYear] = useState<number | null>(null);
  const [theme, toggleTheme] = useTheme();

  useEffect(() => saveFavourites(favourites), [favourites]);
  useEffect(() => savePreferences(preferences), [preferences]);

  useEffect(() => {
    fetch("/api/config")
      .then((response) =>
        response.ok
          ? (response.json() as Promise<{ conversation: boolean; newest_year?: number | null }>)
          : null,
      )
      .then((config) => {
        setChatAvailable(config?.conversation ?? false);
        setNewestYear(config?.newest_year ?? null);
      })
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
    setPicks([]);
    setStatus("idle");
  }

  function removeFavourite(movieId: number) {
    setFavourites((current) => current.filter((film) => film.movie_id !== movieId));
    setPicks([]);
    setStatus("idle");
  }

  async function fetchPicks(
    rejectedIds: number[],
    count: number,
    recent: number,
    showing: number[] = [],
  ): Promise<Pick[] | "none-left" | "too-fast"> {
    const response = await fetch("/api/pick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        liked: favourites.map((film) => film.movie_id),
        rejected: rejectedIds,
        count,
        showing,
        recent,
      }),
    });
    if (response.status === 404) return "none-left";
    if (response.status === 429) return "too-fast";
    if (!response.ok) throw new Error(`Pick failed with ${response.status}`);
    return ((await response.json()) as { picks: Pick[] }).picks;
  }

  async function pickFilms(rejectedIds: number[]) {
    setStatus("picking");
    try {
      const result = await fetchPicks(rejectedIds, PICK_COUNT, RECENT_PICKS);
      if (result === "too-fast") {
        setStatus("too-fast");
        return;
      }
      setPicks(result === "none-left" ? [] : result);
      setStatus(result === "none-left" ? "none-left" : "idle");
    } catch {
      setPicks([]);
      setStatus("error");
    }
  }

  function showOthers() {
    const next = [...rejected, ...picks.map((pick) => pick.film.movie_id)];
    setRejected(next);
    void pickFilms(next);
  }

  // Swap one pick for the next best film, keeping the others where they are. A recent film is
  // swapped for another recent one.
  async function replacePick(replaced: Pick) {
    const movieId = replaced.film.movie_id;
    const next = [...rejected, movieId];
    const others = picks.map((pick) => pick.film.movie_id).filter((id) => id !== movieId);
    setRejected(next);
    setReplacing(movieId);
    setStatus("idle");
    try {
      const result = await fetchPicks(next, 1, replaced.recent ? 1 : 0, others);
      if (result === "too-fast") {
        setStatus("too-fast");
        return;
      }
      setPicks((current) =>
        result === "none-left"
          ? current.filter((pick) => pick.film.movie_id !== movieId)
          : current.map((pick) => (pick.film.movie_id === movieId ? result[0] : pick)),
      );
      if (result === "none-left" && others.length === 0) setStatus("none-left");
    } catch {
      setStatus("error");
    } finally {
      setReplacing(null);
    }
  }

  const busy = status === "picking" || replacing !== null;
  const favouriteIds = new Set(favourites.map((film) => film.movie_id));
  const suggestions = matches.filter((film) => !favouriteIds.has(film.movie_id));

  return (
    <>
      <header className="masthead">
        <div className="dateline">
          <span>{today}</span>
          <span className="dateline-end">
            {newestYear && <span>Films up to {newestYear}</span>}
            <button className="theme-toggle" type="button" onClick={toggleTheme}>
              {theme === "dark" ? "Light mode" : "Dark mode"}
            </button>
          </span>
        </div>
        <h1>
          Tonight’s <em>Pick</em>
        </h1>
        <p className="intro">
          Name a few films you love, and get three to choose from tonight, one of them a new
          release.
        </p>
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
            disabled={favourites.length === 0 || busy}
            onClick={() => void pickFilms(rejected)}
          >
            {status === "picking" ? "Choosing…" : "Pick three films"}
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

        {picks.length > 0 && (
          <section className="picks" aria-label="Your picks">
            <div className="picks-head">
              <p className="kicker">Tonight · {COUNT_WORDS[picks.length]} to choose from</p>
              <button type="button" disabled={busy} onClick={showOthers}>
                Three others
              </button>
            </div>
            <ol className="pick-list">
              {picks.map((pick) => (
                <li key={pick.film.movie_id}>
                  <article className="pick">
                    {pick.recent && <p className="flag">New release</p>}
                    <h2>
                      <FilmTitle film={pick.film} />
                    </h2>
                    <p className="details">{details(pick.film)}</p>
                    <p className="reason">
                      Closest to <strong>{pick.reason.similar_to}</strong> among your favourites.
                      {pick.reason.themes.length > 0 &&
                        ` Shared themes: ${pick.reason.themes.join(", ")}.`}
                    </p>
                    <div className="actions">
                      <button
                        type="button"
                        aria-label={`Not this one: ${pick.film.title}`}
                        disabled={busy}
                        onClick={() => void replacePick(pick)}
                      >
                        {replacing === pick.film.movie_id ? "Finding another…" : "Not this one"}
                      </button>
                      {/* Grouped so that on a phone both links wrap together under the button. */}
                      <span className="links">
                        <a href={pick.film.where_to_watch_url} target="_blank" rel="noreferrer">
                          Where to watch
                        </a>
                        {pick.film.wikipedia_url && (
                          <a href={pick.film.wikipedia_url} target="_blank" rel="noreferrer">
                            Wikipedia
                          </a>
                        )}
                      </span>
                    </div>
                  </article>
                </li>
              ))}
            </ol>
          </section>
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
        <a href="https://en.wikipedia.org/">Wikipedia</a> (CC BY-SA), with popularity from
        Wikipedia page views. Free and non-commercial. MovieLens ratings end in October 2023, so
        newer films are matched by their descriptions alone.
      </footer>
    </>
  );
}
