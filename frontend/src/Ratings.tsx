import type { ReactNode } from "react";

import type { PickedFilm } from "./types";

type Film = Pick<
  PickedFilm,
  "imdb_rating" | "imdb_url" | "rotten_tomatoes" | "rotten_tomatoes_url"
>;

// Rotten Tomatoes calls a film fresh when at least this share of critics' reviews are positive.
const FRESH_FROM = 60;

// A film's IMDb rating and Rotten Tomatoes Tomatometer, each linking to its page there. A score
// that isn't known is left out, and so is the whole line when neither is.
export default function Ratings({ film }: { film: Film }) {
  const imdb = film.imdb_rating ?? null;
  const tomatometer = film.rotten_tomatoes ?? null;
  if (imdb === null && tomatometer === null) return null;

  return (
    <p className="ratings">
      {imdb !== null && (
        <Score href={film.imdb_url ?? null} title="IMDb rating, out of 10">
          <span className="imdb-mark" aria-hidden="true">
            IMDb
          </span>
          {/* Read as "IMDb rating 8.1 out of 10". The spaces stay outside the spans, because
              accessible names trim each element's text. */}
          <span className="visually-hidden">IMDb rating</span> {imdb.toFixed(1)}{" "}
          <span className="visually-hidden">out of 10</span>
        </Score>
      )}
      {tomatometer !== null && (
        <Score
          href={film.rotten_tomatoes_url ?? null}
          title="Rotten Tomatoes Tomatometer: the share of critics' reviews that are positive"
        >
          {tomatometer >= FRESH_FROM ? <Tomato /> : <Splat />}
          <span className="visually-hidden">Rotten Tomatoes</span> {tomatometer}%
        </Score>
      )}
    </p>
  );
}

function Score({ href, title, children }: { href: string | null; title: string; children: ReactNode }) {
  return href ? (
    <a className="score" href={href} title={title} target="_blank" rel="noreferrer">
      {children}
    </a>
  ) : (
    <span className="score" title={title}>
      {children}
    </span>
  );
}

// Drawn for this app: a ripe tomato for a fresh score.
function Tomato() {
  return (
    <svg className="score-icon tomato" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <circle cx="8" cy="9.4" r="6.1" fill="#e0392b" />
      <ellipse cx="5.4" cy="7.4" rx="1.5" ry="0.9" fill="#ffffff" opacity="0.35" transform="rotate(-35 5.4 7.4)" />
      <ellipse cx="6" cy="3.7" rx="2.3" ry="0.95" fill="#3f9b3a" transform="rotate(-22 6 3.7)" />
      <ellipse cx="10" cy="3.7" rx="2.3" ry="0.95" fill="#3f9b3a" transform="rotate(22 10 3.7)" />
      <rect x="7.4" y="1.2" width="1.2" height="3" rx="0.6" fill="#3f9b3a" />
    </svg>
  );
}

// And a green splat for a rotten one.
function Splat() {
  return (
    <svg className="score-icon splat" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <g fill="#5e9a2c">
        <circle cx="8" cy="8.5" r="4.6" />
        <circle cx="3.1" cy="4.6" r="1.4" />
        <circle cx="13.2" cy="4.4" r="1.1" />
        <circle cx="2.6" cy="12.4" r="1" />
        <circle cx="13.4" cy="12.6" r="1.5" />
        <circle cx="8.6" cy="1.9" r="0.9" />
        <circle cx="7.2" cy="14.8" r="0.8" />
        <ellipse cx="4.6" cy="6.3" rx="1.6" ry="0.8" transform="rotate(40 4.6 6.3)" />
        <ellipse cx="11.6" cy="11.4" rx="1.8" ry="0.9" transform="rotate(40 11.6 11.4)" />
      </g>
    </svg>
  );
}
