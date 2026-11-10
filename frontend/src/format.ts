import type { PickedFilm, Preferences, Proposal } from "./types";

export function withYear(film: { title: string; year: number | null }): string {
  return film.year ? `${film.title} (${film.year})` : film.title;
}

export function details(film: PickedFilm): string {
  return [
    film.runtime ? `${film.runtime} min` : null,
    film.certification,
    film.directors.length ? `Directed by ${film.directors.join(", ")}` : null,
    film.genres.join(", ") || null,
  ]
    .filter(Boolean)
    .join(" · ");
}

export function describeProposal(proposal: Proposal): string {
  return proposal.kind === "avoid_genre"
    ? `Stop suggesting ${proposal.genre} films`
    : `Only suggest films up to ${proposal.minutes} minutes`;
}

export function applyProposal(preferences: Preferences, proposal: Proposal): Preferences {
  if (proposal.kind === "avoid_genre" && proposal.genre) {
    const genres = new Set([...preferences.avoid_genres, proposal.genre]);
    return { ...preferences, avoid_genres: [...genres].sort() };
  }
  if (proposal.kind === "max_runtime" && proposal.minutes) {
    return { ...preferences, max_runtime: proposal.minutes };
  }
  return preferences;
}
