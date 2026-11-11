import type { FilmMatch, Preferences } from "./types";

// Browser storage can be unavailable (private windows, blocked site data), so every access is
// guarded and the app works the same without it.

const FAVOURITES_KEY = "tonights-pick:favourites";
const PREFERENCES_KEY = "tonights-pick:preferences";
// public/theme.js reads this too, to apply the theme before the page is drawn.
const THEME_KEY = "tonights-pick:theme";

export type Theme = "light" | "dark";

export const NO_PREFERENCES: Preferences = { avoid_genres: [], max_runtime: null };

function read(key: string): unknown {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function write(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage is unavailable; the value just won't survive a reload.
  }
}

function isFilm(value: unknown): value is FilmMatch {
  const film = value as FilmMatch;
  return typeof film?.movie_id === "number" && typeof film.title === "string";
}

export function loadFavourites(): FilmMatch[] {
  const stored = read(FAVOURITES_KEY);
  return Array.isArray(stored) ? stored.filter(isFilm) : [];
}

export function saveFavourites(favourites: FilmMatch[]) {
  write(FAVOURITES_KEY, favourites);
}

export function loadPreferences(): Preferences {
  const stored = read(PREFERENCES_KEY) as Partial<Preferences> | null;
  return {
    avoid_genres: Array.isArray(stored?.avoid_genres)
      ? stored.avoid_genres.filter((genre) => typeof genre === "string")
      : [],
    max_runtime: typeof stored?.max_runtime === "number" ? stored.max_runtime : null,
  };
}

export function savePreferences(preferences: Preferences) {
  write(PREFERENCES_KEY, preferences);
}

// The theme someone chose with the switch, or null to follow the system setting.
export function loadTheme(): Theme | null {
  const stored = read(THEME_KEY);
  return stored === "light" || stored === "dark" ? stored : null;
}

export function saveTheme(theme: Theme) {
  write(THEME_KEY, theme);
}
