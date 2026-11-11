import { useEffect, useState } from "react";

import { loadTheme, saveTheme, type Theme } from "./storage";

const DARK_QUERY = "(prefers-color-scheme: dark)";

function systemTheme(): Theme {
  return window.matchMedia?.(DARK_QUERY).matches ? "dark" : "light";
}

// The theme in use and a function that switches to the other one. Until someone switches, it
// follows the system setting. A choice is saved, set on <html> as data-theme for styles.css,
// and applied before the page is drawn on later visits by public/theme.js.
export function useTheme(): [Theme, () => void] {
  const [chosen, setChosen] = useState<Theme | null>(loadTheme);
  const [system, setSystem] = useState<Theme>(systemTheme);

  useEffect(() => {
    const query = window.matchMedia?.(DARK_QUERY);
    if (!query) return;
    const update = () => setSystem(query.matches ? "dark" : "light");
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    if (chosen) document.documentElement.dataset.theme = chosen;
  }, [chosen]);

  const theme = chosen ?? system;
  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    setChosen(next);
    saveTheme(next);
  }
  return [theme, toggle];
}
