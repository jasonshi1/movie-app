// Applies a theme chosen on an earlier visit before the page is drawn, so it never flashes the
// other one first. It's a separate file because the Content-Security-Policy blocks inline
// scripts. The key and values match src/storage.ts.
try {
  var theme = JSON.parse(window.localStorage.getItem("tonights-pick:theme"));
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
} catch (error) {
  // Storage is unavailable, so the page follows the system setting.
}
