// script.js - shared UI behaviours for TrendIQ Terminal

document.addEventListener("DOMContentLoaded", function () {
  // Auto-dismiss alerts after 5 seconds
  document.querySelectorAll(".alert").forEach(function (alertEl) {
    setTimeout(function () {
      if (window.bootstrap && bootstrap.Alert) {
        const instance = bootstrap.Alert.getOrCreateInstance(alertEl);
        instance.close();
      }
    }, 5000);
  });

  // Live terminal clock in the topbar
  const timeEl = document.getElementById("topbarTime");
  const dateEl = document.getElementById("topbarDate");
  if (timeEl && dateEl) {
    function tick() {
      const now = new Date();
      const timeStr = now.toLocaleTimeString("en-GB", { hour12: false });
      const dateStr = now.toLocaleDateString("en-US", {
        weekday: "short", year: "numeric", month: "short", day: "2-digit"
      }).toUpperCase();
      timeEl.textContent = timeStr;
      dateEl.textContent = dateStr;
    }
    tick();
    setInterval(tick, 1000);
  }

  // Global Chart.js defaults so every chart reads correctly on the dark
  // terminal background without needing per-chart color overrides.
  if (window.Chart) {
    Chart.defaults.color = "#6B7688";
    Chart.defaults.borderColor = "#1C2330";
    Chart.defaults.font.family = "'IBM Plex Mono', 'Roboto Mono', monospace";
    Chart.defaults.font.size = 11;
    Chart.defaults.plugins.legend.labels.color = "#9AA3B5";
  }
});
