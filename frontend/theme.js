(function () {
    "use strict";

    const STORAGE_KEY = "technews_theme";
    const root = document.documentElement;
    const toggle = document.getElementById("theme-toggle");
    const media = window.matchMedia("(prefers-color-scheme: dark)");

    function getStoredTheme() {
        let value = "";

        try {
            value = localStorage.getItem(STORAGE_KEY);
        } catch {
            value = "";
        }

        return value === "dark" || value === "light" ? value : "";
    }

    function storeTheme(theme) {
        try {
            localStorage.setItem(STORAGE_KEY, theme);
        } catch {
            // Keep the visual toggle working even when storage is unavailable.
        }
    }

    function getSystemTheme() {
        return media.matches ? "dark" : "light";
    }

    function setToggleState(theme) {
        if (!toggle) return;

        const isDark = theme === "dark";
        const label = isDark ? "切换浅色模式" : "切换深色模式";
        const icon = toggle.querySelector(".material-symbols-outlined");

        toggle.setAttribute("aria-label", label);
        toggle.setAttribute("title", label);
        toggle.setAttribute("aria-pressed", String(isDark));

        if (icon) {
            icon.textContent = isDark ? "light_mode" : "dark_mode";
        }
    }

    function applyTheme(theme, shouldPersist) {
        const resolvedTheme = theme || getSystemTheme();

        root.dataset.theme = resolvedTheme;
        root.style.colorScheme = resolvedTheme;
        setToggleState(resolvedTheme);

        if (shouldPersist && theme) {
            storeTheme(theme);
        }
    }

    applyTheme(getStoredTheme(), false);

    media.addEventListener("change", () => {
        if (!getStoredTheme()) {
            applyTheme("", false);
        }
    });

    if (toggle) {
        toggle.addEventListener("click", () => {
            const activeTheme = root.dataset.theme || getSystemTheme();
            const nextTheme = activeTheme === "dark" ? "light" : "dark";
            applyTheme(nextTheme, true);
        });
    }
})();
