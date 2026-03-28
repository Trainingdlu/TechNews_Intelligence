/* ===== TechNews Brief Subscription ===== */
(function () {
    "use strict";

    const API_BASE = "https://agentapi.trainingcqy.com";

    const form = document.getElementById("subscribe-form");
    const emailInput = document.getElementById("email-input");
    const nameInput = document.getElementById("name-input");
    const sourceList = document.getElementById("source-list");
    const frequencySelect = document.getElementById("frequency-select");
    const timezoneInput = document.getElementById("timezone-input");
    const saveBtn = document.getElementById("save-btn");
    const loadBtn = document.getElementById("load-btn");
    const unsubscribeBtn = document.getElementById("unsubscribe-btn");
    const statusText = document.getElementById("status-text");

    let availableSources = ["HackerNews", "TechCrunch"];
    const SOURCE_ICON_MAP = {
        hackernews: "forum",
        techcrunch: "newspaper",
    };

    function setStatus(text, type) {
        const level = type || "info";
        statusText.textContent = text;
        statusText.className = "status-msg " + level;
    }

    function setBusy(isBusy) {
        saveBtn.disabled = isBusy;
        loadBtn.disabled = isBusy;
        unsubscribeBtn.disabled = isBusy;
    }

    function renderSourceOptions(selected = []) {
        sourceList.innerHTML = "";
        availableSources.forEach((source) => {
            const id = "source-" + source.toLowerCase().replace(/[^a-z0-9]+/g, "-");
            const iconName = SOURCE_ICON_MAP[source.toLowerCase()] || "rss_feed";

            const label = document.createElement("label");
            label.className = "source-chip";
            label.setAttribute("for", id);
            label.dataset.source = source.toLowerCase();

            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.name = "sources";
            checkbox.value = source;
            checkbox.id = id;
            checkbox.checked = selected.includes(source);

            const icon = document.createElement("span");
            icon.className = "material-symbols-outlined source-chip-icon";
            icon.textContent = iconName;

            const check = document.createElement("span");
            check.className = "material-symbols-outlined source-chip-check";
            check.textContent = "check";

            const textNode = document.createElement("span");
            textNode.className = "source-chip-text";
            textNode.textContent = source;

            label.appendChild(checkbox);
            label.appendChild(icon);
            label.appendChild(textNode);
            label.appendChild(check);
            sourceList.appendChild(label);
        });
    }

    function getSelectedSources() {
        const checks = sourceList.querySelectorAll('input[name="sources"]:checked');
        return Array.from(checks).map((node) => node.value);
    }

    async function apiFetch(path, options = {}) {
        const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
        return fetch(API_BASE + path, { ...options, headers });
    }

    async function parseError(res) {
        const payload = await res.json().catch(() => ({}));
        return payload.detail || "request_failed";
    }

    async function loadOptions() {
        try {
            const res = await apiFetch("/subscription-options");
            if (!res.ok) return;
            const data = await res.json();
            if (Array.isArray(data.sources) && data.sources.length > 0) {
                availableSources = data.sources;
            }
            if (Array.isArray(data.frequencies) && data.frequencies.length > 0) {
                const current = frequencySelect.value;
                frequencySelect.innerHTML = "";
                data.frequencies.forEach((freq) => {
                    const option = document.createElement("option");
                    option.value = freq;
                    option.textContent = freq === "daily" ? "每天" : (freq === "weekday" ? "工作日" : "每周");
                    if (freq === current) option.selected = true;
                    frequencySelect.appendChild(option);
                });
            }
            if (data.default_timezone && !timezoneInput.value) {
                timezoneInput.value = data.default_timezone;
            }
        } catch {
            // keep fallback options
        } finally {
            renderSourceOptions(availableSources);
        }
    }

    async function loadCurrentSettings() {
        const email = emailInput.value.trim();
        if (!email) {
            setStatus("请先填写邮箱。", "error");
            return;
        }

        setBusy(true);
        setStatus("正在读取当前设置...", "");
        try {
            const res = await apiFetch("/subscriptions?email=" + encodeURIComponent(email));
            if (!res.ok) {
                const detail = await parseError(res);
                if (res.status === 404) {
                    setStatus("这个邮箱目前还没有订阅记录。", "error");
                    return;
                }
                setStatus("读取失败：" + detail, "error");
                return;
            }

            const data = await res.json();
            nameInput.value = data.name || "";
            frequencySelect.value = data.frequency || "daily";
            timezoneInput.value = data.timezone || "Asia/Shanghai";
            renderSourceOptions(Array.isArray(data.sources) ? data.sources : availableSources);

            if (data.is_active) {
                setStatus("已读取当前订阅偏好。", "success");
            } else {
                setStatus("此邮箱已退订，可点击“保存订阅”重新启用。", "error");
            }
        } catch {
            setStatus("网络异常，请稍后重试。", "error");
        } finally {
            setBusy(false);
        }
    }

    async function submitSubscription(event) {
        event.preventDefault();

        const email = emailInput.value.trim();
        if (!email) {
            setStatus("请先填写邮箱。", "error");
            return;
        }

        const selectedSources = getSelectedSources();
        if (selectedSources.length === 0) {
            setStatus("至少选择一个新闻来源。", "error");
            return;
        }

        const payload = {
            email,
            name: nameInput.value.trim() || null,
            sources: selectedSources,
            frequency: frequencySelect.value,
            timezone: timezoneInput.value.trim() || "Asia/Shanghai",
        };

        setBusy(true);
        setStatus("正在保存订阅...", "");
        try {
            const res = await apiFetch("/subscriptions", {
                method: "POST",
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                const detail = await parseError(res);
                setStatus("保存失败：" + detail, "error");
                return;
            }

            const data = await res.json();
            renderSourceOptions(Array.isArray(data.sources) ? data.sources : selectedSources);
            setStatus("订阅已保存，日报会按你的偏好推送。", "success");
        } catch {
            setStatus("网络异常，请稍后重试。", "error");
        } finally {
            setBusy(false);
        }
    }

    async function unsubscribe() {
        const email = emailInput.value.trim();
        if (!email) {
            setStatus("退订前请先填写邮箱。", "error");
            return;
        }

        setBusy(true);
        setStatus("正在退订...", "");
        try {
            const res = await apiFetch("/subscriptions/unsubscribe", {
                method: "POST",
                body: JSON.stringify({ email }),
            });
            if (!res.ok) {
                const detail = await parseError(res);
                setStatus("退订失败：" + detail, "error");
                return;
            }
            setStatus("已退订成功。", "success");
        } catch {
            setStatus("网络异常，请稍后重试。", "error");
        } finally {
            setBusy(false);
        }
    }

    form.addEventListener("submit", submitSubscription);
    loadBtn.addEventListener("click", loadCurrentSettings);
    unsubscribeBtn.addEventListener("click", unsubscribe);

    loadOptions();
})();
