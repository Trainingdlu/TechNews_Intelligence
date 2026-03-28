/* ===== TechNews Brief Subscription ===== */
(function () {
    "use strict";

    const API_BASE = "https://agentapi.trainingcqy.com";

    const form = document.getElementById("subscribe-form");
    const emailInput = document.getElementById("email-input");
    const nameInput = document.getElementById("name-input");
    const frequencySelect = document.getElementById("frequency-select");
    const timezoneInput = document.getElementById("timezone-input");
    const sourceOptions = document.getElementById("source-options");
    const saveBtn = document.getElementById("save-btn");
    const loadBtn = document.getElementById("load-btn");
    const unsubscribeBtn = document.getElementById("unsubscribe-btn");
    const statusText = document.getElementById("status-text");

    const defaultEmailPlaceholder =
        emailInput.getAttribute("placeholder") || "输入邮箱地址订阅日报";

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

    function clearEmailInlineHint() {
        emailInput.classList.remove("input-inline-hint");
        emailInput.placeholder = defaultEmailPlaceholder;
    }

    function showEmailInlineHint(message) {
        emailInput.classList.add("input-inline-hint");
        emailInput.value = "";
        emailInput.placeholder = message;
        emailInput.focus();
    }

    async function apiFetch(path, options = {}) {
        const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
        return fetch(API_BASE + path, { ...options, headers });
    }

    async function parseError(res) {
        const payload = await res.json().catch(() => ({}));
        return payload.detail || "request_failed";
    }

    function renderSourceOptions(sources) {
        const sourceList = Array.isArray(sources) ? sources.filter(Boolean) : [];

        if (!sourceList.length) {
            sourceOptions.innerHTML = '<span class="source-loading">暂无可选来源</span>';
            return;
        }

        sourceOptions.innerHTML = "";

        sourceList.forEach((source) => {
            const label = document.createElement("label");
            label.className = "source-option";

            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.value = source;
            checkbox.checked = true;

            const text = document.createElement("span");
            text.textContent = source;

            label.appendChild(checkbox);
            label.appendChild(text);
            sourceOptions.appendChild(label);
        });
    }

    function getSelectedSources() {
        const nodes = sourceOptions.querySelectorAll('input[type="checkbox"]:checked');
        return Array.from(nodes)
            .map((node) => node.value)
            .filter(Boolean);
    }

    function setSelectedSources(sources) {
        const selected = new Set((Array.isArray(sources) ? sources : []).filter(Boolean));
        const nodes = sourceOptions.querySelectorAll('input[type="checkbox"]');

        if (!nodes.length) return;

        if (selected.size === 0) {
            nodes.forEach((node) => {
                node.checked = true;
            });
            return;
        }

        nodes.forEach((node) => {
            node.checked = selected.has(node.value);
        });
    }

    async function loadOptions() {
        try {
            const res = await apiFetch("/subscription-options");
            if (!res.ok) return;

            const data = await res.json();

            if (Array.isArray(data.frequencies) && data.frequencies.length > 0) {
                const current = frequencySelect.value;
                frequencySelect.innerHTML = "";

                data.frequencies.forEach((freq) => {
                    const option = document.createElement("option");
                    option.value = freq;
                    option.textContent = freq === "daily" ? "每天推送（当前可用）" : freq;
                    if (freq === current) option.selected = true;
                    frequencySelect.appendChild(option);
                });
            }

            renderSourceOptions(data.sources || []);

            if (data.default_timezone && !timezoneInput.value) {
                timezoneInput.value = data.default_timezone;
            }
        } catch {
            renderSourceOptions([]);
        }
    }

    async function loadCurrentSettings() {
        const email = emailInput.value.trim();
        if (!email) {
            setStatus("", "");
            showEmailInlineHint("请先填写邮箱");
            return;
        }

        if (!sourceOptions.querySelector('input[type="checkbox"]')) {
            await loadOptions();
        }

        setBusy(true);
        setStatus("正在读取当前设置...", "");
        try {
            const res = await apiFetch("/subscriptions?email=" + encodeURIComponent(email));
            if (!res.ok) {
                if (res.status === 404) {
                    setStatus("", "");
                    showEmailInlineHint("该邮箱暂无订阅记录");
                    return;
                }
                setStatus("", "");
                showEmailInlineHint("读取失败，请检查邮箱");
                return;
            }

            const data = await res.json();
            nameInput.value = data.name || "";
            frequencySelect.value = data.frequency || "daily";
            timezoneInput.value = data.timezone || "Asia/Shanghai";
            setSelectedSources(data.sources || []);

            if (data.is_active) {
                setStatus("已读取当前订阅偏好。", "success");
            } else {
                setStatus("此邮箱已退订，可点击“保存订阅”重新启用。", "info");
            }
        } catch {
            setStatus("", "");
            showEmailInlineHint("读取失败，请稍后再试");
        } finally {
            setBusy(false);
        }
    }

    async function submitSubscription(event) {
        event.preventDefault();

        const email = emailInput.value.trim();
        if (!email) {
            setStatus("", "");
            showEmailInlineHint("请先填写邮箱");
            return;
        }

        const selectedSources = getSelectedSources();
        if (!sourceOptions.querySelector('input[type="checkbox"]')) {
            setStatus("来源列表尚未加载完成，请稍后重试。", "error");
            return;
        }
        if (!selectedSources.length) {
            setStatus("请至少选择一个来源。", "error");
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
                setStatus("保存失败: " + detail, "error");
                return;
            }

            await res.json();
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
            setStatus("", "");
            showEmailInlineHint("请先填写邮箱");
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
                setStatus("退订失败: " + detail, "error");
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
    emailInput.addEventListener("input", clearEmailInlineHint);
    emailInput.addEventListener("focus", () => {
        if (!emailInput.value.trim()) clearEmailInlineHint();
    });

    loadOptions();
})();
