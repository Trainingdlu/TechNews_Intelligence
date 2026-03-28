/* ===== TechNews Brief Subscription ===== */
(function () {
    "use strict";

    const API_BASE = "https://agentapi.trainingcqy.com";

    const form = document.getElementById("subscribe-form");
    const emailInput = document.getElementById("email-input");
    const nameInput = document.getElementById("name-input");
    const frequencySelect = document.getElementById("frequency-select");
    const timezoneInput = document.getElementById("timezone-input");
    const saveBtn = document.getElementById("save-btn");
    const loadBtn = document.getElementById("load-btn");
    const unsubscribeBtn = document.getElementById("unsubscribe-btn");
    const statusText = document.getElementById("status-text");
    const defaultEmailPlaceholder = emailInput.getAttribute("placeholder") || "输入邮箱地址";

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
        }
    }

    async function loadCurrentSettings() {
        const email = emailInput.value.trim();
        if (!email) {
            setStatus("", "");
            showEmailInlineHint("请先填写邮箱");
            return;
        }

        setBusy(true);
        setStatus("正在读取当前设置...", "");
        try {
            const res = await apiFetch("/subscriptions?email=" + encodeURIComponent(email));
            if (!res.ok) {
                const detail = await parseError(res);
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

        const payload = {
            email,
            name: nameInput.value.trim() || null,
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
    emailInput.addEventListener("input", clearEmailInlineHint);
    emailInput.addEventListener("focus", () => {
        if (!emailInput.value.trim()) clearEmailInlineHint();
    });

    loadOptions();
})();
