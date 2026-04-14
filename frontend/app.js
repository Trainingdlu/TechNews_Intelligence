/* ===== TechNews Agent - Frontend App ===== */
(function () {
    'use strict';

    // API 后端地址（通过 Cloudflare Tunnel 暴露）
    const API_BASE = 'https://agentapi.trainingcqy.com';

    // == State ==
    let token = localStorage.getItem('agent_token') || '';
    let history = [];
    let isLoading = false;
    let remaining = null;
    const defaultLoadingStatus = '正在理解问题';

    // == DOM refs ==
    const authView = document.getElementById('auth-view');
    const chatView = document.getElementById('chat-view');
    const emailForm = document.getElementById('email-form');
    const emailInput = document.getElementById('email-input');
    const emailBtn = document.getElementById('email-btn');
    const emailStatus = document.getElementById('email-status');
    const tokenForm = document.getElementById('token-form');
    const tokenInput = document.getElementById('token-input');
    const messagesEl = document.getElementById('messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    const logoutBtn = document.getElementById('logout-btn');
    const quotaDisplay = document.getElementById('quota-display');
    const quotaOverlay = document.getElementById('quota-overlay');
    const inputBarInner = document.querySelector('.input-bar-inner');
    const defaultChatPlaceholder = chatInput.placeholder || '输入你的问题...';
    const exhaustedChatPlaceholder = '额度已耗尽，请求邮件已发送，通过后将通过邮件告知';

    // == Marked.js: 链接在新标签页打开 ==
    const renderer = new marked.Renderer();
    renderer.link = function ({ href, title, text }) {
        const titleAttr = title ? ` title="${title}"` : '';
        return `<a href="${href}"${titleAttr} target="_blank" rel="noopener noreferrer">${text}</a>`;
    };
    marked.setOptions({ renderer });

    // == Init ==
    if (token) {
        showChat();
        fetchQuota();
    }

    // == View Switching ==
    function showAuth() {
        chatView.classList.remove('active');
        // small delay so CSS transition runs
        requestAnimationFrame(() => {
            authView.classList.add('active');
        });
    }

    function showChat() {
        authView.classList.remove('active');
        requestAnimationFrame(() => {
            chatView.classList.add('active');
            chatInput.focus();
        });
    }

    // == Status Messages ==
    function setStatus(msg, type) {
        emailStatus.textContent = msg;
        emailStatus.className = 'status-msg ' + type;
    }

    async function playButtonPress(btn) {
        if (!btn) return;
        btn.classList.add('btn-pressing');
        await new Promise((resolve) => setTimeout(resolve, 90));
        btn.classList.remove('btn-pressing');
    }

    // == API Helpers ==
    async function apiFetch(path, options = {}) {
        const url = API_BASE + path;
        const headers = { 'Content-Type': 'application/json', ...options.headers };
        if (token) headers['Authorization'] = 'Bearer ' + token;
        const res = await fetch(url, { ...options, headers });
        return res;
    }

    // == Email Form ==
    emailForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = emailInput.value.trim();
        if (!email) return;

        await playButtonPress(emailBtn);
        emailBtn.classList.add('btn-loading');
        emailBtn.disabled = true;
        setStatus('', '');

        try {
            const res = await apiFetch('/request-access', {
                method: 'POST',
                body: JSON.stringify({ email }),
            });
            const data = await res.json();
            if (res.ok) {
                setStatus(data.message, 'success');
                tokenInput.focus();
            } else {
                setStatus(data.detail || '请求失败', 'error');
            }
        } catch {
            setStatus('网络错误，请检查连接', 'error');
        } finally {
            emailBtn.classList.remove('btn-loading');
            emailBtn.disabled = false;
        }
    });

    // == Token Form ==
    tokenForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const t = tokenInput.value.trim();
        if (!t) return;
        const tokenSubmitBtn = tokenForm.querySelector('button[type=\"submit\"]');
        await playButtonPress(tokenSubmitBtn);
        token = t;
        localStorage.setItem('agent_token', token);
        showChat();
        fetchQuota();
    });

    // == Quota ==
    async function fetchQuota() {
        try {
            const res = await apiFetch('/quota/' + encodeURIComponent(token));
            if (res.status === 404 || res.status === 401) {
                handleInvalidToken();
                return;
            }
            const data = await res.json();
            updateQuotaUI(data.remaining, data.quota, data.status);
        } catch { /* silent */ }
    }

    function updateQuotaUI(rem, total, status) {
        remaining = rem;
        quotaDisplay.textContent = `剩余 ${rem}/${total} 次`;
        quotaDisplay.classList.toggle('low', rem <= 3);

        if (status === 'exhausted' && rem <= 0) {
            showQuotaExhausted();
        } else {
            setQuotaInputLocked(false);
        }
    }

    function showQuotaExhausted() {
        setQuotaInputLocked(true);
    }

    function setQuotaInputLocked(locked) {
        quotaOverlay.classList.remove('visible');

        if (inputBarInner) {
            inputBarInner.classList.toggle('quota-locked', locked);
        }

        chatInput.disabled = locked;
        chatInput.placeholder = locked ? exhaustedChatPlaceholder : defaultChatPlaceholder;

        if (locked) {
            chatInput.value = '';
            chatInput.style.height = 'auto';
        }

        updateSendBtn();
    }

    function handleInvalidToken() {
        token = '';
        localStorage.removeItem('agent_token');
        history = [];
        setQuotaInputLocked(false);
        showAuth();
        setStatus('Token 无效或已过期，请重新获取', 'error');
    }

    // == Chat Input ==
    chatInput.addEventListener('input', () => {
        // Auto-resize textarea
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        updateSendBtn();
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    function updateSendBtn() {
        sendBtn.disabled = chatInput.disabled || !chatInput.value.trim() || isLoading;
    }

    function applyChatSuccess(text, data) {
        const reply = String(data.reply || '').trim();
        const citationUrls = Array.isArray(data.citation_urls)
            ? data.citation_urls.map((u) => String(u || '').trim()).filter(Boolean)
            : [];
        history.push({ role: 'user', parts: [{ text }] });
        history.push({ role: 'model', parts: [{ text: reply }] });

        appendMessage('agent', reply, { citationUrls });
        const fallbackTotal = remaining !== null ? parseInt(quotaDisplay.textContent.split('/')[1]) : 10;
        const totalQuota = Number.isFinite(data.quota) ? data.quota : fallbackTotal;
        updateQuotaUI(data.remaining, totalQuota, data.remaining > 0 ? 'active' : 'exhausted');
    }

    function applyChatClarification(text, data) {
        const clarification = data && typeof data.clarification === 'object' ? data.clarification : {};
        const question = String((clarification && clarification.question) || data.reply || '').trim();
        const hints = Array.isArray(clarification.hints)
            ? clarification.hints.map((item) => String(item || '').trim()).filter(Boolean)
            : [];
        const clarificationText = hints.length
            ? `${question}\n\n你可以补充以下任意一项：\n${hints.map((hint) => `- ${hint}`).join('\n')}`
            : question;

        history.push({ role: 'user', parts: [{ text }] });
        history.push({
            role: 'model',
            kind: 'clarification_required',
            clarification: clarification,
            parts: [{ text: question }],
        });

        appendMessage('agent', clarificationText || '请补充分析范围后我再继续。');
        const fallbackTotal = remaining !== null ? parseInt(quotaDisplay.textContent.split('/')[1]) : 10;
        const totalQuota = Number.isFinite(data.quota) ? data.quota : fallbackTotal;
        updateQuotaUI(data.remaining, totalQuota, data.remaining > 0 ? 'active' : 'exhausted');
    }

    function applyChatPayload(text, data) {
        const kind = String((data && data.kind) || 'answer').toLowerCase();
        if (kind === 'clarification_required') {
            applyChatClarification(text, data || {});
            return;
        }
        applyChatSuccess(text, data || {});
    }

    async function handleChatHttpError(res) {
        if (res.status === 401) {
            handleInvalidToken();
            return true;
        }
        if (res.status === 403) {
            showQuotaExhausted();
            return true;
        }
        if (res.status === 429) {
            appendMessage('agent', '请求过于频繁，请稍后再试。');
            return true;
        }
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            appendMessage('agent', '出错了：' + (err.detail || '未知错误'));
            return true;
        }
        return false;
    }

    function updateTypingStatus(typingEl, statusText) {
        const labelEl = typingEl?.querySelector('.typing-label');
        if (labelEl && statusText) {
            labelEl.textContent = statusText;
        }
    }

    async function consumeChatStream(res, typingEl) {
        if (!res.body) return { final: null, error: null };

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalPayload = null;
        let errorPayload = null;

        const processBlock = (block) => {
            const lines = block.split(/\r?\n/);
            let eventName = 'message';
            const dataLines = [];
            for (const line of lines) {
                if (line.startsWith('event:')) {
                    eventName = line.slice(6).trim();
                } else if (line.startsWith('data:')) {
                    dataLines.push(line.slice(5).trim());
                }
            }
            if (!dataLines.length) return;

            let payload = {};
            try {
                payload = JSON.parse(dataLines.join('\n'));
            } catch {
                return;
            }

            if (eventName === 'status' && payload.text) {
                updateTypingStatus(typingEl, payload.text);
                return;
            }
            if (eventName === 'final') {
                finalPayload = payload;
                return;
            }
            if (eventName === 'error') {
                errorPayload = payload;
            }
        };

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            let splitAt = buffer.indexOf('\n\n');
            while (splitAt !== -1) {
                const block = buffer.slice(0, splitAt).trim();
                buffer = buffer.slice(splitAt + 2);
                if (block) processBlock(block);
                splitAt = buffer.indexOf('\n\n');
            }

            if (finalPayload || errorPayload) {
                try {
                    await reader.cancel();
                } catch { /* ignore */ }
                break;
            }
        }

        return { final: finalPayload, error: errorPayload };
    }

    // == Send Message ==
    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isLoading) return;

        isLoading = true;
        updateSendBtn();

        const welcome = messagesEl.querySelector('.welcome-msg');
        if (welcome) welcome.remove();

        appendMessage('user', text);
        chatInput.value = '';
        chatInput.style.height = 'auto';

        const typingEl = showTyping(defaultLoadingStatus);
        let streamRes = null;

        try {
            streamRes = await apiFetch('/chat-stream', {
                method: 'POST',
                body: JSON.stringify({ message: text, history }),
            });

            // Compatibility fallback: older backend may not have /chat-stream yet.
            if (streamRes.status === 404 || streamRes.status === 405) {
                streamRes = await apiFetch('/chat', {
                    method: 'POST',
                    body: JSON.stringify({ message: text, history }),
                });
            }

            const streamType = (streamRes.headers.get('content-type') || '').toLowerCase();
            const canUseStream = streamRes.ok
                && streamType.includes('text/event-stream')
                && !!streamRes.body;

            if (canUseStream) {
                const streamed = await consumeChatStream(streamRes, typingEl);
                if (streamed.error) {
                    appendMessage('agent', '出错了：' + (streamed.error.detail || '未知错误'));
                    return;
                }
                if (!streamed.final) {
                    appendMessage('agent', '出错了：未收到完整回复');
                    return;
                }
                applyChatPayload(text, streamed.final);
                return;
            }

            if (await handleChatHttpError(streamRes)) return;
            const fallbackData = await streamRes.json();
            applyChatPayload(text, fallbackData);
        } catch {
            // If streaming request itself failed before getting a response object,
            // fallback to non-streaming chat once.
            if (!streamRes) {
                try {
                    const nonStreamRes = await apiFetch('/chat', {
                        method: 'POST',
                        body: JSON.stringify({ message: text, history }),
                    });
                    if (await handleChatHttpError(nonStreamRes)) return;
                    const nonStreamData = await nonStreamRes.json();
                    applyChatPayload(text, nonStreamData);
                    return;
                } catch {
                    // continue to unified network error below
                }
            }
            appendMessage('agent', '网络错误，请检查连接后重试。');
        } finally {
            removeTyping(typingEl);
            isLoading = false;
            updateSendBtn();
            chatInput.focus();
        }
    }

    // == Message Rendering ==
    const sourceHeaderRe = /^\s{0,3}(?:#{1,6}\s*)?(?:来源|证据来源|source(?:s)?|evidence\s+sources?)\s*:?\s*$/i;
    const sourceBulletRe = /^\s*(?:-\s*)?\[(\d{1,3})\]\s+.+$/;

    function _splitBodyAndSource(rawText) {
        const text = String(rawText || '');
        const lines = text.split('\n');
        let sourceStart = -1;
        for (let i = 0; i < lines.length; i++) {
            if (sourceHeaderRe.test(lines[i].trim())) {
                sourceStart = i;
                break;
            }
        }
        if (sourceStart < 0) {
            const bulletIndexes = [];
            for (let i = 0; i < lines.length; i++) {
                if (sourceBulletRe.test(lines[i])) {
                    bulletIndexes.push(i);
                }
            }
            if (bulletIndexes.length >= 2) {
                sourceStart = bulletIndexes[0];
            }
        }
        if (sourceStart < 0) {
            return { body: text, source: '' };
        }
        return {
            body: lines.slice(0, sourceStart).join('\n'),
            source: lines.slice(sourceStart).join('\n'),
        };
    }

    function _extractCitationUrlMap(sourceText) {
        const map = {};
        const lines = String(sourceText || '').split('\n');
        for (const line of lines) {
            const m = line.match(/^\s*(?:-\s*)?\[(\d{1,3})\]\s+(.+)$/);
            if (!m) continue;
            const idx = m[1];
            const rest = m[2];
            let url = null;

            const markdownLink = rest.match(/\]\((https?:\/\/[^\s)]+)\)/i);
            if (markdownLink) {
                url = markdownLink[1];
            } else {
                const plainUrl = rest.match(/https?:\/\/[^\s)]+/i);
                if (plainUrl) url = plainUrl[0];
            }

            if (url) {
                map[idx] = url.replace(/[),.;!?]+$/, '');
            }
        }
        return map;
    }

    function _prepareAgentMarkdown(rawText, citationUrls = []) {
        const { body, source } = _splitBodyAndSource(rawText);
        const citationMap = _extractCitationUrlMap(source);
        if (Array.isArray(citationUrls) && citationUrls.length) {
            citationUrls.forEach((url, i) => {
                const normalized = String(url || '').trim();
                if (normalized) {
                    citationMap[String(i + 1)] = normalized;
                }
            });
        }

        // Body: [n] -> linked [n](url) when url exists; otherwise keep as literal [n].
        const linkedBody = String(body || '').replace(/\[(\d{1,3})\](?!\()/g, (full, idx) => {
            const url = citationMap[idx];
            if (!url) return `\\[${idx}\\]`;
            return `[${idx}](${url})`;
        });

        if (!source) {
            return linkedBody;
        }

        // Source section: keep numbering literal, avoid accidental relative links like /[5].
        const normalizedSource = String(source)
            .replace(/^(\s*-\s*)\[(\d{1,3})\]/gm, '$1\\[$2\\]')
            .replace(/^(\s*)\[(\d{1,3})\]/gm, '$1\\[$2\\]');
        return `${linkedBody}\n${normalizedSource}`.trim();
    }

    function appendMessage(role, text, meta = {}) {
        const msg = document.createElement('div');
        msg.className = 'msg ' + role;

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined';
        icon.textContent = role === 'agent' ? 'smart_toy' : 'person';
        avatar.appendChild(icon);

        const content = document.createElement('div');
        content.className = 'msg-content';
        if (role === 'agent') {
            const citationUrls = Array.isArray(meta.citationUrls) ? meta.citationUrls : [];
            content.innerHTML = marked.parse(_prepareAgentMarkdown(text, citationUrls));
        } else {
            content.textContent = text;
        }

        msg.appendChild(avatar);
        msg.appendChild(content);
        messagesEl.appendChild(msg);

        // Smooth scroll to bottom
        requestAnimationFrame(() => {
            messagesEl.scrollTo({
                top: messagesEl.scrollHeight,
                behavior: 'smooth',
            });
        });
    }

    // == Typing Indicator ==
    function showTyping(statusText = defaultLoadingStatus) {
        const el = document.createElement('div');
        el.className = 'msg agent typing-msg';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = `
            <span class="typing-spinner" aria-hidden="true"></span>
            <div class="typing-label" role="status" aria-live="polite">${statusText}</div>
        `;

        el.appendChild(indicator);
        messagesEl.appendChild(el);

        requestAnimationFrame(() => {
            messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
        });

        return el;
    }

    function removeTyping(el) {
        if (el && el.parentNode) {
            el.style.opacity = '0';
            el.style.transform = 'translateY(-8px)';
            el.style.transition = 'all 0.2s ease';
            setTimeout(() => el.remove(), 200);
        }
    }

    // == Clear / Logout ==
    clearBtn.addEventListener('click', () => {
        history = [];
        messagesEl.innerHTML = '';
        // Re-add welcome
        messagesEl.innerHTML = `
            <div class="welcome-msg">
                <span class="material-symbols-outlined welcome-icon">waving_hand</span>
                <p>你好！我是 TechNews 智能分析助手。</p>
                <p class="welcome-hint">试试问我："最近AI领域有什么大事？"</p>
            </div>`;
    });

    logoutBtn.addEventListener('click', () => {
        token = '';
        localStorage.removeItem('agent_token');
        history = [];
        messagesEl.innerHTML = '';
        setQuotaInputLocked(false);
        showAuth();
    });
})();
