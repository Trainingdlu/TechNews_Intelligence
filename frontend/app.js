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

    // == Send Message ==
    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isLoading) return;

        isLoading = true;
        updateSendBtn();

        // Clear welcome if first message
        const welcome = messagesEl.querySelector('.welcome-msg');
        if (welcome) welcome.remove();

        appendMessage('user', text);
        chatInput.value = '';
        chatInput.style.height = 'auto';

        const typingEl = showTyping();

        try {
            const res = await apiFetch('/chat', {
                method: 'POST',
                body: JSON.stringify({ message: text, history }),
            });

            removeTyping(typingEl);

            if (res.status === 401) {
                handleInvalidToken();
                return;
            }
            if (res.status === 403) {
                showQuotaExhausted();
                return;
            }
            if (res.status === 429) {
                appendMessage('agent', '请求过于频繁，请稍后再试。');
                return;
            }
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                appendMessage('agent', '出错了：' + (err.detail || '未知错误'));
                return;
            }

            const data = await res.json();

            // Update history
            history.push({ role: 'user', parts: [{ text }] });
            history.push({ role: 'model', parts: [{ text: data.reply }] });

            appendMessage('agent', data.reply);
            updateQuotaUI(data.remaining, remaining !== null ? parseInt(quotaDisplay.textContent.split('/')[1]) : 15, data.remaining > 0 ? 'active' : 'exhausted');
        } catch {
            removeTyping(typingEl);
            appendMessage('agent', '网络错误，请检查连接后重试。');
        } finally {
            isLoading = false;
            updateSendBtn();
            chatInput.focus();
        }
    }

    // == Message Rendering ==
    function appendMessage(role, text) {
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
            content.innerHTML = marked.parse(text);
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
    function showTyping() {
        const el = document.createElement('div');
        el.className = 'msg agent';

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined';
        icon.textContent = 'smart_toy';
        avatar.appendChild(icon);

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';

        el.appendChild(avatar);
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

