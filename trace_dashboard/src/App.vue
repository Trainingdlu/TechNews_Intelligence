<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import { createTraceClient, TraceApiError } from "./api";
import JsonBlock from "./components/JsonBlock.vue";
import SpanTree from "./components/SpanTree.vue";

const STORAGE_KEY = "technews_trace_dashboard_token";

const token = ref(sessionStorage.getItem(STORAGE_KEY) || "");
const tokenInput = ref("");
const authError = ref("");
const errorMessage = ref("");
const meta = ref(null);
const filters = ref({
  status: "all",
  q: ""
});
const runs = ref([]);
const total = ref(0);
const selectedRunId = ref("");
const runDetail = ref(null);
const spanTree = ref([]);
const selectedSpan = ref(null);
const modelIoBySpanId = ref({});
const loadingModelIoBySpanId = ref({});
const modelIoErrorsBySpanId = ref({});
const loadingRuns = ref(false);
const loadingRun = ref(false);
const loadingSpan = ref(false);
const detailPanelRef = ref(null);

let detailScrollTimer = 0;
let programmaticDetailScroll = false;
let modelIoObserver = null;

const hasToken = computed(() => Boolean(token.value));
const selectedRun = computed(() => runDetail.value?.run || runs.value.find((item) => item.request_id === selectedRunId.value) || null);
const detailSpans = computed(() => runDetail.value?.spans || []);
const spanMap = computed(() => Object.fromEntries(detailSpans.value.map((span) => [span.span_id, span])));
const activeSpan = computed(() => {
  const selectedId = selectedSpan.value?.span_id;
  if (!selectedId) return null;
  return spanMap.value[selectedId] || selectedSpan.value;
});
const activeLangSmith = computed(() => selectedRun.value?.langsmith || meta.value?.langsmith || {});
const errorToast = computed(() => {
  const message = String(errorMessage.value || "").trim();
  if (!message) return null;
  if (message.toLowerCase().includes("failed to fetch")) {
    return {
      title: "Trace API 连接失败",
      detail: "无法连接 Trace API，请确认服务已启动或稍后刷新重试。"
    };
  }
  return {
    title: "请求失败",
    detail: message
  };
});

function client() {
  return createTraceClient(token.value);
}

function setError(error) {
  if (error instanceof TraceApiError) {
    if (error.status === 401) {
      clearToken();
      authError.value = "Token 无效，请重新输入。";
      return;
    }
    errorMessage.value = String(error.detail || error.message);
    return;
  }
  errorMessage.value = error?.message || String(error);
}

function dismissError() {
  errorMessage.value = "";
}

async function submitToken() {
  const value = tokenInput.value.trim();
  if (!value) {
    authError.value = "请输入 TRACE_DASHBOARD_TOKEN。";
    return;
  }
  sessionStorage.setItem(STORAGE_KEY, value);
  token.value = value;
  tokenInput.value = "";
  authError.value = "";
  await loadInitialData();
}

function clearToken() {
  disconnectModelIoObserver();
  sessionStorage.removeItem(STORAGE_KEY);
  token.value = "";
  meta.value = null;
  runs.value = [];
  total.value = 0;
  selectedRunId.value = "";
  runDetail.value = null;
  spanTree.value = [];
  selectedSpan.value = null;
  modelIoBySpanId.value = {};
  loadingModelIoBySpanId.value = {};
  modelIoErrorsBySpanId.value = {};
}

async function loadInitialData() {
  await loadMeta();
  await loadRuns();
}

async function loadMeta() {
  if (!hasToken.value) return;
  try {
    meta.value = await client().meta();
  } catch (error) {
    setError(error);
  }
}

async function loadRuns() {
  if (!hasToken.value) return;
  loadingRuns.value = true;
  errorMessage.value = "";
  try {
    const params = new URLSearchParams();
    params.set("limit", "50");
    params.set("offset", "0");
    if (filters.value.status && filters.value.status !== "all") params.set("status", filters.value.status);
    if (filters.value.q.trim()) params.set("q", filters.value.q.trim());
    const payload = await client().runs(params);
    runs.value = payload.items || [];
    total.value = payload.total || 0;
    if (!selectedRunId.value && runs.value.length) {
      await selectRun(runs.value[0]);
    }
  } catch (error) {
    setError(error);
  } finally {
    loadingRuns.value = false;
  }
}

async function selectRun(run) {
  if (!run?.request_id) return;
  disconnectModelIoObserver();
  selectedRunId.value = run.request_id;
  loadingRun.value = true;
  errorMessage.value = "";
  runDetail.value = null;
  spanTree.value = [];
  selectedSpan.value = null;
  modelIoBySpanId.value = {};
  loadingModelIoBySpanId.value = {};
  modelIoErrorsBySpanId.value = {};
  try {
    const payload = await client().run(run.request_id);
    runDetail.value = payload;
    spanTree.value = payload.span_tree || [];
    const defaultSpan = findDefaultSpan(payload.spans || []);
    await nextTick();
    setupModelIoObserver();
    if (defaultSpan) {
      await selectSpan(defaultSpan);
    }
  } catch (error) {
    setError(error);
  } finally {
    loadingRun.value = false;
  }
}

function findDefaultSpan(spans) {
  return (
    spans.find((span) => span.error_code || ["error", "failed", "blocked"].includes(String(span.status).toLowerCase())) ||
    spans[0] ||
    null
  );
}

async function selectSpan(span) {
  if (!span?.span_id || !selectedRunId.value) return;
  selectedSpan.value = spanMap.value[span.span_id] || span;
  loadingSpan.value = span.span_type === "model_call" && !modelIoBySpanId.value[span.span_id];
  errorMessage.value = "";
  try {
    if (span.span_type === "model_call") {
      await ensureModelIo(span);
    }
  } catch (error) {
    setError(error);
  } finally {
    loadingSpan.value = false;
    await nextTick();
    scrollSpanCardIntoView(span.span_id);
    scrollSelectedTreeNodeIntoView();
  }
}

async function ensureModelIo(span, options = {}) {
  const spanId = span?.span_id;
  if (!spanId || span?.span_type !== "model_call" || !selectedRunId.value) return;
  if (modelIoBySpanId.value[spanId] || loadingModelIoBySpanId.value[spanId]) return;
  const requestId = selectedRunId.value;
  loadingModelIoBySpanId.value = { ...loadingModelIoBySpanId.value, [spanId]: true };
  modelIoErrorsBySpanId.value = { ...modelIoErrorsBySpanId.value, [spanId]: "" };
  try {
    const payload = await client().modelIo(spanId, requestId);
    if (selectedRunId.value !== requestId) return;
    modelIoBySpanId.value = { ...modelIoBySpanId.value, [spanId]: payload.model_io || null };
  } catch (error) {
    if (selectedRunId.value !== requestId) return;
    modelIoErrorsBySpanId.value = {
      ...modelIoErrorsBySpanId.value,
      [spanId]: error?.detail || error?.message || String(error)
    };
    if (!options.silent) throw error;
  } finally {
    if (selectedRunId.value === requestId) {
      loadingModelIoBySpanId.value = { ...loadingModelIoBySpanId.value, [spanId]: false };
    }
  }
}

function disconnectModelIoObserver() {
  if (!modelIoObserver) return;
  modelIoObserver.disconnect();
  modelIoObserver = null;
}

function setupModelIoObserver() {
  disconnectModelIoObserver();
  const panel = detailPanelRef.value;
  if (!panel || typeof IntersectionObserver === "undefined") return;
  modelIoObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const spanId = entry.target.dataset.spanId;
        const span = spanMap.value[spanId];
        if (span?.span_type === "model_call") {
          void ensureModelIo(span, { silent: true });
        }
      }
    },
    {
      root: panel,
      rootMargin: "240px 0px",
      threshold: 0.01
    }
  );
  panel
    .querySelectorAll('.span-detail-card[data-span-type="model_call"]')
    .forEach((element) => modelIoObserver.observe(element));
}

function scrollSpanCardIntoView(spanId) {
  const card = document.getElementById(`span-detail-${spanId}`);
  if (!card) return;
  programmaticDetailScroll = true;
  card.scrollIntoView({ block: "start", inline: "nearest" });
  window.setTimeout(() => {
    programmaticDetailScroll = false;
    syncActiveSpanFromScroll();
  }, 260);
}

function handleDetailScroll() {
  if (programmaticDetailScroll) return;
  window.clearTimeout(detailScrollTimer);
  detailScrollTimer = window.setTimeout(syncActiveSpanFromScroll, 80);
}

function syncActiveSpanFromScroll() {
  const panel = detailPanelRef.value;
  if (!panel || !detailSpans.value.length) return;
  const cards = Array.from(panel.querySelectorAll(".span-detail-card"));
  if (!cards.length) return;
  const panelTop = panel.getBoundingClientRect().top + 58;
  let current = cards[0];
  for (const card of cards) {
    if (card.getBoundingClientRect().top <= panelTop) {
      current = card;
    } else {
      break;
    }
  }
  const spanId = current.dataset.spanId;
  if (!spanId || activeSpan.value?.span_id === spanId) return;
  const span = spanMap.value[spanId];
  if (!span) return;
  selectedSpan.value = span;
  if (span.span_type === "model_call") {
    void ensureModelIo(span, { silent: true });
  }
  nextTick(scrollSelectedTreeNodeIntoView);
}

function scrollSelectedTreeNodeIntoView() {
  document.querySelector(".span-node.selected")?.scrollIntoView({
    block: "center",
    inline: "nearest"
  });
}

function statusLabel(status) {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "success") return "成功";
  if (normalized === "error" || normalized === "failed") return "失败";
  if (normalized === "blocked") return "已拦截";
  if (normalized === "running") return "运行中";
  return status || "未知";
}

function statusClass(status) {
  return `status-${String(status || "unknown").toLowerCase()}`;
}

function isModelSpanFor(span) {
  return span?.span_type === "model_call";
}

function isToolSpanFor(span) {
  return span?.span_type === "tool_call";
}

function isGuardSpanFor(span) {
  return ["guard", "postprocess"].includes(span?.span_type);
}

function isContextSpanFor(span) {
  return span?.span_type === "context";
}

function formatLatency(ms) {
  if (ms === null || ms === undefined) return "-";
  const value = Number(ms);
  if (!Number.isFinite(value)) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${Math.round(value)}ms`;
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(date);
}

function compactId(value) {
  const text = String(value || "");
  if (text.length <= 18) return text;
  return `${text.slice(0, 8)}…${text.slice(-8)}`;
}

function previewText(value, limit = 96) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}…`;
}

function normalizeMessages(value) {
  return Array.isArray(value) ? value : [];
}

function messageRole(message) {
  if (typeof message === "string") return "message";
  return message?.role || message?.type || message?._type || message?.id?.[2] || "message";
}

function sanitizeProviderInternalPayload(value) {
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeProviderInternalPayload(item));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        key === "thought_signature"
          ? `[provider_internal_${key}_omitted chars=${String(item ?? "").length}]`
          : sanitizeProviderInternalPayload(item)
      ])
    );
  }
  return value;
}

function messageContent(message) {
  if (typeof message === "string") return message;
  const content = message?.content ?? message?.kwargs?.content ?? message?.data?.content ?? message;
  const cleanContent = sanitizeProviderInternalPayload(content);
  if (typeof cleanContent === "string") return cleanContent;
  return JSON.stringify(cleanContent, null, 2);
}

function collectUrls(span) {
  if (!span) return [];
  const buckets = [
    span.metadata?.evidence_urls,
    span.metadata?.evidence,
    span.output_summary?.evidence_urls,
    span.output_summary?.urls,
    span.output_summary?.evidence
  ];
  const urls = [];
  for (const bucket of buckets) {
    if (Array.isArray(bucket)) {
      for (const item of bucket) {
        if (typeof item === "string") urls.push(item);
        if (item && typeof item === "object" && item.url) urls.push(item.url);
      }
    }
  }
  return [...new Set(urls.filter(Boolean))];
}

function diagnosticsFor(span) {
  return (
    span?.metadata?.diagnostics ||
    span?.output_summary?.diagnostics ||
    span?.input_summary?.diagnostics ||
    null
  );
}

onMounted(() => {
  if (hasToken.value) {
    loadInitialData();
  }
});

onBeforeUnmount(() => {
  window.clearTimeout(detailScrollTimer);
  disconnectModelIoObserver();
});
</script>

<template>
  <main v-if="!hasToken" class="auth-page">
    <section class="auth-panel">
      <p class="eyebrow">Trace Console</p>
      <h1>输入访问 Token</h1>
      <form class="auth-form" @submit.prevent="submitToken">
        <input
          v-model="tokenInput"
          type="password"
          autocomplete="current-password"
          placeholder="请输入token"
        />
        <button type="submit">进入</button>
      </form>
      <p v-if="authError" class="inline-error">{{ authError }}</p>
    </section>
  </main>

  <main v-else class="trace-shell">
    <header class="trace-topbar">
      <div class="brand-block">
        <div>
          <h1>Trace Console</h1>
          <p>{{ meta?.admin_email || "管理员" }}</p>
        </div>
      </div>

      <div class="top-actions">
        <span class="langsmith-state" :class="{ enabled: activeLangSmith.enabled }">
          LangSmith {{ activeLangSmith.enabled ? "开启" : "关闭" }}
        </span>
        <button type="button" class="toolbar-button" @click="loadRuns">刷新</button>
        <button type="button" class="toolbar-button secondary" @click="clearToken">退出</button>
      </div>
    </header>

    <section class="filterbar">
      <div class="filter-left">
        <label class="status-filter">
          <span>状态</span>
          <select v-model="filters.status" @change="loadRuns">
            <option value="all">全部</option>
            <option value="success">成功</option>
            <option value="error">失败</option>
            <option value="blocked">已拦截</option>
          </select>
        </label>
      </div>
      <form class="filter-center" @submit.prevent="loadRuns">
        <input
          v-model="filters.q"
          type="search"
          placeholder="request_id / thread_id / 用户问题 / error_code"
        />
        <button type="submit" class="toolbar-button">筛选</button>
      </form>
      <div class="filter-right">
        <span class="filter-count">共 {{ total }} 条</span>
      </div>
    </section>

    <aside v-if="errorToast" class="global-error" role="status" aria-live="polite">
      <span class="error-mark" aria-hidden="true">!</span>
      <span class="error-copy">
        <strong>{{ errorToast.title }}</strong>
        <span>{{ errorToast.detail }}</span>
      </span>
      <button type="button" class="error-dismiss" aria-label="关闭错误提示" @click="dismissError">×</button>
    </aside>

    <section class="trace-grid">
      <aside class="runs-panel">
        <div class="panel-header">
          <h2>请求列表</h2>
          <span>{{ loadingRuns ? "加载中" : `${runs.length} / ${total}` }}</span>
        </div>
        <div class="run-list">
          <button
            v-for="run in runs"
            :key="run.request_id"
            type="button"
            class="run-row"
            :class="{ selected: run.request_id === selectedRunId }"
            @click="selectRun(run)"
          >
            <span class="run-primary">
              <span class="status-dot" :class="statusClass(run.final_status)"></span>
              <strong>{{ previewText(run.user_message || "无用户问题", 46) }}</strong>
              <span>{{ formatLatency(run.latency_ms) }}</span>
            </span>
            <span class="row-foot">
              <span>{{ formatDate(run.created_at) }}</span>
              <span v-if="run.error_code">{{ run.error_code }}</span>
              <span v-else>{{ run.evidence_count || 0 }} 条证据</span>
            </span>
            <span class="run-id">{{ compactId(run.request_id) }}</span>
          </button>
        </div>
      </aside>

      <section class="detail-panel" ref="detailPanelRef" @scroll="handleDetailScroll">
        <div class="panel-header">
          <h2>节点详情</h2>
          <span v-if="loadingRun || loadingSpan">加载中</span>
          <span v-else-if="activeSpan">{{ activeSpan.span_type_label }}</span>
        </div>

        <div v-if="!detailSpans.length" class="empty-state">
          选择一次请求后查看连续调用详情。
        </div>

        <template v-else>
          <article
            v-for="span in detailSpans"
            :id="`span-detail-${span.span_id}`"
            :key="span.span_id"
            class="span-detail-card"
            :class="{ active: span.span_id === activeSpan?.span_id }"
            :data-span-id="span.span_id"
            :data-span-type="span.span_type"
          >
            <section class="detail-title">
              <div>
                <h3>{{ span.display_name }}</h3>
                <p>{{ span.name }} · {{ span.span_id }}</p>
              </div>
              <div class="detail-badges">
                <span class="status-pill" :class="statusClass(span.status)">
                  {{ statusLabel(span.status) }}
                </span>
                <span>{{ formatLatency(span.latency_ms) }}</span>
              </div>
            </section>

            <details v-if="span.error_code || span.error_message" class="error-box" open>
              <summary>
                <strong>{{ span.error_code || "执行失败" }}</strong>
              </summary>
              <p>{{ span.error_message || "该节点记录了异常状态。" }}</p>
              <JsonBlock
                v-if="span.exception_chain && span.exception_chain.length"
                title="异常链"
                :value="span.exception_chain"
                :open="false"
              />
            </details>

            <section v-if="isModelSpanFor(span)" class="model-section">
              <details class="info-block" open>
                <summary>模型信息</summary>
                <div class="kv-grid">
                  <span>Provider</span><strong>{{ modelIoBySpanId[span.span_id]?.provider || "-" }}</strong>
                  <span>Model</span><strong>{{ modelIoBySpanId[span.span_id]?.model || "-" }}</strong>
                  <span>Node</span><strong>{{ modelIoBySpanId[span.span_id]?.node || span.name }}</strong>
                  <span>Token</span><strong>{{ modelIoBySpanId[span.span_id]?.token_usage?.total_tokens ?? "-" }}</strong>
                </div>
              </details>

              <div v-if="modelIoBySpanId[span.span_id]" class="message-stack">
                <h4>模型输入 messages</h4>
                <details
                  v-for="(message, index) in normalizeMessages(modelIoBySpanId[span.span_id].input_messages)"
                  :key="index"
                  class="message-card"
                  open
                >
                  <summary>{{ messageRole(message) }}</summary>
                  <pre>{{ messageContent(message) }}</pre>
                </details>
              </div>
              <div v-else class="empty-state compact">
                <span v-if="loadingModelIoBySpanId[span.span_id]">模型 I/O 加载中。</span>
                <span v-else-if="modelIoErrorsBySpanId[span.span_id]">模型 I/O 加载失败：{{ modelIoErrorsBySpanId[span.span_id] }}</span>
                <span v-else>模型 I/O 尚未加载，滚动到该节点或点击右侧节点后加载。</span>
              </div>

              <JsonBlock v-if="modelIoBySpanId[span.span_id]" title="模型原始输出 raw_output" :value="modelIoBySpanId[span.span_id].raw_output" />
              <JsonBlock v-if="modelIoBySpanId[span.span_id]?.parsed_output" title="解析结果 parsed_output" :value="modelIoBySpanId[span.span_id].parsed_output" />
              <JsonBlock v-if="modelIoBySpanId[span.span_id]?.token_usage" title="Token Usage" :value="modelIoBySpanId[span.span_id].token_usage" :open="false" />
            </section>

            <section v-else-if="isToolSpanFor(span)" class="tool-section">
              <details class="info-block" open>
                <summary>工具信息</summary>
                <div class="kv-grid">
                  <span>工具</span><strong>{{ span.name }}</strong>
                  <span>状态</span><strong>{{ statusLabel(span.status) }}</strong>
                  <span>错误码</span><strong>{{ span.error_code || "-" }}</strong>
                  <span>耗时</span><strong>{{ formatLatency(span.latency_ms) }}</strong>
                </div>
              </details>
              <details v-if="collectUrls(span).length" class="url-list" open>
                <summary>证据 URL</summary>
                <div class="url-items">
                  <a v-for="url in collectUrls(span)" :key="url" :href="url" target="_blank" rel="noreferrer">{{ url }}</a>
                </div>
              </details>
              <JsonBlock title="工具输入摘要（非完整输入）" :value="span.input_summary" />
              <JsonBlock title="工具输出摘要（非完整输出）" :value="span.output_summary" />
              <JsonBlock v-if="diagnosticsFor(span)" title="Diagnostics" :value="diagnosticsFor(span)" />
            </section>

            <section v-else-if="isGuardSpanFor(span)" class="guard-section">
              <JsonBlock title="检查输入摘要" :value="span.input_summary" />
              <JsonBlock title="检查输出摘要" :value="span.output_summary" />
              <JsonBlock title="调试元数据" :value="span.metadata" />
            </section>

            <section v-else-if="isContextSpanFor(span)" class="context-section">
              <details class="info-block" open>
                <summary>上下文信息</summary>
                <div class="kv-grid">
                  <span>策略</span><strong>{{ span.output_summary?.strategy || "-" }}</strong>
                  <span>选中历史</span><strong>{{ span.output_summary?.selected_turn_count ?? "-" }}</strong>
                  <span>选中证据</span><strong>{{ span.output_summary?.selected_evidence_count ?? "-" }}</strong>
                  <span>依赖历史</span><strong>{{ span.output_summary?.depends_on_history ?? "-" }}</strong>
                </div>
              </details>
              <JsonBlock title="上下文输入摘要" :value="span.input_summary" />
              <JsonBlock title="上下文输出摘要" :value="span.output_summary" />
              <JsonBlock title="调试元数据" :value="span.metadata" />
            </section>

            <section v-else class="generic-section">
              <JsonBlock title="输入摘要" :value="span.input_summary" />
              <JsonBlock title="输出摘要" :value="span.output_summary" />
              <JsonBlock title="调试元数据" :value="span.metadata" />
            </section>

            <JsonBlock title="完整节点记录" :value="span" :open="false" />
          </article>
        </template>
      </section>

      <aside class="tree-panel">
        <div class="panel-header">
          <h2>调用链</h2>
          <span>{{ spanTree.length ? "完整链路" : "无数据" }}</span>
        </div>
        <div v-if="!spanTree.length" class="empty-state">
          {{ loadingRun ? "调用链加载中。" : "该请求没有 span 记录。" }}
        </div>
        <SpanTree
          v-else
          :nodes="spanTree"
          :selected-span-id="activeSpan?.span_id || ''"
          @select="selectSpan"
        />
      </aside>
    </section>
  </main>
</template>
