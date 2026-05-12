<script setup>
import { computed, nextTick, onMounted, ref } from "vue";
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
const spanDetail = ref(null);
const modelIo = ref(null);
const loadingRuns = ref(false);
const loadingRun = ref(false);
const loadingSpan = ref(false);

const hasToken = computed(() => Boolean(token.value));
const selectedRun = computed(() => runDetail.value?.run || runs.value.find((item) => item.request_id === selectedRunId.value) || null);
const activeSpan = computed(() => spanDetail.value?.span || selectedSpan.value);
const activeLangSmith = computed(() => selectedRun.value?.langsmith || meta.value?.langsmith || {});

const isModelSpan = computed(() => activeSpan.value?.span_type === "model_call");
const isToolSpan = computed(() => activeSpan.value?.span_type === "tool_call");
const isGuardSpan = computed(() => ["guard", "postprocess"].includes(activeSpan.value?.span_type));
const isContextSpan = computed(() => activeSpan.value?.span_type === "context");

const evidenceUrls = computed(() => collectUrls(activeSpan.value));
const diagnostics = computed(() => {
  const span = activeSpan.value || {};
  return (
    span.metadata?.diagnostics ||
    span.output_summary?.diagnostics ||
    span.input_summary?.diagnostics ||
    null
  );
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
  sessionStorage.removeItem(STORAGE_KEY);
  token.value = "";
  meta.value = null;
  runs.value = [];
  total.value = 0;
  selectedRunId.value = "";
  runDetail.value = null;
  spanTree.value = [];
  selectedSpan.value = null;
  spanDetail.value = null;
  modelIo.value = null;
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
  selectedRunId.value = run.request_id;
  loadingRun.value = true;
  errorMessage.value = "";
  runDetail.value = null;
  spanTree.value = [];
  selectedSpan.value = null;
  spanDetail.value = null;
  modelIo.value = null;
  try {
    const payload = await client().run(run.request_id);
    runDetail.value = payload;
    spanTree.value = payload.span_tree || [];
    const defaultSpan = findDefaultSpan(payload.spans || []);
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
  selectedSpan.value = span;
  spanDetail.value = null;
  modelIo.value = null;
  loadingSpan.value = true;
  errorMessage.value = "";
  try {
    spanDetail.value = await client().span(span.span_id, selectedRunId.value);
    if (span.span_type === "model_call") {
      const payload = await client().modelIo(span.span_id, selectedRunId.value);
      modelIo.value = payload.model_io;
    }
  } catch (error) {
    setError(error);
  } finally {
    loadingSpan.value = false;
    await nextTick();
    scrollDetailToTop();
    scrollSelectedTreeNodeIntoView();
  }
}

function scrollDetailToTop() {
  document.querySelector(".detail-panel")?.scrollTo({ top: 0 });
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

onMounted(() => {
  if (hasToken.value) {
    loadInitialData();
  }
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

    <p v-if="errorMessage" class="global-error">{{ errorMessage }}</p>

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

      <section class="detail-panel">
        <div class="panel-header">
          <h2>节点详情</h2>
          <span v-if="loadingRun || loadingSpan">加载中</span>
          <span v-else-if="activeSpan">{{ activeSpan.span_type_label }}</span>
        </div>

        <div v-if="!activeSpan" class="empty-state">
          选择一次请求和调用链节点后查看详情。
        </div>

        <template v-else>
          <section class="detail-title">
            <div>
              <h3>{{ activeSpan.display_name }}</h3>
              <p>{{ activeSpan.name }} · {{ activeSpan.span_id }}</p>
            </div>
            <div class="detail-badges">
              <span class="status-pill" :class="statusClass(activeSpan.status)">
                {{ statusLabel(activeSpan.status) }}
              </span>
              <span>{{ formatLatency(activeSpan.latency_ms) }}</span>
            </div>
          </section>

          <details v-if="activeSpan.error_code || activeSpan.error_message" class="error-box" open>
            <summary>
              <strong>{{ activeSpan.error_code || "执行失败" }}</strong>
            </summary>
            <p>{{ activeSpan.error_message || "该节点记录了异常状态。" }}</p>
            <JsonBlock
              v-if="activeSpan.exception_chain && activeSpan.exception_chain.length"
              title="异常链"
              :value="activeSpan.exception_chain"
              :open="false"
            />
          </details>

          <section v-if="isModelSpan" class="model-section">
            <details class="info-block" open>
              <summary>模型信息</summary>
              <div class="kv-grid">
                <span>Provider</span><strong>{{ modelIo?.provider || "-" }}</strong>
                <span>Model</span><strong>{{ modelIo?.model || "-" }}</strong>
                <span>Node</span><strong>{{ modelIo?.node || activeSpan.name }}</strong>
                <span>Token</span><strong>{{ modelIo?.token_usage?.total_tokens ?? "-" }}</strong>
              </div>
            </details>

            <div v-if="modelIo" class="message-stack">
              <h4>模型输入 messages</h4>
              <details v-for="(message, index) in normalizeMessages(modelIo.input_messages)" :key="index" class="message-card" open>
                <summary>{{ messageRole(message) }}</summary>
                <pre>{{ messageContent(message) }}</pre>
              </details>
            </div>
            <div v-else class="empty-state compact">模型 I/O 加载中或不存在。</div>

            <JsonBlock v-if="modelIo" title="模型原始输出 raw_output" :value="modelIo.raw_output" />
            <JsonBlock v-if="modelIo?.parsed_output" title="解析结果 parsed_output" :value="modelIo.parsed_output" />
            <JsonBlock v-if="modelIo?.token_usage" title="Token Usage" :value="modelIo.token_usage" :open="false" />
          </section>

          <section v-else-if="isToolSpan" class="tool-section">
            <details class="info-block" open>
              <summary>工具信息</summary>
              <div class="kv-grid">
                <span>工具</span><strong>{{ activeSpan.name }}</strong>
                <span>状态</span><strong>{{ statusLabel(activeSpan.status) }}</strong>
                <span>错误码</span><strong>{{ activeSpan.error_code || "-" }}</strong>
                <span>耗时</span><strong>{{ formatLatency(activeSpan.latency_ms) }}</strong>
              </div>
            </details>
            <details v-if="evidenceUrls.length" class="url-list" open>
              <summary>证据 URL</summary>
              <div class="url-items">
                <a v-for="url in evidenceUrls" :key="url" :href="url" target="_blank" rel="noreferrer">{{ url }}</a>
              </div>
            </details>
            <JsonBlock title="工具输入摘要（非完整输入）" :value="activeSpan.input_summary" />
            <JsonBlock title="工具输出摘要（非完整输出）" :value="activeSpan.output_summary" />
            <JsonBlock v-if="diagnostics" title="Diagnostics" :value="diagnostics" />
          </section>

          <section v-else-if="isGuardSpan" class="guard-section">
            <JsonBlock title="检查输入摘要" :value="activeSpan.input_summary" />
            <JsonBlock title="检查输出摘要" :value="activeSpan.output_summary" />
            <JsonBlock title="调试元数据" :value="activeSpan.metadata" />
          </section>

          <section v-else-if="isContextSpan" class="context-section">
            <details class="info-block" open>
              <summary>上下文信息</summary>
              <div class="kv-grid">
                <span>策略</span><strong>{{ activeSpan.output_summary?.strategy || "-" }}</strong>
                <span>选中历史</span><strong>{{ activeSpan.output_summary?.selected_turn_count ?? "-" }}</strong>
                <span>选中证据</span><strong>{{ activeSpan.output_summary?.selected_evidence_count ?? "-" }}</strong>
                <span>依赖历史</span><strong>{{ activeSpan.output_summary?.depends_on_history ?? "-" }}</strong>
              </div>
            </details>
            <JsonBlock title="上下文输入摘要" :value="activeSpan.input_summary" />
            <JsonBlock title="上下文输出摘要" :value="activeSpan.output_summary" />
            <JsonBlock title="调试元数据" :value="activeSpan.metadata" />
          </section>

          <section v-else class="generic-section">
            <JsonBlock title="输入摘要" :value="activeSpan.input_summary" />
            <JsonBlock title="输出摘要" :value="activeSpan.output_summary" />
            <JsonBlock title="调试元数据" :value="activeSpan.metadata" />
          </section>

          <JsonBlock title="完整节点记录" :value="activeSpan" :open="false" />
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
