<script setup>
import { ArrowLeft, FileText, List as ListIcon, PanelLeft, PanelRight, Workflow } from "@lucide/vue";
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
const selectedSpan = ref(null);
const modelIoBySpanId = ref({});
const loadingModelIoBySpanId = ref({});
const modelIoErrorsBySpanId = ref({});
const loadingRuns = ref(false);
const loadingRun = ref(false);
const loadingSpan = ref(false);
const runsPanelCollapsed = ref(false);
const treePanelCollapsed = ref(false);
const mobileView = ref("runs");
const errorToastKey = ref(0);
const detailPanelRef = ref(null);
const treePanelRef = ref(null);

let detailScrollTimer = 0;
let treeScrollTimer = 0;
let programmaticDetailScroll = false;
let programmaticDetailScrollTimer = 0;
let errorToastTimer = 0;
let modelIoObserver = null;

const TREE_SCROLL_DELAY_MS = 120;
const TREE_SCROLL_SAFE_GAP = 72;
const DETAIL_PROGRAMMATIC_SCROLL_MS = 900;
const ERROR_TOAST_AUTO_DISMISS_MS = 5200;
const statusOptions = [
  { value: "all", label: "全部" },
  { value: "success", label: "成功" },
  { value: "error", label: "失败" },
  { value: "blocked", label: "已拦截" }
];

// Local-only demo data for previewing Trace Console states when the Trace API has no rows.
// Keep this aligned with the real agent graph_node flow so the navigator hierarchy stays representative.
const demoRuns = [
  {
    request_id: "demo-trace-success-001",
    thread_id: "demo-thread-mobile",
    user_message: "梳理最近两个月 OpenAI 时间线",
    final_status: "success",
    latency_ms: 64280,
    created_at: "2026-05-16T10:12:30+08:00",
    evidence_count: 6,
    is_demo: true
  },
  {
    request_id: "demo-trace-blocked-002",
    thread_id: "demo-thread-guard",
    user_message: "测试安全策略拦截路径",
    final_status: "blocked",
    latency_ms: 1210,
    created_at: "2026-05-16T10:08:10+08:00",
    evidence_count: 0,
    error_code: "POLICY_BLOCKED",
    is_demo: true
  }
];
const demoDetailsByRequestId = {
  "demo-trace-success-001": {
    run: demoRuns[0],
    spans: [
      {
        span_id: "demo-prepare-context",
        parent_span_id: null,
        span_type: "graph_node",
        span_type_label: "流程节点",
        display_name: "准备上下文",
        name: "prepare_context",
        status: "success",
        latency_ms: 1080,
        input_summary: { question: "梳理最近两个月 OpenAI 时间线" },
        output_summary: { context_strategy: "recent_context", selected_turn_count: 3 },
        metadata: { demo: true }
      },
      {
        span_id: "demo-context",
        parent_span_id: "demo-prepare-context",
        span_type: "context",
        span_type_label: "上下文整理",
        display_name: "生成上下文包",
        name: "context_pack_builder",
        status: "success",
        latency_ms: 18,
        input_summary: { recent_question: "OpenAI 最近两个月发生了什么？" },
        output_summary: {
          strategy: "recent_context",
          selected_turn_count: 3,
          selected_evidence_count: 2,
          depends_on_history: true
        },
        metadata: { demo: true }
      },
      {
        span_id: "demo-intent-router",
        parent_span_id: null,
        span_type: "graph_node",
        span_type_label: "流程节点",
        display_name: "判断问题类型",
        name: "intent_router",
        status: "success",
        latency_ms: 420,
        input_summary: { question: "梳理最近两个月 OpenAI 时间线" },
        output_summary: { route: "needs_tools", confidence: 0.92 },
        metadata: { demo: true }
      },
      {
        span_id: "demo-tool-selection",
        parent_span_id: null,
        span_type: "graph_node",
        span_type_label: "流程节点",
        display_name: "选择工具",
        name: "tool_selection",
        status: "success",
        latency_ms: 62,
        input_summary: { intent_route: "needs_tools" },
        output_summary: { selected_tools: ["search_news"] },
        metadata: { demo: true }
      },
      {
        span_id: "demo-tool-worker",
        parent_span_id: null,
        span_type: "graph_node",
        span_type_label: "流程节点",
        display_name: "规划工具调用",
        name: "tool_worker",
        status: "success",
        latency_ms: 760,
        input_summary: { selected_tools: ["search_news"] },
        output_summary: { pending_tool_count: 1 },
        metadata: { demo: true }
      },
      {
        span_id: "demo-tool-policy",
        parent_span_id: null,
        span_type: "graph_node",
        span_type_label: "流程节点",
        display_name: "工具策略检查",
        name: "tool_policy",
        status: "success",
        latency_ms: 90,
        input_summary: { pending_tool_count: 1 },
        output_summary: { allowed: true },
        metadata: { demo: true }
      },
      {
        span_id: "demo-tool-executor",
        parent_span_id: null,
        span_type: "graph_node",
        span_type_label: "流程节点",
        display_name: "执行工具",
        name: "tool_executor",
        status: "success",
        latency_ms: 1530,
        input_summary: { pending_tool_count: 1 },
        output_summary: { tool_result_count: 1, evidence_count: 6 },
        metadata: { demo: true }
      },
      {
        span_id: "demo-tool-news",
        parent_span_id: "demo-tool-executor",
        span_type: "tool_call",
        span_type_label: "工具执行",
        display_name: "工具执行：search_news",
        name: "search_news",
        status: "success",
        latency_ms: 1520,
        input_summary: { query: "OpenAI news May 2026 timeline" },
        output_summary: {
          result_count: 6,
          urls: [
            "https://example.com/openai-timeline",
            "https://example.com/ai-market-news"
          ]
        },
        metadata: { demo: true }
      },
      {
        span_id: "demo-evidence-normalizer",
        parent_span_id: null,
        span_type: "graph_node",
        span_type_label: "流程节点",
        display_name: "归一化证据",
        name: "evidence_normalizer",
        status: "success",
        latency_ms: 34,
        input_summary: { tool_result_count: 1 },
        output_summary: { evidence_count: 6, brief_chars: 420 },
        metadata: { demo: true }
      },
      {
        span_id: "demo-loop-decider",
        parent_span_id: null,
        span_type: "graph_node",
        span_type_label: "流程节点",
        display_name: "判断是否继续调用工具",
        name: "tool_loop_decider",
        status: "success",
        latency_ms: 12,
        input_summary: { evidence_count: 6, tool_round: 1 },
        output_summary: { next_step: "enough_evidence" },
        metadata: { demo: true }
      },
      {
        span_id: "demo-final-synthesizer",
        parent_span_id: null,
        span_type: "graph_node",
        span_type_label: "流程节点",
        display_name: "生成最终回答",
        name: "final_synthesizer",
        status: "success",
        latency_ms: 2270,
        input_summary: { evidence_count: 6 },
        output_summary: { final_text_chars: 860 },
        metadata: { demo: true }
      },
      {
        span_id: "demo-model-summary",
        parent_span_id: "demo-final-synthesizer",
        span_type: "model_call",
        span_type_label: "模型调用",
        display_name: "模型调用：总结时间线",
        name: "timeline_summary_model",
        status: "success",
        latency_ms: 2270,
        input_summary: { instruction: "按时间线合并新闻证据" },
        output_summary: { format: "timeline", bullets: 5 },
        metadata: { provider: "demo", model: "gpt-demo" }
      },
      {
        span_id: "demo-output-guard",
        parent_span_id: null,
        span_type: "graph_node",
        span_type_label: "流程节点",
        display_name: "输出检查",
        name: "output_guard",
        status: "success",
        latency_ms: 24,
        input_summary: { text_chars: 860, valid_url_count: 6 },
        output_summary: { guarded_text_chars: 858 },
        metadata: { demo: true }
      }
    ]
  },
  "demo-trace-blocked-002": {
    run: demoRuns[1],
    spans: [
      {
        span_id: "demo-block-guard",
        parent_span_id: null,
        span_type: "guard",
        span_type_label: "安全检查",
        display_name: "安全策略检查",
        name: "policy_guard",
        status: "blocked",
        latency_ms: 1210,
        error_code: "POLICY_BLOCKED",
        error_message: "演示数据：该请求被策略拦截。",
        input_summary: { category: "demo_policy" },
        output_summary: { decision: "blocked", reason: "演示拦截路径" },
        metadata: { demo: true }
      }
    ]
  }
};
const demoModelIoBySpanId = {
  "demo-model-summary": {
    provider: "demo",
    model: "gpt-demo",
    node: "timeline_summary_model",
    input_messages: [
      { role: "system", content: "你是新闻时间线整理助手。" },
      { role: "user", content: "请根据证据整理 OpenAI 最近两个月时间线。" }
    ],
    raw_output: {
      content: "演示输出：OpenAI 动态主要集中在产品更新、公司治理、市场竞争和安全讨论。"
    },
    parsed_output: {
      summary: "这是本地演示数据，用于查看 Trace Console 手机端交互。",
      timeline_items: 5
    },
    token_usage: {
      input_tokens: 186,
      output_tokens: 92,
      total_tokens: 278
    }
  }
};

const hasToken = computed(() => Boolean(token.value));
const selectedRun = computed(() => runDetail.value?.run || runs.value.find((item) => item.request_id === selectedRunId.value) || null);
const detailSpans = computed(() => runDetail.value?.spans || []);
const spanMap = computed(() => Object.fromEntries(detailSpans.value.map((span) => [span.span_id, span])));
const activeSpan = computed(() => {
  const selectedId = selectedSpan.value?.span_id;
  if (!selectedId) return null;
  return spanMap.value[selectedId] || selectedSpan.value;
});
const spanNavigatorSteps = computed(() => buildLogicalSteps(detailSpans.value));
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

function cloneDemo(value) {
  return JSON.parse(JSON.stringify(value));
}

function buildLogicalSteps(spans) {
  const entries = (spans || []).filter((span) => span?.span_id);
  const byId = new Map(entries.map((span) => [String(span.span_id), span]));

  const childrenByGraph = new Map();
  for (const span of entries) {
    if (span.span_type === "graph_node") continue;
    const ancestor = closestGraphNodeAncestor(span, byId);
    if (!ancestor) continue;
    const key = String(ancestor.span_id);
    if (!childrenByGraph.has(key)) childrenByGraph.set(key, []);
    childrenByGraph.get(key).push(span);
  }

  const repeatByName = new Map();
  const steps = [];
  for (const span of entries) {
    if (span.span_type !== "graph_node") continue;
    const children = childrenByGraph.get(String(span.span_id)) || [];
    const primary =
      children.find((c) => c.span_type === "model_call") ||
      children.find((c) => c.span_type === "tool_call") ||
      null;
    const secondary = children.filter((c) => c !== primary);
    let kind = "plumbing";
    if (primary?.span_type === "model_call") kind = "llm";
    else if (primary?.span_type === "tool_call") kind = "tool";
    const hasError =
      isErrorStatus(span.status) || children.some((c) => isErrorStatus(c.status));
    const anchor = primary || span;
    const repeatKey = span.name || span.span_id;
    const repeatIndex = (repeatByName.get(repeatKey) || 0) + 1;
    repeatByName.set(repeatKey, repeatIndex);
    steps.push({
      id: String(anchor.span_id),
      anchor,
      graphNode: span,
      primary,
      secondary,
      kind,
      hasError,
      title: navigatorStepTitle(span),
      status: span.status,
      latency_ms: span.latency_ms,
      repeat_index: repeatIndex,
      signal: computeStepSignal(span, primary, secondary, kind)
    });
  }
  return steps;
}

function isErrorStatus(status) {
  const value = String(status || "").toLowerCase();
  return value === "error" || value === "failed" || value === "blocked";
}

function computeStepSignal(graphNode, primary, secondary, kind) {
  const g = graphNode?.output_summary || {};
  if (g.intent_route) return `→ ${g.intent_route}`;
  if (g.next_step) return `→ ${g.next_step}`;
  if (kind === "tool" && primary) {
    const o = primary.output_summary || {};
    const n = o.result_count ?? o.evidence_count;
    if (typeof n === "number") return `命中 ${n}`;
  }
  for (const child of secondary || []) {
    const o = child.output_summary || {};
    if (typeof o.allowed === "boolean") return o.allowed ? "放行" : (o.reason || "拦截");
    if (typeof o.removed_unknown_url_count === "number") return `删 ${o.removed_unknown_url_count} URL`;
    if (typeof o.evidence_count === "number") return `证据 ${o.evidence_count}`;
  }
  return "";
}

function stepKindLabel(kind) {
  if (kind === "llm") return "模型调用";
  if (kind === "tool") return "工具执行";
  return "流程节点";
}

function childTypeLabel(child) {
  const map = { context: "上下文", postprocess: "后处理", guard: "检查", model_call: "模型", tool_call: "工具" };
  return map[child?.span_type] || child?.span_type || "子步骤";
}

function stepErrorCode(step) {
  const all = [step?.graphNode, step?.primary, ...(step?.secondary || [])].filter(Boolean);
  const errored = all.find((s) => isErrorStatus(s.status) || s.error_code);
  return errored?.error_code || "";
}

function stepErrorMessage(step) {
  const all = [step?.graphNode, step?.primary, ...(step?.secondary || [])].filter(Boolean);
  const errored = all.find((s) => s.error_message);
  return errored?.error_message || "";
}

function closestGraphNodeAncestor(span, byId) {
  let parentId = span?.parent_span_id ? String(span.parent_span_id) : "";
  const seen = new Set();
  while (parentId && !seen.has(parentId)) {
    seen.add(parentId);
    const parent = byId.get(parentId);
    if (!parent) return null;
    if (parent.span_type === "graph_node") return parent;
    parentId = parent.parent_span_id ? String(parent.parent_span_id) : "";
  }
  return null;
}

function navigatorStepTitle(span) {
  return span?.display_name || span?.name || span?.span_id || "执行步骤";
}

function setError(error) {
  if (error instanceof TraceApiError) {
    if (error.status === 401) {
      clearToken();
      authError.value = "Token 无效，请重新输入。";
      return;
    }
    showErrorToast(String(error.detail || error.message));
    return;
  }
  showErrorToast(error?.message || String(error));
}

function showErrorToast(message) {
  window.clearTimeout(errorToastTimer);
  errorMessage.value = String(message || "");
  errorToastKey.value += 1;
  errorToastTimer = window.setTimeout(dismissError, ERROR_TOAST_AUTO_DISMISS_MS);
}

function dismissError() {
  window.clearTimeout(errorToastTimer);
  errorToastTimer = 0;
  errorMessage.value = "";
}

function setStatusFilter(status) {
  if (filters.value.status === status) return;
  filters.value.status = status;
  void loadRuns();
}

function loadDemoRuns() {
  dismissError();
  disconnectModelIoObserver();
  runs.value = cloneDemo(demoRuns);
  total.value = demoRuns.length;
  selectedRunId.value = "";
  runDetail.value = null;
  selectedSpan.value = null;
  modelIoBySpanId.value = {};
  loadingModelIoBySpanId.value = {};
  modelIoErrorsBySpanId.value = {};
  mobileView.value = "runs";
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
  dismissError();
  sessionStorage.removeItem(STORAGE_KEY);
  token.value = "";
  meta.value = null;
  runs.value = [];
  total.value = 0;
  selectedRunId.value = "";
  runDetail.value = null;
  selectedSpan.value = null;
  modelIoBySpanId.value = {};
  loadingModelIoBySpanId.value = {};
  modelIoErrorsBySpanId.value = {};
  mobileView.value = "runs";
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
  dismissError();
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

async function selectRun(run, options = {}) {
  if (!run?.request_id) return;
  if (options.nextMobileView) {
    mobileView.value = options.nextMobileView;
  }
  disconnectModelIoObserver();
  selectedRunId.value = run.request_id;
  loadingRun.value = true;
  dismissError();
  runDetail.value = null;
  selectedSpan.value = null;
  modelIoBySpanId.value = {};
  loadingModelIoBySpanId.value = {};
  modelIoErrorsBySpanId.value = {};
  const demoDetail = demoDetailsByRequestId[run.request_id];
  if (demoDetail) {
    runDetail.value = cloneDemo(demoDetail);
    modelIoBySpanId.value = cloneDemo(demoModelIoBySpanId);
    const defaultSpan = findDefaultSpan(demoDetail.spans || []);
    await nextTick();
    setupModelIoObserver();
    if (defaultSpan) {
      await selectSpan(defaultSpan, { skipDetailScroll: true });
    }
    loadingRun.value = false;
    return;
  }
  try {
    const payload = await client().run(run.request_id);
    runDetail.value = payload;
    const defaultSpan = findDefaultSpan(payload.spans || []);
    await nextTick();
    setupModelIoObserver();
    if (defaultSpan) {
      await selectSpan(defaultSpan, { skipDetailScroll: true });
    }
  } catch (error) {
    setError(error);
  } finally {
    loadingRun.value = false;
  }
}

function findDefaultSpan(spans) {
  const steps = buildLogicalSteps(spans);
  if (!steps.length) return null;
  const errored = steps.find((step) => step.hasError);
  return (errored || steps[0]).anchor;
}

async function selectSpan(span, options = {}) {
  if (!span?.span_id || !selectedRunId.value) return;
  if (options.nextMobileView) {
    mobileView.value = options.nextMobileView;
  }
  if (!options.skipDetailScroll) {
    pauseDetailScrollSpy();
  }
  selectedSpan.value = spanMap.value[span.span_id] || span;
  loadingSpan.value = span.span_type === "model_call" && !modelIoBySpanId.value[span.span_id];
  dismissError();
  try {
    if (span.span_type === "model_call") {
      await ensureModelIo(span);
    }
  } catch (error) {
    setError(error);
  } finally {
    loadingSpan.value = false;
    await nextTick();
    if (!options.skipDetailScroll) {
      scrollSpanCardIntoView(span.span_id);
    }
    queueSelectedTreeNodeIntoView();
  }
}

async function ensureModelIo(span, options = {}) {
  const spanId = span?.span_id;
  if (!spanId || span?.span_type !== "model_call" || !selectedRunId.value) return;
  if (demoModelIoBySpanId[spanId]) {
    modelIoBySpanId.value = { ...modelIoBySpanId.value, [spanId]: cloneDemo(demoModelIoBySpanId[spanId]) };
    return;
  }
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
  pauseDetailScrollSpy();
  card.scrollIntoView({ block: "start", inline: "nearest", behavior: "smooth" });
}

function pauseDetailScrollSpy() {
  window.clearTimeout(detailScrollTimer);
  window.clearTimeout(programmaticDetailScrollTimer);
  programmaticDetailScroll = true;
  programmaticDetailScrollTimer = window.setTimeout(() => {
    programmaticDetailScroll = false;
  }, DETAIL_PROGRAMMATIC_SCROLL_MS);
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
  nextTick(queueSelectedTreeNodeIntoView);
}

function queueSelectedTreeNodeIntoView() {
  window.clearTimeout(treeScrollTimer);
  treeScrollTimer = window.setTimeout(scrollSelectedTreeNodeIntoView, TREE_SCROLL_DELAY_MS);
}

function scrollSelectedTreeNodeIntoView() {
  if (treePanelCollapsed.value) return;
  const panel = treePanelRef.value;
  const selectedNode = panel?.querySelector(".span-node.selected");
  if (!panel || !selectedNode) return;

  const panelRect = panel.getBoundingClientRect();
  const nodeRect = selectedNode.getBoundingClientRect();
  const safeTop = panelRect.top + TREE_SCROLL_SAFE_GAP;
  const safeBottom = panelRect.bottom - TREE_SCROLL_SAFE_GAP;
  const isInsideSafeArea = nodeRect.top >= safeTop && nodeRect.bottom <= safeBottom;
  if (isInsideSafeArea) return;

  selectedNode.scrollIntoView({
    block: "nearest",
    inline: "nearest",
    behavior: "smooth"
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

function modelOutputContent(span) {
  const io = modelIoBySpanId.value[span?.span_id];
  const raw = io?.raw_output;
  if (raw && typeof raw === "object" && !Array.isArray(raw) && "content" in raw) {
    return raw.content;
  }
  return raw ?? null;
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
  window.clearTimeout(treeScrollTimer);
  window.clearTimeout(programmaticDetailScrollTimer);
  window.clearTimeout(errorToastTimer);
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

  <main
    v-else
    class="trace-shell"
    :class="{
      'runs-collapsed': runsPanelCollapsed,
      'tree-collapsed': treePanelCollapsed,
      'mobile-view-runs': mobileView === 'runs',
      'mobile-view-detail': mobileView === 'detail',
      'mobile-view-tree': mobileView === 'tree'
    }"
  >
    <header class="trace-topbar">
      <div class="brand-block">
        <div>
          <h1>Trace Console</h1>
        </div>
      </div>

      <form class="top-filterbar" @submit.prevent="loadRuns">
        <span class="filter-count">共 {{ total }} 条</span>
        <div class="status-filter" role="group" aria-label="状态筛选">
          <span class="status-filter-label">状态</span>
          <span class="status-options">
            <button
              v-for="option in statusOptions"
              :key="option.value"
              type="button"
              class="status-option"
              :class="{ active: filters.status === option.value }"
              :aria-pressed="filters.status === option.value"
              @click="setStatusFilter(option.value)"
            >
              {{ option.label }}
            </button>
          </span>
        </div>
        <input
          v-model="filters.q"
          type="search"
          placeholder="request_id / thread_id / 用户问题 / error_code"
        />
        <button type="submit" class="toolbar-button">筛选</button>
      </form>

      <div class="top-actions">
        <button type="button" class="toolbar-button" @click="loadRuns">刷新</button>
        <button type="button" class="toolbar-button secondary" @click="clearToken">退出</button>
      </div>
    </header>

    <section v-if="mobileView !== 'runs'" class="mobile-workspace-nav">
      <button type="button" class="mobile-back-button" aria-label="返回请求列表" title="返回请求列表" @click="mobileView = 'runs'">
        <ArrowLeft class="panel-icon" aria-hidden="true" />
      </button>
      <div class="mobile-run-context">
        <strong>{{ previewText(selectedRun?.user_message || "当前请求", 24) }}</strong>
        <span v-if="selectedRun">{{ formatDate(selectedRun.created_at) }} · {{ statusLabel(selectedRun.final_status) }}</span>
      </div>
    </section>

    <nav v-if="mobileView !== 'runs'" class="mobile-workspace-tabs" aria-label="当前请求视图切换">
      <button
        type="button"
        :class="{ active: mobileView === 'detail' }"
        @click="mobileView = 'detail'"
      >
        <FileText class="panel-icon" aria-hidden="true" />
        <span>详情</span>
      </button>
      <button
        type="button"
        :class="{ active: mobileView === 'tree' }"
        @click="mobileView = 'tree'"
      >
        <Workflow class="panel-icon" aria-hidden="true" />
        <span>调用链</span>
      </button>
    </nav>

    <aside v-if="errorToast" :key="errorToastKey" class="global-error" role="status" aria-live="polite">
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
          <h2>
            <ListIcon class="panel-icon" aria-hidden="true" />
            <span>请求列表</span>
          </h2>
          <button
            type="button"
            class="panel-toggle panel-toggle-left"
            :aria-label="runsPanelCollapsed ? '展开请求列表' : '折叠请求列表'"
            :title="runsPanelCollapsed ? '展开请求列表' : '折叠请求列表'"
            @click="runsPanelCollapsed = !runsPanelCollapsed"
          >
            <PanelLeft class="panel-icon" aria-hidden="true" />
          </button>
        </div>
        <button
          type="button"
          class="collapsed-panel-button runs-collapsed-button"
          aria-label="展开请求列表"
          title="展开请求列表"
          @click="runsPanelCollapsed = false"
        >
          <ListIcon class="panel-icon" aria-hidden="true" />
        </button>
        <div class="run-list">
          <div v-if="!runs.length" class="empty-state runs-empty-state">
            <p>{{ loadingRuns ? "请求列表加载中。" : "暂无匹配请求。" }}</p>
            <button v-if="!loadingRuns" type="button" class="empty-action" @click="loadDemoRuns">
              查看演示数据
            </button>
          </div>
          <button
            v-for="run in runs"
            :key="run.request_id"
            type="button"
            class="run-row"
            :class="{ selected: run.request_id === selectedRunId }"
            @click="selectRun(run, { nextMobileView: 'detail' })"
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
          <h2>
            <FileText class="panel-icon" aria-hidden="true" />
            <span>节点详情</span>
          </h2>
          <span v-if="loadingRun || loadingSpan">加载中</span>
          <span v-else-if="activeSpan">{{ activeSpan.span_type_label }}</span>
        </div>

        <section v-if="selectedRun" class="request-overview">
          <div class="request-overview-head">
            <div>
              <p>请求概览</p>
              <h3>{{ selectedRun.user_message || "无用户问题" }}</h3>
            </div>
            <span class="status-pill" :class="statusClass(selectedRun.final_status)">
              {{ statusLabel(selectedRun.final_status) }}
            </span>
          </div>
          <div class="request-overview-grid">
            <span>创建时间</span><strong>{{ formatDate(selectedRun.created_at) }}</strong>
            <span>总耗时</span><strong>{{ formatLatency(selectedRun.latency_ms) }}</strong>
            <span>证据数</span><strong>{{ selectedRun.evidence_count ?? 0 }}</strong>
            <span>错误码</span><strong>{{ selectedRun.error_code || "-" }}</strong>
            <span>request_id</span><strong>{{ selectedRun.request_id || "-" }}</strong>
            <span>thread_id</span><strong>{{ selectedRun.thread_id || "-" }}</strong>
          </div>
        </section>

        <div v-if="!detailSpans.length" class="empty-state">
          选择一次请求后查看连续调用详情。
        </div>

        <template v-else>
          <template v-for="step in spanNavigatorSteps" :key="step.id">
            <details
              v-if="step.kind === 'plumbing' && !step.hasError"
              :id="`span-detail-${step.id}`"
              class="span-detail-card plumbing"
              :class="{ active: step.id === activeSpan?.span_id }"
              :data-span-id="step.anchor.span_id"
              :data-span-type="step.anchor.span_type"
            >
              <summary class="plumbing-summary">
                <span class="plumbing-name">{{ step.title }}</span>
                <span v-if="step.repeat_index > 1" class="span-repeat">第 {{ step.repeat_index }} 次</span>
                <span v-if="step.signal" class="plumbing-signal">{{ step.signal }}</span>
                <span class="plumbing-latency">{{ formatLatency(step.latency_ms) }}</span>
              </summary>
              <JsonBlock title="输出摘要" :value="step.graphNode.output_summary" :open="false" />
              <JsonBlock
                v-for="child in step.secondary"
                :key="child.span_id"
                :title="`${childTypeLabel(child)} · 输出摘要`"
                :value="child.output_summary"
                :open="false"
              />
            </details>

            <article
              v-else
              :id="`span-detail-${step.id}`"
              class="span-detail-card"
              :class="[`span-kind-${step.kind}`, { active: step.id === activeSpan?.span_id }]"
              :data-span-id="step.anchor.span_id"
              :data-span-type="step.anchor.span_type"
            >
              <section class="detail-title">
                <div>
                  <h3>{{ step.title }}</h3>
                  <p>
                    {{ stepKindLabel(step.kind) }}
                    <span v-if="step.signal"> · {{ step.signal }}</span>
                    <span v-if="step.repeat_index > 1"> · 第 {{ step.repeat_index }} 次</span>
                  </p>
                </div>
                <div class="detail-badges">
                  <span class="status-pill" :class="statusClass(step.status)">
                    {{ statusLabel(step.status) }}
                  </span>
                  <span>{{ formatLatency(step.latency_ms) }}</span>
                </div>
              </section>

              <details v-if="step.hasError" class="error-box" open>
                <summary>
                  <strong>{{ stepErrorCode(step) || "执行失败" }}</strong>
                </summary>
                <p>{{ stepErrorMessage(step) || "该步骤记录了异常状态。" }}</p>
              </details>

              <section v-if="step.kind === 'llm' && step.primary" class="model-section">
                <JsonBlock
                  v-if="modelIoBySpanId[step.primary.span_id]"
                  title="模型输出 · 决策"
                  :value="modelOutputContent(step.primary)"
                />
                <div v-else class="empty-state compact">
                  <span v-if="loadingModelIoBySpanId[step.primary.span_id]">模型 I/O 加载中。</span>
                  <span v-else-if="modelIoErrorsBySpanId[step.primary.span_id]">模型 I/O 加载失败：{{ modelIoErrorsBySpanId[step.primary.span_id] }}</span>
                  <span v-else>模型 I/O 尚未加载，滚动到该节点或点击右侧节点后加载。</span>
                </div>

                <details class="info-block">
                  <summary>模型信息</summary>
                  <div class="kv-grid">
                    <span>Provider</span><strong>{{ modelIoBySpanId[step.primary.span_id]?.provider || "-" }}</strong>
                    <span>Model</span><strong>{{ modelIoBySpanId[step.primary.span_id]?.model || "-" }}</strong>
                    <span>Node</span><strong>{{ modelIoBySpanId[step.primary.span_id]?.node || step.graphNode.name }}</strong>
                    <span>Token</span><strong>{{ modelIoBySpanId[step.primary.span_id]?.token_usage?.total_tokens ?? "-" }}</strong>
                  </div>
                </details>

                <details v-if="modelIoBySpanId[step.primary.span_id]" class="info-block">
                  <summary>模型输入 messages</summary>
                  <div class="message-stack">
                    <details
                      v-for="(message, index) in normalizeMessages(modelIoBySpanId[step.primary.span_id].input_messages)"
                      :key="index"
                      class="message-card"
                    >
                      <summary>{{ messageRole(message) }}</summary>
                      <pre>{{ messageContent(message) }}</pre>
                    </details>
                  </div>
                </details>

                <JsonBlock v-if="modelIoBySpanId[step.primary.span_id]" title="模型原始输出 raw_output" :value="modelIoBySpanId[step.primary.span_id].raw_output" :open="false" />
                <JsonBlock v-if="modelIoBySpanId[step.primary.span_id]?.parsed_output" title="解析结果 parsed_output" :value="modelIoBySpanId[step.primary.span_id].parsed_output" :open="false" />
                <JsonBlock v-if="modelIoBySpanId[step.primary.span_id]?.token_usage" title="Token Usage" :value="modelIoBySpanId[step.primary.span_id].token_usage" :open="false" />
              </section>

              <section v-else-if="step.kind === 'tool' && step.primary" class="tool-section">
                <details class="info-block" open>
                  <summary>工具信息</summary>
                  <div class="kv-grid">
                    <span>工具</span><strong>{{ step.primary.name }}</strong>
                    <span>状态</span><strong>{{ statusLabel(step.primary.status) }}</strong>
                    <span>错误码</span><strong>{{ step.primary.error_code || "-" }}</strong>
                    <span>耗时</span><strong>{{ formatLatency(step.primary.latency_ms) }}</strong>
                  </div>
                </details>
                <details v-if="collectUrls(step.primary).length" class="url-list" open>
                  <summary>证据 URL</summary>
                  <div class="url-items">
                    <a v-for="url in collectUrls(step.primary)" :key="url" :href="url" target="_blank" rel="noreferrer">{{ url }}</a>
                  </div>
                </details>
                <JsonBlock title="工具输入摘要" :value="step.primary.input_summary" />
                <JsonBlock title="工具输出摘要" :value="step.primary.output_summary" />
                <JsonBlock v-if="diagnosticsFor(step.primary)" title="Diagnostics" :value="diagnosticsFor(step.primary)" />
              </section>

              <section v-else class="generic-section">
                <JsonBlock title="输出摘要" :value="step.graphNode.output_summary" />
              </section>

              <details v-if="step.secondary.length" class="info-block">
                <summary>子步骤（{{ step.secondary.length }}）</summary>
                <JsonBlock
                  v-for="child in step.secondary"
                  :key="child.span_id"
                  :title="`${childTypeLabel(child)} · ${child.name}`"
                  :value="child.output_summary"
                  :open="false"
                />
              </details>
            </article>
          </template>
        </template>
      </section>

      <aside class="tree-panel" ref="treePanelRef">
        <div class="panel-header">
          <h2>
            <Workflow class="panel-icon" aria-hidden="true" />
            <span>调用链</span>
          </h2>
          <button
            type="button"
            class="panel-toggle panel-toggle-right"
            :aria-label="treePanelCollapsed ? '展开调用链' : '折叠调用链'"
            :title="treePanelCollapsed ? '展开调用链' : '折叠调用链'"
            @click="treePanelCollapsed = !treePanelCollapsed"
          >
            <PanelRight class="panel-icon" aria-hidden="true" />
          </button>
        </div>
        <button
          type="button"
          class="collapsed-panel-button tree-collapsed-button"
          aria-label="展开调用链"
          title="展开调用链"
          @click="treePanelCollapsed = false"
        >
          <Workflow class="panel-icon" aria-hidden="true" />
        </button>
        <div v-if="!spanNavigatorSteps.length" class="empty-state">
          {{ loadingRun ? "调用链加载中。" : "该请求没有 span 记录。" }}
        </div>
        <SpanTree
          v-else
          :steps="spanNavigatorSteps"
          :selected-span-id="activeSpan?.span_id || ''"
          @select="(step) => selectSpan(step.anchor, { nextMobileView: 'detail' })"
        />
      </aside>
    </section>
  </main>
</template>
