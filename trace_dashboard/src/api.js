export class TraceApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = "TraceApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function createTraceClient(token) {
  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${token}`,
        ...(options.headers || {})
      }
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      const detail = typeof payload === "object" && payload !== null ? payload.detail : payload;
      throw new TraceApiError(`Trace API failed: ${response.status}`, response.status, detail);
    }
    return payload;
  }

  return {
    meta: () => request("/trace-api/meta"),
    runs: (params) => request(`/trace-api/runs?${params.toString()}`),
    run: (requestId) => request(`/trace-api/runs/${encodeURIComponent(requestId)}`),
    span: (spanId, requestId) => {
      const params = new URLSearchParams();
      if (requestId) params.set("request_id", requestId);
      const suffix = params.toString() ? `?${params.toString()}` : "";
      return request(`/trace-api/spans/${encodeURIComponent(spanId)}${suffix}`);
    },
    modelIo: (spanId, requestId) => {
      const params = new URLSearchParams();
      if (requestId) params.set("request_id", requestId);
      const suffix = params.toString() ? `?${params.toString()}` : "";
      return request(`/trace-api/spans/${encodeURIComponent(spanId)}/model-io${suffix}`);
    }
  };
}
