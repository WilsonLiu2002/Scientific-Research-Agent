const state = { result: null, experiment: null, activeEvidence: null, reviewThread: null, review: null, requestController: null, stopRequested: false };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const pipelineStages = [
  ["Plan research", "Goal and search queries"],
  ["Search literature", "MCP / OpenAlex"],
  ["Retrieve evidence", "Hybrid RAG"],
  ["Read full text", "Selected papers"],
  ["Synthesize", "Evidence context"],
  ["Verify grounding", "Claim entailment"],
];

document.addEventListener("DOMContentLoaded", () => {
  renderPipeline(-1);
  drawIdleMap();
  bindEvents();
  loadSystemStatus();
  if (window.lucide) window.lucide.createIcons();
});

async function loadSystemStatus() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    const llm = health.llm_provider ? `${health.llm_provider} · ${health.llm_model}` : "local inference";
    $("#system-status").innerHTML = `<span class="status-dot"></span>MCP connected · ${escapeHtml(llm)}`;
  } catch {
    $("#system-status").innerHTML = '<span class="status-dot"></span>Local service';
  }
}

function bindEvents() {
  $("#research-form").addEventListener("submit", runResearch);
  $("#stop-run").addEventListener("click", stopActiveResearch);
  $$("[data-question]").forEach((button) => button.addEventListener("click", () => {
    $("#question").value = button.dataset.question;
    $("#question").focus();
  }));
  $$("[data-tab]").forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tab)));
  $("#theme-toggle").addEventListener("click", toggleTheme);
  $("#copy-report").addEventListener("click", copyReport);
  $("#experiment-demo").addEventListener("click", runExperimentDemo);
  $("#screening-form").addEventListener("submit", runCandidateScreen);
  $("#review-approve").addEventListener("click", () => submitReview({ action: "approve" }));
  $("#review-edit").addEventListener("click", submitReviewEdits);
  $("#review-instruct").addEventListener("click", submitReviewInstruction);
  $("#review-skip").addEventListener("click", () => submitReview({ action: "skip" }));
  $("#review-stop").addEventListener("click", () => {
    if (state.requestController) stopActiveResearch();
    else submitReview({ action: "stop" });
  });
  $("#review-dialog").addEventListener("cancel", (event) => event.preventDefault());
  window.addEventListener("resize", () => { state.result ? drawEvidenceMap() : drawIdleMap(); if (state.experiment) drawExperimentChart(); });
}

async function runCandidateScreen(event) {
  event.preventDefault();
  const parseElements = (value) => value.split(/[,\s]+/).map((item) => item.trim()).filter(Boolean);
  const request = {
    min_band_gap_ev: Number($("#gap-min").value),
    max_band_gap_ev: Number($("#gap-max").value),
    include_elements: parseElements($("#include-elements").value),
    exclude_elements: parseElements($("#exclude-elements").value),
    top_k: Number($("#candidate-limit").value),
  };
  const button = $("#screening-form .screen-button");
  button.disabled = true;
  try {
    const response = await fetch("/api/materials/screen", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail?.[0]?.msg || result.detail || "Screening failed");
    $("#screen-match-count").textContent = result.matched_count.toLocaleString();
    $("#screen-dataset-size").textContent = result.dataset_size.toLocaleString();
    $("#candidate-rows").innerHTML = result.candidates.map((item) => `<tr><td>${item.rank}</td><td>${escapeHtml(item.formula)}</td><td>${item.band_gap_ev.toFixed(3)} eV</td><td>${item.score.toFixed(3)}</td><td>${escapeHtml(item.value_type)} · Matbench</td></tr>`).join("") || '<tr><td colspan="5">No candidates match these constraints.</td></tr>';
    $("#material-rag-context").textContent = result.rag_context;
    showToast(`${result.matched_count} measured candidates matched`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function runExperimentDemo() {
  const button = $("#experiment-demo");
  button.disabled = true;
  try {
    const response = await fetch("/api/demo/experiment");
    if (!response.ok) throw new Error("Local experiment demo failed");
    state.experiment = await response.json();
    renderExperiment(state.experiment);
    $("#workspace").classList.add("has-results");
    $("#empty-state").classList.add("hidden");
    activateTab("experiment");
    showToast("Synthetic experiment generated locally");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function runResearch(event) {
  event.preventDefault();
  const question = $("#question").value.trim();
  if (question.length < 8) return;
  state.reviewThread = crypto.randomUUID().replaceAll("-", "");
  state.stopRequested = false;
  state.requestController = new AbortController();
  setLoading(true);
  animatePipeline();
  try {
    const response = await fetch("/api/research/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: state.reviewThread }),
      signal: state.requestController.signal,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Research run failed");
    handleReviewSession(payload);
  } catch (error) {
    renderPipeline(-1);
    if (error.name !== "AbortError") showToast(error.message, true);
  } finally {
    state.requestController = null;
    setLoading(false);
  }
}

async function stopActiveResearch() {
  if (!state.reviewThread) return;
  state.stopRequested = true;
  state.requestController?.abort();
  const threadId = state.reviewThread;
  setGenerationActive(false);
  clearInterval(window.pipelineTimer);
  renderPipeline(-1);
  $("#run-id").textContent = "STOPPED";
  if ($("#review-dialog").open) $("#review-dialog").close();
  showToast("Stop requested. No further workflow steps will start.");
  try {
    await fetch(`/api/research/sessions/${encodeURIComponent(threadId)}/cancel`, {
      method: "POST",
      keepalive: true,
    });
  } catch {
    // The local request may already have closed after the browser abort.
  }
}

function handleReviewSession(payload) {
  if (state.stopRequested) return;
  state.reviewThread = payload.thread_id;
  if (payload.status === "review_required" && payload.review) {
    state.review = payload.review;
    renderReview(payload.review);
    setLoading(false);
    return;
  }
  state.review = null;
  state.result = payload.state;
  if ($("#review-dialog").open) $("#review-dialog").close();
  renderResult(payload.state);
  setLoading(false);
  showToast(payload.state.cancelled ? "Research stopped" : "Research run complete");
}

function renderReview(review) {
  const isSearch = review.review_type === "search_plan";
  $("#review-stage").textContent = isSearch ? "Checkpoint · before external search" : "Checkpoint · before full-text access";
  $("#review-title").textContent = isSearch ? "Review the search strategy" : "Choose papers for deep reading";
  $("#review-purpose").textContent = isSearch
    ? review.research_intent?.objective || review.question
    : `The agent proposes full-text retrieval for ${(review.papers || []).length} paper(s).`;
  $("#review-current").innerHTML = isSearch
    ? (review.queries || []).map((query, index) => `<label class="review-query"><span>${String(index + 1).padStart(2, "0")}</span><input data-review-query value="${escapeHtml(query)}" aria-label="Search query ${index + 1}"></label>`).join("")
    : (review.papers || []).map((paper) => `<label class="review-paper"><input type="checkbox" data-review-paper value="${escapeHtml(paper.id)}" checked><span><strong>${escapeHtml(paper.title)}</strong><small>${escapeHtml(paper.doi || paper.id)}</small></span></label>`).join("");
  const instructions = review.reviewer_instructions || [];
  $("#review-history").textContent = instructions.length ? `Active guidance: ${instructions.join(" · ")}` : "No reviewer guidance added yet.";
  $("#review-instruction").value = "";
  $("#review-skip").classList.toggle("hidden", isSearch);
  $("#review-edit").innerHTML = isSearch ? '<i data-lucide="pencil"></i>Apply query edits' : '<i data-lucide="list-checks"></i>Use selection';
  $("#run-id").textContent = `REVIEW-${state.reviewThread.slice(0, 6).toUpperCase()}`;
  clearInterval(window.pipelineTimer);
  if (!$("#review-dialog").open) $("#review-dialog").showModal();
  if (window.lucide) window.lucide.createIcons();
}

function submitReviewEdits() {
  if (state.review?.review_type === "search_plan") {
    const queries = $$('[data-review-query]').map((input) => input.value.trim()).filter(Boolean);
    if (!queries.length) return showToast("Keep at least one search query", true);
    submitReview({ action: "edit", queries });
    return;
  }
  const paperIds = $$('[data-review-paper]:checked').map((input) => input.value);
  submitReview({ action: "edit", paper_ids: paperIds });
}

function submitReviewInstruction() {
  const instruction = $("#review-instruction").value.trim();
  if (!instruction) return showToast("Enter an instruction first", true);
  submitReview({ action: "add_instruction", instruction });
}

async function submitReview(decision) {
  if (!state.reviewThread) return;
  setReviewBusy(true);
  state.stopRequested = false;
  state.requestController = new AbortController();
  setGenerationActive(true);
  animatePipeline();
  try {
    const response = await fetch(`/api/research/sessions/${encodeURIComponent(state.reviewThread)}/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(decision),
      signal: state.requestController.signal,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Could not resume research");
    handleReviewSession(payload);
  } catch (error) {
    clearInterval(window.pipelineTimer);
    if (error.name !== "AbortError") showToast(error.message, true);
  } finally {
    state.requestController = null;
    setGenerationActive(false);
    setReviewBusy(false);
  }
}

function setReviewBusy(busy) {
  $$(".review-button").forEach((button) => { button.disabled = busy && button.id !== "review-stop"; });
  $("#review-approve").innerHTML = busy ? '<i data-lucide="loader-circle"></i>Working…' : '<i data-lucide="check"></i>Approve';
  if (window.lucide) window.lucide.createIcons();
}

function setLoading(loading) {
  setGenerationActive(loading);
  const button = $(".run-button");
  button.querySelector("span").textContent = loading ? "Researching…" : "Run research";
  if (loading) {
    $("#run-id").textContent = "RUNNING";
    $("#workspace").classList.remove("has-results");
    $("#empty-state").classList.remove("hidden");
  }
}

function setGenerationActive(active) {
  $(".run-button").disabled = active;
  $("#stop-run").classList.toggle("hidden", !active);
}

function animatePipeline() {
  let index = 0;
  renderPipeline(index);
  clearInterval(window.pipelineTimer);
  window.pipelineTimer = setInterval(() => {
    index = Math.min(index + 1, pipelineStages.length - 1);
    renderPipeline(index);
  }, 1200);
}

function renderPipeline(activeIndex, completed = false) {
  $("#pipeline").innerHTML = pipelineStages.map(([name, detail], index) => {
    const mode = completed || index < activeIndex ? "done" : index === activeIndex ? "active" : "";
    return `<div class="pipeline-step ${mode}">${escapeHtml(name)}<small>${escapeHtml(detail)}</small></div>`;
  }).join("");
}

function renderResult(result) {
  clearInterval(window.pipelineTimer);
  renderPipeline(-1, true);
  $("#workspace").classList.add("has-results");
  $("#empty-state").classList.add("hidden");
  $("#run-id").textContent = `RUN-${Date.now().toString().slice(-6)}`;
  $("#metric-coverage").textContent = `${Math.round((result.evidence_coverage || 0) * 100)}%`;
  $("#metric-papers").textContent = (result.retrieved_papers || []).length;
  $("#metric-evidence").textContent = (result.retrieved_passages || []).length;
  $("#metric-iterations").textContent = result.search_iteration || 1;
  const progress = Math.round((result.goal?.progress || 0) * 100);
  $("#goal-percent").textContent = `${progress}%`;
  $("#goal-progress").style.width = `${progress}%`;
  $("#goal-objective").textContent = result.goal?.objective || result.question;
  renderReport(result.answer || "No answer generated.");
  renderEvidence(result.retrieved_passages || []);
  renderPapers(result.retrieved_papers || result.papers || []);
  renderTrace(result);
  activateTab("report");
  if (window.lucide) window.lucide.createIcons();
}

function renderReport(answer) {
  const headings = new Set(["Overview", "Main Approaches", "Applications / Predicted Properties", "Key Observations", "Limitations", "References"]);
  const html = answer.split("\n").map((raw) => {
    const line = raw.trim();
    if (!line) return "";
    if (headings.has(line)) return `<h3>${escapeHtml(line)}</h3>`;
    const withCitations = escapeHtml(line).replace(/\[(P\d+)\]/g, '<button class="citation" data-citation="$1">[$1]</button>');
    return `<p class="${line.startsWith("[P") ? "reference" : ""}">${withCitations}</p>`;
  }).join("");
  $("#report").innerHTML = html;
  $$(".citation").forEach((button) => button.addEventListener("click", () => focusEvidence(button.dataset.citation)));
}

function renderEvidence(passages) {
  $("#evidence-list").innerHTML = passages.map((passage, index) => `
    <article class="evidence-item" data-label="P${index + 1}">
      <header><strong>P${index + 1}</strong><span>${escapeHtml(passage.section || "Abstract")}${passage.page ? ` · p.${passage.page}` : ""}</span></header>
      <p>${escapeHtml(passage.text)}</p>
    </article>`).join("") || '<div class="evidence-item"><p>No evidence passages were selected.</p></div>';
  $$(".evidence-item[data-label]").forEach((item) => item.addEventListener("click", () => {
    state.activeEvidence = item.dataset.label;
    $$(".evidence-item").forEach((node) => node.classList.toggle("selected", node === item));
    drawEvidenceMap();
  }));
  requestAnimationFrame(drawEvidenceMap);
}

function renderPapers(papers) {
  $("#paper-count").textContent = `${papers.length} sources`;
  $("#paper-list").innerHTML = papers.map((paper, index) => `
    <article class="paper">
      <div class="paper-top"><span class="paper-index">${String(index + 1).padStart(2, "0")}</span><span class="paper-source">${escapeHtml(paper.source || "source")}</span></div>
      <h3>${escapeHtml(paper.title)}</h3>
      <p>${escapeHtml(paper.abstract || "No abstract available.")}</p>
      <footer><span>${paper.year || "n.d."}</span><span>${escapeHtml(paper.venue || "Unspecified venue")}</span>${paper.url ? `<a href="${safeUrl(paper.url)}" target="_blank" rel="noreferrer">Open source</a>` : ""}</footer>
    </article>`).join("");
}

function renderTrace(result) {
  const tasks = result.subtasks || [];
  const critiques = result.critique_history || [];
  const items = [
    { time: "PLAN", title: `Purpose: ${result.research_intent?.purpose || "research"}`, detail: result.research_intent?.objective || `${(result.search_queries || []).length} search queries generated`, status: result.goal?.status || "complete" },
    ...tasks.map((task) => ({ time: `ITER ${task.iteration}`, title: task.objective, detail: `${task.result_count || 0} results${task.error ? ` · ${task.error}` : ""}`, status: task.status })),
    ...critiques.map((critique) => ({ time: `CRITIC ${critique.iteration}`, title: `Evidence verdict: ${critique.verdict}`, detail: critique.reason, status: `${Math.round(critique.confidence * 100)}% confidence` })),
    { time: "CONTEXT", title: "Evidence packet assembled", detail: `${result.context_stats?.estimated_tokens || 0} estimated tokens from ${result.context_stats?.distinct_papers || 0} papers`, status: "completed" },
    { time: "VERIFY", title: "Claims checked against cited evidence", detail: `${(result.grounding_verdicts || []).length} verdicts · ${(result.grounding_warnings || []).length} repairs`, status: "completed" },
  ];
  $("#trace-list").innerHTML = items.map((item) => `
    <div class="trace-item"><time>${escapeHtml(item.time)}</time><span class="trace-node"></span><div><h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.detail)}</p></div><span class="trace-status">${escapeHtml(item.status)}</span></div>`).join("");
  renderLocalTelemetry(result.run_trace, result.run_metrics);
}

function renderLocalTelemetry(trace, metrics = {}) {
  if (!trace) {
    $("#trace-correlation").textContent = "No local trace captured";
    $("#telemetry-metrics").innerHTML = "";
    $("#telemetry-list").innerHTML = '<p class="telemetry-empty">Run research to generate a local trace.</p>';
    return;
  }
  $("#trace-correlation").textContent = `RUN ${trace.run_id.slice(0, 8)} · TRACE ${trace.trace_id.slice(0, 8)}`;
  const values = [
    ["Duration", `${Math.round(metrics.total_duration_ms || 0)} ms`],
    ["LLM calls", metrics.llm_calls || 0],
    ["Tool calls", metrics.tool_calls || 0],
    ["Errors", metrics.errors || 0],
    ["Input tokens", metrics.input_tokens || 0],
    ["Output tokens", metrics.output_tokens || 0],
  ];
  $("#telemetry-metrics").innerHTML = values.map(([label, value]) => `<div><span>${escapeHtml(value)}</span><small>${escapeHtml(label)}</small></div>`).join("");
  const parents = new Map((trace.events || []).filter((event) => event.event === "operation.started").map((event) => [event.span_id, event.parent_span_id]));
  const depthOf = (spanId) => { let depth = 0; let parent = parents.get(spanId); while (parent && depth < 4) { depth += 1; parent = parents.get(parent); } return depth; };
  const visible = (trace.events || []).filter((event) => event.event !== "operation.started");
  $("#telemetry-list").innerHTML = visible.map((event) => {
    const title = event.name || event.tool || event.operation || event.event;
    const detail = event.decision ? `branch → ${event.decision}` : event.result_count !== undefined ? `${event.result_count} result(s)` : event.error_type || event.provider_status || event.status || "event";
    const duration = event.duration_ms !== undefined ? `${event.duration_ms.toFixed(1)} ms` : "";
    return `<div class="telemetry-event severity-${escapeHtml(event.severity.toLowerCase())}" style="--trace-depth:${depthOf(event.span_id)}"><code>${escapeHtml(event.event)}</code><div><strong>${escapeHtml(title)}</strong><small>${escapeHtml(detail)}</small></div><time>${escapeHtml(duration)}</time></div>`;
  }).join("") || '<p class="telemetry-empty">No operation events recorded.</p>';
}

function renderExperiment(experiment) {
  $("#experiment-id").textContent = `${experiment.id} · seed ${experiment.seed}`;
  $("#experiment-title").textContent = experiment.title;
  $("#experiment-summary").textContent = experiment.summary;
  const dataset = experiment.dataset;
  $("#experiment-dataset").innerHTML = [
    ["Samples", dataset.samples.toLocaleString()], ["Train / test", `${dataset.train_samples} / ${dataset.test_samples}`],
    ["Repeated splits", dataset.splits], ["Target", dataset.target],
  ].map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  $("#experiment-rows").innerHTML = experiment.models.map((model) => `
    <tr><td><i class="model-key" style="background:${model.color}"></i>${escapeHtml(model.name)}</td><td>${model.mean_mae.toFixed(4)} eV/atom</td><td>±${model.std_mae.toFixed(4)}</td><td>${model.runtime_seconds.toFixed(1)} s</td><td>${model.mae_runs.map((value) => value.toFixed(3)).join(" · ")}</td></tr>`).join("");
  $("#experiment-provenance").textContent = `Synthetic data notice: ${experiment.provenance}`;
  requestAnimationFrame(drawExperimentChart);
}

function drawExperimentChart() {
  const canvas = $("#experiment-canvas");
  if (!canvas || !state.experiment) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, rect.width * ratio); canvas.height = Math.max(1, rect.height * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.clearRect(0, 0, rect.width, rect.height);
  const styles = getComputedStyle(document.documentElement);
  const ink = styles.getPropertyValue("--ink").trim();
  const line = styles.getPropertyValue("--line").trim();
  const models = state.experiment.models;
  const margin = { left: 128, right: 44, top: 22, bottom: 34 };
  const width = rect.width - margin.left - margin.right;
  const height = rect.height - margin.top - margin.bottom;
  const max = Math.max(...models.map((model) => model.mean_mae)) * 1.18;
  ctx.font = "10px SFMono-Regular, monospace"; ctx.fillStyle = ink;
  [0, .25, .5, .75, 1].forEach((tick) => {
    const x = margin.left + width * tick;
    ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + height); ctx.strokeStyle = line; ctx.stroke();
    ctx.fillText((max * tick).toFixed(3), x - 13, rect.height - 8);
  });
  models.forEach((model, index) => {
    const rowHeight = height / models.length;
    const y = margin.top + index * rowHeight + rowHeight * .25;
    const barHeight = rowHeight * .48;
    ctx.fillStyle = ink; ctx.textAlign = "right"; ctx.fillText(model.name, margin.left - 12, y + barHeight * .68);
    ctx.fillStyle = model.color; ctx.fillRect(margin.left, y, width * model.mean_mae / max, barHeight);
    ctx.textAlign = "left"; ctx.fillStyle = ink; ctx.fillText(model.mean_mae.toFixed(4), margin.left + width * model.mean_mae / max + 8, y + barHeight * .68);
  });
  ctx.textAlign = "left";
}

function activateTab(tab) {
  if (tab === "screening") {
    $("#workspace").classList.add("has-results");
    $("#empty-state").classList.add("hidden");
  }
  $$("[data-tab]").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  $$(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${tab}`));
  if (tab === "evidence") requestAnimationFrame(drawEvidenceMap);
  if (tab === "experiment") requestAnimationFrame(drawExperimentChart);
}

function focusEvidence(label) {
  state.activeEvidence = label;
  activateTab("evidence");
  $$(".evidence-item").forEach((item) => item.classList.toggle("selected", item.dataset.label === label));
  const selected = $(`.evidence-item[data-label="${label}"]`);
  if (selected) selected.scrollIntoView({ behavior: "smooth", block: "nearest" });
  drawEvidenceMap();
}

function drawEvidenceMap() {
  const canvas = $("#evidence-map");
  if (!canvas || !state.result) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, rect.width * ratio);
  canvas.height = Math.max(1, rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const styles = getComputedStyle(document.documentElement);
  const ink = styles.getPropertyValue("--ink").trim();
  const line = styles.getPropertyValue("--line").trim();
  const green = styles.getPropertyValue("--green").trim();
  const coral = styles.getPropertyValue("--coral").trim();
  const blue = styles.getPropertyValue("--blue").trim();
  const passages = state.result.retrieved_passages || [];
  const width = rect.width;
  const height = rect.height;
  const center = { x: width * .48, y: height * .5 };
  ctx.clearRect(0, 0, width, height);
  ctx.font = "10px SFMono-Regular, monospace";
  passages.forEach((passage, index) => {
    const angle = (Math.PI * 2 * index / Math.max(passages.length, 1)) - Math.PI / 2;
    const radius = Math.min(width, height) * (.28 + (index % 3) * .045);
    const node = { x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius };
    const active = state.activeEvidence === `P${index + 1}`;
    ctx.beginPath(); ctx.moveTo(center.x, center.y); ctx.lineTo(node.x, node.y); ctx.strokeStyle = active ? green : line; ctx.lineWidth = active ? 2 : 1; ctx.stroke();
    ctx.beginPath(); ctx.arc(node.x, node.y, active ? 8 : 5, 0, Math.PI * 2); ctx.fillStyle = passage.section || passage.page ? coral : blue; ctx.fill();
    ctx.fillStyle = ink; ctx.fillText(`P${index + 1}`, node.x + 10, node.y + 3);
  });
  ctx.beginPath(); ctx.arc(center.x, center.y, 28, 0, Math.PI * 2); ctx.fillStyle = green; ctx.fill();
  ctx.fillStyle = styles.getPropertyValue("--surface").trim(); ctx.textAlign = "center"; ctx.fillText("ANSWER", center.x, center.y + 3); ctx.textAlign = "left";
}

function drawIdleMap() {
  const canvas = $("#idle-canvas");
  if (!canvas || state.result) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, rect.width * ratio); canvas.height = Math.max(1, rect.height * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.clearRect(0, 0, rect.width, rect.height);
  const color = getComputedStyle(document.documentElement).getPropertyValue("--line").trim();
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--green").trim();
  const nodes = Array.from({ length: 18 }, (_, i) => ({ x: ((i * 137) % 89) / 89 * rect.width, y: ((i * 73) % 61) / 61 * rect.height }));
  nodes.forEach((node, i) => nodes.slice(i + 1).forEach((other) => {
    if (Math.hypot(node.x - other.x, node.y - other.y) < 150) { ctx.beginPath(); ctx.moveTo(node.x, node.y); ctx.lineTo(other.x, other.y); ctx.strokeStyle = color; ctx.stroke(); }
  }));
  nodes.forEach((node, i) => { ctx.beginPath(); ctx.arc(node.x, node.y, i % 5 === 0 ? 5 : 2.5, 0, Math.PI * 2); ctx.fillStyle = i % 5 === 0 ? accent : color; ctx.fill(); });
}

function toggleTheme() {
  const dark = document.documentElement.dataset.theme === "dark";
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  $("#theme-toggle").innerHTML = `<i data-lucide="${dark ? "sun" : "moon"}"></i>`;
  if (window.lucide) window.lucide.createIcons();
  state.result ? drawEvidenceMap() : drawIdleMap();
  if (state.experiment) drawExperimentChart();
}

async function copyReport() {
  await navigator.clipboard.writeText(state.result?.answer || "");
  showToast("Report copied");
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.style.background = error ? "var(--coral)" : "var(--ink)";
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2800);
}

function safeUrl(value) {
  try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) ? url.href : "#"; } catch { return "#"; }
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]);
}
