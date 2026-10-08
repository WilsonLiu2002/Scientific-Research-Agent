const state = { result: null, experiment: null, activeEvidence: null, reviewThread: null, review: null, requestController: null, stopRequested: false, liveTrace: null, traceCursor: 0, traceTimer: null, tracePolling: false, projects: [], chats: [], activeProject: null, activeChat: null };
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
  renderAccessSummary();
  loadSystemStatus();
  if (window.innerWidth > 1100) document.body.classList.add("memory-open");
  loadWorkspaceMemory();
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
  $("#memory-toggle").addEventListener("click", () => document.body.classList.toggle("memory-open"));
  $("#new-project").addEventListener("click", () => { $("#project-create-form").classList.toggle("hidden"); $("#project-name").focus(); });
  $("#project-create-form").addEventListener("submit", createProject);
  $("#new-chat").addEventListener("click", createChat);
  $("#research-form").addEventListener("submit", runResearch);
  $("#access-token").addEventListener("change", () => {
    renderAccessSummary();
    state.activeProject = null;
    state.activeChat = null;
    loadWorkspaceMemory();
  });
  $("#stop-run").addEventListener("click", stopActiveResearch);
  $$("[data-tab]").forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tab)));
  $("#theme-toggle").addEventListener("click", toggleTheme);
  $("#copy-report").addEventListener("click", copyReport);
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

async function loadWorkspaceMemory() {
  try {
    const response = await memoryFetch("/api/projects");
    if (!response.ok) throw new Error("Could not load projects");
    state.projects = await response.json();
    renderProjects();
    const remembered = localStorage.getItem("research-project-id");
    const project = state.projects.find((item) => item.id === remembered) || state.projects[0];
    if (project) await selectProject(project.id, true);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function createProject(event) {
  event.preventDefault();
  const name = $("#project-name").value.trim();
  if (!name) return;
  const response = await memoryFetch("/api/projects", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
  if (!response.ok) return showToast("Could not create project", true);
  const project = await response.json();
  $("#project-name").value = "";
  $("#project-create-form").classList.add("hidden");
  await loadWorkspaceMemory();
  await selectProject(project.id, false);
}

async function selectProject(projectId, restoreChat = false) {
  state.activeProject = state.projects.find((item) => item.id === projectId) || null;
  if (!state.activeProject) return;
  localStorage.setItem("research-project-id", projectId);
  $("#active-project-name").textContent = state.activeProject.name;
  renderProjects();
  const response = await memoryFetch(`/api/projects/${encodeURIComponent(projectId)}/chats`);
  state.chats = response.ok ? await response.json() : [];
  renderChats();
  const remembered = localStorage.getItem(`research-chat-${projectId}`);
  const chat = restoreChat ? state.chats.find((item) => item.id === remembered) || state.chats[0] : null;
  if (chat) await openChat(chat.id);
  else if (!state.chats.length) await createChat();
}

async function createChat() {
  if (!state.activeProject) return;
  const response = await memoryFetch(`/api/projects/${encodeURIComponent(state.activeProject.id)}/chats`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: "New chat" }) });
  if (!response.ok) return showToast("Could not create chat", true);
  const chat = await response.json();
  const [listResponse, projectsResponse] = await Promise.all([
    memoryFetch(`/api/projects/${encodeURIComponent(state.activeProject.id)}/chats`),
    memoryFetch("/api/projects"),
  ]);
  state.chats = listResponse.ok ? await listResponse.json() : [chat, ...state.chats];
  if (projectsResponse.ok) {
    state.projects = await projectsResponse.json();
    state.activeProject = state.projects.find((item) => item.id === state.activeProject.id) || state.activeProject;
    renderProjects();
  }
  renderChats();
  await openChat(chat.id);
}

async function ensureActiveChat() {
  if (!state.activeProject) await loadWorkspaceMemory();
  if (!state.activeChat) await createChat();
  if (!state.activeProject || !state.activeChat) throw new Error("Select a project and chat first");
}

async function openChat(chatId) {
  const response = await memoryFetch(`/api/chats/${encodeURIComponent(chatId)}`);
  if (!response.ok) return showToast("Could not open chat", true);
  state.activeChat = await response.json();
  state.reviewThread = state.activeChat.active_thread_id || null;
  localStorage.setItem(`research-chat-${state.activeChat.project_id}`, chatId);
  renderChats();
  renderConversation(state.activeChat);
  const latestResult = [...state.activeChat.messages].reverse().find((message) => message.result)?.result;
  if (latestResult) {
    state.result = latestResult;
    state.liveTrace = latestResult.run_trace || null;
    renderResult(latestResult);
    renderConversation(state.activeChat);
  } else {
    $("#workspace").classList.add("has-results");
    $("#empty-state").classList.add("hidden");
  }
  activateTab("chat");
  if (state.activeChat.status === "review_required" && state.activeChat.active_thread_id) {
    restoreReviewSession(state.activeChat.active_thread_id);
  }
}

async function restoreReviewSession(threadId) {
  const response = await memoryFetch(`/api/research/sessions/${encodeURIComponent(threadId)}`);
  if (!response.ok) return;
  const payload = await response.json();
  if (payload.status === "review_required") handleReviewSession(payload);
}

function renderProjects() {
  $("#project-list").innerHTML = state.projects.map((project) => `<button class="project-row ${project.id === state.activeProject?.id ? "active" : ""}" type="button" data-project-id="${project.id}"><i data-lucide="folder"></i><span><strong>${escapeHtml(project.name)}</strong><small>${accessLevelLabel(project.minimum_access_level)} · ${project.chat_count || 0} chats · ${project.message_count || 0} messages</small></span></button>`).join("");
  $$('[data-project-id]').forEach((button) => button.addEventListener("click", () => selectProject(button.dataset.projectId, true)));
  if (window.lucide) window.lucide.createIcons();
}

function renderChats() {
  $("#chat-list").innerHTML = state.chats.map((chat) => `<button class="chat-row ${chat.id === state.activeChat?.id ? "active" : ""}" type="button" data-chat-id="${chat.id}"><i data-lucide="message-square"></i><span><strong>${escapeHtml(chat.title)}</strong><small>${accessLevelLabel(chat.minimum_access_level)} · ${escapeHtml(chat.status)} · ${chat.message_count || 0} messages</small></span></button>`).join("") || '<p class="telemetry-empty">No chats available at this access level.</p>';
  $$('[data-chat-id]').forEach((button) => button.addEventListener("click", () => openChat(button.dataset.chatId)));
  if (window.lucide) window.lucide.createIcons();
}

function renderConversation(chat) {
  $("#chat-panel-title").textContent = chat.title;
  $("#chat-status").textContent = chat.status === "review_required" ? "Paused for review" : "Saved locally";
  $("#conversation").innerHTML = chat.messages.map((message) => `<article class="message ${message.role}"><header><strong>${message.role === "user" ? "Question" : "Research response"}</strong><time>${formatMemoryTime(message.created_at)}</time></header><p>${escapeHtml(message.content)}</p>${message.result ? `<button class="message-result-button" type="button" data-result-message="${message.id}"><i data-lucide="file-search"></i>Open full result</button>` : ""}</article>`).join("") || '<p class="telemetry-empty">Ask a question to begin this chat.</p>';
  $$('[data-result-message]').forEach((button) => button.addEventListener("click", () => {
    const message = chat.messages.find((item) => item.id === button.dataset.resultMessage);
    if (!message?.result) return;
    state.result = message.result;
    renderResult(message.result);
    activateTab("report");
  }));
  if (window.lucide) window.lucide.createIcons();
}

function formatMemoryTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function accessLevelLabel(level) {
  return ({ local_reader: "Local", researcher: "Researcher", principal_investigator: "PI" })[level] || "Local";
}

async function refreshWorkspaceMemory() {
  if (!state.activeProject || !state.activeChat) return;
  const [projectsResponse, chatsResponse, chatResponse] = await Promise.all([
    memoryFetch("/api/projects"),
    memoryFetch(`/api/projects/${encodeURIComponent(state.activeProject.id)}/chats`),
    memoryFetch(`/api/chats/${encodeURIComponent(state.activeChat.id)}`),
  ]);
  if (projectsResponse.ok) state.projects = await projectsResponse.json();
  if (chatsResponse.ok) state.chats = await chatsResponse.json();
  if (chatResponse.ok) state.activeChat = await chatResponse.json();
  renderProjects();
  renderChats();
  renderConversation(state.activeChat);
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
    const response = await fetch("/api/materials/screen", { method: "POST", headers: { "Content-Type": "application/json", "X-Simulated-Authorization": selectedAccessToken() }, body: JSON.stringify(request) });
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

async function runResearch(event) {
  event.preventDefault();
  const question = $("#question").value.trim();
  if (question.length < 8) return;
  try {
    await ensureActiveChat();
  } catch (error) {
    return showToast(error.message, true);
  }
  state.reviewThread = crypto.randomUUID().replaceAll("-", "");
  state.stopRequested = false;
  state.requestController = new AbortController();
  setLoading(true);
  startLiveTracePolling(true);
  renderPipeline(0);
  try {
    const response = await fetch("/api/research/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: state.reviewThread, access_token: selectedAccessToken(), project_id: state.activeProject.id, chat_id: state.activeChat.id }),
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
  stopLiveTracePolling("cancelling");
  renderPipeline(-1);
  $("#run-id").textContent = "STOPPED";
  if ($("#review-dialog").open) $("#review-dialog").close();
  showToast("Stop requested. No further workflow steps will start.");
  try {
    await memoryFetch(`/api/research/sessions/${encodeURIComponent(threadId)}/cancel`, {
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
  if (payload.project_id && state.activeProject?.id !== payload.project_id) {
    state.activeProject = state.projects.find((item) => item.id === payload.project_id) || state.activeProject;
  }
  if (payload.status === "review_required" && payload.review) {
    stopLiveTracePolling("review_required");
    state.review = payload.review;
    renderReview(payload.review);
    setLoading(false);
    refreshWorkspaceMemory();
    return;
  }
  state.review = null;
  stopLiveTracePolling(payload.status || "completed");
  state.result = payload.state;
  if ($("#review-dialog").open) $("#review-dialog").close();
  renderResult(payload.state);
  refreshWorkspaceMemory();
  setLoading(false);
  showToast(payload.state.cancelled ? "Research stopped" : "Research run complete");
}

function renderReview(review) {
  const isSearch = review.review_type === "search_plan";
  const plannedTools = (review.tool_plan?.choices || []).filter((tool) => tool.selected).map((tool) => tool.tool_id);
  $("#review-stage").textContent = isSearch ? "Checkpoint · before external search" : "Checkpoint · before full-text access";
  $("#review-title").textContent = isSearch ? "Review the search strategy" : "Choose papers for deep reading";
  $("#review-purpose").textContent = isSearch
    ? `${review.research_intent?.objective || review.question}${plannedTools.length ? ` Selected tools: ${plannedTools.join(", ")}.` : ""}`
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
  const pendingReview = state.review;
  setReviewBusy(true);
  if ($("#review-dialog").open) $("#review-dialog").close();
  state.stopRequested = false;
  state.requestController = new AbortController();
  setGenerationActive(true);
  startLiveTracePolling(false);
  renderPipelineFromTrace(state.liveTrace?.events || []);
  showToast(decision.action === "stop" ? "Stopping research…" : "Research resumed");
  try {
    const response = await memoryFetch(`/api/research/sessions/${encodeURIComponent(state.reviewThread)}/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Simulated-Authorization": selectedAccessToken() },
      body: JSON.stringify(decision),
      signal: state.requestController.signal,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Could not resume research");
    handleReviewSession(payload);
  } catch (error) {
    if (error.name !== "AbortError") {
      showToast(error.message, true);
      if (pendingReview && !state.stopRequested) renderReview(pendingReview);
    }
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
  }
}

function startLiveTracePolling(reset) {
  if (reset) {
    state.liveTrace = { events: [], metrics: {} };
    state.traceCursor = 0;
    $("#trace-list").innerHTML = '<p class="telemetry-empty">Waiting for the first workflow event…</p>';
    renderLocalTelemetry(null);
  }
  clearInterval(state.traceTimer);
  $("#workspace").classList.add("has-results");
  $("#empty-state").classList.add("hidden");
  $("#trace-badge").textContent = "Live";
  $("#trace-badge").classList.add("is-live");
  activateTab("trace");
  pollLiveTrace();
  state.traceTimer = setInterval(pollLiveTrace, 600);
}

function stopLiveTracePolling(status) {
  clearInterval(state.traceTimer);
  state.traceTimer = null;
  const live = status === "running" || status === "cancelling";
  $("#trace-badge").textContent = status === "review_required" ? "Paused for review" : live ? "Stopping" : "Stored locally";
  $("#trace-badge").classList.toggle("is-live", live);
  pollLiveTrace();
}

async function pollLiveTrace() {
  if (!state.reviewThread || state.tracePolling) return;
  state.tracePolling = true;
  try {
    const response = await memoryFetch(`/api/research/sessions/${encodeURIComponent(state.reviewThread)}/trace?after=${state.traceCursor}`);
    if (!response.ok) return;
    const update = await response.json();
    const events = [...(state.liveTrace?.events || []), ...(update.events || [])];
    state.traceCursor = update.event_count || events.length;
    state.liveTrace = { ...update, events, metrics: update.metrics || state.liveTrace?.metrics || {} };
    renderLiveWorkflow(events);
    renderPipelineFromTrace(events);
    renderLocalTelemetry(state.liveTrace, state.liveTrace.metrics);
    if (["completed", "cancelled", "failed"].includes(update.status)) {
      clearInterval(state.traceTimer);
      state.traceTimer = null;
      $("#trace-badge").textContent = "Stored locally";
      $("#trace-badge").classList.remove("is-live");
    }
  } catch {
    // Polling is best-effort and the next interval retries automatically.
  } finally {
    state.tracePolling = false;
  }
}

function renderLiveWorkflow(events) {
  const operations = new Map();
  events.forEach((event) => {
    if (event.event === "operation.started") {
      operations.set(event.span_id, { ...event, status: "running" });
    } else if (event.event.startsWith("operation.")) {
      const prior = operations.get(event.span_id) || event;
      operations.set(event.span_id, { ...prior, ...event, status: event.status || event.event.replace("operation.", "") });
    }
  });
  const visible = [...operations.values()].slice(-12);
  $("#trace-list").innerHTML = visible.map((item) => {
    const name = String(item.name || item.operation || "workflow operation").replace(/^langgraph\.(node|route)\./, "").replaceAll("_", " ");
    const duration = item.duration_ms !== undefined ? `${Math.round(item.duration_ms)} ms` : "in progress";
    return `<div class="trace-item"><time>${escapeHtml(item.kind || "operation")}</time><span class="trace-node ${item.status === "running" ? "running" : ""}"></span><div><h3>${escapeHtml(name)}</h3><p>${escapeHtml(duration)}</p></div><span class="trace-status">${escapeHtml(item.status)}</span></div>`;
  }).join("") || '<p class="telemetry-empty">Waiting for the first workflow event…</p>';
}

function setGenerationActive(active) {
  $(".run-button").disabled = active;
  $("#stop-run").classList.toggle("hidden", !active);
}

function selectedAccessToken() {
  return $("#access-token").value;
}

function memoryFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Simulated-Authorization", selectedAccessToken());
  return fetch(url, { ...options, headers });
}

function renderAccessSummary() {
  const descriptions = {
    "sim-local-reader": "Local Reader · embedded literature RAG only",
    "sim-researcher": "Researcher · local RAG + OpenAlex + materials DB",
    "sim-principal-investigator": "Principal Investigator · local RAG + OpenAlex + full text",
  };
  $("#access-summary").textContent = descriptions[selectedAccessToken()];
  if (!$('.run-button').disabled) renderPipeline(-1);
}

function renderPipeline(activeIndex, completed = false) {
  const statuses = pipelineStages.map((_, index) => completed ? "done" : index === activeIndex ? "active" : index < activeIndex ? "done" : "pending");
  renderPipelineStatuses(statuses);
}

function renderPipelineFromTrace(events) {
  const stageByOperation = [
    [/langgraph\.node\.(plan_research|select_tools)|llm\.research_planning/, 0],
    [/langgraph\.node\.(search_papers|check_paper_integrity|expand_citation_graph)|subtask\.literature_search|mcp\.client\.(search_papers|check_research_integrity|expand_citation_graph)/, 1],
    [/langgraph\.node\.(index_papers|retrieve_relevant_papers|evaluate_evidence)|rag\.retrieve_for_intent/, 2],
    [/langgraph\.node\.(review_deep_read|deep_read_papers)|subtask\.deep_read|mcp\.client\.get_full_text/, 3],
    [/langgraph\.node\.(prepare_context|write_answer)|llm\.answer_synthesis/, 4],
    [/langgraph\.node\.(verify_grounding|validate_answer)|llm\.grounding/, 5],
  ];
  const statuses = pipelineStages.map(() => "pending");
  events.forEach((event) => {
    if (!event.event?.startsWith("operation.")) return;
    const name = String(event.name || event.operation || "");
    const match = stageByOperation.find(([pattern]) => pattern.test(name));
    if (!match) return;
    const stage = match[1];
    if (event.event === "operation.started" && statuses[stage] !== "done") statuses[stage] = "active";
    if (event.event === "operation.completed") statuses[stage] = "done";
    if (event.event === "operation.failed") statuses[stage] = "failed";
  });
  if (statuses[3] === "pending" && statuses.slice(4).some((status) => status !== "pending")) {
    statuses[3] = "skipped";
  }
  renderPipelineStatuses(statuses);
}

function renderPipelineStatuses(statuses) {
  $("#pipeline").innerHTML = pipelineStages.map(([name, detail], index) => {
    if (name === "Search literature" && selectedAccessToken() === "sim-local-reader") detail = "Embedded corpus only";
    const mode = statuses[index] === "pending" ? "" : statuses[index];
    return `<div class="pipeline-step ${mode}">${escapeHtml(name)}<small>${escapeHtml(detail)}</small></div>`;
  }).join("");
}

function renderResult(result) {
  const traceEvents = state.liveTrace?.events || result.run_trace?.events || [];
  if (traceEvents.length) renderPipelineFromTrace(traceEvents);
  else renderPipeline(-1, true);
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
  const profile = result.access_profile;
  if (profile) $("#access-summary").textContent = `${profile.label} · ${result.local_corpus_paper_count || 0} local corpus papers indexed`;
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
      <div class="paper-top"><span class="paper-index">${String(index + 1).padStart(2, "0")}</span><span class="paper-source">${escapeHtml(paper.source || "source")}</span>${paper.integrity_status && paper.integrity_status !== "unchecked" ? `<span class="paper-integrity integrity-${escapeHtml(paper.integrity_status)}">${escapeHtml(paper.integrity_status.replaceAll("_", " "))}</span>` : ""}</div>
      <h3>${escapeHtml(paper.title)}</h3>
      <p>${escapeHtml(paper.abstract || "No abstract available.")}</p>
      ${paper.discovered_from ? `<small class="paper-discovery">Citation graph from ${escapeHtml(paper.discovered_from)}</small>` : ""}
      ${(paper.integrity_updates || []).length ? `<small class="paper-warning">${escapeHtml(paper.integrity_updates.join(" · "))}</small>` : ""}
      <footer><span>${paper.year || "n.d."}</span><span>${escapeHtml(paper.venue || "Unspecified venue")}</span>${paper.url ? `<a href="${safeUrl(paper.url)}" target="_blank" rel="noreferrer">Open source</a>` : ""}</footer>
    </article>`).join("");
}

function renderTrace(result) {
  const tasks = result.subtasks || [];
  const critiques = result.critique_history || [];
  const selectedTools = (result.tool_plan?.choices || []).filter((tool) => tool.selected).map((tool) => tool.tool_id);
  const items = [
    { time: "PLAN", title: `Purpose: ${result.research_intent?.purpose || "research"}`, detail: result.research_intent?.objective || `${(result.search_queries || []).length} search queries generated`, status: result.goal?.status || "complete" },
    { time: "TOOLS", title: `${selectedTools.length} tools selected`, detail: selectedTools.join(" · ") || "No authorized tools selected", status: "completed" },
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
