import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  Database,
  FileText,
  Gauge,
  LoaderCircle,
  LockKeyhole,
  LogOut,
  MessageSquareText,
  Network,
  RotateCw,
  Sparkles,
  Ticket,
  Workflow,
} from "lucide-react";

type JiraStatus = {
  stage: string;
  message: string;
  connected: boolean;
  project_count: number;
  issue_count: number;
  chunk_count: number;
  confluence_error?: string | null;
  error?: string | null;
};

type ConfluenceStatus = {
  stage: string;
  message: string;
  connected: boolean;
  space_count: number;
  page_count: number;
  chunk_count: number;
  error?: string | null;
};

type PipelineStatus = {
  stage: string;
  message: string;
  file_name: string;
  source_folder?: string;
  page_count: number;
  chunk_count: number;
  embedding_model: string;
  generation_model?: string;
  vector_store: string;
  retrieval_mode?: RetrievalMode;
  retrieval_candidates?: number;
  result_limit?: number;
  total_chunk_count?: number;
  jira?: JiraStatus;
  confluence?: ConfluenceStatus;
  error?: string | null;
};

type RetrievalMode = "vector" | "keyword" | "hybrid";

type RetrievedChunk = {
  id: string;
  source_id: string;
  citation_id?: string;
  text: string;
  page: number | null;
  chunk: number;
  section_title?: string | null;
  source: string;
  source_type: "pdf" | "jira" | "confluence";
  issue_key?: string | null;
  issue_summary?: string | null;
  issue_url?: string | null;
  page_id?: string | null;
  page_title?: string | null;
  page_url?: string | null;
  project_key?: string | null;
  space_key?: string | null;
  space_name?: string | null;
  score: number | null;
  vector_rank?: number;
  keyword_rank?: number;
  rrf_score?: number;
  rank: number;
  retrieval_mode: RetrievalMode | "exact_issue";
  exact_issue_match?: boolean;
};

type QueryDiagnostics = {
  mode: string;
  candidate_count: number;
  vector_candidate_count?: number;
  keyword_candidate_count?: number;
  returned_count: number;
  exact_issue_key?: string;
  exact_issue_lookup?: "found_locally" | "not_found_locally";
  embedding_model: string;
  generation_model: string;
  timings_ms: {
    retrieval: number;
    generation: number;
    total: number;
  };
};

type QueryResult = {
  answer: string;
  model: string;
  chunks: RetrievedChunk[];
  citations: RetrievedChunk[];
  retrieval_mode: string;
  request_id: string;
  diagnostics: QueryDiagnostics;
};

const API = "/api";
const suggestions = [
  "What is the main goal of this product?",
  "Which features are required for the shopping cart?",
  "What does the document say about checkout?",
];

const stages = [
  { id: "reading", name: "PDF", detail: "load document" },
  { id: "chunking", name: "Chunk", detail: "split text" },
  { id: "embedding", name: "Embed", detail: "Nomic vectors" },
  { id: "storing", name: "Store", detail: "ChromaDB" },
  { id: "retrieving", name: "Retrieve", detail: "top results" },
  { id: "answering", name: "Answer", detail: "Groq LLM" },
];

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.headers.get("content-type")?.includes("application/json")) {
    throw new Error("The RAG API returned an unexpected response. Check that the backend is running.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail ??
        (response.status >= 500
          ? "The RAG API could not be reached. Start the backend with `uvicorn backend.main:app --reload`."
          : `Request failed (${response.status})`),
    );
  }
  return response.json() as Promise<T>;
}

function isQueryResult(value: unknown): value is QueryResult {
  if (!value || typeof value !== "object") return false;
  const result = value as Partial<QueryResult>;
  const diagnostics = result.diagnostics;
  const timings = diagnostics?.timings_ms;
  return (
    typeof result.answer === "string" &&
    Array.isArray(result.chunks) &&
    Array.isArray(result.citations) &&
    typeof result.retrieval_mode === "string" &&
    typeof result.request_id === "string" &&
    !!diagnostics &&
    typeof diagnostics.candidate_count === "number" &&
    typeof diagnostics.returned_count === "number" &&
    !!timings &&
    typeof timings.retrieval === "number" &&
    typeof timings.generation === "number"
  );
}

function App() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [question, setQuestion] = useState("");
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode | "">("");
  const [sourceFilter, setSourceFilter] = useState<"all" | "pdf" | "jira" | "confluence">("all");
  const [projectKey, setProjectKey] = useState("");
  const [issueTypeFilter, setIssueTypeFilter] = useState("");
  const [issueStatusFilter, setIssueStatusFilter] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState("");
  const [asking, setAsking] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [connectingJira, setConnectingJira] = useState(false);
  const [disconnectingJira, setDisconnectingJira] = useState(false);
  const [syncingAtlassian, setSyncingAtlassian] = useState(false);
  const [jiraSiteUrl, setJiraSiteUrl] = useState("");
  const [jiraEmail, setJiraEmail] = useState("");
  const [jiraApiToken, setJiraApiToken] = useState("");
  const answerRef = useRef<HTMLDivElement>(null);

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await request<PipelineStatus>("/status"));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not connect to the API.");
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    const timer = window.setInterval(() => void refreshStatus(), 2000);
    return () => window.clearInterval(timer);
  }, [refreshStatus]);

  const submitQuestion = async (value = question) => {
    const normalized = value.trim();
    if (!normalized || asking) return;
    setQuestion(normalized);
    setError("");
    setResult(null);
    setAsking(true);
    try {
      const answer = await request<unknown>("/query", {
        method: "POST",
        body: JSON.stringify({
          question: normalized,
          retrieval_mode: retrievalMode || undefined,
          source_type: sourceFilter === "all" ? undefined : sourceFilter,
          project_key: sourceFilter === "pdf" || sourceFilter === "confluence" ? undefined : projectKey.trim() || undefined,
          issue_type: sourceFilter === "pdf" || sourceFilter === "confluence" ? undefined : issueTypeFilter.trim() || undefined,
          issue_status: sourceFilter === "pdf" || sourceFilter === "confluence" ? undefined : issueStatusFilter.trim() || undefined,
        }),
      });
      if (!isQueryResult(answer)) {
        throw new Error(
          "The backend returned an outdated query response. Restart the backend to load the current API code, then try again.",
        );
      }
      setResult(answer);
      window.setTimeout(
        () => answerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
        60,
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not generate an answer.");
    } finally {
      setAsking(false);
    }
  };

  const ingest = async () => {
    setIngesting(true);
    setError("");
    try {
      setStatus(await request<PipelineStatus>("/ingest", { method: "POST" }));
      setResult(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Document ingestion failed.");
      void refreshStatus();
    } finally {
      setIngesting(false);
    }
  };

  const connectJira = async () => {
    if (connectingJira) return;
    setConnectingJira(true);
    setError("");
    try {
      const jiraStatus = await request<JiraStatus>("/jira/connect", {
        method: "POST",
        body: JSON.stringify({
          site_url: jiraSiteUrl.trim(),
          email: jiraEmail.trim(),
          api_token: jiraApiToken,
        }),
      });
      setStatus((current) => current ? { ...current, jira: jiraStatus } : current);
      setJiraApiToken("");
      void refreshStatus();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not connect to Jira.");
      void refreshStatus();
    } finally {
      setJiraApiToken("");
      setConnectingJira(false);
    }
  };

  const disconnectJira = async () => {
    if (disconnectingJira) return;
    setDisconnectingJira(true);
    setError("");
    try {
      const jiraStatus = await request<JiraStatus>("/jira/disconnect", {
        method: "POST",
      });
      setStatus((current) => current ? { ...current, jira: jiraStatus } : current);
      await refreshStatus();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not disconnect Jira.");
      void refreshStatus();
    } finally {
      setDisconnectingJira(false);
    }
  };

  const syncAtlassian = async () => {
    if (syncingAtlassian) return;
    setSyncingAtlassian(true);
    setError("");
    try {
      const jiraStatus = await request<JiraStatus>("/jira/sync", { method: "POST" });
      setStatus((current) => current ? { ...current, jira: jiraStatus } : current);
      void refreshStatus();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not sync Atlassian content.");
    } finally {
      setSyncingAtlassian(false);
    }
  };

  const isReady = status?.stage === "ready";
  const jira = status?.jira;
  const confluence = status?.confluence;
  const jiraHasIndexedData = (jira?.chunk_count ?? 0) > 0;
  const confluenceHasIndexedData = (confluence?.chunk_count ?? 0) > 0;
  const jiraIsIndexed = Boolean(jira?.connected || jiraHasIndexedData);
  const currentStage = stages.findIndex((stage) => stage.id === status?.stage);
  const activeStage = asking ? 4 : currentStage;
  const isBusy =
    ingesting ||
    (!!status && ["starting", "reading", "chunking", "embedding", "storing"].includes(status.stage));

  return (
    <div className="app-shell">
      <div className="main-panel">
        <main className="main-content" id="playground">
          <header className="app-header">
            <div className="app-brand">
              <span className="header-brand-mark" aria-hidden="true">◆</span>
              <span>RAG Explorer</span>
            </div>
            <div className="model-status-list" aria-label="Connected RAG components">
              <span className="model-status"><i className={isReady ? "is-online" : ""} />ChromaDB</span>
              <span className="model-status"><i className={isReady ? "is-online" : ""} />{status?.embedding_model.split("/").pop() ?? "Nomic embeddings"}</span>
              <span className="model-status"><i className="is-online" />{status?.generation_model ?? "Groq LLM"}</span>
            </div>
            <p className="app-tagline">PDF <span>→</span> chunk <span>→</span> Nomic embed <span>→</span> ChromaDB <span>→</span> retrieve top {status?.result_limit ?? 4} <span>→</span> Groq answer</p>
          </header>

          <section className="pipeline-card" aria-label="RAG pipeline status">
            <div className="pipeline-heading">
              <div className="pipeline-title"><span className="pipeline-title-icon"><Workflow size={15} /></span><span>RAG pipeline</span><span className="pipeline-title-sep">·</span><span className="pipeline-subtitle">Live status</span></div>
              <span className={`status-pill ${isReady ? "status-ready" : status?.stage === "error" ? "status-error" : ""}`}>
                <span className="status-dot" />
                {isReady ? "Indexed" : status?.stage === "error" ? "Needs attention" : "Processing"}
              </span>
            </div>
            <div className="steps-row">
              {stages.map((stage, index) => {
                const complete = (isReady && index < 4) || (asking && index < 4) || (!!result && index >= 4) || (!asking && currentStage > index && currentStage >= 0);
                const active = activeStage === index;
                return (
                  <div className={`pipeline-step ${complete ? "step-done" : ""} ${active ? "step-active" : ""}`} key={stage.id}>
                    <div className="step-icon">{complete ? <Check size={15} /> : <span>{index + 1}</span>}</div>
                    <span className="step-name">{stage.name}<small>{stage.detail}</small></span>
                    {index < stages.length - 1 && <span className={`step-connector ${complete ? "connector-done" : ""}`} />}
                  </div>
                );
              })}
            </div>
            <div className="pipeline-foot">
              <span className="pipeline-message">{asking ? "Retrieving the top results and generating your answer…" : status?.message ?? "Connecting to the RAG API…"}</span>
              <span className="pipeline-meta"><Database size={13} /> {status?.total_chunk_count ? `${status.total_chunk_count} vectors stored locally` : "Waiting for ingestion"}</span>
            </div>
          </section>

          <div className="workspace-grid">
            <aside className="source-column" id="source">
              <div className="panel-heading">
                <div><span className="section-label">1 · INGESTION</span><h2>Connected sources</h2></div>
                <button className="outline-button" onClick={() => void ingest()} disabled={isBusy}>
                  {isBusy ? <LoaderCircle className="spin" size={14} /> : <RotateCw size={14} />}
                  {isBusy ? "Indexing…" : "Re-index PDF"}
                </button>
              </div>
              <div className="source-folder"><span>Source folder:</span> {status?.source_folder ?? "Configured local folder"}</div>
              <div className="document-card">
                <div className="document-main">
                  <div className="pdf-icon"><FileText size={21} /></div>
                  <div className="document-info">
                    <div className="file-name">{status?.file_name ?? "Shopping_Cart_BRD.pdf"}</div>
                    <div className="file-meta">PDF document <span className="meta-bullet" /> {status?.page_count || "—"} pages</div>
                  </div>
                  <span className={`indexed-badge ${isReady ? "" : "indexing-badge"}`}><span />{isReady ? "Indexed" : isBusy ? "Indexing" : "Pending"}</span>
                </div>
                <div className="source-divider" />
                <div className="source-metrics">
                  <div className="metric-item"><strong>{status?.page_count ?? "—"}</strong><span>Pages</span></div>
                  <div className="metric-item"><strong>{status?.chunk_count ?? "—"}</strong><span>Chunks</span></div>
                  <div className="metric-item"><strong>{status?.total_chunk_count ?? status?.chunk_count ?? "—"}</strong><span>Stored</span></div>
                  <div className="metric-item"><strong>{status?.result_limit ?? 4}</strong><span>Top results</span></div>
                </div>
                <div className="source-divider" />
                <div className="source-config">
                  <div className="config-row"><span className="config-icon purple"><Network size={14} /></span><span className="config-label">Embedding model</span><strong>{status?.embedding_model.split("/").pop() ?? "—"}</strong></div>
                  <div className="config-row"><span className="config-icon green"><Database size={14} /></span><span className="config-label">Vector database</span><strong>ChromaDB <span className="local-tag">LOCAL</span></strong></div>
                  <div className="config-row"><span className="config-icon amber"><Gauge size={14} /></span><span className="config-label">Retrieval mode</span><strong>{status?.retrieval_mode ?? "—"}</strong></div>
                  <div className="config-row"><span className="config-icon amber"><Gauge size={14} /></span><span className="config-label">Answer model</span><strong>{status?.generation_model ?? "—"}</strong></div>
                </div>
              </div>
              <div className="document-card jira-document-card" id="jira-source">
                <div className="document-main">
                  <div className="pdf-icon jira-icon"><Ticket size={20} /></div>
                  <div className="document-info">
                    <div className="file-name">Jira issues</div>
                    <div className="file-meta">All accessible projects <span className="meta-bullet" /> Read-only sync</div>
                  </div>
                  <span className={`indexed-badge ${jiraIsIndexed ? "" : "indexing-badge"}`}>
                    <span />{jira?.connected ? "Synced" : jira?.stage === "syncing" ? "Syncing" : jiraHasIndexedData ? "Cached" : "Optional"}
                  </span>
                </div>
                <div className="source-divider" />
                <div className="source-metrics">
                  <div className="metric-item"><strong>{jira?.project_count ?? "—"}</strong><span>Projects</span></div>
                  <div className="metric-item"><strong>{jira?.issue_count ?? "—"}</strong><span>Issues</span></div>
                  <div className="metric-item"><strong>{jira?.chunk_count ?? "—"}</strong><span>Chunks</span></div>
                </div>
                <div className="source-divider" />
                {jira?.connected ? (
                  <div className="jira-connection-row">
                    <span className="jira-connection-copy">{jira.message}</span>
                    <button
                      className="outline-button jira-connect-button"
                      onClick={() => void syncAtlassian()}
                      disabled={syncingAtlassian || jira?.stage === "syncing"}
                    >
                      {syncingAtlassian || jira?.stage === "syncing"
                        ? <LoaderCircle className="spin" size={14} />
                        : <RotateCw size={14} />}
                      {syncingAtlassian || jira?.stage === "syncing" ? "Syncing…" : "Sync now"}
                    </button>
                    <button
                      className="outline-button jira-connect-button jira-disconnect-button"
                      onClick={() => void disconnectJira()}
                      disabled={disconnectingJira}
                    >
                      {disconnectingJira
                        ? <LoaderCircle className="spin" size={14} />
                        : <LogOut size={14} />}
                      {disconnectingJira ? "Disconnecting…" : "Disconnect"}
                    </button>
                  </div>
                ) : (
                  <form
                    className="jira-credentials-form"
                    onSubmit={(event) => {
                      event.preventDefault();
                      void connectJira();
                    }}
                  >
                    <label className="jira-field">
                      <span>Atlassian Cloud site URL</span>
                      <input
                        type="url"
                        autoComplete="url"
                        placeholder="https://your-company.atlassian.net"
                        value={jiraSiteUrl}
                        onChange={(event) => setJiraSiteUrl(event.target.value)}
                        required
                      />
                    </label>
                    <label className="jira-field">
                      <span>Atlassian account email</span>
                      <input
                        type="email"
                        autoComplete="username"
                        placeholder="you@company.com"
                        value={jiraEmail}
                        onChange={(event) => setJiraEmail(event.target.value)}
                        required
                      />
                    </label>
                    <label className="jira-field">
                      <span>API token</span>
                      <input
                        type="password"
                        autoComplete="new-password"
                        placeholder="Paste an Atlassian API token"
                        value={jiraApiToken}
                        onChange={(event) => setJiraApiToken(event.target.value)}
                        required
                      />
                    </label>
                    <div className="jira-connection-row jira-form-actions">
                      <span className="jira-connection-copy">
                        {jira?.stage === "syncing"
                          ? jira.message
                          : jira?.error || (
                            <>
                              Create an{" "}
                              <a
                                href="https://id.atlassian.com/manage-profile/security/api-tokens"
                                target="_blank"
                                rel="noreferrer"
                              >
                                Atlassian API token
                              </a>
                              .
                            </>
                          )}
                      </span>
                      <button
                        className="outline-button jira-connect-button"
                        type="submit"
                        disabled={connectingJira || jira?.stage === "syncing"}
                      >
                        {connectingJira || jira?.stage === "syncing"
                          ? <LoaderCircle className="spin" size={14} />
                          : <Ticket size={14} />}
                        {connectingJira
                          ? "Checking…"
                          : jira?.stage === "syncing"
                            ? "Syncing…"
                            : jiraHasIndexedData
                              ? "Reconnect"
                              : "Connect Atlassian"}
                      </button>
                    </div>
                  </form>
                )}
                <div className="jira-permission-note">Uses your Atlassian account’s read permissions to sync Jira issues and accessible Confluence pages. Credentials stay in the OS keyring; the API token is sent only to this local backend and is cleared from the form after submission. Disconnect removes credentials but keeps indexed content locally.</div>
              </div>
              <div className="document-card confluence-document-card" id="confluence-source">
                <div className="document-main">
                  <div className="pdf-icon confluence-icon"><BookOpen size={20} /></div>
                  <div className="document-info">
                    <div className="file-name">Confluence pages</div>
                    <div className="file-meta">Accessible spaces <span className="meta-bullet" /> Read-only sync</div>
                  </div>
                  <span className={`indexed-badge ${confluenceHasIndexedData ? "" : "indexing-badge"}`}>
                    <span />
                    {confluence?.error
                      ? "No access"
                      : confluence?.stage === "syncing"
                        ? "Syncing"
                        : confluenceHasIndexedData
                          ? "Indexed"
                          : "Optional"}
                  </span>
                </div>
                <div className="source-divider" />
                <div className="source-metrics">
                  <div className="metric-item"><strong>{confluence?.space_count ?? "—"}</strong><span>Spaces</span></div>
                  <div className="metric-item"><strong>{confluence?.page_count ?? "—"}</strong><span>Pages</span></div>
                  <div className="metric-item"><strong>{confluence?.chunk_count ?? "—"}</strong><span>Chunks</span></div>
                </div>
                {confluence?.error && (
                  <div className="confluence-error" role="status">{confluence.error}</div>
                )}
              </div>
              <div className="privacy-note"><LockKeyhole size={13} /><span>Vectors stay local; selected excerpts are sent to Groq when you ask a question.</span></div>
              <div className="source-callout"><span className="callout-mark"><BookOpen size={15} /></span><span><strong>Answers stay grounded</strong><small>Responses use only the retrieved source excerpts.</small></span></div>
            </aside>

            <section className="query-column" aria-label="Source question and answer">
              <div className="panel-heading query-panel-heading">
                <div><span className="section-label">2 · ASK THE DOCUMENT</span><h2>Ask a question</h2></div>
                <span className="model-select"><span className="model-spark"><Sparkles size={13} /></span>{status?.generation_model ?? "Groq"}</span>
              </div>

              <form className="query-box" onSubmit={(event) => { event.preventDefault(); void submitQuestion(); }}>
                <label className="sr-only" htmlFor="question-input">Ask a question about connected sources</label>
                <textarea
                  id="question-input"
                  placeholder="Ask a question about your connected sources…"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void submitQuestion();
                    }
                  }}
                  disabled={!isReady || asking}
                  rows={3}
                />
                <div className="retrieval-options" aria-label="Retrieval options">
                  <label>
                    <span>Search mode</span>
                    <select
                      aria-label="Search mode"
                      value={retrievalMode || status?.retrieval_mode || "hybrid"}
                      onChange={(event) => setRetrievalMode(event.target.value as RetrievalMode)}
                    >
                      <option value="hybrid">Hybrid (recommended)</option>
                      <option value="vector">Vector only</option>
                      <option value="keyword">Keyword only</option>
                    </select>
                  </label>
                  <label>
                    <span>Source</span>
                    <select
                      aria-label="Source filter"
                      value={sourceFilter}
                      onChange={(event) => setSourceFilter(event.target.value as "all" | "pdf" | "jira" | "confluence")}
                    >
                      <option value="all">All sources</option>
                      <option value="pdf">PDF only</option>
                      {jiraIsIndexed && <option value="jira">Jira only</option>}
                      {confluenceHasIndexedData && <option value="confluence">Confluence only</option>}
                    </select>
                  </label>
                  <label>
                    <span>Jira project key</span>
                    <input
                      aria-label="Jira project key filter"
                      value={projectKey}
                      onChange={(event) => setProjectKey(event.target.value)}
                      placeholder="Optional"
                      maxLength={50}
                      disabled={sourceFilter === "pdf" || sourceFilter === "confluence" || !jiraIsIndexed}
                    />
                  </label>
                  <label>
                    <span>Issue type</span>
                    <input
                      aria-label="Jira issue type filter"
                      value={issueTypeFilter}
                      onChange={(event) => setIssueTypeFilter(event.target.value)}
                      placeholder="Optional"
                      maxLength={100}
                      disabled={sourceFilter === "pdf" || sourceFilter === "confluence" || !jiraIsIndexed}
                    />
                  </label>
                  <label>
                    <span>Issue status</span>
                    <input
                      aria-label="Jira issue status filter"
                      value={issueStatusFilter}
                      onChange={(event) => setIssueStatusFilter(event.target.value)}
                      placeholder="Optional"
                      maxLength={100}
                      disabled={sourceFilter === "pdf" || sourceFilter === "confluence" || !jiraIsIndexed}
                    />
                  </label>
                </div>
                <div className="query-box-footer">
                  <span className="grounded-label"><LockKeyhole size={13} /> Grounded in retrieved sources</span>
                  <div className="query-actions"><span className="keyboard-hint">Enter to ask · Shift + Enter for a new line</span><button className="send-button" type="submit" disabled={!isReady || asking || !question.trim()} aria-label="Send question">{asking ? <LoaderCircle className="spin" size={16} /> : <ArrowUp size={17} />}<span>Ask</span></button></div>
                </div>
              </form>

              {(error || (status?.stage === "error" ? status.error : null) || jira?.error) && (
                <div className="error-banner" role="alert">{error || status?.error || jira?.error}</div>
              )}

              {!result && !asking && (
                <div className="suggestion-area">
                  <div className="suggestion-label">SUGGESTED QUESTIONS</div>
                  <div className="suggestion-list">
                    {suggestions.map((suggestion) => (
                      <button className="suggestion-chip" key={suggestion} onClick={() => { setQuestion(suggestion); void submitQuestion(suggestion); }} disabled={!isReady}>
                        <MessageSquareText size={13} />{suggestion}<ArrowRight size={13} />
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {(result || asking) && (
                <div className="answer-section" ref={answerRef}>
                  <div className="answer-topline">
                    <div><span className="section-label">ANSWER</span><h2>{question}</h2></div>
                    {result && <span className="answer-model"><Sparkles size={12} /> Generated with Groq</span>}
                  </div>
                  <div className="answer-card">
                    <div className="answer-mark"><Sparkles size={15} /></div>
                    {asking ? <div className="answer-loading"><LoaderCircle className="spin" size={16} /> Searching connected sources and composing an answer…</div> : <p>{result?.answer}</p>}
                    {result && <div className="answer-citations"><BookOpen size={13} /> {result.citations.length} verified citations · {result.chunks.length} retrieved excerpts <ChevronDown size={13} /></div>}
                  </div>
                  {result && (
                    <details className="diagnostics-panel">
                      <summary>Retrieval diagnostics</summary>
                      <div className="diagnostics-grid">
                        <span>Request ID</span><code>{result.request_id}</code>
                        <span>Mode</span><strong>{result.retrieval_mode}</strong>
                        <span>Candidates</span><strong>{result.diagnostics.candidate_count}</strong>
                        <span>Vector / keyword</span><strong>{result.diagnostics.vector_candidate_count ?? 0} / {result.diagnostics.keyword_candidate_count ?? 0}</strong>
                        <span>Retrieval / generation</span><strong>{result.diagnostics.timings_ms.retrieval} / {result.diagnostics.timings_ms.generation} ms</strong>
                        {result.diagnostics.exact_issue_lookup && (
                          <>
                            <span>Issue-key lookup</span>
                            <strong>{result.diagnostics.exact_issue_key}: {result.diagnostics.exact_issue_lookup}</strong>
                          </>
                        )}
                      </div>
                    </details>
                  )}
                </div>
              )}
              {result && (
                <section className="retrieval-section">
                  <div className="retrieval-heading">
                    <div><span className="section-label"><Database size={13} /> RETRIEVED CONTEXT</span><h2>Sources used to answer</h2></div>
                    <span className="retrieval-caption">Top {result.chunks.length} matches <span className="meta-bullet" /> Ranked by {result.retrieval_mode}</span>
                  </div>
                  <div className="chunk-grid">
                    {result.chunks.map((chunk, index) => (
                      <article className="chunk-card" key={`${chunk.source}-${chunk.chunk}`}>
                        <div className="chunk-top">
                          <span className="chunk-rank">{index + 1}</span>
                          <span className="chunk-page">
                            {chunk.source_type === "jira"
                              ? <Ticket size={13} />
                              : chunk.source_type === "confluence"
                                ? <BookOpen size={13} />
                                : <FileText size={13} />}
                            {chunk.source_type === "jira"
                              ? chunk.issue_key
                              : chunk.source_type === "confluence"
                                ? `${chunk.space_key || chunk.space_name || "Confluence"} · ${chunk.page_title || chunk.source}`
                                : `Page ${chunk.page}`}
                          </span>
                          <span className="relevance-score">{chunk.citation_id ? `[${chunk.citation_id}]` : `Rank ${chunk.rank}`}</span>
                        </div>
                        {chunk.section_title && (
                          <div className="chunk-section">{chunk.section_title}</div>
                        )}
                        <p>{chunk.text}</p>
                        <div className="chunk-footer">
                          {chunk.source_type === "jira" && chunk.issue_url
                            ? <a href={chunk.issue_url} target="_blank" rel="noreferrer">{chunk.issue_summary || chunk.issue_key}</a>
                            : chunk.source_type === "confluence" && chunk.page_url
                              ? <a href={chunk.page_url} target="_blank" rel="noreferrer">Open Confluence page</a>
                              : <span>Chunk {chunk.chunk}</span>}
                          <span className="chunk-rank-detail">
                            {chunk.exact_issue_match
                              ? "Exact local issue-key match"
                              : [`vector ${chunk.vector_rank ? `#${chunk.vector_rank}` : "—"}`, `keyword ${chunk.keyword_rank ? `#${chunk.keyword_rank}` : "—"}`].join(" · ")}
                          </span>
                        </div>
                      </article>
                    ))}
                  </div>
                </section>
              )}
            </section>
          </div>

          <footer className="app-footer">
            <span><span className="footer-mark">r</span> RAG Explorer</span>
            <span>Local index <span className="footer-separator">·</span> Groq receives retrieved excerpts for answers</span>
            <span>PDF + Jira <ArrowRight size={11} /> embeddings <ArrowRight size={11} /> answer</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

export default App;
