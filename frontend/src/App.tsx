import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  Check,
  ChevronDown,
  CircleHelp,
  Database,
  FileText,
  LoaderCircle,
  MessageSquareText,
  Network,
  Plus,
  Sparkles,
  Upload,
  Workflow,
} from "lucide-react";

type PipelineStatus = {
  stage: string;
  message: string;
  file_name: string;
  page_count: number;
  chunk_count: number;
  embedding_model: string;
  vector_store: string;
  error?: string | null;
};

type RetrievedChunk = {
  text: string;
  page: number;
  chunk: number;
  source: string;
  score: number;
};

type QueryResult = {
  answer: string;
  model: string;
  chunks: RetrievedChunk[];
};

const API = "/api";
const suggestions = [
  "What is the main goal of this product?",
  "Which features are required for the shopping cart?",
  "What does the document say about checkout?",
];

const stages = [
  { id: "reading", name: "Read PDF", icon: FileText },
  { id: "chunking", name: "Chunk text", icon: Workflow },
  { id: "embedding", name: "Create embeddings", icon: Network },
  { id: "storing", name: "Store vectors", icon: Database },
  { id: "retrieving", name: "Retrieve context", icon: ArrowDown },
  { id: "answering", name: "Generate answer", icon: Sparkles },
];

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
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

function App() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState("");
  const [asking, setAsking] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const answerRef = useRef<HTMLDivElement>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const current = await request<PipelineStatus>("/status");
      setStatus(current);
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
      const answer = await request<QueryResult>("/query", {
        method: "POST",
        body: JSON.stringify({ question: normalized }),
      });
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
      const updated = await request<PipelineStatus>("/ingest", { method: "POST" });
      setStatus(updated);
      setResult(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Document ingestion failed.");
      void refreshStatus();
    } finally {
      setIngesting(false);
    }
  };

  const isReady = status?.stage === "ready";
  const currentStage = stages.findIndex((stage) => stage.id === status?.stage);
  const activeStage = asking ? 4 : currentStage;
  const isBusy = ingesting || (!!status && ["starting", "reading", "chunking", "embedding", "storing"].includes(status.stage));

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="RAG Explorer home">
          <span className="brand-mark"><Network size={19} strokeWidth={2.2} /></span>
          <span>rag<span className="brand-light">explorer</span></span>
        </a>
        <div className="topbar-right">
          <span className="workspace-label"><span className="live-dot" /> Local workspace</span>
          <span className="topbar-divider" />
          <button className="help-button" title="About RAG Explorer"><CircleHelp size={17} /></button>
          <div className="avatar">R</div>
        </div>
      </header>

      <main className="main-content">
        <div className="page-intro">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> YOUR DOCUMENT, IN CONTEXT</div>
            <h1>RAG Explorer</h1>
            <p className="intro-copy">See how your documents become answers. Ask a question and follow every step of the retrieval pipeline.</p>
          </div>
          <button className="outline-button" onClick={() => void ingest()} disabled={isBusy}>
            {isBusy ? <LoaderCircle className="spin" size={16} /> : <Upload size={16} />}
            {isBusy ? "Processing…" : "Re-ingest document"}
          </button>
        </div>

        <section className="pipeline-card" aria-label="RAG pipeline status">
          <div className="pipeline-heading">
            <div className="section-label"><Workflow size={15} /> PIPELINE</div>
            <span className={`status-pill ${isReady ? "status-ready" : status?.stage === "error" ? "status-error" : ""}`}>
              <span className="status-dot" />
              {isReady ? "Ready" : status?.stage === "error" ? "Needs attention" : "In progress"}
            </span>
          </div>
          <div className="steps-row">
            {stages.map((stage, index) => {
              const Icon = stage.icon;
              const complete = (isReady && index < 4) || (asking && index < 4) || (!!result && index >= 4) || (!asking && currentStage > index && currentStage >= 0);
              const active = activeStage === index;
              return (
                <div className={`pipeline-step ${complete ? "step-done" : ""} ${active ? "step-active" : ""}`} key={stage.id}>
                  <div className="step-icon">{complete ? <Check size={17} /> : <Icon size={17} />}</div>
                  <span className="step-name">{stage.name}</span>
                  {index < stages.length - 1 && <span className={`step-connector ${complete ? "connector-done" : ""}`} />}
                </div>
              );
            })}
          </div>
          <div className="pipeline-foot">
            <span>{asking ? "Retrieving the top 4 chunks and generating your answer…" : status?.message ?? "Connecting to the RAG API…"}</span>
            <span className="pipeline-meta">{status?.chunk_count ? `${status.chunk_count} chunks indexed` : "Waiting for ingestion"}</span>
          </div>
        </section>

        <div className="workspace-grid">
          <aside className="source-column">
            <div className="panel-heading">
              <div><span className="section-label">SOURCE</span><h2>Your document</h2></div>
              <button className="icon-button" title="Add a document"><Plus size={17} /></button>
            </div>
            <div className="document-card">
              <div className="pdf-preview">
                <div className="pdf-sheet">
                  <div className="pdf-tag">BRD</div>
                  <div className="pdf-rule rule-long" />
                  <div className="pdf-rule" />
                  <div className="pdf-rule rule-short" />
                  <div className="pdf-gap" />
                  <div className="pdf-rule rule-long" />
                  <div className="pdf-rule" />
                  <div className="pdf-rule rule-short" />
                  <div className="pdf-lines"><span /><span /><span /><span /></div>
                </div>
              </div>
              <div className="document-info">
                <div className="file-name">{status?.file_name ?? "Shopping_Cart_BRD.pdf"}</div>
                <div className="file-meta"><span>PDF document</span><span className="meta-bullet" />{status?.page_count || "—"} pages</div>
              </div>
              <div className="source-divider" />
              <div className="source-stats">
                <div className="source-stat"><span className="stat-icon"><FileText size={15} /></span><span>Text chunks</span><strong>{status?.chunk_count ?? "—"}</strong></div>
                <div className="source-stat"><span className="stat-icon purple"><Network size={15} /></span><span>Embedding model</span><strong className="model-value">Nomic Embed</strong></div>
                <div className="source-stat"><span className="stat-icon green"><Database size={15} /></span><span>Vector store</span><strong className="model-value">ChromaDB <span className="local-tag">LOCAL</span></strong></div>
              </div>
            </div>
            <div className="privacy-note"><span className="lock-icon">⌑</span><span>Your vectors are stored locally on this machine.</span></div>
          </aside>

          <section className="query-column">
            <div className="panel-heading query-panel-heading">
              <div><span className="section-label">RETRIEVAL PLAYGROUND</span><h2>Ask your document</h2></div>
              <button className="model-select" title="Answer model">
                <span className="model-spark"><Sparkles size={13} /></span>
                <span>Groq · GPT-OSS 120B</span><ChevronDown size={14} />
              </button>
            </div>

            <form className="query-box" onSubmit={(event) => { event.preventDefault(); void submitQuestion(); }}>
              <textarea
                aria-label="Ask a question about the document"
                placeholder="Ask anything about your document…"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void submitQuestion();
                  }
                }}
                disabled={!isReady || asking}
                rows={2}
              />
              <div className="query-box-footer">
                <span><MessageSquareText size={14} /> Answers grounded in your document</span>
                <button className="send-button" type="submit" disabled={!isReady || asking || !question.trim()} aria-label="Send question">
                  {asking ? <LoaderCircle className="spin" size={17} /> : <ArrowUp size={18} />}
                </button>
              </div>
            </form>

            {error && <div className="error-banner" role="alert">{error}</div>}

            {!result && !asking && (
              <div className="suggestion-area">
                <div className="suggestion-label">TRY ASKING</div>
                <div className="suggestion-list">
                  {suggestions.map((suggestion) => (
                    <button className="suggestion-chip" key={suggestion} onClick={() => { setQuestion(suggestion); void submitQuestion(suggestion); }} disabled={!isReady}>
                      {suggestion}<ArrowRight size={14} />
                    </button>
                  ))}
                </div>
              </div>
            )}

            {(result || asking) && (
              <div className="answer-section" ref={answerRef}>
                <div className="answer-topline">
                  <div className="section-label"><Sparkles size={14} /> GENERATED ANSWER</div>
                  {result && <span className="answer-model">{result.model}</span>}
                </div>
                <div className="answer-card">
                  <div className="answer-mark"><Sparkles size={16} /></div>
                  {asking ? (
                    <div className="answer-loading"><LoaderCircle className="spin" size={16} /> Searching the document and composing an answer…</div>
                  ) : <p>{result?.answer}</p>}
                </div>
              </div>
            )}
          </section>
        </div>

        {result && (
          <section className="retrieval-section">
            <div className="retrieval-heading">
              <div><span className="section-label"><Database size={14} /> RETRIEVAL</span><h2>Top 4 relevant chunks</h2></div>
              <span className="retrieval-caption">Semantic search · Nomic Embed · ChromaDB</span>
            </div>
            <div className="chunk-grid">
              {result.chunks.map((chunk, index) => (
                <article className="chunk-card" key={`${chunk.source}-${chunk.chunk}`}>
                  <div className="chunk-top">
                    <span className="chunk-rank">0{index + 1}</span>
                    <span className="chunk-page"><FileText size={13} /> Page {chunk.page}</span>
                    <span className="relevance-score">{Math.round(chunk.score * 100)}% match</span>
                  </div>
                  <p>{chunk.text}</p>
                  <div className="chunk-footer"><span>Chunk {chunk.chunk}</span><span className="score-track"><i style={{ width: `${Math.min(100, Math.round(chunk.score * 100))}%` }} /></span></div>
                </article>
              ))}
            </div>
          </section>
        )}

        <footer className="app-footer">
          <span><span className="footer-mark">r</span> RAG Explorer</span>
          <span>PDF <ArrowRight size={12} /> chunks <ArrowRight size={12} /> embeddings <ArrowRight size={12} /> retrieval <ArrowRight size={12} /> answer</span>
          <span>Built for learning</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
