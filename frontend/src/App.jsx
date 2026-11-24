import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const SECTIONS = [
  { id: "overview", label: "Dashboard", icon: "⌂" },
  { id: "documents", label: "Documents", icon: "▤" },
  { id: "assistant", label: "Assistant", icon: "✧" },
  { id: "help", label: "Help", icon: "?" },
];
const ACCEPTED_FILES = ".pdf,.txt,.md,.csv,.json";

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((issue) => issue.msg).join(" ")
      : data.detail;
    throw new Error(detail || "The request could not be completed.");
  }
  return data;
}

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [queuedFiles, setQueuedFiles] = useState([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState("");
  const [asking, setAsking] = useState(false);
  const [question, setQuestion] = useState("");
  const [sentQuestion, setSentQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [figures, setFigures] = useState([]);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");
  const [activeSection, setActiveSection] = useState("overview");
  const [sidebarWidth, setSidebarWidth] = useState(228);
  const [documentWidth, setDocumentWidth] = useState(600);
  const splitDrag = useRef(null);
  const selectedDocument = documents.find((item) => item.id === selectedDocumentId) || documents[0] || null;

  useEffect(() => {
    let active = true;
    Promise.all([apiRequest("/health"), apiRequest("/documents")])
      .then(([, result]) => {
        if (!active) return;
        setDocuments(result.documents || []);
        setSelectedDocumentId(result.documents?.[0]?.id || "");
        setApiStatus("online");
      })
      .catch((requestError) => {
        if (!active) return;
        setApiStatus("offline");
        setError(requestError.message || "Could not connect to the backend.");
      })
      .finally(() => active && setDocumentsLoading(false));

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const handlePointerMove = (event) => {
      if (!splitDrag.current) return;
      const { type, startX, startValue } = splitDrag.current;
      const delta = event.clientX - startX;
      if (type === "sidebar") {
        setSidebarWidth(Math.min(340, Math.max(180, startValue + delta)));
      } else {
        setDocumentWidth(Math.min(1000, Math.max(360, startValue + delta)));
      }
    };
    const stopDragging = () => {
      splitDrag.current = null;
      document.body.classList.remove("is-resizing");
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopDragging);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopDragging);
    };
  }, []);

  const beginResize = (event, type, value) => {
    event.preventDefault();
    splitDrag.current = { type, startX: event.clientX, startValue: value };
    document.body.classList.add("is-resizing");
  };

  const changeSection = (section) => {
    setActiveSection(section);
    if (section === "assistant" && window.innerWidth > 820) {
      window.setTimeout(() => document.getElementById("chat-input")?.focus(), 0);
    }
  };

  const addFiles = (selectedFiles) => {
    const accepted = [];
    const rejected = [];
    for (const file of selectedFiles) {
      const extension = file.name.split(".").pop()?.toLowerCase();
      if (["pdf", "txt", "md", "csv", "json"].includes(extension)) accepted.push(file);
      else rejected.push(file.name);
    }
    if (rejected.length) setError(`Unsupported file type: ${rejected.join(", ")}`);
    else setError("");
    if (accepted.length) {
      setQueuedFiles((current) => {
        const combined = [...current];
        for (const file of accepted) {
          if (!combined.some((item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified)) {
            combined.push(file);
          }
        }
        return combined;
      });
    }
  };

  const handleFileChange = (event) => {
    addFiles(Array.from(event.target.files || []));
    event.target.value = "";
  };

  const handleUpload = async () => {
    if (!queuedFiles.length) {
      setError("Choose one or more files before starting indexing.");
      return;
    }
    setUploading(true);
    setError("");
    const formData = new FormData();
    queuedFiles.forEach((file) => formData.append("files", file));

    try {
      const result = await apiRequest("/upload", { method: "POST", body: formData });
      setDocuments((current) => [...result.documents, ...current]);
      setSelectedDocumentId(result.documents[0]?.id || "");
      setQueuedFiles([]);
      setActiveSection("documents");
    } catch (uploadError) {
      setError(uploadError.message || "Could not upload the selected files.");
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDocument = async (documentId) => {
    setDeletingDocumentId(documentId);
    setError("");
    try {
      await apiRequest(`/documents/${documentId}`, { method: "DELETE" });
      const remaining = documents.filter((item) => item.id !== documentId);
      setDocuments(remaining);
      if (selectedDocumentId === documentId) {
        setSelectedDocumentId(remaining[0]?.id || "");
      }
      setAnswer("");
      setSources([]);
      setFigures([]);
      setSentQuestion("");
    } catch (deleteError) {
      setError(deleteError.message || "Could not remove the document.");
    } finally {
      setDeletingDocumentId("");
    }
  };

  const handleAsk = async (value = question) => {
    const sentQuestion = value.trim();
    if (!sentQuestion) {
      setError("Enter a question to get started.");
      return;
    }
    if (!documents.length) {
      setError("Upload and index a document before asking questions.");
      setActiveSection("documents");
      return;
    }
    setAsking(true);
    setQuestion("");
    setSentQuestion(sentQuestion);
    setError("");
    setAnswer("");
    setSources([]);
    setFigures([]);
    setActiveSection("assistant");

    try {
      const response = await fetch(`${API_URL}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: sentQuestion }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || "Could not get an answer.");
      }
      if (!response.body) {
        throw new Error("Streaming responses are not supported by this browser.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;
      const consumeEvent = (line) => {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === "token") {
          setAnswer((current) => current + event.content);
        } else if (event.type === "complete") {
          completed = true;
          if (event.answer) setAnswer(event.answer);
          setSources(Array.isArray(event.sources) ? event.sources : []);
          setFigures(Array.isArray(event.figures) ? event.figures : []);
        } else if (event.type === "error") {
          throw new Error(event.message || "Answer generation failed.");
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        lines.forEach(consumeEvent);
        if (done) break;
      }
      consumeEvent(buffer);
      if (!completed) throw new Error("The answer stream ended unexpectedly.");
    } catch (askError) {
      setError(askError.message || "Could not get an answer.");
    } finally {
      setAsking(false);
    }
  };

  const fileSize = (bytes) => `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  const pageCount = selectedDocument?.page_count;
  const pageChart = selectedDocument?.page_chart || [];

  return (
    <div
      className={`app-shell ${activeSection === "help" ? "help-focus" : ""}`}
      style={{ "--sidebar-width": `${sidebarWidth}px`, "--document-width": `${documentWidth}px` }}
    >
      <aside className="sidebar">
        <a className="brand" href="#overview" onClick={() => changeSection("overview")}>
          <span className="brand-icon">K</span>
          <span className="brand-copy"><strong>Knowledge<span>Hub</span></strong><small>KNOWLEDGE WORKSPACE</small></span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <div className="workspace-selector">
          <span className="workspace-icon">◈</span>
          <span><strong>Shared workspace</strong><small>Single document library</small></span>
          <span className="selector-chevron">⌄</span>
        </div>
        <div className="sidebar-heading">NAVIGATION</div>
        <nav className="navigation" aria-label="Workspace sections">
          {SECTIONS.map((section) => (
            <button className={`nav-item ${activeSection === section.id ? "active" : ""}`} key={section.id} onClick={() => changeSection(section.id)}>
              <span className="nav-icon">{section.icon}</span>{section.label}
            </button>
          ))}
        </nav>
        <section className="sidebar-document">
          <div className="sidebar-heading">DOCUMENT LIBRARY</div>
          <div className="sidebar-status"><span className={`status-dot ${documents.length ? "online" : ""}`} />{documents.length} indexed document{documents.length === 1 ? "" : "s"}</div>
          <p>{selectedDocument?.filename || "Upload files to build the shared knowledge base."}</p>
        </section>
        <div className="sidebar-bottom">
          <div className="user-avatar">K</div>
          <div className="user-details"><strong>KnowledgeHub</strong><small>Shared workspace</small></div>
        </div>
      </aside>

      <div className="splitter sidebar-splitter" role="separator" aria-label="Resize navigation panel" aria-orientation="vertical" onPointerDown={(event) => beginResize(event, "sidebar", sidebarWidth)} />

      <main className="main-column">
        <header className="topbar">
          <div className="mobile-menu-wrap">
            <button className="mobile-menu-button" aria-label="Open dashboard" onClick={() => changeSection("overview")}>⌂</button>
          </div>
          <div className="topbar-heading">
            <div className="breadcrumb">Workspace <span>/</span> {SECTIONS.find((item) => item.id === activeSection)?.label}</div>
            <h1>{activeSection === "help" ? "Help & guidance" : "Knowledge Workspace"}</h1>
          </div>
          <div className="topbar-actions">
            <div className={`system-status ${apiStatus}`}><span />{apiStatus === "online" ? "API CONNECTED" : apiStatus === "checking" ? "CONNECTING…" : "API OFFLINE"}</div>
            <button className="new-document" onClick={() => changeSection("documents")}>＋ Add documents</button>
          </div>
        </header>

        <div className="dashboard-content">
          <section className={`overview-dashboard mobile-section ${activeSection === "overview" ? "mobile-active" : ""}`} id="overview">
            <div className="dashboard-heading">
              <div><span className="panel-eyebrow">YOUR WORKSPACE</span><h2>Dashboard</h2><p>Choose what you want to do next.</p></div>
              <div className="dashboard-actions">
                <button onClick={() => changeSection("documents")}><span className="action-icon">＋</span><span><strong>Upload documents</strong><small>Add one or many files</small></span><b>→</b></button>
                <button onClick={() => changeSection("assistant")}><span className="action-icon">✧</span><span><strong>Ask a question</strong><small>Search across your files</small></span><b>→</b></button>
              </div>
            </div>
            <section className="stats-grid" aria-label="Workspace summary">
              <div className="stat-card"><span className="stat-icon">▤</span><span className="stat-label">DOCUMENTS</span><strong>{String(documents.length).padStart(2, "0")}</strong><small>{documentsLoading ? "Loading library…" : "Shared files indexed"}</small></div>
              <div className="stat-card"><span className="stat-icon">◈</span><span className="stat-label">KNOWLEDGE BASE</span><strong>{documents.length ? "Ready" : "Empty"}</strong><small>Semantic search index</small></div>
              <div className="stat-card"><span className="stat-icon">✧</span><span className="stat-label">AI ASSISTANT</span><strong>{asking ? "Active" : "Ready"}</strong><small>Document-grounded answers</small></div>
            </section>
            <section className="panel recent-panel">
              <div className="section-title"><div><span className="panel-eyebrow">DOCUMENT LIBRARY</span><h3>Recent documents</h3></div><button className="text-button" onClick={() => changeSection("documents")}>View all →</button></div>
              {documents.length ? (
                <div className="recent-list">{documents.slice(0, 5).map((item) => (
                  <button className="recent-document" key={item.id} onClick={() => { setSelectedDocumentId(item.id); changeSection("documents"); }}>
                    <span className="recent-file-icon">▤</span><span><strong>{item.filename}</strong><small>{item.file_type} · {fileSize(item.size_bytes)} · {item.chunks_count} chunks</small></span><b>→</b>
                  </button>
                ))}</div>
              ) : <p className="empty-library">{documentsLoading ? "Loading documents…" : "No documents yet. Upload files to get started."}</p>}
            </section>
          </section>

          <section className={`workspace-grid ${activeSection === "documents" ? "documents-focus" : ""} ${activeSection === "assistant" ? "assistant-focus" : ""}`}>
            <section className={`document-column mobile-section ${activeSection === "documents" ? "mobile-active" : ""}`} id="documents">
              <div className="panel document-panel">
                <header className="panel-header">
                  <div><span className="panel-eyebrow">DOCUMENT INTELLIGENCE</span><h2>Upload documents</h2></div>
                  <span className={`document-badge ${documents.length ? "ready" : ""}`}><span />{documents.length} INDEXED</span>
                </header>
                <div className="upload-zone" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); addFiles(Array.from(event.dataTransfer.files)); }}>
                  <div className="upload-prompt">
                    <div className="upload-symbol">↑</div>
                    <h3>Drop files here or choose files</h3>
                    <p>Select multiple PDF, TXT, Markdown, CSV, or JSON files · up to 25 MB each</p>
                    <label className="upload-button">
                      Choose files
                      <input type="file" accept={ACCEPTED_FILES} multiple onChange={handleFileChange} />
                    </label>
                  </div>
                  {queuedFiles.length > 0 && (
                    <div className="upload-queue" aria-label="Files waiting to upload">
                      {queuedFiles.map((file, index) => (
                        <div className="queued-file" key={`${file.name}-${file.size}-${file.lastModified}`}>
                          <span className="recent-file-icon">▤</span><span><strong>{file.name}</strong><small>{fileSize(file.size)}</small></span>
                          <button aria-label={`Remove ${file.name}`} onClick={() => setQueuedFiles((current) => current.filter((_, fileIndex) => fileIndex !== index))}>×</button>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="upload-actions">
                    <span>{queuedFiles.length ? `${queuedFiles.length} file${queuedFiles.length === 1 ? "" : "s"} selected` : "Shared document library"}</span>
                    <button className="process-button" onClick={handleUpload} disabled={!queuedFiles.length || uploading}>
                      {uploading ? <><span className="button-spinner" /> Indexing…</> : `Upload & index ${queuedFiles.length ? `(${queuedFiles.length})` : ""} →`}
                    </button>
                  </div>
                </div>
                <div className="metadata-section">
                  <div className="section-title">
                    <div><span className="panel-eyebrow">FILE DETAILS</span><h3>Selected document</h3></div>
                    <div className="metadata-actions">
                      <span className="metadata-status"><i className={`status-dot ${selectedDocument ? "online" : ""}`} />{selectedDocument ? "Indexed" : "No selection"}</span>
                      {selectedDocument && (
                        <button
                          className="delete-selected-document"
                          onClick={() => handleDeleteDocument(selectedDocument.id)}
                          disabled={Boolean(deletingDocumentId)}
                        >
                          {deletingDocumentId === selectedDocument.id ? "Deleting…" : "Delete document"}
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="metadata-grid">
                    <div className="metadata-item"><span>SIZE</span><strong>{selectedDocument ? fileSize(selectedDocument.size_bytes) : "—"}</strong></div>
                    <div className="metadata-item"><span>TYPE</span><strong>{selectedDocument?.file_type || "—"}</strong></div>
                    <div className="metadata-item"><span>PAGES</span><strong>{selectedDocument?.page_count ?? (selectedDocument ? "N/A" : "—")}</strong></div>
                    <div className="metadata-item"><span>STATUS</span><strong className={selectedDocument ? "text-success" : ""}>{selectedDocument ? "Indexed" : "Waiting"}</strong></div>
                  </div>
                </div>
                <div className="combined-library">
                  <div className="section-title"><div><span className="panel-eyebrow">YOUR FILES</span><h3>Document library</h3></div><span className="pipeline-count">{documents.length} total</span></div>
                  {documents.length ? (
                    <div className="document-list">{documents.map((item) => (
                      <div className={`library-document ${selectedDocument?.id === item.id ? "selected" : ""}`} key={item.id}>
                        <button className="library-select" onClick={() => setSelectedDocumentId(item.id)}>
                          <span className="recent-file-icon">â–¤</span><span><strong>{item.filename}</strong><small>{item.file_type} Â· {fileSize(item.size_bytes)} Â· {item.chunks_count} chunks</small></span>
                        </button>
                        <button className="delete-document" onClick={() => handleDeleteDocument(item.id)} disabled={Boolean(deletingDocumentId)} aria-label={`Delete ${item.filename}`} title="Delete document">Delete</button>
                      </div>
                    ))}</div>
                  ) : <p className="empty-library">Your uploaded documents will appear here.</p>}
                </div>
              </div>

              <div className="panel processing-panel">
                <div className="section-title"><div><span className="panel-eyebrow">PIPELINE</span><h3>Processing status</h3></div><span className="pipeline-count">{selectedDocument ? `${selectedDocument.chunks_count} chunks` : "Awaiting upload"}</span></div>
                <div className="processing-grid">
                  <article className="process-card">
                    <div className="process-card-heading"><span className="process-icon">▤</span><span className={`process-state ${selectedDocument ? "complete" : ""}`}>{selectedDocument ? "COMPLETE" : "PENDING"}</span></div>
                    <span className="process-label">{selectedDocument?.file_type === "PDF" ? "PAGES PROCESSED" : "DOCUMENT PROCESSED"}</span>
                    <strong>{selectedDocument?.pages_processed ?? "—"}<small>{pageCount ? ` / ${pageCount}` : ""}</small></strong>
                    <div className="mini-chart page-chart" aria-label={`${selectedDocument?.pages_processed || 0} pages processed`}>
                      {(pageChart.length ? pageChart.slice(0, 24) : [0, 0, 0, 0, 0, 0]).map((processed, index) => <i className={processed ? "processed" : ""} key={index} />)}
                    </div>
                  </article>
                  <article className="process-card">
                    <div className="process-card-heading"><span className="process-icon">◈</span><span className={`process-state ${selectedDocument ? "complete" : ""}`}>{selectedDocument ? "READY" : "PENDING"}</span></div>
                    <span className="process-label">VECTOR STATUS</span><strong>{selectedDocument ? "Indexed" : "—"}</strong>
                    <div className="vector-chart" aria-label={selectedDocument ? "Semantic index ready" : "Semantic index pending"}><span className={selectedDocument ? "chart-ready" : ""}>◈</span><div><b>{selectedDocument ? `${selectedDocument.chunks_count} vectors` : "No vectors"}</b><small>Semantic indexing</small></div></div>
                  </article>
                  <article className="process-card entity-card">
                    <div className="process-card-heading"><span className="process-icon">⌘</span><span className="process-state unavailable">NOT ENABLED</span></div>
                    <span className="process-label">ENTITY EXTRACTION</span><strong className="entity-value">Not configured</strong>
                    <div className="entity-chart"><span>—</span><div><b>Extraction service</b><small>Not available in this pipeline</small></div></div>
                  </article>
                </div>
              </div>
            </section>

            <div className="splitter content-splitter" role="separator" aria-label="Resize document panel" aria-orientation="vertical" onPointerDown={(event) => beginResize(event, "document", documentWidth)} />

            <section className={`panel chat-panel mobile-section ${activeSection === "assistant" ? "mobile-active" : ""}`} id="assistant">
              <header className="panel-header"><div><span className="panel-eyebrow">INTELLIGENT SEARCH</span><h2>AI Assistant</h2></div><div className="ai-avatar">✧</div></header>
              {!documents.length && <div className="chat-notice">Upload and index documents to enable shared, document-based answers.</div>}
              <div className="chat-messages" aria-live="polite">
                {!answer && !asking ? (
                  <div className="chat-welcome">
                    <div className="chat-orb">✦</div><h3>Ask across your documents</h3>
                    <p>Answers search all indexed files and include source references.</p>
                    <button className="suggestion" disabled={!documents.length} onClick={() => { setQuestion("Summarize my documents"); handleAsk("Summarize my documents"); }}>Summarize my documents <span>→</span></button>
                    <button className="suggestion" disabled={!documents.length} onClick={() => { setQuestion("What are the key points?"); handleAsk("What are the key points?"); }}>What are the key points? <span>→</span></button>
                  </div>
                ) : asking && !answer ? (
                  <div className="chat-response">
                    {sentQuestion && <div className="sent-question">{sentQuestion}</div>}
                    <div className="thinking"><span /><span /><span />Searching your knowledge base…</div>
                  </div>
                ) : (
                  <div className="chat-response">
                    {sentQuestion && <div className="sent-question">{sentQuestion}</div>}
                    <div className="answer-heading">✦ AI RESPONSE</div><p className="answer-text">{answer}</p>
                    {asking && <div className="thinking answer-thinking"><span /><span /><span />Writing answer…</div>}
                    {figures.length > 0 && <div className="figures"><h3>Figures</h3>{figures.map((figure, index) => (
                      <figure key={`${figure.document_id}-${figure.page}-${index}`}>
                        <img src={`${API_URL}${figure.url}`} alt={`Figure from ${figure.filename}, page ${figure.page}`} />
                        <figcaption>{figure.filename} · Page {figure.page}</figcaption>
                      </figure>
                    ))}</div>}
                    {sources.length > 0 && <div className="sources"><h3>Retrieved sources</h3>{sources.map((source, index) => {
                      const sourceDocument = documents.find((item) => item.id === source.document_id);
                      return <div className="source-item" key={`${source.document_id}-${source.chunk_id}-${index}`}><span>{index + 1}</span><div><strong>{sourceDocument?.filename || source.source || "Document"}</strong><small>Chunk {source.chunk_id}</small></div></div>;
                    })}</div>}
                  </div>
                )}
              </div>
              <form className="chat-input-area" onSubmit={(event) => { event.preventDefault(); handleAsk(); }}>
                <textarea id="chat-input" placeholder="Ask anything across your documents…" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); handleAsk(); } }} disabled={!documents.length || asking} />
                <div className="chat-input-footer"><span>Enter to send · Shift + Enter for a new line</span><button type="submit" disabled={!documents.length || asking || !question.trim()} aria-label="Send question">{asking ? "…" : "↑"}</button></div>
              </form>
            </section>
          </section>

          <section className={`panel help-section mobile-section ${activeSection === "help" ? "mobile-active" : ""}`} id="help">
            <div className="help-title"><span className="panel-eyebrow">GETTING STARTED</span><h2>Help &amp; guidance</h2><p>How to use this shared KnowledgeHub workspace.</p></div>
            <div className="help-grid">
              <article><span>01</span><h3>Upload documents</h3><p>Open Documents, choose or drop one or more PDF, TXT, Markdown, CSV, or JSON files, then select Upload &amp; index. Each file can be up to 25 MB.</p></article>
              <article><span>02</span><h3>Check processing</h3><p>Select a document to see its size, type, page count, extraction status, and vector-index status. Remove files from the library when you no longer need them.</p></article>
              <article><span>03</span><h3>Ask questions</h3><p>Open Assistant and ask a question. Answers search all documents in this workspace and show the source files used. Ollama must be running for answer generation.</p></article>
              <article><span>04</span><h3>Switch sections</h3><p>Use the sidebar on desktop or the bottom navigation on a phone to move between Dashboard, Documents, Assistant, and Help.</p></article>
            </div>
            <div className="help-note"><strong>Shared workspace:</strong> everyone using this app sees the same document library. Avoid uploading private files.</div>
          </section>

          {error && <div className="error-message" role="alert">⚠ {error}<button onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}
          <footer className="app-footer"><span>KnowledgeHub AI</span><span>Shared document library · Semantic search</span></footer>
        </div>
        <nav className="mobile-navigation" aria-label="Workspace sections">
          {SECTIONS.map((section) => (
            <button className={activeSection === section.id ? "active" : ""} key={section.id} onClick={() => changeSection(section.id)}>
              <span>{section.icon}</span>{section.label}
            </button>
          ))}
        </nav>
      </main>
    </div>
  );
}

export default App;
