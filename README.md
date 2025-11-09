## Task

Design a **multi-agent Retrieval-Augmented Generation (RAG) system** that can:
1. **Ingest**, index, and retrieve information from *Policy documents* (OECD, EU, UN, NTIA, etc.),
2. **Analyze and debate** governance trade-offs (transparency, risk, privacy, accountability),
3. **Verify and summarize** outcomes using autonomous agents,
4. **Produce an explainable, auditable policy brief** grounded in cited sources.

The system must combine **classical NLP**, **transformer embeddings**, and **agentic reasoning** within a reproducible GitHub repository.



## Architecture
You will implement the following agents:

| Agent | Function | Techniques |
|--------|-----------|-------------|
| **PDFIngestionAgent** | Parse PDFs, extract text & tables, chunk semantically | PyMuPDF / pdfplumber + spaCy |
| **RetrieverAgent** | Hybrid RAG: BM25 + dense embeddings | LangChain + Chroma |
| **SummarizerAgent** | Summarize retrieved context; generate structured policy brief with citations | Flan-T5 / Mistral 7B |
| **DebateAgents A & B** | Produce opposing stances (e.g. pro-innovation vs risk-control); reach consensus | LangGraph loops |
| **VerifierAgent** | Cross-check factual grounding (NLI, cosine similarity ≥ 0.8) | `facebook/bart-large-mnli` |
| **EvaluatorAgent** | Judge clarity, factuality, and balance; record metrics | LLM-based evaluation |
| **MemoryAgent** | Store run history (α, β fusion weights, retrieval k) and adapt next run | JSON memory store |
| **VisualizerAgent** | Generate charts for retrieval coverage, confidence, and debate flow | matplotlib / networkx |

---

### **Task 1 — PDF Ingestion & Pre-Processing**
- Parse all PDFs under `data/pdfs/` using `pdfplumber` or `PyPDFLoader`.
- Chunk semantically (`RecursiveCharacterTextSplitter`).
- Perform tokenization, POS tagging, lemmatization, NER.
- Save outputs to `results/classical_output.json`.

### **Task 2 — Topic Modeling & Representation Diagnostics**
- Run **LDA / NMF** to identify 10+ topics.
- Train **Word2Vec or GloVe**, compare to BERT/SBERT embeddings.
- Visualize via t-SNE / PCA → `results/embedding_map.png`.


### **Task 3 — Hybrid Retrieval (BM25 + Pinecone)**
- Implement in `src/agents/retriever_agent.py`.
- Combine sparse (BM25) and dense (Pinecone) similarity:
  \[
  S_\text{final} = \alpha S_\text{dense} + (1-\alpha) S_\text{sparse}
  \]
- Store retrieval diagnostics in `results/retrieval_ablation.json`.

### **Task 4 — Planning & Multilingual Query Routing**
- Detect query language (EN/DE).
- Decompose complex policy questions into sub-queries via `PlannerAgent`.
- Save to `results/plans.json`.

### **Task 5 — Synthesis & Debate**
- **SummarizerAgent:** produce structured summaries with citations `[src: file.pdf, p.X]`.
- **DebateAgents A/B:** argue contrasting positions; consensus → `results/final_policy_brief.txt`.

### **Task 6 — Verification & Guardrails**
- **VerifierAgent:** check factuality (NLI), semantic alignment (cos ≥ 0.8), and temporal consistency.
- **GuardrailsAgent:** redact PII and filter injected prompts.
- Metrics → `results/metrics.json`.

### **Task 7 — Adaptivity & Visualization**
- **MemoryAgent:** log parameters `(α, k, latency, confidence)`.
- Auto-tune for improved factual precision.
- **VisualizerAgent:** plot confidence trajectories & agent graph → `results/plots/`.

### **Task 8 — Advanced Retrieval Architectures (Beyond Hybrid + Pinecone)**
> *Research-grade challenge*

Implement **one** alternative retrieval architecture in  
`src/agents/retriever_experiment_agent.py`.

#### Options
| Paradigm | Description | Hints |
|-----------|-------------|-------|
| **Cross-Encoder Reranking** | Re-score top-k results with a transformer (`cross-encoder/ms-marco-MiniLM-L-6-v2`). | `sentence-transformers` |
| **GraphRAG** | Build entity graphs with `spaCy` + `networkx`; retrieve subgraphs. | entity co-occurrence edges |
| **Self-RAG** | Generate → re-retrieve → refine in a feedback loop. | two-stage generation |
|  **ColBERT / Late Interaction** | Fine-grained token-level matching for semantic precision. | `colbert-ai` |
| **Long-Context RAG** | Use long-context LLMs (e.g., Mistral 7B Instruct, LongT5). | full-context inference |
| **Multi-Retriever Ensemble** | Combine multiple retrievers with learned weights. | adaptive fusion |

#### Deliverables
| File | Description |
|------|--------------|
| `src/agents/retriever_experiment_agent.py` | your advanced retriever |
| `results/retrieval_comparison.json` | metrics vs baseline |
| `results/retrieval_plot.png` | visualization |
| `results/example_hits.txt` | qualitative examples |

