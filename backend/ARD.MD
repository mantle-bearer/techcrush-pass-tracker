# How `main.py` turns the course materials into high-quality exam questions — explained simply (but technically)

Nice — this file is a compact, practical pipeline that turns your course documents and screenshots into a searchable knowledge base, then asks a model to generate exam items grounded in that knowledge, and finally filters/normalizes the model output into usable JSONL questions. Below I walk through the **whole system** from data → embeddings → retrieval → prompt → generation → parsing → filtering → storage. I explain the *why* and *how* of each technical piece (not line-by-line syntax), tradeoffs, and what to tune when things go wrong.

---

## High-level architecture (one-sentence summary)

1. **Ingest** course files (PDFs, text, screenshots).
2. **Chunk & embed** those texts with a sentence embedding model.
3. **Index** embeddings using FAISS for fast similarity search.
4. **Retrieve** relevant context for a requested topic.
5. **Prompt** a generative model (Groq) with the retrieved context + strict format instructions.
6. **Parse** the plain-text response into structured question records.
7. **Normalize & filter** records into a clean question bank (JSONL).
8. **Serve** batches or single questions to the frontend on demand.

---

# 1) Ingestion & text preparation — why and how

**What it does**

* Walks the `data/` directory and reads files: `.txt`, `.md`, `.pdf`, and images (screenshots).
* For images it runs OCR (`pytesseract`) to extract text (style hints or question text from screenshots).
* Long files are broken into overlapping chunks (function `chunk_text`).

**Why chunking & overlap**

* Embedding models have limited context usefulness: embedding a whole book is less precise for short question grounding.
* Overlap helps because a useful fact might sit near a chunk boundary. Overlap ensures that fact appears in at least one chunk.

**Key design choices**

* `chunk_size=900`, `overlap=120` — these set chunk length and redundancy. Larger chunks = more context per vector but coarser retrieval; smaller chunks = more precise but more vectors (storage + compute).
* Style OCR chunks are stored separately (`style_hints`) to encourage the generative model to mimic phrasing/format.

**Pitfalls & tips**

* OCR can be noisy; keep style hints short (the code grabs at most a few chunks).
* PDFs with poor text extraction may produce empty chunks — the code gracefully skips them.
* If you have lots of long docs, consider deduplicating / removing boilerplate (copyrights, headers).

---

# 2) Embeddings — model and normalization

**What it does**

* Uses a SentenceTransformer (`MODEL_EMB = "sentence-transformers/all-MiniLM-L6-v2"`) to convert each chunk to a numeric vector and normalizes them.

**Why embeddings**

* They convert semantic meaning into vector space where similar pieces of text are near each other; this makes retrieval by similarity fast and semantically meaningful.

**Important details**

* `normalize_embeddings=True` produces unit vectors enabling the use of inner product (dot product) as cosine similarity.
* Embeddings are stored as `float32` arrays for memory efficiency and FAISS compatibility.

**Tuning tips**

* `all-MiniLM-L6-v2` is fast and cheap; for higher quality retrieval (especially for nuanced course material), a larger model (e.g., `all-mpnet-base-v2`) can improve results but costs more CPU/RAM.
* If retrieval seems off, try increasing chunk overlap or using a better embedding model.

---

# 3) FAISS index — why and what kind

**What it does**

* Builds a FAISS index (`IndexFlatIP`) for inner-product search (works with normalized embeddings -> cosine similarity).
* Writes index files to `index/` so you persist between runs.

**Why FAISS**

* FAISS is a battle-tested library for nearest-neighbor search on vectors. `IndexFlatIP` is exact search (no approximate hash) — simple and accurate for small-medium scale.

**Tradeoffs**

* Exact index (`IndexFlatIP`) is simple and accurate but memory-heavy for large corpora. For big banks consider IVF/PQ indexes (approximate) for memory/time tradeoffs.
* Persisting both index and metadata with `pickle` lets you restart the service quickly without recomputing embeddings.

**Pitfalls**

* If you change the embedding model, older indexes become incompatible — you must re-embed and rebuild indexes.

---

# 4) Retrieval — how context is chosen and deduplicated

**What it does**

* `_search` embeds the query and asks FAISS for top-K similar chunks.
* Retrieval for a request pulls both **track** (course-specific) and **general** (external or generic) contexts.
* Results are deduplicated by path + text prefix to avoid repeated context.

**Why two indices (track vs general)**

* The system enforces a *90% track / 10% general* split: most questions must be grounded in course material; a few general questions come from broader ML references. This prevents the model from inventing things that don't match the course.

**How it's used in prompts**

* Retrieved text blocks are included in the prompt as `TRACK CONTEXT` and `GENERAL CONTEXT` to anchor generation.

**Tuning**

* `TOP_K` controls how many chunks you retrieve per call. Raising it gives the generator more context but increases prompt size and cost.

---

# 5) Prompting the generator (Groq) — format, style, constraints

**What it does**

* `build_batch_prompt` composes:

  * style hints (short samples from screenshots),
  * the selected retrieved contexts,
  * a strict format specification (KEY: VALUE blocks, with `TYPE:`, `QUESTION:`, `OPTIONS:`, `CORRECT_INDEX:` etc.),
  * and target counts for each question type (mcq, objectives, fill, coding).

**Why the strict plain-text format**

* You force the language model to output in a machine-parseable layout (no JSON fences). This makes parsing simpler and more robust than asking for free JSON (LLMs sometimes hallucinate syntax).
* The `FORMAT` block reduces downstream parsing errors.

**Prompt engineering best practices used**

* Provide context + explicit instruction (system prompt + user prompt).
* Set `temperature=0.2` to reduce randomness; deterministic outputs are easier to parse and verify.
* Provide *targets* — the generator knows how many of each type you want.
* Include `STYLE HINTS` so the model mimics vocabulary from screenshots.

**Risks**

* The model may not always obey formatting perfectly — thus the parser has to be forgiving (see below).
* Prompt length must be kept below provider token limits; retrieve only the most relevant chunks.

---

# 6) Parsing plain text -> structured items

**What it does**

* `parse_kv_plain` uses line regexes to split the generator output into blocks starting with `TYPE:` and gathers key/value pairs (TYPE, QUESTION, OPTIONS, CORRECT_INDEX, ANSWER, TESTS, EXPLANATION, CITATIONS).
* `_clean_bullets` strips leading bullet markers and A)/B) prefixes from option lines.

**Why not JSON from the model?**

* LLMs sometimes produce nearly-correct JSON with trailing commas or stray text. Asking for plain key/value blocks is often more reliably parsed than raw JSON.

**Parsing robustness**

* The parser:

  * removes triple-backtick fenced code,
  * tolerates different option bullet styles (`-`, `•`, `A)`),
  * extracts integer `CORRECT_INDEX` where possible.
* Still: parsing will fail when the model uses completely different keys or malformed blocks. Your code handles such cases by ignoring invalid blocks.

**Improvements**

* A small post-processing LLM pass could be used to "repair" near-miss items (e.g., if options are missing but question type is MCQ, ask the model to regenerate the OPTIONS only).
* Add more regex patterns to handle odd candidate outputs.

---

# 7) Normalization, filtering, and quotas

**What it does**

* `_normalize_item` converts raw parsed dicts into the canonical schema used by the frontend:

  * `type` normalized (mcq, objectives, fill_blank, coding, general),
  * ensures MCQs have at least 2 options and a valid `correct_index`,
  * ensures fill_blank has an `answer`,
  * packs coding items with `prompt`, `starter_code`, and `tests`.

* `_make_counts` / `_counts_from_total` compute how many items of each type should be generated in a batch (the 90/10 split and within-track distribution: ~55% MCQ, 15% objectives, 20% fill, rest coding).

* During batch generation, the code enforces quotas by keeping `type_buckets` and skipping items exceeding the desired counts.

**Why enforce quotas**

* Ensures variety and that exam-like distribution matches your teaching plan.
* Prevents mode collapse where the model generates only one question type.

**Filtering**

* Any item failing normalization is dropped. This keeps only usable questions in the bank.

**Tuning**

* Change the ratios in `_counts_from_total` to reflect your exam style (more coding, fewer objectives, etc.).

---

# 8) Batch vs single generation endpoints

**`/api/generate-batch`**

* Called at session start to return 10–20 starter questions (the frontend requests `starter = min(20, max(10, requested_total))`).
* Uses `build_batch_prompt` to request multiple items at once.

**`/api/generate-single`**

* Called to fetch one new question on-demand (when the user hits Next).
* Chooses type probabilistically with tuned weights (e.g., mcq 45%, fill 20%, objectives 15%, coding 10%, general 10%).
* Uses `build_single_prompt` which asks the model to output a single item.

**Why both**

* Batch gives the user an initial pool quickly and keeps network overhead low.
* Single supports progressive fetching and adapts to user pacing; it also avoids costing too much token-budget at session start.

---

# 9) Reliability, retries, and rate-limiting

**What it does**

* The `generate_all` bulk loop uses retries with exponential backoff (sleep, then double delay) when the Groq call fails.
* The single/batch endpoints return friendly HTTP 502 errors when generation fails, prompting the frontend to retry.

**Why**

* Remote LLM APIs have transient failures and rate limits. Exponential backoff + limited retries reduces failed runs and prevents tight retry loops.

**Production considerations**

* Add circuit-breaker logic to stop generating when too many failures occur.
* Track request quotas and costs; bulk generation should be used offline during off-peak hours.

---

# 10) Persistence: indexes, metadata, and question bank

**What it does**

* FAISS index files and metadata (pickles) are stored under `index/` for fast startup.
* Generated question bank is appended as NDJSON to `generated/question_bank.jsonl` when using `generate_all`.

**Why**

* Rebuilding embeddings is expensive. Persisting index + metadata enables quick restarts.
* NDJSON is simple to append to and easy to stream read later.

**Notes**

* If you change embedding model or data, you must rebuild indexes and reindex files.
* Pickle is convenient but consider a safer serialization (JSONL metadata or SQLite) for cross-language compatibility and security.

---

# 11) Security & operational cautions

* **API Keys:** `GROQ_API_KEY` is read from environment variables. Never commit API keys to source control. For deployment, put secrets in secure env stores.
* **CORS:** currently wide open (`allow_origins=["*"]`) — tighten in production.
* **Pickle risks:** loading pickles from unknown sources is unsafe. If the server receives uploaded pickles, it's a risk. Here you create and load your own pickles, which is fine.
* **Input validation:** the generator can hallucinate plausible-sounding but incorrect facts. Because questions are automatically served, you need human review/QA for high-stakes exams.

---

# 12) How to debug quality problems (practical checklist)

If questions are low quality or irrelevant, go through these steps in order:

1. **Inspect retrieval**: log retrieved context blocks (`build_context_block`) for sample failed questions. If the retrieved context is irrelevant, adjust `TOP_K`, embedding model, or chunking.
2. **Check embeddings**: re-embed a few chunks and the failing query; compute cosine similarity manually to see if retrieval makes sense.
3. **Examine STYLE HINTS**: noisy OCR may bias phrasing. Try removing style hints to see how generation changes.
4. **Prompt tweaks**:

   * Reduce model `temperature`.
   * Narrow the context or prefer a one-sentence summary of context.
   * Make the `FORMAT` stricter or shorter if the model returns garbage.
5. **Parser robustness**: check the raw text returned by the model — often small format violations break parsing. Improve `_clean_bullets` and KEY_RE patterns.
6. **Quota tuning**: maybe the model had trouble with many coding items—reduce coding quota per batch and fetch coding items separately.
7. **Manual review step**: add a human vetting pipeline for generated items before they enter the production bank.

---

# 13) Extensions & improvements (next-level ideas)

* **Automatic validation**: for MCQs, run a secondary model to verify that the `correct_index` answer is actually supported by context; discard conflicts.
* **Self-consistency**: ask the model to rephrase or produce a second independent candidate and compare.
* **Chain-of-thought constrained generation**: for coding tasks, request simple tests and run them in a sandbox to verify.
* **Vector metadata**: store more metadata (source section title, page number) to produce better citations.
* **Human-in-the-loop UI**: a reviewer dashboard to accept/reject/edit items before they’re published.
* **Evaluation metrics**: track generation yield (items accepted / items generated), average parsing errors, and which sources produce highest-quality questions.
* **Model choice & ensemble**: combine outputs from two models to reduce hallucination (e.g., Groq for speed + an OpenAI/Anthropic model for quality checking).

---

# 14) Summary of the most technical pieces (TL;DR for the engineer)

* **Embeddings** map text → vector. Normalized vectors let you use inner-product as cosine similarity.
* **FAISS** indexes these embeddings for fast similarity search; you persist the index to disk to avoid re-embedding.
* **Retrieval** picks relevant chunks for the prompt (90% track / 10% general). Deduplicate by path+prefix.
* **Prompting**: include retrieved context + style hints + strict format spec + type quotas to constrain LLM output.
* **Parsing**: transform plain key/value blocks into structured items; cleaning options and extracting indices are the tricky parts.
* **Normalization** enforces schema and quotas; drop bad items.
* **Robustness**: retries, backoff, and defensive parsing are required because the LLM output can be flaky.

---

## If you want, next I can:

* Annotate the exact lines in your file and show where to add logging, metrics, or unit tests.
* Provide a checklist and a short script to **rebuild indexes** safely when you change embeddings.
* Draft a small review UI (React) for vetting generated questions before they go into `question_bank.jsonl`.

Which of those would you like me to do next?
