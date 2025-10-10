import os, json, pickle, glob, random, re
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

import time
from pathlib import Path
import logging


# -------------------- Config --------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini")  # Good default
MODEL_EMB = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 6
INDEX_DIR = "index"
DATA_DIR = "data"

# Persisted index paths
TRACK_INDEX_FILE = os.path.join(INDEX_DIR, "track.faiss")
TRACK_META_FILE  = os.path.join(INDEX_DIR, "track.pkl")
GEN_INDEX_FILE   = os.path.join(INDEX_DIR, "general.faiss")
GEN_META_FILE    = os.path.join(INDEX_DIR, "general.pkl")
STYLE_META_FILE  = os.path.join(INDEX_DIR, "style.pkl")  # OCR of quiz screenshots, prompts style only

# -------------------- App --------------------
app = FastAPI(title="PassPilot Practice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax during dev; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("passpilot")

GENERATED_DIR = Path("generated")
GENERATED_DIR.mkdir(exist_ok=True)
QUESTION_BANK_FILE = GENERATED_DIR / "question_bank.jsonl"  # plain text (NDJSON)


# -------------------- Globals --------------------
emb_model: Optional[SentenceTransformer] = None
track_index = None
track_meta: List[dict] = []
gen_index = None
gen_meta: List[dict] = []
style_hints: List[str] = []  # OCR’d text from quiz screenshots

# -------------------- Utils --------------------
def load_text_from_pdf(path: str) -> str:
    from pypdf import PdfReader
    r = PdfReader(path)
    return "\n".join([p.extract_text() or "" for p in r.pages])

def load_text_from_image(path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(path))
    except Exception:
        return ""

def chunk_text(text: str, chunk_size=900, overlap=120):
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return [c for c in chunks if c.strip()]

def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = emb_model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")

def _is_general_path(path: str) -> bool:
    p = path.lower()
    return ("general" in p) or ("techcrush" in p and "track" not in p)

def _is_style_path(path: str) -> bool:
    p = path.lower()
    return any(x in p for x in ["screenshot", "quiz", ".png", ".jpg", ".jpeg", ".webp"])

def _ingest_data() -> Dict[str, Any]:
    texts_track, meta_track = [], []
    texts_gen, meta_gen = [], []
    style_texts = []

    for path in glob.glob(os.path.join(DATA_DIR, "**/*"), recursive=True):
        if os.path.isdir(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        raw = ""
        try:
            if ext in [".txt", ".md"]:
                raw = open(path, "r", encoding="utf-8", errors="ignore").read()
            elif ext == ".pdf":
                raw = load_text_from_pdf(path)
            elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
                raw = load_text_from_image(path)
            else:
                continue
        except Exception:
            continue

        chunks = chunk_text(raw)
        if not chunks:
            continue

        if _is_style_path(path):
            style_texts.extend(chunks[:4])  # small
        elif _is_general_path(path):
            for c in chunks:
                texts_gen.append(c)
                meta_gen.append({"text": c, "source": {"path": path}})
        else:
            for c in chunks:
                texts_track.append(c)
                meta_track.append({"text": c, "source": {"path": path}})

    return {
        "track_texts": texts_track, "track_meta": meta_track,
        "gen_texts": texts_gen, "gen_meta": meta_gen,
        "style_texts": style_texts,
    }

def _build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vectors)
    return idx

def ensure_indices():
    global emb_model, track_index, track_meta, gen_index, gen_meta, style_hints

    os.makedirs(INDEX_DIR, exist_ok=True)

    if emb_model is None:
        emb_model = SentenceTransformer(MODEL_EMB)

    if all(os.path.exists(p) for p in [TRACK_INDEX_FILE, TRACK_META_FILE, GEN_INDEX_FILE, GEN_META_FILE]):
        track_index = faiss.read_index(TRACK_INDEX_FILE)
        gen_index = faiss.read_index(GEN_INDEX_FILE)
        with open(TRACK_META_FILE, "rb") as f:
            globals()["track_meta"] = pickle.load(f)
        with open(GEN_META_FILE, "rb") as f:
            globals()["gen_meta"] = pickle.load(f)
        if os.path.exists(STYLE_META_FILE):
            with open(STYLE_META_FILE, "rb") as f:
                globals()["style_hints"] = pickle.load(f)
        return

    data = _ingest_data()
    track_texts, globals()["track_meta"] = data["track_texts"], data["track_meta"]
    gen_texts,   globals()["gen_meta"]   = data["gen_texts"],   data["gen_meta"]
    globals()["style_hints"]             = data["style_texts"]

    if len(track_texts) == 0:
        dummy = np.zeros((1, emb_model.get_sentence_embedding_dimension()), dtype="float32")
        globals()["track_index"] = _build_index(dummy)
        globals()["track_meta"] = []
    else:
        Xt = embed_texts(track_texts)
        globals()["track_index"] = _build_index(Xt)

    if len(gen_texts) == 0:
        dummy = np.zeros((1, emb_model.get_sentence_embedding_dimension()), dtype="float32")
        globals()["gen_index"] = _build_index(dummy)
        globals()["gen_meta"] = []
    else:
        Xg = embed_texts(gen_texts)
        globals()["gen_index"] = _build_index(Xg)

    faiss.write_index(track_index, TRACK_INDEX_FILE)
    faiss.write_index(gen_index,   GEN_INDEX_FILE)
    with open(TRACK_META_FILE, "wb") as f:
        pickle.dump(track_meta, f)
    with open(GEN_META_FILE, "wb") as f:
        pickle.dump(gen_meta, f)
    with open(STYLE_META_FILE, "wb") as f:
        pickle.dump(style_hints, f)

def _search(index, meta, query: str, k=TOP_K) -> List[dict]:
    qv = embed_texts([query])
    scores, ids = index.search(qv, k)
    out = []
    for i, sc in zip(ids[0], scores[0]):
        if i == -1:
            continue
        if 0 <= i < len(meta):
            out.append({"score": float(sc), **meta[i]})
    return out




# -------------------- Schemas --------------------
class GenerateBatchRequest(BaseModel):
    minutes: int = Field(45, ge=10, le=45)
    total_questions: int = Field(100, ge=10, le=100)
    difficulty: Optional[str] = "mixed"
    topic: Optional[str] = None

class GenerateBatchResponse(BaseModel):
    session_id: str
    items: List[Dict[str, Any]]

class GenerateSingleRequest(BaseModel):
    session_id: str
    difficulty: Optional[str] = "mixed"
    topic: Optional[str] = None

class GenerateSingleResponse(BaseModel):
    session_id: str
    item: Dict[str, Any]

# -------------------- Groq call (plain text back) --------------------
def groq_chat(system_prompt: str, user_prompt: str) -> str:
    import requests
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "max_tokens": 3000,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if r.status_code >= 400:
        raise RuntimeError("upstream_generation_error")
    return r.json()["choices"][0]["message"]["content"]

# -------------------- Plain text → JSON parser --------------------
KEY_RE = re.compile(r"^\s*([A-Z_]+)\s*:\s*(.*)$", re.IGNORECASE)

def _clean_bullets(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        ln = re.sub(r"^[-•]\s*", "", ln)
        ln = re.sub(r"^[A-D]\)\s*", "", ln, flags=re.IGNORECASE)
        ln = re.sub(r"^[A-D]\.\s*", "", ln, flags=re.IGNORECASE)
        out.append(ln)
    return out

def _normalize_type(t: str) -> Optional[str]:
    t = (t or "").strip().lower()
    if t.startswith("obj"): return "objectives"
    if t in ["tf", "true/false", "truefalse"]: return "objectives"
    if t in ["mcq", "multiple choice", "multiple-choice"]: return "mcq"
    if t.startswith("fill"): return "fill_blank"
    if t.startswith("code") or t == "coding": return "coding"
    if t.startswith("general"): return "general"
    return None

def parse_kv_plain(text: str) -> Dict[str, Any]:
    t = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    blocks = re.split(r"(?=^\s*TYPE\s*:)", t, flags=re.MULTILINE)
    items: List[Dict[str, Any]] = []

    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue

        lines = blk.splitlines()
        current_key = None
        current_buf: List[str] = []
        record: Dict[str, Any] = {}

        def flush():
            nonlocal current_key, current_buf, record
            if not current_key:
                return
            raw = "\n".join(current_buf).strip()
            key = current_key.upper()

            if key == "TYPE":
                record["type"] = _normalize_type(raw)
            elif key in ["QUESTION", "PROMPT", "EXPLANATION", "STARTER_CODE", "ANSWER"]:
                record[key.lower()] = raw
            elif key == "OPTIONS":
                opts = _clean_bullets(raw.splitlines())
                record["options"] = opts
            elif key == "CITATIONS":
                cits = _clean_bullets(raw.splitlines())
                record["citations"] = cits
            elif key == "TESTS":
                tests = _clean_bullets(raw.splitlines())
                record["tests"] = tests
            elif key == "CORRECT_INDEX":
                try:
                    record["correct_index"] = int(re.findall(r"-?\d+", raw)[0])
                except Exception:
                    pass
            current_key, current_buf = None, []

        for ln in lines:
            m = KEY_RE.match(ln)
            if m:
                flush()
                current_key = m.group(1)
                current_buf = [m.group(2) or ""]
            else:
                if current_key is None:
                    continue
                current_buf.append(ln)
        flush()

        if record:
            items.append(record)

    return {"items": items}

# -------------------- Prompt builders --------------------
def build_style_hint_block(n=2) -> str:
    if not style_hints:
        return "No style hints available."
    sample = random.sample(style_hints, k=min(n, len(style_hints)))
    return " | ".join(s.strip().replace("\n", " ")[:300] for s in sample)

def build_context_block(retrieved: List[dict]) -> str:
    if not retrieved:
        return "No context."
    cite_paths = [r["source"]["path"] for r in retrieved]
    ctx = "\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(retrieved)])
    ctx += "\nCITABLE_SOURCES:\n" + "\n".join(cite_paths)
    return ctx

def _counts_from_total(total: int) -> Dict[str, int]:
    # 90% track, 10% general; within track: 55% MCQ, 15% TF, 20% fill, rest coding.
    general = max(1, round(total * 0.10))
    track_total = total - general
    c_obj = max(0, round(track_total * 0.15))
    c_mcq = max(0, round(track_total * 0.55))
    c_fill = max(0, round(track_total * 0.20))
    c_code = max(0, track_total - (c_obj + c_mcq + c_fill))
    return dict(objectives=c_obj, mcq=c_mcq, fill_blank=c_fill, coding=c_code, general=general, track_total=track_total)

def build_batch_prompt(track_ctx, gen_ctx, difficulty, counts, topic) -> str:
    style_block = build_style_hint_block()
    track_block = build_context_block(track_ctx)
    gen_block = build_context_block(gen_ctx)

    format_spec = """
FORMAT (plain text, NO JSON, NO MARKDOWN FENCES):
Repeat the following block for EACH item you generate:

TYPE: one of [mcq | objectives | fill_blank | coding | general]
QUESTION: (for mcq/objectives/fill_blank/general)
PROMPT: (for coding)
OPTIONS:
- A) option text
- B) option text
- C) option text
- D) option text
CORRECT_INDEX: 0-based index (mcq/objectives/general)
ANSWER: (fill_blank)
STARTER_CODE: (optional for coding)
TESTS:
- brief idea lines (optional)
EXPLANATION: rationale grounded in context
CITATIONS:
- path/to/source1
- path/to/source2

(blank line between items)
"""
    return f"""
You are an exam generator.

Rules:
- Use ONLY the TRACK CONTEXT for the {counts.get('track_total', 0)} track questions.
- Use ONLY the GENERAL CONTEXT for the {counts.get('general', 0)} general questions.
- Stay within the course outline implied by the contexts and more from Google ML materials.
- Mimic phrasing from STYLE HINTS; do not invent facts.

Difficulty: {difficulty}. Topic focus: {topic or "general course topics"}.

Target counts (try to meet them; if impossible, give as many valid items as possible):
- Objectives (True/False): {counts.get('objectives', 0)}
- Multiple Choice (A–D):   {counts.get('mcq', 0)}
- Fill in the Blank:       {counts.get('fill_blank', 0)}
- Coding tasks:            {counts.get('coding', 0)}
- General MCQ (GENERAL):   {counts.get('general', 0)}

{format_spec}

STYLE HINTS: {style_block}

TRACK CONTEXT:
{track_block}

GENERAL CONTEXT:
{gen_block}
"""

def build_single_prompt(track_ctx, gen_ctx, difficulty, qtype, topic) -> str:
    # For a single item, we set a 1 in the chosen bucket
    counts = dict(objectives=0, mcq=0, fill_blank=0, coding=0, general=0, track_total= (1 if qtype!="general" else 0))
    if qtype in counts:
        counts[qtype] = 1
    return build_batch_prompt(track_ctx, gen_ctx, difficulty, counts, topic)

# -------------------- Endpoint helpers --------------------
def _normalize_item(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = _normalize_type(it.get("type", ""))
    if not t:
        return None
    if t in ["mcq", "general", "objectives"]:
        opts = it.get("options") or []
        if not isinstance(opts, list) or len(opts) < 2:
            return None
        ci = it.get("correct_index", None)
        if ci is None:
            ans = (it.get("answer") or "").strip().lower()
            if ans in ["true", "false"] and len(opts) == 2:
                ci = 0 if ans == "true" else 1
            else:
                ci = 0
        if not isinstance(ci, int) or ci < 0 or ci >= len(opts):
            return None
    elif t == "fill_blank":
        if "answer" not in it or not str(it["answer"]).strip():
            return None
    # coding can be loose

    norm = {
        "type": t,
        "question": it.get("question") or it.get("prompt") or "",
        "options": it.get("options", []),
        "correct_index": it.get("correct_index", 0),
        "explanation": it.get("explanation", ""),
        "citations": it.get("citations", []),
    }
    if t == "fill_blank":
        norm["answer"] = it.get("answer", "")
    if t == "coding":
        norm["prompt"] = it.get("prompt", it.get("question", ""))
        norm["starter_code"] = it.get("starter_code", "")
        norm["tests"] = it.get("tests", [])
    return norm

def _retrieve(topic: Optional[str], difficulty: Optional[str]):
    topic_phrase = topic or "course outline and core concepts"
    tctx = _search(track_index, track_meta, f"{topic_phrase}", k=TOP_K) + \
           _search(track_index, track_meta, f"{difficulty or 'mixed'} key facts", k=max(2, TOP_K//2))
    seen = set(); uniq = []
    for r in tctx:
        key = r["source"]["path"] + r["text"][:80]
        if key not in seen:
            seen.add(key); uniq.append(r)
    tctx = uniq[:12]
    gctx = _search(gen_index, gen_meta, "general tech concepts and policies", k=TOP_K)[:8]
    return tctx, gctx


class GenerateAllRequest(BaseModel):
    # knobs for offline bank generation
    per_batch: int = Field(15, ge=5, le=25)   # ≈15 per loop
    loops: int = Field(20, ge=1, le=200)      # default 20 loops  => ~300 items
    sleep_sec: float = Field(8.0, ge=0.0, le=30.0)
    difficulty: Optional[str] = "mixed"
    topic: Optional[str] = None
    overwrite: bool = True  # overwrite the file before starting

class GenerateAllResponse(BaseModel):
    file: str
    total_saved: int
    message: str

def _make_counts(total_track: int, total_general: int) -> Dict[str, int]:
    """Split a small batch into your standard mix."""
    obj = max(0, round(total_track * 0.15))
    mcq = max(0, round(total_track * 0.55))
    fib = max(0, round(total_track * 0.20))
    coding = max(0, total_track - (obj + mcq + fib))
    return {
        "objectives": obj,
        "mcq": mcq,
        "fill_blank": fib,
        "coding": coding,
        "general": total_general,
        "track_total": total_track,
    }

def _retrieve_context(difficulty: str, topic: Optional[str]):
    """Reuse your retrieval; returns (track_ctx, gen_ctx)."""
    topic_phrase = topic or "course outline and core concepts"
    tctx = _search(track_index, track_meta, f"{topic_phrase}", k=TOP_K) \
         + _search(track_index, track_meta, f"{difficulty or 'mixed'} key facts", k=max(2, TOP_K//2))
    seen = set(); uniq = []
    for r in tctx:
        key = r["source"]["path"] + r["text"][:80]
        if key not in seen:
            seen.add(key); uniq.append(r)
    tctx = uniq[:12]
    gctx = _search(gen_index, gen_meta, "general tech concepts and policies", k=TOP_K)[:8]
    return tctx, gctx

def _generate_once(per_batch: int, difficulty: str, topic: Optional[str]) -> List[Dict[str, Any]]:
    """One small generation (≈ per_batch items) -> normalized list ready for file append."""
    # 90/10 split inside this small batch
    general = max(1, round(per_batch * 0.10))
    track_total = max(0, per_batch - general)
    counts = _make_counts(track_total, general)

    track_ctx, gen_ctx = _retrieve_context(difficulty, topic)
    sys = "You are an assessment designer. Keep outputs factual, derived from the provided contexts and Google ML materials." 
    user = build_batch_prompt(track_ctx, gen_ctx, difficulty or "mixed", counts, topic)

    raw = groq_chat(sys, user)          # plain text (your existing function)
    parsed = parse_kv_plain(raw)        # your plain-text -> JSON parser
    items = parsed.get("items", [])

    # Normalize/filter using your existing rules
    out = []
    type_targets = {
        "objectives": counts["objectives"],
        "mcq": counts["mcq"],
        "fill_blank": counts["fill_blank"],
        "coding": counts["coding"],
        "general": counts["general"],
    }
    type_buckets = {k: 0 for k in type_targets}

    for it in items:
        t = _normalize_type(it.get("type", ""))
        if not t:
            continue

        if t in ["mcq", "general", "objectives"]:
            opts = it.get("options") or []
            if not isinstance(opts, list) or len(opts) < 2:
                continue
            ci = it.get("correct_index", None)
            if ci is None:
                ans = (it.get("answer") or "").strip().lower()
                if ans in ["true", "false"] and len(opts) == 2:
                    ci = 0 if ans == "true" else 1
                else:
                    ci = 0
            if not isinstance(ci, int) or ci < 0 or ci >= len(opts):
                continue
            it["correct_index"] = ci
        elif t == "fill_blank":
            if "answer" not in it or not str(it["answer"]).strip():
                continue
        # coding: accept as-is

        if type_buckets[t] >= type_targets[t]:
            continue

        norm = {
            "type": t,
            "question": it.get("question") or it.get("prompt") or "",
            "options": it.get("options", []),
            "correct_index": it.get("correct_index", 0),
            "explanation": it.get("explanation", ""),
            "citations": it.get("citations", []),
        }
        if t == "fill_blank":
            norm["answer"] = it.get("answer", "")
        if t == "coding":
            norm["prompt"] = it.get("prompt", it.get("question", ""))
            norm["starter_code"] = it.get("starter_code", "")
            norm["tests"] = it.get("tests", [])

        out.append(norm)
        type_buckets[t] += 1

        if sum(type_buckets.values()) >= per_batch:
            break

    return out


# -------------------- Startup --------------------
@app.on_event("startup")
def on_startup():
    ensure_indices()

# -------------------- Batch (starter 10–20) --------------------
@app.post("/api/generate-batch", response_model=GenerateBatchResponse)
def generate_batch(req: GenerateBatchRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Setup incomplete on the server.")

    requested_total = int(req.total_questions)
    # Starter chunk: 10..20 (or exactly requested_total if <= 10)
    starter = requested_total if requested_total <= 10 else min(20, max(10, requested_total))

    counts = _counts_from_total(starter)
    track_ctx, gen_ctx = _retrieve(req.topic, req.difficulty)

    sys = "You are an assessment designer. Output plain text key/value blocks exactly in the requested format."
    user = build_batch_prompt(track_ctx, gen_ctx, req.difficulty or "mixed", counts, req.topic)

    try:
        raw = groq_chat(sys, user)
        parsed = parse_kv_plain(raw)
        items = parsed.get("items", [])

        out, buckets = [], {k:0 for k in ["objectives","mcq","fill_blank","coding","general"]}
        targets = {k:counts[k] for k in buckets.keys()}

        for it in items:
            norm = _normalize_item(it)
            if not norm: 
                continue
            t = norm["type"]
            if buckets[t] >= targets.get(t, 0): 
                continue
            out.append(norm)
            buckets[t] += 1
            if sum(buckets.values()) >= starter:
                break

        if not out:
            raise ValueError("no_items")

        session_id = f"PP-{random.randint(100000,999999)}"
        return GenerateBatchResponse(session_id=session_id, items=out[:starter])

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="We had trouble preparing the first set of questions. Please try again."
        )

# -------------------- Single (on-demand) --------------------
@app.post("/api/generate-single", response_model=GenerateSingleResponse)
def generate_single(req: GenerateSingleRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Setup incomplete on the server.")

    # Choose a type with ~10% general; within track prefer mcq, then others
    # weights tuned for variety
    choices = [
        ("general", 0.10),
        ("objectives", 0.15),
        ("fill_blank", 0.20),
        ("coding", 0.10),
        ("mcq", 0.45),
    ]
    r = random.random()
    acc = 0.0
    qtype = "mcq"
    for t, w in choices:
        acc += w
        if r <= acc:
            qtype = t; break

    track_ctx, gen_ctx = _retrieve(req.topic, req.difficulty)

    sys = "You are an assessment designer. Output one item as plain text key/value block."
    user = build_single_prompt(track_ctx, gen_ctx, req.difficulty or "mixed", qtype, req.topic)

    try:
        raw = groq_chat(sys, user)
        parsed = parse_kv_plain(raw)
        items = parsed.get("items", [])

        # Prefer the target type; if not found, accept any valid item
        norm = None
        for it in items:
            if _normalize_type(it.get("type","")) == qtype:
                norm = _normalize_item(it)
                if norm: break
        if not norm:
            for it in items:
                norm = _normalize_item(it)
                if norm: break
        if not norm:
            raise ValueError("no_item")

        return GenerateSingleResponse(session_id=req.session_id, item=norm)

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="We couldn’t load the next question right now. Please tap Next again."
        )


# -------------------- Full bank (offline) --------------------
@app.post("/api/generate-all", response_model=GenerateAllResponse)
def generate_all(req: GenerateAllRequest):
    """
    Synchronously generates a large question bank by looping 'loops' times,
    ~per_batch items each time, appending NDJSON lines to generated/question_bank.jsonl.
    Not intended for frontend usage. Watch server logs for progress.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Setup incomplete on the server.")
    ensure_indices()

    # (Re)start the file if requested
    if req.overwrite and QUESTION_BANK_FILE.exists():
        try:
            QUESTION_BANK_FILE.unlink()
        except Exception:
            pass

    total_saved = 0
    logger.info(f"[BANK] Starting bulk generation: loops={req.loops}, per_batch={req.per_batch}, file={QUESTION_BANK_FILE}")

    for i in range(int(req.loops)):

        max_tries = 3
        delay = float(req.sleep_sec)
        for attempt in range(1, max_tries+1):
            try:
                batch = _generate_once(int(req.per_batch), req.difficulty or "mixed", req.topic)
                if not batch:
                    logger.warning(f"[BANK] Loop {i+1}/{req.loops}: no items parsed, skipping write.")
                else:
                    with QUESTION_BANK_FILE.open("a", encoding="utf-8") as f:
                        for q in batch:
                            f.write(json.dumps(q, ensure_ascii=False) + "\n")
                    total_saved += len(batch)
                    logger.info(f"[BANK] Loop {i+1}/{req.loops}: saved {len(batch)} (cumulative {total_saved}).")
            except RuntimeError:

                if attempt == max_tries:
                    logger.warning(f"[BANK] Loop {i+1}/{req.loops}: generation issue after retries; continuing.")
                    batch = []
                else:
                    time.sleep(delay)            # wait before retry
                    delay *= 2                   # exponential backoff
                # Upstream generation issue hidden from client; just log and continue
                # logger.warning(f"[BANK] Loop {i+1}/{req.loops}: generation issue; continuing.")
            except Exception as e:
                logger.warning(f"[BANK] Loop {i+1}/{req.loops}: unexpected error: {e!r}")

            # Gentle pause to avoid upstream rate issues
            time.sleep(float(req.sleep_sec))

    msg = f"Completed. Total saved: {total_saved}. File: {QUESTION_BANK_FILE}"
    logger.info(f"[BANK] {msg}")
    return GenerateAllResponse(
        file=str(QUESTION_BANK_FILE),
        total_saved=total_saved,
        message="Question bank generation finished."
    )



