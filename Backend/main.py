"""
main.py — AI Interview Simulator API
=====================================
Run: uvicorn main:app --reload --port 8000

Requires a .env file containing:
    OPENROUTER_API_KEY=sk-or-v1-...
"""

from __future__ import annotations
import os, uuid, json, re, time, logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("interview_sim")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it to a .env file — never hardcode API keys."
    )
os.environ["OPENAI_API_KEY"]  = OPENROUTER_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

VECTOR_DB_PATH = os.getenv("CHROMA_DB_PATH", "./question_vectors")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/mistral-large-2512")
MODELS = {
    "parser": os.getenv("MODEL_PARSER", DEFAULT_MODEL),
    "OS":     os.getenv("MODEL_OS",     DEFAULT_MODEL),
    "CN":     os.getenv("MODEL_CN",     DEFAULT_MODEL),
    "DBMS":   os.getenv("MODEL_DBMS",   DEFAULT_MODEL),
    "SD":     os.getenv("MODEL_SD",     DEFAULT_MODEL),
    "DSA":    os.getenv("MODEL_DSA",    DEFAULT_MODEL),
    "grader": os.getenv("MODEL_GRADER", DEFAULT_MODEL),
}

SUBJECT_NAMES = {
    "OS":   "Operating Systems",
    "CN":   "Computer Networks",
    "DBMS": "Database Management Systems",
    "SD":   "System Design",
    "DSA":  "Data Structures & Algorithms",
}

BIG4     = ["OS", "CN", "DBMS", "SD"]
ALL_SUBJ = BIG4 + ["DSA"]

OPTIMAL_THRESHOLD     = 0.72
BRUTE_FORCE_THRESHOLD = 0.50
DSA_SCORE_MAP = {"Optimal":10, "Brute Force":5, "Partial":3, "Off-topic":0}

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STORE
# ═══════════════════════════════════════════════════════════════════════════════
SESSIONS: dict[str, dict] = {}

# ═══════════════════════════════════════════════════════════════════════════════
# CHROMADB  (singleton)
# ═══════════════════════════════════════════════════════════════════════════════
_client = _dsa_col = _core_col = None

def _get_cols():
    global _client, _dsa_col, _core_col
    if _client is None:
        emb = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2")
        _client   = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        _dsa_col  = _client.get_or_create_collection("dsa_questions",  embedding_function=emb)
        _core_col = _client.get_or_create_collection("core_questions", embedding_function=emb)
    return _dsa_col, _core_col

# ═══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _llm(key: str, temp: float = 0.3, max_tok: int = 600) -> ChatOpenAI:
    return ChatOpenAI(model=MODELS[key], temperature=temp, max_tokens=max_tok)

def _run(llm: ChatOpenAI, prompt: PromptTemplate, **kw) -> str:
    res = llm.invoke(prompt.format(**kw))
    return res.content if hasattr(res, "content") else str(res)

def _json(raw: str) -> dict:
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM JSON output (%s). Raw (first 200 chars): %r", e, cleaned[:200])
        return {}

# ═══════════════════════════════════════════════════════════════════════════════
# RESUME PARSER
# ═══════════════════════════════════════════════════════════════════════════════
_P_RESUME = PromptTemplate(input_variables=["resume"], template=
"""You are an elite Senior Technical Recruiter at a top-tier tech company. Your task is to rigorously analyze the provided resume.

Extract the following:
1. Identify the candidate's full name.
2. Extract up to 3 of the most technically complex project titles or professional experiences.
3. Infer subject matter expertise specifically mapping to these categories: OS, CN, DBMS, SD, DSA.

Resume content:
{resume}

Reply ONLY with valid JSON (no markdown formatting, no backticks, no explanations). Use exactly this schema:
{{"name":"...","known_subjects":["OS","DSA"],"projects":["ProjectA"]}}""")

def parse_resume(txt: str) -> dict:
    p = _json(_run(_llm("parser", 0.1, 300), _P_RESUME, resume=txt))
    if not p:
        logger.warning("parse_resume: LLM parse failed or returned nothing.")
    known = [s for s in p.get("known_subjects", []) if s in ALL_SUBJ]
    return {
        "name":           p.get("name", "Candidate"),
        "known_subjects": known,
        "projects":       p.get("projects", ["a software project"])[:3],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MoE QUESTION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
_P_QUESTION = PromptTemplate(input_variables=["subject","project"], template=
"""You are a Principal Engineer at a FAANG company conducting an intensive technical deep-dive. 
The candidate has built the following project: {project}

Your goal is to test their fundamental understanding of {subject} by applying it directly to their project. 
Do not ask generic trivia. Formulate ONE highly specific, challenging scenario-based question.

Provide a strict 4-level grading rubric:
0: Complete misunderstanding or irrelevant answer.
1: Surface-level knowledge; lacks practical application to the project.
2: Solid understanding; applies concepts correctly but misses edge cases or scalability concerns.
3: Exceptional; identifies bottlenecks, discusses trade-offs, and proposes highly scalable/optimal solutions.

Reply ONLY with valid JSON (no markdown formatting, no backticks). Schema:
{{"question":"...","rubric":{{"0":"...","1":"...","2":"...","3":"..."}}}}""")

def gen_question(subject_code: str, project: str) -> dict:
    subject_label = SUBJECT_NAMES.get(subject_code, subject_code)
    r = _json(_run(_llm(subject_code, 0.5, 500), _P_QUESTION, subject=subject_label, project=project))
    if not r.get("question"):
        logger.warning("gen_question: empty response for subject=%s — using fallback question.", subject_code)
        r = {"question": f"Explain how {subject_label} concepts apply to '{project}'.",
             "rubric": {"0":"Wrong","1":"Basic","2":"Good","3":"Excellent"}}
    return r

# ═══════════════════════════════════════════════════════════════════════════════
# DSA EVALUATOR (ChromaDB cosine + LCEL)
# ═══════════════════════════════════════════════════════════════════════════════
_P_DSA_FB = PromptTemplate(
    input_variables=["question_text","transcript","approach_label","score"],
    template=
"""You are a strict FAANG interviewer conducting a technical phone screen. Provide concise, actionable feedback (2-3 sentences max) directly addressing the candidate.

Question: {question_text}
Candidate's Answer: {transcript}
Matched Approach: {approach_label} | Score: {score}/10

Structure your feedback as follows:
1. Acknowledge what they did correctly or the validity of their core logic.
2. Point out the critical logical flaw, missing edge case, or time/space complexity inefficiency.
3. Provide one concrete, actionable tip for reaching the optimal solution.

Feedback:""")

_dsa_chain = None
def _dsa_feedback_chain():
    global _dsa_chain
    if _dsa_chain is None:
        _dsa_chain = _P_DSA_FB | _llm("DSA", 0.3, 200)
    return _dsa_chain

def _classify_dsa(distance: float, label: str) -> tuple[str, int]:
    sim = max(0.0, 1.0 - distance)
    if sim >= OPTIMAL_THRESHOLD and label == "Optimal":
        return "Optimal",    DSA_SCORE_MAP["Optimal"]
    if sim >= BRUTE_FORCE_THRESHOLD:
        lbl = "Brute Force" if label == "Brute Force" else "Partial"
        return lbl, DSA_SCORE_MAP[lbl]
    if sim >= 0.30:
        return "Partial", DSA_SCORE_MAP["Partial"]
    return "Off-topic", DSA_SCORE_MAP["Off-topic"]

def evaluate_dsa(question_text: str, transcript: str, question_id: str | None = None) -> dict[str, Any]:
    dsa_col, _ = _get_cols()

    where = None
    if question_id:
        parts = question_id.split("_")
        if len(parts) >= 2:
            parent_id = f"{parts[0]}_{parts[1]}"
            where = {"parent_question_id": parent_id}
            logger.info("evaluate_dsa: filtering by parent_question_id=%s", parent_id)

    try:
        results = dsa_col.query(
            query_texts=[transcript],
            n_results=2,
            **({"where": where} if where else {}),
        )
    except Exception as e:
        logger.error("evaluate_dsa: ChromaDB query failed: %s", e)
        return {"approach": "Off-topic", "score": 0, "max_score": 10,
                "similarity": 0.0, "feedback": "DB query failed.", "matched_label": ""}

    if not results["ids"] or not results["ids"][0]:
        return {"approach": "Off-topic", "score": 0, "max_score": 10, "similarity": 0.0,
                "feedback": "No matching question found in the database.", "matched_label": ""}

    best_dist  = float("inf")
    best_label = ""
    for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
        logger.info("evaluate_dsa: approach=%s dist=%.4f sim=%.4f", meta.get("approach_label"), dist, 1.0 - dist)
        if dist < best_dist:
            best_dist  = dist
            best_label = meta.get("approach_label", "")

    approach, score = _classify_dsa(best_dist, best_label)
    feedback = ""
    if approach != "Off-topic":
        resp = _dsa_feedback_chain().invoke({
            "question_text":  question_text,
            "transcript":     transcript,
            "approach_label": approach,
            "score":          score,
        })
        feedback = resp.content.strip() if hasattr(resp, "content") else str(resp)

    return {
        "approach":      approach,
        "score":         score,
        "max_score":     10,
        "similarity":    round(max(0.0, 1.0 - best_dist), 4),
        "matched_label": best_label,
        "feedback":      feedback,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CORE SUBJECT EVALUATOR (rubric grader)
# ═══════════════════════════════════════════════════════════════════════════════
_P_EVAL = PromptTemplate(input_variables=["question","r0","r1","r2","r3","transcript"], template=
"""You are an impartial Calibration Committee member at a top tech firm. Your job is to strictly grade the candidate's answer against the predefined rubric. Do not be overly generous; actively penalize hallucinations or "bluffing."

Question: {question}
Candidate's Answer: {transcript}

Grading Rubric:
Level 0: {r0}
Level 1: {r1}
Level 2: {r2}
Level 3: {r3}

Determine the score (0-3). Provide a concise, 2-sentence justification explaining exactly why the answer met this level and why it failed to reach the next level.

Reply ONLY with valid JSON (no markdown formatting, no backticks). Schema:
{{"score":2,"max_score":3,"justification":"..."}}""")

def evaluate_core(question: str, rubric: dict, transcript: str) -> dict:
    r = _json(_run(_llm("grader", 0.2, 300), _P_EVAL,
                   question=question,
                   r0=rubric.get("0",""), r1=rubric.get("1",""),
                   r2=rubric.get("2",""), r3=rubric.get("3",""),
                   transcript=transcript))
    return r if "score" in r else {"score":0,"max_score":3,"justification":"Parse error."}

# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL TRAP QUESTION (ChromaDB fetch for unknown subjects)
# ═══════════════════════════════════════════════════════════════════════════════
def get_trap_question(subject_code: str) -> dict:
    subject_label = SUBJECT_NAMES.get(subject_code, subject_code)
    _, core_col = _get_cols()
    try:
        # Use varied query templates to get different candidates
        query_templates = [
            f"fundamental basics of {subject_label}",
            f"core {subject_label} concepts interview question",
            f"{subject_label} fundamental principles and mechanisms",
        ]
        import random
        query = random.choice(query_templates)

        # Fetch more candidates (3) and pick one randomly
        res = core_col.query(
            query_texts=[query],
            n_results=3,
            where={"subject": subject_code},
        )
        if res["documents"] and res["documents"][0]:
            idx = random.randrange(len(res["documents"][0]))
            doc  = res["documents"][0][idx]
            meta = res["metadatas"][0][idx]

            m = re.search(r"Question:\s*(.*?)\n", doc)
            q = m.group(1).strip() if m else f"Explain a fundamental {subject_label} concept."
            return {
                "question": q,
                "rubric": {
                    "0": "No understanding.",
                    "1": "Basic definition only.",
                    "2": "Correct but incomplete.",
                    "3": "Complete with example."
                },
                "source": "db",
                "topic": meta.get("topic", subject_label)
            }
    except Exception as e:
        logger.warning("get_trap_question: ChromaDB query failed for subject=%s (%s)", subject_code, e)

    return {
        "question": f"Explain a core concept in {subject_label} you've used.",
        "rubric": {"0": "Wrong", "1": "Basic", "2": "Good", "3": "Excellent"},
        "source": "db",
        "topic": subject_label
    }

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="AI Interview Simulator", version="2.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)

class StartReq(BaseModel):
    resume: str
    mode:   str = "full"

class EvalReq(BaseModel):
    session_id:     str
    question_index: int
    transcript:     str

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def frontend():
    f = STATIC / "index.html"
    return HTMLResponse(f.read_text() if f.exists() else "<h1>Place index.html in /static/</h1>")

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.1"}

@app.post("/api/session/start")
async def start_session(body: StartReq):
    if not body.resume.strip():
        raise HTTPException(400, "Resume required.")

    sid      = str(uuid.uuid4())[:8]
    parsed   = parse_resume(body.resume)
    known    = parsed["known_subjects"]
    projects = parsed["projects"]
    proj_ctx = ", ".join(projects)
    qs: list[dict] = []

    logger.info("session %s: resume parsed -> known_subjects=%s, projects=%s", sid, known, projects)

    # ── DSA: 3 questions from vector DB ──────────────────────────────────────
    if body.mode in ("full", "dsa_only"):
        dsa_col, _ = _get_cols()
        # Fetch a larger pool, then randomly sample 3
        res = dsa_col.query(
            query_texts=["coding algorithm data structures problem"],
            n_results=min(20, dsa_col.count()),
            where={"is_optimal": True},
        )
        import random
        total = len(res["ids"][0])
        if total == 0:
            logger.warning("No DSA questions found in vector DB")
        else:
            pick = min(3, total)
            indices = random.sample(range(total), pick)
            for i in indices:
                meta  = res["metadatas"][0][i]
                q_id  = res["ids"][0][i]
                doc   = res["documents"][0][i]

                m = re.search(r"Question:\s*(.*?)\n", doc)
                q_txt = m.group(1).strip() if m else "Solve an algorithm problem."

                qs.append({
                    "subject":        "DSA",
                    "subject_label":  SUBJECT_NAMES["DSA"],
                    "topic":          meta.get("topic", "Algorithms"),
                    "question":       q_txt,
                    "question_id":    q_id,
                    "rubric":         {},
                    "source":         "db",
                    "routing_reason": "DSA is always served from the curated vector DB (optimal-approach tagged questions), independent of resume content.",
                    "time_limit":     300,
                })

    # ── Core subjects ─────────────────────────────────────────────────────────
    if body.mode in ("full", "core_only"):
        for subj in BIG4:
            label = SUBJECT_NAMES[subj]
            if subj not in known:
                q = get_trap_question(subj)
                reason = (f"Resume did not indicate familiarity with {label} → served a fixed "
                          f"fundamentals question from the vector DB (0 LLM tokens spent).")
            else:
                q = gen_question(subj, proj_ctx)
                q["source"] = "moe"
                reason = (f"Resume indicates familiarity with {label} → routed to the MoE generator "
                          f"(model '{MODELS[subj]}'), scoped to the candidate's project(s).")
            qs.append({
                "subject":        subj,
                "subject_label":  label,
                "topic":          q.get("topic", label),
                "question":       q["question"],
                "question_id":    None,
                "rubric":         q.get("rubric", {}),
                "source":         q.get("source", "moe"),
                "routing_reason": reason,
                "time_limit":     180 if q.get("source") == "moe" else 120,
            })

    SESSIONS[sid] = {
        "sid": sid, "candidate": parsed["name"], "resume": body.resume,
        "projects": projects, "known": known, "questions": qs,
        "answers": [], "current_idx": 0, "started_at": time.time(),
    }

    return {
        "session_id":      sid,
        "candidate":       parsed["name"],
        "known_subjects":  known,
        "projects":        projects,
        "total_questions": len(qs),
        "plan": [{"index": i, "subject": q["subject"], "subject_label": q["subject_label"],
                  "topic": q["topic"], "source": q["source"], "routing_reason": q["routing_reason"]}
                 for i, q in enumerate(qs)],
        "message": f"Interview ready for {parsed['name']} — {len(qs)} questions.",
    }

@app.get("/api/question/{session_id}/{index}")
async def get_question(session_id: str, index: int):
    sess = SESSIONS.get(session_id)
    if not sess: raise HTTPException(404, "Session not found.")
    qs = sess["questions"]
    if index >= len(qs): raise HTTPException(400, f"Index {index} out of range.")
    q = qs[index]
    return {
        "session_id":         session_id,
        "question_index":     index,
        "total_questions":    len(qs),
        "subject":            q["subject"],
        "subject_label":      q["subject_label"],
        "topic":              q["topic"],
        "question":           q["question"],
        "source":             q["source"],
        "routing_reason":     q["routing_reason"],
        "time_limit_seconds": q["time_limit"],
        "is_last":            index == len(qs) - 1,
    }

@app.post("/api/evaluate")
async def evaluate_answer(body: EvalReq):
    sess = SESSIONS.get(body.session_id)
    if not sess: raise HTTPException(404, "Session not found.")
    qs = sess["questions"]
    if body.question_index >= len(qs): raise HTTPException(400, "Invalid index.")

    q          = qs[body.question_index]
    transcript = body.transcript.strip()

    if not transcript:
        result = {"score": 0, "max_score": 10 if q["subject"] == "DSA" else 3,
                  "feedback": "No answer.", "approach": "None"}
    elif q["subject"] == "DSA":
        result = evaluate_dsa(q["question"], transcript, q.get("question_id"))
    else:
        result = evaluate_core(q["question"], q.get("rubric", {}), transcript)

    sess["answers"].append({
        "question_index": body.question_index,
        "subject": q["subject"], "topic": q["topic"],
        "question": q["question"], "transcript": transcript,
        **result,
    })
    sess["current_idx"] = body.question_index + 1
    nxt = body.question_index + 1
    has_next = nxt < len(qs)

    return {
        "evaluation":     result,
        "question_index": body.question_index,
        "has_next":       has_next,
        "next_index":     nxt if has_next else None,
        "next_subject":   qs[nxt]["subject"] if has_next else None,
    }

@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess: raise HTTPException(404, "Session not found.")
    answers = sess["answers"]
    if not answers:
        return {"message": "No answers yet.", "session_id": session_id}

    by_subj: dict[str, list] = {}
    for a in answers:
        by_subj.setdefault(a["subject"], []).append(a)

    summary, total = {}, 0.0
    for subj, lst in by_subj.items():
        earned = sum(a.get("score", 0) for a in lst)
        mx     = sum(a.get("max_score", 3) for a in lst)
        pct    = round(earned / mx * 100 if mx else 0, 1)
        total += pct
        summary[subj] = {
            "percent": pct,
            "questions": [{
                "topic":      a["topic"],
                "question":   a["question"][:120] + "...",
                "transcript": a["transcript"][:200] + "...",
                "score":      a.get("score", 0),
                "max_score":  a.get("max_score", 3),
                "feedback":   a.get("feedback") or a.get("justification", ""),
                "approach":   a.get("approach", ""),
            } for a in lst],
        }

    overall = round(total / len(by_subj), 1) if by_subj else 0
    verdict = ("Strong Hire" if overall >= 80 else
               "Leaning Hire" if overall >= 60 else
               "Borderline"   if overall >= 40 else "No Hire")

    return {
        "session_id":      session_id,
        "candidate":       sess["candidate"],
        "overall_percent": overall,
        "verdict":         verdict,
        "duration_minutes": round((time.time() - sess["started_at"]) / 60, 1),
        "subjects":        summary,
        "total_questions": len(sess["questions"]),
        "answered":        len(answers),
    }

@app.get("/api/sessions")
async def list_sessions():
    return {"count": len(SESSIONS),
            "sessions": [{"id": k, "candidate": v["candidate"]} for k, v in SESSIONS.items()]}

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in SESSIONS: raise HTTPException(404, "Not found.")
    del SESSIONS[session_id]
    return {"message": "Deleted."}