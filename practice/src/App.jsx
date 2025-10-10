import React, { useEffect, useMemo, useRef, useState } from "react";
import QuestionCard from "./components/QuestionCard";

const API_BASE = import.meta.env.VITE_API_BASE || ""; // e.g., http://localhost:8000

export default function App() {
  const [minutes, setMinutes] = useState(45);   // 10..45
  const [total, setTotal] = useState(100);      // 10..100
  const [difficulty, setDifficulty] = useState("mixed");
  const [topic, setTopic] = useState("");       // optional

  const [phase, setPhase] = useState("setup");  // setup|running|review
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [sessionId, setSessionId] = useState(null);
  const [items, setItems] = useState([]);
  const [answers, setAnswers] = useState({});   // idx -> value
  const [index, setIndex] = useState(0);

  const [timeLeft, setTimeLeft] = useState(0);
  const timerRef = useRef(null);

  const currentItem = items[index];

  // timer
  useEffect(() => {
    if (phase !== "running") return;
    if (timeLeft <= 0) {
      setPhase("review"); // auto-submit
      return;
    }
    timerRef.current = setTimeout(() => setTimeLeft((t) => t - 1), 1000);
    return () => clearTimeout(timerRef.current);
  }, [phase, timeLeft]);

  const minutesDisplay = Math.floor(timeLeft / 60);
  const secondsDisplay = String(timeLeft % 60).padStart(2, "0");

  const startSession = async () => {
    setLoading(true); setError("");
    try {
      const payload = {
        minutes: Math.max(10, Math.min(45, Number(minutes))),
        total_questions: Math.max(10, Math.min(100, Number(total))),
        difficulty,
        topic: topic || null
      };
      const res = await fetch(`${API_BASE}/api/generate-batch`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      const data = await res.json();
      // add _uid for input grouping
      const withUid = (data.items || []).map((it, i) => ({...it, _uid: `${data.session_id}-${i}`}));
      setItems(withUid);
      setSessionId(data.session_id);
      setAnswers({});
      setIndex(0);
      setTimeLeft(payload.minutes * 60);
      setPhase("running");
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const submitNow = () => setPhase("review");

  const score = useMemo(() => {
    if (phase !== "review") return { correct: 0, total: items.length, blanksCorrect: 0 };
    let correct = 0;
    let blanksCorrect = 0;
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      const ans = answers[i];
      if (["mcq", "general", "objectives"].includes(it.type)) {
        if (typeof ans === "number" && ans === it.correct_index) correct++;
      } else if (it.type === "fill_blank") {
        const gold = (it.answer || "").trim().toLowerCase();
        const guess = (ans || "").trim().toLowerCase();
        if (gold && guess && (gold === guess)) { correct++; blanksCorrect++; }
      } else if (it.type === "coding") {
        // not auto-graded; could award later if needed
      }
    }
    return { correct, total: items.length, blanksCorrect };
  }, [phase, items, answers]);

  const progress = items.length ? Math.round(((index + 1) / items.length) * 100) : 0;

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <header className="mb-6 flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gray-900 grid place-items-center">
          <svg viewBox="0 0 64 64" className="w-7 h-7"><rect rx="12" width="64" height="64" fill="#a3e635"/><path d="M12 32 L52 12 L40 52 L32 36 Z" fill="#0a0a0a"/><path d="M22 44 l3 3 l6-6" stroke="#0a0a0a" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
        </div>
        <div className="flex-1">
          <h1 className="text-2xl font-bold">Practice Exam Session</h1>
          <p className="text-gray-600 -mt-0.5">100 Questions • 45 Minutes • 25 pts (exam style mix)</p>
        </div>
        <a href="/" className="text-sm px-3 py-1.5 rounded-lg bg-white border">← Back to Pass Tracker</a>
      </header>

      {/* Setup */}
      {phase === "setup" && (
        <div className="bg-white rounded-2xl shadow p-5 border border-gray-100">
          <div className="grid sm:grid-cols-2 gap-4">
            <label className="block">
              <div className="text-sm mb-1">Minutes (10–45)</div>
              <input type="range" min={10} max={45} value={minutes} onChange={(e) => setMinutes(e.target.value)} className="w-full"/>
              <div className="text-sm mt-1 font-medium">{minutes} minutes</div>
            </label>
            <label className="block">
              <div className="text-sm mb-1">Questions (10–100)</div>
              <input type="range" min={10} max={100} value={total} onChange={(e) => setTotal(e.target.value)} className="w-full"/>
              <div className="text-sm mt-1 font-medium">{total} questions</div>
            </label>
            <label className="block">
              <div className="text-sm mb-1">Difficulty</div>
              <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)} className="px-3 py-2 rounded-lg border bg-white w-full">
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
                <option value="mixed">Mixed</option>
              </select>
            </label>
            <label className="block">
              <div className="text-sm mb-1">Topic (optional track focus)</div>
              <input value={topic} onChange={(e)=>setTopic(e.target.value)} placeholder="e.g., Logistic regression, Python variables" className="px-3 py-2 rounded-lg border bg-white w-full"/>
            </label>
          </div>

          {error && <div className="mt-4 p-3 rounded-lg border border-red-200 bg-red-50 text-sm text-red-700">{error}</div>}

          <div className="mt-4">
            <button onClick={startSession} disabled={loading} className="px-4 py-2 rounded-lg bg-gray-900 text-white disabled:opacity-60">
              {loading ? "Preparing..." : "Start Session"}
            </button>
          </div>
        </div>
      )}

      {/* Running */}
      {phase === "running" && (
        <>
          {/* Timer + progress */}
          <div className="mb-4 flex items-center gap-3 text-sm">
            <div className="px-3 py-1.5 rounded-lg border bg-white">⏱️ {minutesDisplay}:{secondsDisplay}</div>
            <div className="flex-1 h-2.5 rounded-full bg-gray-200">
              <div className="h-2.5 rounded-full" style={{width: `${progress}%`, background: "#16a34a"}}/>
            </div>
            <div className="text-xs text-gray-600">{index+1} / {items.length}</div>
          </div>

          {/* Question */}
          {currentItem && (
            <QuestionCard
              item={currentItem}
              answer={answers[index]}
              onAnswer={(val) => setAnswers(prev => ({...prev, [index]: val}))}
              review={false}
            />
          )}

          {/* Nav */}
          <div className="mt-4 flex items-center gap-2">
            <button className="px-3 py-1.5 rounded-lg bg-white border disabled:opacity-50"
              onClick={()=> setIndex(i => Math.max(0, i-1))}
              disabled={index === 0}
            >Previous</button>
            <button className="px-3 py-1.5 rounded-lg bg-white border disabled:opacity-50"
              onClick={()=> setIndex(i => Math.min(items.length-1, i+1))}
              disabled={index >= items.length-1}
            >Next</button>
            <div className="ml-auto"/>
            <button className="px-3 py-1.5 rounded-lg bg-gray-900 text-white" onClick={submitNow}>Submit</button>
          </div>
        </>
      )}

      {/* Review */}
      {phase === "review" && (
        <>
          <div className="mb-4 p-3 rounded-xl bg-white border flex items-center gap-3">
            <div className="text-sm">
              <span className="font-semibold">{score.correct}</span> / {score.total} correct
              <span className="text-gray-500"> (Fill-in-the-blank auto-matched exactly; coding not auto-graded)</span>
            </div>
            <div className="ml-auto text-sm">
              Accuracy: <span className="font-semibold">{score.total ? Math.round((score.correct/score.total)*100) : 0}%</span>
            </div>
          </div>
          <div className="space-y-4">
            {items.map((it, i) => (
              <QuestionCard
                key={it._uid}
                item={it}
                answer={answers[i]}
                onAnswer={()=>{}}
                review
              />
            ))}
          </div>
          <div className="mt-6">
            <button className="px-4 py-2 rounded-lg bg-white border" onClick={()=>{ setPhase("setup"); setItems([]); setAnswers({}); }}>New Session</button>
          </div>
        </>
      )}
    </div>
  );
}
