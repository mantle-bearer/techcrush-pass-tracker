import React from "react";

export default function QuestionCard({ item, answer, onAnswer, review }) {
  const type = item.type;

  const renderOptions = (options) => {
    return options.map((opt, idx) => {
      const selected = answer === idx;
      const isCorrect = idx === item.correct_index;
      const cls = review
        ? (isCorrect ? "border-green-500 bg-green-50" : (selected ? "border-red-400 bg-red-50" : ""))
        : (selected ? "border-gray-900 bg-gray-50" : "");
      return (
        <label key={idx} className={`flex items-center gap-3 px-3 py-2 rounded-lg border cursor-pointer ${cls}`}>
          <input
            type="radio"
            name={`opt-${item._uid}`}
            className="shrink-0"
            checked={selected}
            onChange={() => onAnswer(idx)}
            disabled={review}
          />
          <span className="whitespace-pre-wrap">{opt}</span>
        </label>
      );
    });
  };

  return (
    <div className="bg-white rounded-2xl shadow p-5 border border-gray-100">
      <div className="text-xs text-gray-500 mb-2 uppercase tracking-wide">Type: {type}</div>
      {["mcq", "general", "objectives"].includes(type) && (
        <>
          <h2 className="text-lg font-semibold mb-3">Question</h2>
          <p className="mb-4">{item.question}</p>
          <div className="space-y-2">
            {renderOptions(item.options || [])}
          </div>
        </>
      )}

      {type === "fill_blank" && (
        <>
          <h2 className="text-lg font-semibold mb-3">Fill in the blank</h2>
          <p className="mb-2">{item.question}</p>
          <input
            type="text"
            className="w-full px-3 py-2 rounded-lg border"
            value={answer || ""}
            onChange={(e) => onAnswer(e.target.value)}
            disabled={review}
            placeholder="Your answer"
          />
        </>
      )}

      {type === "coding" && (
        <>
          <h2 className="text-lg font-semibold mb-3">Coding Task</h2>
          <p className="mb-2">{item.prompt || item.question}</p>
          {item.starter_code && (
            <pre className="bg-gray-50 border rounded-lg p-3 text-xs overflow-auto mb-2">
{item.starter_code}
            </pre>
          )}
          <textarea
            className="w-full min-h-[160px] px-3 py-2 rounded-lg border font-mono text-sm"
            value={answer || ""}
            onChange={(e) => onAnswer(e.target.value)}
            disabled={review}
            placeholder="Write your solution here..."
          />
        </>
      )}

      {review && (
        <div className="mt-4 border-t pt-3 text-sm">
          {["mcq", "general", "objectives"].includes(type) && (
            <div className="mb-2">
              <span className={`inline-flex items-center px-2 py-1 rounded-lg text-xs ${answer === item.correct_index ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                {answer === item.correct_index ? "Correct" : "Not correct"}
              </span>
            </div>
          )}
          {type === "fill_blank" && (
            <div className="mb-2">
              <span className="inline-flex items-center px-2 py-1 rounded-lg text-xs bg-gray-100 text-gray-700">
                Correct answer: {item.answer}
              </span>
            </div>
          )}
          {item.explanation && (
            <>
              <div className="font-medium mb-1">Explanation</div>
              <p className="text-gray-700">{item.explanation}</p>
            </>
          )}
          {Array.isArray(item.citations) && item.citations.length > 0 && (
            <div className="mt-2">
              <div className="font-medium mb-1">Citations</div>
              <ul className="list-disc ml-5 text-xs text-gray-600">
                {item.citations.map((c, i) => <li key={i}>{c}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
