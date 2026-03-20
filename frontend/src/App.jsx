import { useState } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

function PolarityBadge({ polarity }) {
  const colors = {
    Positive: "#22c55e",
    Negative: "#ef4444",
    Neutral: "#94a3b8",
  };
  return (
    <span
      style={{
        background: colors[polarity] || "#94a3b8",
        color: "#fff",
        borderRadius: "12px",
        padding: "2px 10px",
        fontSize: "0.78rem",
        fontWeight: 600,
      }}
    >
      {polarity}
    </span>
  );
}

function AspectTable({ aspects }) {
  if (!aspects.length)
    return <p style={{ color: "#94a3b8" }}>No aspects detected.</p>;
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
      <thead>
        <tr style={{ borderBottom: "1px solid #334155" }}>
          <th style={{ textAlign: "left", padding: "6px 8px" }}>Aspect</th>
          <th style={{ textAlign: "left", padding: "6px 8px" }}>Polarity</th>
          <th style={{ textAlign: "left", padding: "6px 8px" }}>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {aspects.map((a, i) => (
          <tr key={i} style={{ borderBottom: "1px solid #1e293b" }}>
            <td style={{ padding: "6px 8px", fontWeight: 500 }}>{a.term}</td>
            <td style={{ padding: "6px 8px" }}>
              <PolarityBadge polarity={a.polarity} />
              {a.low_confidence && (
                <span style={{ color: "#f59e0b", fontSize: "0.75rem", marginLeft: 6 }}>
                  ⚠ low confidence
                </span>
              )}
            </td>
            <td style={{ padding: "6px 8px", color: "#94a3b8" }}>
              {(a.confidence * 100).toFixed(1)}%
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SummarySection({ summary }) {
  return (
    <div style={{ display: "flex", gap: "24px", marginTop: "8px" }}>
      <div style={{ flex: 1 }}>
        <h3 style={{ color: "#22c55e", marginBottom: 8 }}>✓ Strengths</h3>
        {summary.strengths.length === 0 ? (
          <p style={{ color: "#94a3b8" }}>None detected</p>
        ) : (
          <ul style={{ paddingLeft: 16 }}>
            {summary.strengths.map((s, i) => (
              <li key={i} style={{ marginBottom: 4 }}>
                <strong>{s.aspect}</strong>
                <span style={{ color: "#94a3b8", marginLeft: 6 }}>
                  ({s.count}×)
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div style={{ flex: 1 }}>
        <h3 style={{ color: "#ef4444", marginBottom: 8 }}>✗ Weaknesses</h3>
        {summary.weaknesses.length === 0 ? (
          <p style={{ color: "#94a3b8" }}>None detected</p>
        ) : (
          <ul style={{ paddingLeft: 16 }}>
            {summary.weaknesses.map((w, i) => (
              <li key={i} style={{ marginBottom: 4 }}>
                <strong>{w.aspect}</strong>
                <span style={{ color: "#94a3b8", marginLeft: 6 }}>
                  ({w.count}×)
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleAnalyze() {
    if (!review.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ minHeight: "100vh", background: "#0f172a", color: "#e2e8f0", fontFamily: "system-ui, sans-serif" }}>
      <div style={{ maxWidth: 760, margin: "0 auto", padding: "48px 24px" }}>
        <h1 style={{ fontSize: "1.8rem", fontWeight: 700, marginBottom: 4 }}>
          Opinion Mining
        </h1>
        <p style={{ color: "#94a3b8", marginBottom: 32 }}>
          Aspect-based sentiment analysis for product reviews
        </p>

        <textarea
          value={review}
          onChange={(e) => setReview(e.target.value)}
          placeholder="Paste a product review here..."
          rows={4}
          style={{
            width: "100%",
            background: "#1e293b",
            border: "1px solid #334155",
            borderRadius: 8,
            color: "#e2e8f0",
            padding: "12px",
            fontSize: "0.95rem",
            resize: "vertical",
            boxSizing: "border-box",
          }}
        />

        <button
          onClick={handleAnalyze}
          disabled={loading || !review.trim()}
          style={{
            marginTop: 12,
            background: loading ? "#334155" : "#3b82f6",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            padding: "10px 28px",
            fontSize: "0.95rem",
            fontWeight: 600,
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        {error && (
          <div style={{ marginTop: 24, color: "#ef4444", background: "#1e293b", padding: 16, borderRadius: 8 }}>
            Error: {error}
          </div>
        )}

        {result && (
          <div style={{ marginTop: 32 }}>
            <div style={{ background: "#1e293b", borderRadius: 8, padding: 20, marginBottom: 20 }}>
              <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 16, color: "#94a3b8" }}>
                DETECTED ASPECTS
              </h2>
              <AspectTable aspects={result.aspects} />
            </div>

            <div style={{ background: "#1e293b", borderRadius: 8, padding: 20 }}>
              <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 8, color: "#94a3b8" }}>
                PRODUCT SUMMARY
              </h2>
              <SummarySection summary={result.summary} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
