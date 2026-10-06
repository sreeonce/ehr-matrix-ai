export default function AISuggestions({ patient }) {
  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span className="label">AI suggestions</span>
        <span className="tag ai">AI-generated · may not be accurate</span>
      </div>
      <ul style={{ margin: "10px 0 0", paddingLeft: 18 }}>
        {patient.ai_suggestions.map((s, i) => (
          <li key={i} style={{ marginBottom: 6, fontSize: 13.5, color: "#000" }}>
            {s}
          </li>
        ))}
      </ul>
    </div>
  );
}
