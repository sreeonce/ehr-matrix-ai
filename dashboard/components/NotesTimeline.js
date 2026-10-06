// Phrases the note generator uses to flag an abnormal finding (see
// generate_notes.py). Highlighting these is a display aid, not a claim
// that every clinically meaningful phrase is caught.
const FLAG_PATTERN = new RegExp(
  [
    "tachycardic", "bradycardic", "heart rate elevated", "heart rate low",
    "tachypneic", "breathing fast", "increased work of breathing",
    "hypoxic", "SpO2 low", "saturation dropping",
    "febrile", "hypothermic", "temperature (?:raised|low)",
    "hypotensive", "soft blood pressure", "SBP low", "blood pressure low",
    "lactate elevated", "raised lactate",
    "WBC raised", "WBC low",
    "creatinine elevated",
    "platelets low",
    "glucose high", "glucose low",
    "Several parameters out of range", "Multiple abnormal findings",
    "[Ee]scalate", "urgent review",
  ].join("|"),
  "g"
);

function renderHighlighted(text) {
  const parts = text.split(FLAG_PATTERN);
  const matches = text.match(FLAG_PATTERN) || [];
  const out = [];
  parts.forEach((part, i) => {
    out.push(<span key={`t${i}`}>{part}</span>);
    if (matches[i]) {
      out.push(
        <mark
          key={`m${i}`}
          style={{ background: "var(--amber-tint)", color: "var(--amber)", padding: "0 2px" }}
        >
          {matches[i]}
        </mark>
      );
    }
  });
  return out;
}

export default function NotesTimeline({ patient }) {
  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span className="label">Clinical notes</span>
        <span className="tag">synthetic — not real patient text</span>
      </div>
      <div
        className="scroll-x thin-scroll"
        style={{ marginTop: 10, maxHeight: 220, overflowY: "auto", paddingRight: 4 }}
      >
        {patient.notes.map((n, i) => (
          <div
            key={i}
            style={{
              borderLeft: `2px solid ${n.n_abnormal >= 2 ? "var(--amber)" : "var(--line)"}`,
              paddingLeft: 10,
              marginBottom: 10,
            }}
          >
            <p className="label" style={{ marginBottom: 2 }}>hour {n.hour}</p>
            <p style={{ fontSize: 13, lineHeight: 1.5 }}>{renderHighlighted(n.text)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}