import { useState } from "react";

export default function PhysicianNotes({ patientId }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState("");

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span className="label">Notes from the physician</span>
        <button
          onClick={() => setEditing((v) => !v)}
          aria-label={editing ? "Save note" : "Edit note"}
          style={{
            border: "none",
            background: "none",
            color: "var(--teal-deep)",
            fontSize: 12.5,
            cursor: "pointer",
            padding: 0,
          }}
        >
          {editing ? "Save" : "Edit"}
        </button>
      </div>

      {editing ? (
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Add the final clinical note or plan for this patient…"
          rows={4}
          style={{
            width: "100%",
            marginTop: 10,
            font: "inherit",
            fontSize: 13.5,
            padding: 8,
            borderRadius: 4,
            border: "1px solid var(--line)",
            resize: "vertical",
            color: "#000"
          }}
        />
      ) : (
        <p style={{ marginTop: 10, fontSize: 13.5, color: text ? "var(--ink)" : "var(--muted)" }}>
          {text || "No note added yet for this patient."}
        </p>
      )}
    </div>
  );
}
