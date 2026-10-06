export default function SummaryPanel({ patient }) {
  return (
    <div className="panel">
      <span className="label">Summary &amp; reasoning</span>
      <p style={{ marginTop: 8, maxWidth: 640, color: "#000" }}>{patient.summary}</p>
    </div>
  );
}
