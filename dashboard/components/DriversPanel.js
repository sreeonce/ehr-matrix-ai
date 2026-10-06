export default function DriversPanel({ patient, hoverIdx }) {
  const latestIdx = patient.hours.length - 1;
  const idx = hoverIdx != null ? hoverIdx : latestIdx;
  const hour = patient.hours[idx];
  const items = patient.drivers[idx];
  const isLive = hoverIdx == null;

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span className="label">What stood out at hour {hour}</span>
        <span className="label">{isLive ? "latest hour" : "hovered hour"}</span>
      </div>
      <ul style={{ margin: "10px 0 0", paddingLeft: 18 }}>
        {items.map((text, i) => (
          <li key={i} style={{ marginBottom: 4, fontSize: 13.5, color: "var(--ink)" }}>
            {text}
          </li>
        ))}
      </ul>
      <p className="label" style={{ marginTop: 8 }}>
        Based on how far each value sits from this cohort&rsquo;s typical range —
        not a model attribution score.
      </p>
    </div>
  );
}