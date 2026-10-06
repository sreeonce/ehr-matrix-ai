function riskBand(r) {
  if (r >= 0.5) return { word: "high", color: "var(--red)", tint: "var(--red-tint)" };
  if (r >= 0.25) return { word: "moderate", color: "var(--amber)", tint: "var(--amber-tint)" };
  return { word: "low", color: "var(--teal-deep)", tint: "var(--teal-tint)" };
}

export default function PatientInfoBar({ patient, currentRisk }) {
  const band = riskBand(currentRisk);

  return (
    <div className="row">
      <div className="panel" style={{ display: "flex", gap: 14, alignItems: "center" }}>
        <div
          aria-hidden="true"
          style={{
            width: 44,
            height: 44,
            borderRadius: "50%",
            background: "#d2d2d2",
            color: "#030303",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 15,
            fontWeight: 600,
            flexShrink: 0,
          }}
        >
          {patient.id.slice(-2)}
        </div>
        <div>
          <p style={{ fontWeight: 600, color: "#333334" }}>{patient.id}</p>
          <p className="label mono">
            {patient.age}y &middot; {patient.gender} &middot; hospital {patient.hospital_set}
          </p>
          <p className="label mono" style={{ marginTop: 2 }}>
            admitted {Math.abs(patient.hoursSinceAdmit || 0)}h before this ICU stay
          </p>
        </div>
      </div>

      <div
        className="panel"
        style={{ display: "flex", flexDirection: "column", justifyContent: "center", gap: 4 }}
      >
        <span className="label">Current risk</span>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span className="mono" style={{ fontSize: 26, fontWeight: 600, color: band.color }}>
            {Math.round(currentRisk * 100)}%
          </span>
          <span
            className="mono"
            style={{
              fontSize: 12,
              padding: "2px 8px",
              borderRadius: 999,
              background: band.tint,
              color: band.color,
            }}
          >
            {band.word}
          </span>
        </div>
      </div>
    </div>
  );
}
