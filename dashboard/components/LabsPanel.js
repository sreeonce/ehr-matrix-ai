const LAB_META = {
  Lactate: { label: "Lactate", unit: "mmol/L" },
  WBC: { label: "WBC", unit: "x10³/µL" },
  Creatinine: { label: "Creatinine", unit: "mg/dL" },
  Platelets: { label: "Platelets", unit: "x10³/µL" },
  Glucose: { label: "Glucose", unit: "mg/dL" },
};

export default function LabsPanel({ patient, hourIdx }) {
  return (
    <div className="panel">
      <span className="label">Lab results</span>
      <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8}}>
        {Object.keys(LAB_META).map((key) => {
          const value = patient.labs[key][hourIdx];
          const hoursSince = patient.labs_hours_since[key][hourIdx];
          const meta = LAB_META[key];
          const neverMeasured = hoursSince >= 999;
          return (
            <div
              key={key}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                borderBottom: "1px solid var(--line)",
                paddingBottom: 6,
                color: "var(--muted)"
              }}
            >
              <span style={{ fontSize: 13 }}>{meta.label}</span>
              <span style={{ textAlign: "right" }}>
                <span className="mono" style={{ fontWeight: 600, color: "#f97b7b" }}>
                  {neverMeasured ? "—" : value}
                </span>
                <span className="label" style={{ marginLeft: 6 }}>
                  {neverMeasured ? "not yet measured" : `${meta.unit} · ${Math.round(hoursSince)}h ago`}
                </span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
