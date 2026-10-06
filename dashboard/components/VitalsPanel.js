const VITAL_META = {
  HR: { label: "Heart rate", unit: "bpm" },
  O2Sat: { label: "SpO2", unit: "%" },
  Temp: { label: "Temperature", unit: "°C" },
  SBP: { label: "Systolic BP", unit: "mmHg" },
  DBP: { label: "Diastolic BP", unit: "mmHg" },
  MAP: { label: "MAP", unit: "mmHg" },
  Resp: { label: "Resp. rate", unit: "/min" },
};

export default function VitalsPanel({ patient, hourIdx }) {
  return (
    <div className="panel">
      <span className="label">Vitals</span>
      <div
        style={{
          marginTop: 10,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 8,
        }}
      >
        {Object.keys(VITAL_META).map((key) => {
          const value = patient.vitals[key][hourIdx];
          const hoursSince = patient.vitals_hours_since[key][hourIdx];
          const meta = VITAL_META[key];
          const stale = hoursSince > 6;
          return (
            <div
              key={key}
              style={{
                background: "var(--surface-2)",
                borderRadius: 4,
                padding: "8px 10px",
              }}
            >
              <p className="label">{meta.label}</p>
              <p className="mono" style={{ fontSize: 17, fontWeight: 600, marginTop: 2, color: "#f97b7b" }}>
                {value}
                <span style={{ fontSize: 11, fontWeight: 400, color: "var(--muted)" }}>
                  {" "}
                  {meta.unit}
                </span>
              </p>
              <p
                className="label"
                style={{ marginTop: 2, color: stale ? "var(--amber)" : "var(--muted)" }}
              >
                {hoursSince === 0
                  ? "just now"
                  : hoursSince >= 999
                  ? "not yet measured"
                  : `last measured ${Math.round(hoursSince)}h ago`}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
