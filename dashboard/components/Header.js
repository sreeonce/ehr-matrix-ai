export default function Header({ patients, selectedId, onSelect }) {
  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "18px 0 16px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        {/* Drop your logo file at dashboard/public/logo.png (or .svg - update
            the src below to match) and it will render here. */}
        <img
          src="/logo.png"
          alt="EHR-Matrix AI logo"
          style={{ height: 28, width: "auto" }}
        />
        <h1 style={{ fontSize: 19, fontWeight: 400, color: "var(--naming)" }}>EHR-Matrix AI</h1>
      </div>

      <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span className="label" style={{ color: "var(--naming)" }}>
          Patient
        </span>
        <select
          value={selectedId}
          onChange={(e) => onSelect(e.target.value)}
          style={{
            font: "inherit",
            fontSize: 14,
            padding: "6px 10px",
            borderRadius: "var(--radius)",
            border: "1px solid var(--line)",
            background: "var(--surface)",
            color: "var(--ink)",
          }}
        >
          {patients.map((p) => (
            <option key={p.id} value={p.id}>
              {p.id}
            </option>
          ))}
        </select>
      </label>
    </header>
  );
}
