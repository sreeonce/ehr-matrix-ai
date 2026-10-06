export default function Footer() {
  return (
    <p
      className="label"
      style={{
        textAlign: "center",
        marginTop: 20,
        lineHeight: 1.6,
        color: "#9c9c9c",
      }}
    >
      Research prototype — not a clinical decision tool. Vitals and labs from the
      PhysioNet 2019 sepsis challenge data; clinical notes are synthetically generated.
    </p>
  );
}
