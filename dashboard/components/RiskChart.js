import { useRef } from "react";

const W = 900;
const H = 220;
const PAD = { top: 16, right: 16, bottom: 28, left: 36 };

function pathFor(hours, risk) {
  const minH = hours[0];
  const maxH = hours[hours.length - 1];
  const xAt = (h) =>
    PAD.left + ((h - minH) / Math.max(maxH - minH, 1)) * (W - PAD.left - PAD.right);
  const yAt = (r) => PAD.top + (1 - r) * (H - PAD.top - PAD.bottom);
  const points = hours.map((h, i) => [xAt(h), yAt(risk[i])]);
  return { points, xAt, yAt };
}

export default function RiskChart({ patient, hoverIdx, onHover }) {
  const { hours, risk, sepsis_onset_hour } = patient;
  const svgRef = useRef(null);

  const { points, xAt, yAt } = pathFor(hours, risk);
  const linePath = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ");

  function handleMove(e) {
    const rect = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let nearest = 0;
    let best = Infinity;
    points.forEach(([x], i) => {
      const d = Math.abs(x - px);
      if (d < best) {
        best = d;
        nearest = i;
      }
    });
    onHover(nearest);
  }

  const onsetX = sepsis_onset_hour != null ? xAt(sepsis_onset_hour) : null;
  const hoverPoint = hoverIdx != null ? points[hoverIdx] : null;

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span className="label">Predicted risk over time</span>
        <span className="label">hover a point for detail</span>
      </div>

      <div style={{ position: "relative" }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          height={H}
          onMouseMove={handleMove}
          onMouseLeave={() => onHover(null)}
          role="img"
          aria-label={`Risk over time for ${patient.id}, from ${Math.round(
            risk[0] * 100
          )}% to a peak of ${Math.round(Math.max(...risk) * 100)}%`}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line
                x1={PAD.left}
                x2={W - PAD.right}
                y1={yAt(f)}
                y2={yAt(f)}
                stroke="var(--line)"
                strokeWidth="1"
              />
              <text x={6} y={yAt(f) + 4} fontSize="10" fill="var(--muted)">
                {Math.round(f * 100)}
              </text>
            </g>
          ))}

          {onsetX != null && (
            <line
              x1={onsetX}
              x2={onsetX}
              y1={PAD.top}
              y2={H - PAD.bottom}
              stroke="var(--red)"
              strokeWidth="1"
              strokeDasharray="3,3"
            />
          )}

          <path d={linePath} fill="none" stroke="var(--teal)" strokeWidth="2" />

          {hoverPoint && (
            <>
              <line
                x1={hoverPoint[0]}
                x2={hoverPoint[0]}
                y1={PAD.top}
                y2={H - PAD.bottom}
                stroke="var(--muted)"
                strokeWidth="1"
                strokeDasharray="2,3"
              />
              <circle cx={hoverPoint[0]} cy={hoverPoint[1]} r="4" fill="var(--teal-deep)" />
            </>
          )}

          {hours.map((h, i) =>
            h % Math.max(Math.round(hours.length / 8), 1) === 0 ? (
              <text key={h} x={xAt(h)} y={H - 8} fontSize="10" fill="var(--muted)" textAnchor="middle">
                {h}
              </text>
            ) : null
          )}
        </svg>

        {hoverIdx != null && (
          <div
            className="mono"
            style={{
              position: "absolute",
              left: `${(hoverPoint[0] / W) * 100}%`,
              top: 0,
              transform: "translate(8px, 0)",
              background: "var(--ink)",
              color: "#fff",
              fontSize: 12,
              padding: "6px 9px",
              borderRadius: 4,
              pointerEvents: "none",
              whiteSpace: "nowrap",
            }}
          >
            hour {hours[hoverIdx]} &middot; risk {Math.round(risk[hoverIdx] * 100)}%
          </div>
        )}
      </div>
    </div>
  );
}