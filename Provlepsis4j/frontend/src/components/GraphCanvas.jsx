// frontend/src/components/GraphCanvas.jsx
import React, { useEffect, useRef, useState } from "react";
import cytoscape from "cytoscape";

function getNodeLabel(node) {
  if (!node) return "";

  // 1) explicit business id sent from backend
  if (node.nodeId !== undefined && node.nodeId !== null) {
    return String(node.nodeId);
  }

  // 2) if someone sends nested props with id
  if (node.properties && node.properties.id != null) {
    return String(node.properties.id);
  }

  // 3) fallbacks
  if (Array.isArray(node.labels) && node.labels.length > 0) {
    return String(node.labels[0]);
  }

  if (node.id != null) {
    return String(node.id);
  }

  return "";
}

export default function GraphCanvas({ graph, height = 420 }) {
  const ref = useRef(null);
  const cyRef = useRef(null);
  const [error, setError] = useState("");
  const [selectedEdge, setSelectedEdge] = useState(null);
  const formatValue = (v) =>
    typeof v === "number" ? v.toFixed(3) : v;

  // color helpers
  const lerp = (a, b, t) => a + (b - a) * t;

  const RED = [239, 68, 68];      // #ef4444
  const YELLOW = [250, 204, 21];  // #facc15
  const GREEN = [74, 222, 128];   // #4ade80

  const mixRgb = (from, to, t) => {
    const r = Math.round(lerp(from[0], to[0], t));
    const g = Math.round(lerp(from[1], to[1], t));
    const b = Math.round(lerp(from[2], to[2], t));
    return `rgb(${r}, ${g}, ${b})`;
  };


  useEffect(() => {
    if (!ref.current) return;

    if (cyRef.current) {
      cyRef.current.destroy();
      cyRef.current = null;
    }

    try {
      // Nodes
      const nodeElements = (graph?.nodes || []).map((n) => {
        const id = String(n?.id ?? "");
        const label = getNodeLabel(n);
        return {
          data: {
            id,
            label,
          },
        };
      });

            // Edges
      const nodeIdSet = new Set(nodeElements.map((el) => el.data.id));
      const baseEdgeColor = "#94a3b8";

      // ---- first pass: normalize edge data, keep predicted as a number ----
      const rawEdges = (graph?.edges || [])
        .map((e) => {
          const src = String(e?.source ?? e?.start ?? "");
          const trg = String(e?.target ?? e?.end ?? "");
          if (!src || !trg) return null;
          if (!nodeIdSet.has(src) || !nodeIdSet.has(trg)) return null;

          const id =
            e?.id != null ? String(e.id) : `${src}->${trg}:${e?.type ?? ""}`;

          const predRaw =
            e?.predicted ??
            e?.properties?.predicted ??
            e?.data?.predicted ??
            e?.attrs?.predicted ??
            null;

          const predicted =
            predRaw === null || predRaw === undefined
              ? null
              : Number(predRaw);

          const probability =
            e.probability ?? e.properties?.probability ?? null;
          const timestamp =
            e.timestamp ?? e.properties?.timestamp ?? null;

          return {
            id,
            src,
            trg,
            type: e?.type ? String(e.type) : "",
            predicted,
            probability,
            timestamp,
          };
        })
        .filter(Boolean);

      // ---- compute min / max predicted > 0 ----
      let minPred = Infinity;
      let maxPred = -Infinity;
      for (const e of rawEdges) {
        if (e.predicted != null && e.predicted > 0) {
          if (e.predicted < minPred) minPred = e.predicted;
          if (e.predicted > maxPred) maxPred = e.predicted;
        }
      }
      const havePredicted =
        minPred !== Infinity && maxPred !== -Infinity;

      // ---- second pass: assign colors with red–yellow–green scheme ----
      const edgeElements = rawEdges.map((e) => {
        let color = baseEdgeColor;

        if (havePredicted && e.predicted != null && e.predicted > 0) {
          // clamp to [minPred, maxPred]
          let p = e.predicted;
          if (p < minPred) p = minPred;
          if (p > maxPred) p = maxPred;

          if (maxPred === minPred) {
            // only one predicted value → treat as min/present → green
            color = mixRgb(GREEN, GREEN, 0); // solid green
          } else {
            const midPred = (minPred + maxPred) / 2;

            if (p <= midPred) {
              // [minPred, midPred] → green → yellow  (present-ish)
              const denom = midPred - minPred || 1;
              const t = (p - minPred) / denom; // 0 at min, 1 at mid
              color = mixRgb(GREEN, YELLOW, t);
            } else {
              // (midPred, maxPred] → yellow → red  (further future)
              const denom = maxPred - midPred || 1;
              const t = (p - midPred) / denom; // 0 at mid, 1 at max
              color = mixRgb(YELLOW, RED, t);
            }
          }
        }

        return {
          data: {
            id: e.id,
            source: e.src,
            target: e.trg,
            type: e.type,
            predicted: e.predicted,
            probability: e.probability,
            timestamp: e.timestamp,
            color,
          },
        };
      });

      const elements = [...nodeElements, ...edgeElements];

      const cy = cytoscape({
        container: ref.current,
        elements,
        layout: {
          name: "cose",
          animate: false,
          fit: false,
          padding: 40,
          nodeRepulsion: 200000,
          idealEdgeLength: 100,
          gravity: 0.1,
        },
        style: [
          // Nodes
          {
            selector: "node",
            style: {
              "background-color": "#4f46e5",
              label: "data(label)",
              "font-size": 10,
              color: "#111",
            },
          },
          // Default edges
          {
            selector: "edge",
            style: {
              width: 2,
              "line-color": "data(color)",
              "curve-style": "bezier",
              "target-arrow-shape": "triangle",
              "target-arrow-color": "data(color)",
              "font-size": 9,
              label: "data(label)",
            },
          },

        ],
      });

      cyRef.current = cy;

      // --- edge click → show bubble with details ---
      cy.on("tap", "edge", (evt) => {
        const d = evt.target.data();
        setSelectedEdge({
          id: d.id,
          type: d.type,
          source: d.source,
          target: d.target,
          probability: d.probability,
          //timestamp: d.timestamp,
        });
      });

      // click on background → hide bubble
      cy.on("tap", (evt) => {
        if (evt.target === cy) {
          setSelectedEdge(null);
        }
      });

      setError("");
    } catch (err) {
      console.error(err);
      setError(
        (err && (err.message || String(err))) || "Failed to render graph."
      );
    }

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [graph]);

  if (error) {
    return (
      <div
        style={{
          width: "100%",
          height,
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          display: "grid",
          placeItems: "center",
          color: "#b91c1c",
          fontSize: 13,
          padding: 12,
        }}
      >
        {error}
      </div>
    );
  }


  return (
  <div style={{ position: "relative", width: "100%", height }}>
    <div
      ref={ref}
      style={{ width: "100%", height: "100%", borderRadius: 12, overflow: "hidden" }}
    />

    {error && (
      <div style={{ color: "#b91c1c", fontSize: 12, marginTop: 4 }}>{error}</div>
    )}

        {selectedEdge && (
      <div
        style={{
          position: "absolute",
          top: 8,
          right: 8,
          background: "rgba(255,255,255,0.97)",
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          padding: "8px 10px",
          boxShadow: "0 4px 10px rgba(0,0,0,0.08)",
          fontSize: 12,
          maxWidth: 220,
          zIndex: 10,
        }}
      >
        {/* title */}
        <div
          style={{
            fontWeight: 700,
            fontSize: 13,
            marginBottom: 6,
          }}
        >
          Edge Properties
        </div>

        {/* probability */}
        {selectedEdge.probability != null && (
          <div style={{ marginBottom: 2 }}>
            <strong>probability:</strong>{" "}
            {formatValue(selectedEdge.probability)}
          </div>
        )}

        {/* timestamp */}
        {/* {selectedEdge.timestamp != null && (
          <div style={{ marginBottom: 2 }}>
            <strong>timestamp:</strong>{" "}
            {formatValue(selectedEdge.timestamp)}
          </div>
        )} */}

        <button
          onClick={() => setSelectedEdge(null)}
          style={{
            marginTop: 6,
            border: "1px solid #d1d5db",
            borderRadius: 4,
            padding: "2px 6px",
            fontSize: 11,
            background: "#f9fafb",
            cursor: "pointer",
          }}
        >
          Close
        </button>
      </div>
    )}

  </div>
);

}
