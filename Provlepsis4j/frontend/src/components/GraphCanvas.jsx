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

  // 3) fallbacks – keep these so old graphs still show *something*
  if (Array.isArray(node.labels) && node.labels.length > 0) {
    return String(node.labels[0]);
  }

  if (node.id != null) {
    return String(node.id);
  }

  return "";
}


/**
 * Expects:
 * graph = {
 *   nodes: [{ id, labels?: string[], ... }],
 *   edges: [{ id?, type?, source? | start, target? | end, predicted?, properties?, ... }]
 * }
 */
export default function GraphCanvas({ graph, height = 420 }) {
  const ref = useRef(null);
  const cyRef = useRef(null);
  const [error, setError] = useState("");

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
            // use our helper instead of always taking the first label
            label,
          },
        };
      });

      // Edges
      const nodeIdSet = new Set(nodeElements.map((el) => el.data.id));
      const edgeElements = (graph?.edges || [])
        .map((e) => {
          const src = String(e?.source ?? e?.start ?? "");
          const trg = String(e?.target ?? e?.end ?? "");
          const id =
            e?.id != null ? String(e.id) : `${src}->${trg}:${e?.type ?? ""}`;

          // --- pull predicted from any plausible place and normalize to 0/1 ---
          const rawPred =
            e?.predicted ??
            e?.properties?.predicted ??
            e?.data?.predicted ??
            e?.attrs?.predicted;

          const predicted =
            rawPred === 1 ||
            rawPred === "1" ||
            rawPred === true ||
            rawPred === "true"
              ? 1
              : 0;

          return {
            data: {
              id,
              source: src,
              target: trg,
              //label: e?.type ? String(e.type) : "",
              type: e?.type ? String(e.type) : "",
              predicted,
            },
          };
        })
        .filter(
          (el) =>
            el.data.source &&
            el.data.target &&
            nodeIdSet.has(el.data.source) &&
            nodeIdSet.has(el.data.target)
        );

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
              "line-color": "#94a3b8",
              "curve-style": "bezier",
              "target-arrow-shape": "triangle",
              "target-arrow-color": "#94a3b8",
              "font-size": 9,
              label: "data(label)",
            },
          },
          // Predicted edges (any truthy)
          {
            selector: "edge[predicted > 0]",
            style: {
              "line-color": "red",
              "target-arrow-color": "red",
              width: 2,
            },
          },
          // Optional fallback if you also tag type as 'PREDICTED'
          {
            selector: 'edge[type = "PREDICTED"]',
            style: {
              "line-color": "red",
              "target-arrow-color": "red",
              width: 2,
            },
          },
        ],
      });

      cyRef.current = cy;
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
    <div
      ref={ref}
      style={{
        width: "100%",
        height,
        border: "1px solid #e5e7eb",
        borderRadius: 12,
        minWidth: 0,
        overflow: "hidden",
      }}
    />
  );
}
