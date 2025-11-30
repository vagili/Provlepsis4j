import React, { useEffect, useState } from "react";

import ProjectedGraphsPane from "./components/ProjectedGraphsPane.jsx";
import ConfigModal from "./components/ConfigModal.jsx";
import SidePanel from "./components/SidePanel.jsx";
import { useGraph } from "./graphContext.jsx";
import { isConfigured, clearConfigured, ping } from "./components/api";
import Heatmap from "./pages/Heatmap.jsx";
import Query from "./pages/Query.jsx";

const headerButtonStyle = {
  height: 32,
  borderRadius: 4,
  border: "1px solid #ccc",
  background: "#f8f8f8",
  padding: "0 12px",
  cursor: "pointer",
  fontSize: 12,
};

export default function App() {
  const [page, setPage] = useState("heatmap"); // "query" | "heatmap"
  const [showConfig, setShowConfig] = useState(!isConfigured());
  const [backendUp, setBackendUp] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        await ping();
        setBackendUp(true);
      } catch {
        setBackendUp(false);
      }
    })();
  }, []);

  const headerStyle = {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "8px 12px",
    borderBottom: "1px solid #ddd",
  };

  return (
    <div
      style={{
        height: "100vh",
        display: "grid",
        gridTemplateRows: "56px 1fr",
        fontFamily: "ui-sans-serif, system-ui",
      }}
    >
      <header style={headerStyle}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <img
            src="/Provlepsis.png"
            alt="Provlepsis logo"
            style={{ height: 60, width: 120, marginBottom: 10 }}
          />
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <GraphToolbar />
          <button
            onClick={() => setShowConfig(true)}
            style={headerButtonStyle}
          >
            Change Connection
          </button>
        </div>
      </header>

      {/* Left: compact side panel. Right: query area with padding */}
      <main
        style={{
          height: "100%",
          overflow: "hidden",
          display: "grid",
          gridTemplateColumns: "320px minmax(0, 1fr)",
        }}
      >
        <SidePanel page={page} setPage={setPage} />

        <div style={{ height: "100%", overflow: "auto", padding: 12 }}>
          {page === "query" ? <Query /> : <Heatmap />}
        </div>
      </main>

      <ConfigModal open={showConfig} onClose={() => setShowConfig(false)} />
    </div>
  );
}

function GraphToolbar() {
  const [openGraphs, setOpenGraphs] = useState(false);
  const { graphName } = useGraph();

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <button
        onClick={() => setOpenGraphs(true)}
        style={headerButtonStyle}
      >
        Projected Graphs
      </button>

      <ProjectedGraphsPane
        open={openGraphs}
        onClose={() => setOpenGraphs(false)}
      />
    </div>
  );
}
