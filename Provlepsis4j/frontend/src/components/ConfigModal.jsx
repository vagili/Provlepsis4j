// frontend/src/components/ConfigModal.jsx
import { useState, useEffect } from "react";
import { setNeo4jConfig, markConfigured, isConfigured, clearConfigured } from "./api";

export default function ConfigModal({ open, onClose }) {
  const [uri, setUri] = useState("bolt://host.docker.internal:7687");
  const [user, setUser] = useState("neo4j");
  const [password, setPassword] = useState("mypassword123");
  const [database, setDatabase] = useState("neo4j");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!open) setErr("");
  }, [open]);

  if (!open) return null;

  async function handleSubmit(e) {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      await setNeo4jConfig({ uri, user, password, database });
      markConfigured();
      onClose?.();
    } catch (e) {
      setErr(e?.response?.data ?? e?.message ?? "Failed to connect");
      clearConfigured();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={backdrop}>
      <div style={card}>
        <h3 style={{ marginTop: 0 }}>Connect to Neo4j</h3>
        <p style={{ marginTop: 0, color: "#555" }}>
          Enter your Neo4j connection once. You can change it later from the header.
        </p>

        <form onSubmit={handleSubmit} style={{ display: "grid", gap: 8 }}>
          <label style={label}>
            <span>URI</span>
            <input value={uri} onChange={e=>setUri(e.target.value)} style={input} placeholder="bolt://host:7687" />
          </label>

          <div style={{ display: "grid", gap: 8, gridTemplateColumns: "1fr 1fr" }}>
            <label style={label}>
              <span>User</span>
              <input value={user} onChange={e=>setUser(e.target.value)} style={input} />
            </label>
            <label style={label}>
              <span>Password</span>
              <input type="password" value={password} onChange={e=>setPassword(e.target.value)} style={input} />
            </label>
          </div>

          <label style={label}>
            <span>Database</span>
            <input value={database} onChange={e=>setDatabase(e.target.value)} style={input} />
          </label>

          {err ? <div style={errorBox}>{String(err)}</div> : null}

          <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 8 }}>
            <button type="button" onClick={onClose} disabled={busy}>Cancel</button>
            <button type="submit" disabled={busy}>{busy ? "Connecting…" : "Connect"}</button>
          </div>
        </form>
      </div>
    </div>
  );
}

const backdrop = {
  position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)",
  display: "grid", placeItems: "center", zIndex: 50
};
const card = {
  width: 520, maxWidth: "92vw", background: "#fff", padding: 16,
  borderRadius: 12, boxShadow: "0 10px 30px rgba(0,0,0,0.2)"
};
const label = { display: "grid", gap: 4, fontSize: 14 };
const input = { padding: "8px 10px", border: "1px solid #ccc", borderRadius: 8, fontSize: 14 };
const errorBox = { background: "#fee", border: "1px solid #f99", padding: 8, borderRadius: 6, color: "#900" };
