// src/api.js
import axios from "axios";

const BASE_URL =
  import.meta.env.VITE_API_BASE ||
  import.meta.env.VITE_API ||
  "http://localhost:8000";

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 0,
});

// ---- runtime-config flags ----
const KEY = "neo4jConfigured";
export function markConfigured() { localStorage.setItem(KEY, "true"); }
export function clearConfigured() { localStorage.removeItem(KEY); }
export function isConfigured() { return localStorage.getItem(KEY) === "true"; }

// ---- backend health ----
export async function ping() {
  const { data } = await api.get("/health");
  return data;
}

// ---- backend: runtime Neo4j config ----
export async function setNeo4jConfig({ uri, user, password, database }) {
  const { data } = await api.post("/config/neo4j", { uri, user, password, database });
  // Notify listeners that the active DB changed
  try {
    window.dispatchEvent(
      new CustomEvent("neo4j:db-changed", { detail: { db: data?.database || database } })
    );
  } catch {}
  return data;
}

export async function uploadGraph(
  edgesFile,
  featuresFile,
  opts = { isTemporal: false, timestampColumn: "", dataset_name: undefined }
) {
  const {
    isTemporal = false,
    timestampColumn = "",
    dataset_name,
  } = opts || {};

  const fd = new FormData();

  fd.append("edges", edgesFile);

  if (featuresFile) {
    fd.append("features", featuresFile);
  }

  // temporal flags
  fd.append("isTemporal", isTemporal ? "true" : "false");
  if (isTemporal && timestampColumn.trim()) {
    fd.append("timestampColumn", timestampColumn.trim());
  }

  // custom dataset name
  if (dataset_name && String(dataset_name).trim()) {
    fd.append("dataset_name", String(dataset_name).trim());
  }

  const { data } = await api.post("/load/graph", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}


// --- Split master call: POST /split/execute
export async function executeSplit({
  trainGraphName = "trainGraph",
  testGraphName = "testGraph",
  valGraphName = "valGraph",
  testHoldout,
  valHoldout,
}) {
  if (typeof api?.post === "function") {
    const { data } = await api.post("/split/execute", {
      trainGraphName,
      testGraphName,
      valGraphName,
      testHoldout,
      valHoldout,
    });
    return data;
  }

  // Fallback to fetch if no axios instance exists in this file
  const API_BASE =
    (import.meta?.env?.VITE_API_BASE && import.meta.env.VITE_API_BASE.replace(/\/+$/, "")) ||
    (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

  const res = await fetch(`${API_BASE}/split/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      trainGraphName,
      testGraphName,
      valGraphName,
      testHoldout,
      valHoldout,
    }),
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function runLP(params) {
  // remove null/undefined/empty-string and literal "null"
  const payload = Object.fromEntries(
    Object.entries(params).filter(([_, v]) => {
      if (v === null || v === undefined) return false;
      if (typeof v === "string" && v.trim() === "") return false;
      if (typeof v === "string" && v.trim().toLowerCase() === "null") return false;
      return true;
    })
  );

  // ensure numeric fields are numbers (optional)
  ["negativeRatio", "predictK", "candidateMultiplier", "probThreshold"].forEach((k) => {
    if (payload[k] !== undefined) payload[k] = Number(payload[k]);
  });

  const res = await api.post("/lp/run", payload);
  return res.data;
}

export async function queryBoth(body) {
  const { data } = await api.post("/query/both", body);
  return data;
}

export async function queryBothGraph(body) {
  const { data } = await api.post("/query/both2", { ...body, mode: "graph" });
  return data;
}

export async function getQueryDbs() {
  const { data } = await api.get("/query/dbs");
  return data;
}

export async function queryPresent(payload) {
  const { data } = await api.post("/query/present", payload);
  return data;
}

export async function queryFuture(payload) {
  const { data } = await api.post("/query/future", payload); 
  return data;
}

export async function getPredictedDbOptions() {
  const { data } = await api.get("/lp/predicted/dbs");
  return data; 
}

export async function getPredictedEdgeSets() {
  const { data } = await api.get("/lp/predicted/edges");
  return data; 
}

export async function getPredictedEdgeSetsByDb() {
  const { data } = await api.get("/lp/predicted/edges/by-db");
  return data;
}

export async function iteratePrediction(body, opts = {}) {
  
  const { data } = await api.post("/lp-iter/iterate", body, {
    timeout: 0,
    ...(opts || {}),
  });
  return data;
}

// LP metrics for current DB (per embedding family)
export async function getLpMetrics() {
  const { data } = await api.get("/lp/metrics");
  return data; 
}

// Ground truth (test-set) edges for current DB
export async function getLpGroundTruthEdges() {
  const res = await api.get("/lp/test-edges");
  return res.data; 
}

export async function getPredictedTimestampCount(dbName) {
  if (!dbName) return null;
  const safe = encodeURIComponent(dbName);
  const { data } = await api.get(`/lp/predicted/timestamps/${safe}`);
 
  return data;
}
