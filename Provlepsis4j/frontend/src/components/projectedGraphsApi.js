// projectedGraphsApi.js
import { api } from './api';

export async function fetchProjectedGraphs() {
  try {
    const { data } = await api.get('/gds/graphs');
    return data;
  } catch (e) {
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string') throw new Error(detail);
    if (detail?.message) throw new Error(detail.message + (detail.HINT ? `\n${detail.HINT}` : ''));
    throw new Error(e?.message || 'Failed to fetch projected graphs');
  }
}

export async function setGraphContext(graphName) {
  try {
    const { data } = await api.post('/gds/graph-context', { graphName });
    return data;
  } catch (e) {
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string') throw new Error(detail);
    if (detail?.message) throw new Error(detail.message);
    throw new Error(e?.message || 'Failed to set graph context');
  }
}

// fetch store (DB) summary (nodes, rels)
export async function fetchStoreSummary() {
  const { data } = await api.get('/gds/store/summary')
  return data // { nodes, relationships }
}

// project the store graph to GDS with a given name
export async function projectStoreGraph(name) {
  const { data } = await api.post('/gds/graph/project-store', { name })
  return data
}

// drop a projected graph
export async function dropProjectedGraph(name) {
  const { data } = await api.post('/gds/graph/drop', { name });
  return data;
}

// List databases in the current instance
export async function fetchDatabases() {
  try {
    const { data } = await api.get('/databases');
    return data;
  } catch (e) {
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string') throw new Error(detail);
    if (detail?.message) throw new Error(detail.message);
    throw new Error(e?.message || 'Failed to fetch databases');
  }
}

// Switch the active database (subsequent ops use this DB)
export async function useDatabase(name) {
  try {
    const { data } = await api.post('/databases/use', { name });
    return data;
  } catch (e) {
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string') throw new Error(detail);
    if (detail?.message) throw new Error(detail.message);
    throw new Error(e?.message || `Failed to switch to database "${name}"`);
  }
}

export async function deleteDatabase(name) {
  try {
    const { data } = await api.post('/databases/drop', { name });
    return data; 
  } catch (e) {
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string') throw new Error(detail);
    if (detail?.message) throw new Error(detail.message);
    throw new Error(e?.message || `Failed to delete database "${name}"`);
  }
}

export async function normalizeExistingDatabase(name) {
  try {
    const { data } = await api.post('/load/normalize-existing', { db: name });
    return data; 
  } catch (e) {
    const detail = e?.response?.data?.detail;
    if (typeof detail === 'string') throw new Error(detail);
    if (detail?.message) throw new Error(detail.message);
    throw new Error(e?.message || `Failed to normalize database "${name}"`);
  }
}
