// frontend/src/ProjectedGraphsPane.jsx
import { useEffect, useState } from 'react';
import {
  fetchProjectedGraphs,
  setGraphContext,
  fetchStoreSummary,
  projectStoreGraph,
  dropProjectedGraph,
  fetchDatabases,
  useDatabase,
  deleteDatabase,
  normalizeExistingDatabase
} from './projectedGraphsApi';
import { useGraph } from '../graphContext';

// helpers
const fmt = (n) => (typeof n === 'number' ? n.toLocaleString() : n ?? '—');
const fmtDate = (s) => {
  try {
    if (!s) return '—';
    const d = new Date(s);
    return Number.isNaN(d.getTime()) ? String(s) : d.toLocaleString();
  } catch {
    return String(s ?? '—');
  }
};

export default function ProjectedGraphsPane({ open, onClose }) {
  const [loading, setLoading] = useState(false);
  const [rows, setRows] = useState([]);
  const [error, setError] = useState(null);

  // Store (DB) graph info + projection name input
  const [storeInfo, setStoreInfo] = useState(null);
  const [storeError, setStoreError] = useState(null);
  const [newName, setNewName] = useState('fullGraph');
  const [projecting, setProjecting] = useState(false);

  // Track which projected graph is dropping
  const [dropping, setDropping] = useState(null);

  // Databases state
  const [dbs, setDbs] = useState({ current: null, databases: [] });
  const [dbError, setDbError] = useState(null);
  const [switchingDb, setSwitchingDb] = useState(null);
  const [deletingDb, setDeletingDb] = useState(null); // NEW: which DB is being deleted

  const { graphName, setGraphName } = useGraph();

  useEffect(() => {
    if (!open) return;

    setLoading(true);
    setError(null);
    setStoreError(null);
    setDbError(null);

    Promise.all([
      fetchDatabases().then(setDbs).catch((e) => setDbError(String(e))),
      fetchStoreSummary().then(setStoreInfo).catch((e) => setStoreError(String(e))),
      fetchProjectedGraphs().then(setRows).catch((e) => setError(String(e))),
    ]).finally(() => setLoading(false));
  }, [open]);

    const switchDb = async (name) => {
    try {
      setSwitchingDb(name);

      // 1) switch active DB (what you already had)
      await useDatabase(name);

      // 2) normalize features in that DB (same pipeline as CSV import)
      try {
        await normalizeExistingDatabase(name);
      } catch (e) {
        console.error('Failed to normalize DB', e);
        // optional: surface error
        // setDbError(String(e));
      }

      // 3) notify the rest of the app
      try {
        window.dispatchEvent(
          new CustomEvent('neo4j:db-changed', { detail: { db: name } })
        );
      } catch {}

      // 4) refresh everything after switching DB
      const [store, graphs, dblist] = await Promise.all([
        fetchStoreSummary().catch((e) => {
          setStoreError(String(e));
          return null;
        }),
        fetchProjectedGraphs().catch((e) => {
          setError(String(e));
          return [];
        }),
        fetchDatabases().catch((e) => {
          setDbError(String(e));
          return null;
        }),
      ]);

      if (store) setStoreInfo(store);
      if (graphs) setRows(graphs);
      if (dblist) setDbs(dblist);
    } finally {
      setSwitchingDb(null);
    }
  };


  // NEW: delete a database (guarded)
  const onDeleteDb = async (name) => {
    if (!name) return;
    if (['neo4j', 'system'].includes(name.toLowerCase())) {
      alert(`Cannot delete reserved database "${name}".`);
      return;
    }
    if (name === dbs.current) {
      alert(`Cannot delete the currently selected database "${name}". Switch to another DB first.`);
      return;
    }
    if (!window.confirm(`Permanently delete database "${name}"?\nThis cannot be undone.`)) return;

    setDeletingDb(name);
    try {
      await deleteDatabase(name);
      const freshed = await fetchDatabases();
      setDbs(freshed);
    } catch (e) {
      alert(e?.message || `Failed to delete database "${name}".`);
    } finally {
      setDeletingDb(null);
    }
  };

  const selectGraph = async (name) => {
    await setGraphContext(name);
    setGraphName(name);
    onClose();
  };

  const doProjectStore = async () => {
    try {
      setProjecting(true);
      setStoreError(null);
      await projectStoreGraph(newName || 'fullGraph');
      const freshed = await fetchProjectedGraphs();
      setRows(freshed);
      await selectGraph(newName || 'fullGraph');
    } catch (e) {
      setStoreError(String(e));
    } finally {
      setProjecting(false);
    }
  };

  // drop a projected graph, refresh list, and handle current selection
  const doDrop = async (name) => {
    if (!window.confirm(`Drop projected graph "${name}"? This cannot be undone.`)) return;
    try {
      setDropping(name);
      await dropProjectedGraph(name);
      const freshed = await fetchProjectedGraphs();
      setRows(freshed);

      if (graphName === name) {
        // If we dropped the currently-selected graph, choose a fallback
        if (freshed.length > 0) {
          const fallback = freshed[0].graphName;
          await setGraphContext(fallback);
          setGraphName(fallback);
        } else {
          // Clear selection if nothing remains
          try {
            await setGraphContext(null);
          } catch (_) {
            /* backend may not accept null; ignore */
          }
          setGraphName(null);
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setDropping(null);
    }
  };

  if (!open) return null;

  return (
    <div
      style={{
        position: 'fixed',
        right: 0,
        top: 0,
        height: '100vh',
        width: 420,
        background: '#111827',
        color: '#fff',
        boxShadow: '-8px 0 24px rgba(0,0,0,0.25)',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 50,
      }}
    >
      <div
        style={{
          padding: '12px 16px',
          borderBottom: '1px solid #374151',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <strong>Graphs</strong>
        <button
          onClick={onClose}
          style={{
            background: '#374151',
            color: '#fff',
            border: 'none',
            padding: '6px 10px',
            borderRadius: 6,
            cursor: 'pointer',
          }}
        >
          Close
        </button>
      </div>

      {/* Databases (instance) */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #374151' }}>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>Databases (instance)</div>

        {dbError && (
          <div style={{ color: '#fca5a5', fontSize: 12, marginBottom: 8 }}>
            {dbError}
          </div>
        )}

        <div style={{ fontSize: 12, color: '#9CA3AF', marginBottom: 8 }}>
          Current DB: <b>{dbs.current ?? '—'}</b>
        </div>

        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            maxHeight: 160,
            overflowY: 'auto',
          }}
        >
          {dbs.databases?.map((d) => (
            <div
              key={d.name}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                border: '1px solid #374151',
                borderRadius: 8,
                padding: 8,
                background: '#1F2937',
              }}
            >
              <div>
                <div style={{ fontWeight: 600 }}>
                  {d.name}
                  {d.name === dbs.current ? ' • (current)' : ''}
                </div>
                <div style={{ fontSize: 12, color: '#D1D5DB' }}>
                  Status: {d.currentStatus} · Access: {d.access} · Role: {d.role}
                </div>
              </div>

              {/* Use + Delete buttons */}
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                {/* Use / Select */}
                <button
                  onClick={() => switchDb(d.name)}
                  disabled={switchingDb === d.name || d.name === dbs.current}
                  style={{
                    padding: '6px 10px',
                    borderRadius: 6,
                    border: '1px solid #6B7280',
                    background:
                      switchingDb === d.name ? '#374151' : '#111827',
                    color: '#fff',
                    cursor: 'pointer',
                  }}
                  title={
                    d.name === dbs.current
                      ? 'Already selected'
                      : 'Use this database'
                  }
                >
                  {switchingDb === d.name
                    ? 'Selecting…'
                    : d.name === dbs.current
                    ? 'Selected'
                    : 'Use'}
                </button>

                {/* NEW: Delete */}
                <button
                  onClick={() => onDeleteDb(d.name)}
                  disabled={
                    deletingDb === d.name ||
                    ['neo4j', 'system'].includes(d.name.toLowerCase()) ||
                    d.name === dbs.current
                  }
                  title={
                    d.name === dbs.current
                      ? "Can't delete the current database"
                      : ['neo4j', 'system'].includes(d.name.toLowerCase())
                      ? 'Cannot delete reserved databases'
                      : 'Delete this database'
                  }
                  style={{
                    padding: '6px 10px',
                    borderRadius: 6,
                    border: '1px solid #ef4444',
                    background:
                      deletingDb === d.name ? '#7f1d1d' : '#ef4444',
                    color: '#fff',
                    opacity:
                      ['neo4j', 'system'].includes(d.name.toLowerCase()) ||
                      d.name === dbs.current
                        ? 0.5
                        : 1,
                    cursor: 'pointer',
                  }}
                >
                  {deletingDb === d.name ? 'Deleting…' : 'Delete'}
                </button>
              </div>
            </div>
          ))}
          {!dbs.databases?.length && (
            <div style={{ color: '#9CA3AF' }}>
              No databases found (need admin rights to list).
            </div>
          )}
        </div>
      </div>

      {/* Store (DB) graph section */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #374151' }}>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>Store Graph (DB)</div>
        {storeError && (
          <div style={{ color: '#fca5a5', fontSize: 12, marginBottom: 8 }}>
            {storeError}
          </div>
        )}
        {storeInfo ? (
          <>
            <div style={{ fontSize: 12, color: '#D1D5DB' }}>
              Nodes: {fmt(storeInfo.nodes)} · Rels: {fmt(storeInfo.relationships)}
            </div>
            <div
              style={{
                display: 'flex',
                gap: 8,
                marginTop: 10,
                alignItems: 'center',
              }}
            >
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Projection name"
                style={{
                  flex: 1,
                  padding: '6px 8px',
                  borderRadius: 6,
                  border: '1px solid #6B7280',
                  background: '#111827',
                  color: '#fff',
                }}
              />
              <button
                onClick={doProjectStore}
                disabled={projecting || !newName}
                style={{
                  padding: '6px 10px',
                  borderRadius: 6,
                  border: '1px solid #6B7280',
                  background: projecting ? '#374151' : '#111827',
                  color: '#fff',
                  cursor: 'pointer',
                }}
              >
                {projecting ? 'Projecting…' : 'Project'}
              </button>
            </div>
            <div style={{ fontSize: 11, color: '#9CA3AF', marginTop: 6 }}>
              This will project the current DB into a GDS graph named “
              {newName || 'fullGraph'}”.
            </div>
          </>
        ) : (
          <div style={{ fontSize: 12, color: '#9CA3AF' }}>
            {loading ? 'Loading…' : '—'}
          </div>
        )}
      </div>

      {/* Divider */}
      <div style={{ padding: '8px 16px', color: '#9CA3AF', fontSize: 12, background: '#0f172a' }}>
        Projected Graphs
      </div>

      {/* Projected graphs list summary */}
      <div style={{ padding: '8px 16px', borderBottom: '1px solid #374151' }}>
        {loading ? 'Loading...' : error ? (
          <span style={{ color: '#fca5a5' }}>{error}</span>
        ) : (
          `${rows.length} graph(s)`
        )}
      </div>

      {/* Current selection */}
      <div style={{ padding: '8px 16px', fontSize: 12, color: '#9CA3AF' }}>
        Current: <b>{graphName ?? '— (none selected)'}</b>
      </div>

      {/* Projected graphs list */}
      <div
        style={{
          overflowY: 'auto',
          padding: '8px 16px',
          gap: 8,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {rows.map((g) => (
          <div
            key={g.graphName}
            style={{
              border: '1px solid #374151',
              borderRadius: 8,
              padding: 12,
              background: '#1F2937',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontWeight: 600 }}>{g.graphName}</div>
                <div style={{ fontSize: 12, color: '#D1D5DB' }}>
                  Nodes: {fmt(g.nodeCount)} · Rels: {fmt(g.relationshipCount)}
                </div>
                <div style={{ fontSize: 12, color: '#9CA3AF' }}>
                  Created: {fmtDate(g.creationTime)}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={() => selectGraph(g.graphName)}
                  style={{
                    padding: '6px 10px',
                    borderRadius: 6,
                    border: '1px solid #6B7280',
                    background: '#111827',
                    color: '#fff',
                    cursor: 'pointer',
                  }}
                >
                  Select
                </button>
                <button
                  onClick={() => doDrop(g.graphName)}
                  disabled={dropping === g.graphName}
                  style={{
                    padding: '6px 10px',
                    borderRadius: 6,
                    border: '1px solid #ef4444',
                    background: dropping === g.graphName ? '#7f1d1d' : '#111827',
                    color: '#fecaca',
                    cursor: 'pointer',
                  }}
                  title="Drop this projected graph from GDS"
                >
                  {dropping === g.graphName ? 'Dropping…' : 'Drop'}
                </button>
              </div>
            </div>
          </div>
        ))}
        {!loading && !error && rows.length === 0 && (
          <div style={{ color: '#9CA3AF' }}>
            No projected graphs. Use <b>Project</b> above to create one.
          </div>
        )}
      </div>
    </div>
  );
}
