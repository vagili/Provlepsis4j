import React, { useState } from 'react'
import { api } from '../components/api'

export default function Data() {
  const [log, setLog] = useState('')
  const API = import.meta.env.VITE_API || "http://localhost:8080"; ///////////

  async function loadSample() {
    setLog('Loading sample data...')
    const { data } = await api.post('/ingest/sample')
    setLog(JSON.stringify(data, null, 2))
  }

  return (
    <div style={{padding:16, display:'grid', gap:12}}>
      <button onClick={loadSample}>Load Sample Dataset</button>
      <p style={{opacity:0.8}}>Or ingest your own data via Aura Data Importer or Cypher.</p>
      <pre style={{background:'#f7f7f7', padding:12, minHeight:200}}>{log}</pre>
    </div>
  )
}
