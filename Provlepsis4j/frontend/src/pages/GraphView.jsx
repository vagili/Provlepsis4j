// frontend/src/views/GraphView.jsx
import React, { useEffect, useRef, useState } from 'react'
import cytoscape from 'cytoscape'
import { api } from '../components/api'

export default function GraphView() {
  const containerRef = useRef(null)
  const cyRef = useRef(null)
  const [nodeId, setNodeId] = useState(0)
  const [k, setK] = useState(50)

  useEffect(() => {
    if (!containerRef.current) return
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements: [],
      layout: { name: 'grid' },
      style: [
        { selector: 'node', style: { 'label': 'data(label)' } },
        { selector: 'edge', style: { 'width': 2, 'label': 'data(score)' } }
      ]
    })
    return () => cyRef.current?.destroy()
  }, [])

  async function load() {
    const { data } = await api.post('/metrics/topPredicted', { nodeId, k })
    const nodes = new Map()
    const elements = []
    nodes.set(nodeId, { data: { id: 'n'+nodeId, label: 'Node '+nodeId } })
    for (const row of data.rows || []) {
      const id = row.node?.identity ?? row.node?.identity?.low ?? row.node
      const score = row.score
      const key = 'n'+id
      if (!nodes.has(id)) nodes.set(id, { data: { id: key, label: 'Node '+id } })
      elements.push({ data: { id: 'e'+nodeId+'_'+id, source: 'n'+nodeId, target: key, score } })
    }
    cyRef.current.json({ elements: [...nodes.values(), ...elements] })
    cyRef.current.layout({ name: 'cose' }).run()
  }

  return (
    <div style={{height:'100%', display:'grid', gridTemplateRows:'56px 1fr'}}>
      <div style={{display:'flex', alignItems:'center', gap:8, padding:8, borderBottom:'1px solid #ddd'}}>
        <label>Center node ID: <input type="number" value={nodeId} onChange={e=>setNodeId(parseInt(e.target.value))}/></label>
        <label>k: <input type="number" value={k} onChange={e=>setK(parseInt(e.target.value))}/></label>
        <button onClick={load}>Load Top-k Predicted</button>
      </div>
      <div ref={containerRef} style={{height:'100%'}}/>
    </div>
  )
}
