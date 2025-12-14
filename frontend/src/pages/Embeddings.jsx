// frontend/src/pages/Embeddings.jsx
import React, { useState } from 'react'
import { api } from '../components/api'

export default function Embeddings({ graphName }) {
  const [log, setLog] = useState('')

  const [propertyRatio, setPropertyRatio] = useState(0.5)
  const [writeProperty, setWriteProperty] = useState('FastRP')

  async function fastrp() {
    try {
      setLog('Running FastRP.write...')
      const { data } = await api.post('/emb/fastrp/write', {
        graphName,
        embeddingDimension: 128,
        writeProperty                 : writeProperty,
        propertyRatio
      })
      setLog(JSON.stringify(data, null, 2))
    } catch (e) {
      setLog(`Error: ${String(e)}`)
    }
  }

  async function node2vec() {
    try {
      setLog('Running Node2Vec.write...')
      const { data } = await api.post('/emb/node2vec/write', {
        graphName,
        embeddingDimension: 128,
        walkLength: 80,
        walksPerNode: 12,
        p: 1,
        q: 1,
        writeProperty          : writeProperty  
      })
      setLog(JSON.stringify(data, null, 2))
    } catch (e) {
      setLog(`Error: ${String(e)}`)
    }
  }

  async function sage() {
    try {
      setLog('Running GraphSAGE.train + write...')
      const { data } = await api.post('/emb/graphsage/trainWrite', {
        graphName,
        writeProperty                 : writeProperty,  
        aggregator: 'mean',
        sampleSizes: [25, 10],
        epochs: 10,
        learningRate: 0.01,
        embeddingDimension: 128
      })
      setLog(JSON.stringify(data, null, 2))
    } catch (e) {
      setLog(`Error: ${String(e)}`)
    }
  }

  async function hash() {
    try {
      setLog('Running HashGNN.write...')
      const { data } = await api.post('/emb/hashgnn/trainWrite', {
        graphName,
        outputDimension: 128,
        embeddingDensity: 10,
        iterations: 10,
        writeProperty                 : writeProperty  
      })
      setLog(JSON.stringify(data, null, 2))
    } catch (e) {
      setLog(`Error: ${String(e)}`)
    }
  }

  return (
    <div style={{ padding:16, display:'grid', gap:12 }}>
      <div style={{ display:'grid', gap:8, gridTemplateColumns:'1fr 1fr' }}>
        <label style={{ gridColumn:'span 2' }}>
          writeProperty
          <input
            value={writeProperty}
            onChange={(e) => setWriteProperty(e.target.value)}
            placeholder="FastRP / GraphSAGE / Node2Vec / HashGNN"
            style={{ width:'100%', padding:6, marginTop:4 }}
          />
        </label>
        <label>
          FastRP propertyRatio (0.0–1.0)
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={propertyRatio}
            onChange={(e) => setPropertyRatio(parseFloat(e.target.value || '0'))}
            style={{ width:'100%', padding:6, marginTop:4 }}
          />
        </label>
        <div />
      </div>

      <div style={{display:'flex', gap:8, flexWrap:'wrap', marginTop:8}}>
        <button onClick={fastrp}>FastRP</button>
        <button onClick={node2vec}>Node2Vec</button>
        <button onClick={sage}>GraphSAGE</button>
        <button onClick={hash}>HashGNN</button>
      </div>

      <pre style={{ background:'#f7f7f7', padding:12, minHeight:200 }}>
        {log}
      </pre>
    </div>
  )
}
