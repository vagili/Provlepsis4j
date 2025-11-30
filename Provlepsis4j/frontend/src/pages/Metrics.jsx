import React, { useState } from 'react'
import { api } from '../components/api'
import { LineChart, XAxis, YAxis, Tooltip, Line, CartesianGrid, ResponsiveContainer } from 'recharts'

export default function Metrics() {
  const [th, setTh] = useState(0.5)
  const [res, setRes] = useState(null)

  async function run() {
    const { data } = await api.post('/metrics/lp', { threshold: th })
    setRes(data)
  }

  const chartData = res ? [
    { name: 'F1', value: res.f1 },
    { name: 'Precision', value: res.precision },
    { name: 'Recall', value: res.recall }
  ] : []

  return (
    <div style={{padding:16, display:'grid', gap:12, gridTemplateColumns:'320px 1fr'}}>
      <div>
        <div>
          <label>Threshold:&nbsp;</label>
          <input type="number" min="0" max="1" step="0.01" value={th} onChange={e=>setTh(parseFloat(e.target.value))}/>
          <button onClick={run} style={{marginLeft:8}}>Compute</button>
        </div>
        <pre style={{background:'#f7f7f7', padding:12, minHeight:200}}>{res ? JSON.stringify(res, null, 2) : 'Run to see results'}</pre>
      </div>
      <div style={{height:320}}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={[0,1]} />
            <Tooltip />
            <Line type="monotone" dataKey="value" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
