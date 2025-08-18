import React, { useState } from 'react'

type Mode = 'boxes' | 'seg' | 'points' | 'boxes3d'

type ModeResult = { image_url?: string; payload?: any }

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8010'
const DEFAULT_MAX_INPUT = Number(import.meta.env.VITE_DEFAULT_MAX_INPUT_SIZE || 640)

const models = [
  { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { id: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
  { id: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
]

const button: React.CSSProperties = {
  padding: '10px 16px',
  borderRadius: 10,
  background: '#3B68FF',
  color: 'white',
  border: 'none',
  cursor: 'pointer',
}

const pill = (active: boolean): React.CSSProperties => ({
  padding: '8px 12px',
  borderRadius: 999,
  border: `1px solid ${active ? '#3B68FF' : '#334155'}`,
  background: active ? '#1d2b64' : '#0b1220',
  color: active ? '#e2e8f0' : '#94a3b8',
  cursor: 'pointer',
})

const card: React.CSSProperties = {
  background: '#0b1220',
  border: '1px solid #1f2937',
  borderRadius: 12,
  padding: 16,
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [fileUrl, setFileUrl] = useState<string | null>(null)
  const [mode, setMode] = useState<Mode>('boxes')
  const [model, setModel] = useState<string>(models[0].id)
  const [items, setItems] = useState<string>('items')
  const [maxInput, setMaxInput] = useState<number>(DEFAULT_MAX_INPUT)
  const [temperature, setTemperature] = useState<number>(0.3)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showJson, setShowJson] = useState(false)

  // Cache results per mode
  const [modeResults, setModeResults] = useState<Record<Mode, ModeResult>>({
    boxes: {}, seg: {}, points: {}, boxes3d: {}
  })

  const activeResult = modeResults[mode]

  const onChoose = (f: File | null) => {
    setFile(f)
    setError(null)
    if (fileUrl) URL.revokeObjectURL(fileUrl)
    setFileUrl(f ? URL.createObjectURL(f) : null)
  }

  const onChangeMode = (m: Mode) => {
    setMode(m)
    setShowJson(false)
  }

  async function onSubmit() {
    if (!file) return
    setIsLoading(true)
    setError(null)
    setShowJson(false)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('mode', mode)
    fd.append('model', model)
    fd.append('items', items)
    fd.append('temperature', String(temperature))
    fd.append('max_input_size', String(maxInput))
    try {
      const r = await fetch(`${API_BASE}/detect`, { method: 'POST', body: fd })
      if (!r.ok) {
        const txt = await r.text().catch(()=>'')
        throw new Error(`API ${r.status}: ${txt || r.statusText}`)
      }
      const json = await r.json()
      setModeResults(prev => ({ ...prev, [mode]: { image_url: json.image_url, payload: json } }))
    } catch (e:any) {
      setError(String(e?.message || e))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '24px auto', padding: 16, color: '#e5e7eb', background: '#0f172a', fontFamily: 'Inter, ui-sans-serif, system-ui' }}>
      <h2 style={{ marginBottom: 16 }}>Auto Vision</h2>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Left: controls */}
        <div style={{ display: 'grid', gap: 12 }}>
          <div style={card}>
            <div style={{ marginBottom: 8, color: '#94a3b8' }}>Upload image</div>
            <input type="file" accept="image/*" onChange={e => onChoose(e.target.files?.[0] || null)} />
          </div>

          <div style={card}>
            <div style={{ marginBottom: 8, color: '#94a3b8' }}>Mode</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {(['boxes','seg','points','boxes3d'] as Mode[]).map(m => (
                <button key={m} onClick={()=>onChangeMode(m)} style={pill(mode===m)}>{m}</button>
              ))}
            </div>
          </div>

          <div style={card}>
            <div style={{ marginBottom: 8, color: '#94a3b8' }}>Model</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {models.map(m => (
                <button key={m.id} onClick={()=>setModel(m.id)} style={pill(model===m.id)}>{m.label}</button>
              ))}
            </div>
          </div>

          <div style={card}>
            <div style={{ marginBottom: 8, color: '#94a3b8' }}>Items (what to detect)</div>
            <input value={items} onChange={e=>setItems(e.target.value)} style={{ width: '100%', padding: 10, borderRadius: 10, border: '1px solid #334155', background: '#0b1220', color: '#e5e7eb' }} />
          </div>

          <div style={card}>
            <div>Max input size for model: {maxInput}px</div>
            <input type="range" min={320} max={1280} step={32} value={maxInput} onChange={e=>setMaxInput(parseInt(e.target.value))} />
          </div>

          <div style={card}>
            <div>Temperature: {temperature.toFixed(2)}</div>
            <input type="range" min={0} max={2} step={0.05} value={temperature} onChange={e=>setTemperature(parseFloat(e.target.value))} />
          </div>

          <div>
            <button disabled={!file || isLoading} onClick={onSubmit} style={{ ...button, opacity: !file || isLoading ? 0.6 : 1 }}>
              {isLoading ? 'Running…' : 'Run'}
            </button>
          </div>

          {error && (
            <div style={card}>
              <div style={{ color: '#fca5a5' }}>{error}</div>
            </div>
          )}

          {activeResult?.payload && (
            <div style={card}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <div style={{ color: '#94a3b8' }}>Results</div>
                <button onClick={() => setShowJson(!showJson)} style={{ ...button, padding: '4px 8px', fontSize: '12px' }}>
                  {showJson ? 'Hide' : 'Show'} JSON
                </button>
              </div>
              {showJson && (
                <pre style={{ background:'#0a0f1a', padding: 12, borderRadius: 8, overflow: 'auto', maxHeight: 400 }}>{JSON.stringify(activeResult.payload, null, 2)}</pre>
              )}
            </div>
          )}
        </div>

        {/* Right: image display */}
        <div style={card}>
          <div style={{ marginBottom: 8, color: '#94a3b8' }}>
            {activeResult?.image_url ? 'Result' : 'Preview'}
          </div>
          {activeResult?.image_url ? (
            <img src={`${API_BASE}${activeResult.image_url}`} style={{ maxWidth: '100%', borderRadius: 8 }} />
          ) : fileUrl ? (
            <img src={fileUrl} style={{ maxWidth: '100%', borderRadius: 8 }} />
          ) : (
            <div style={{ color: '#64748b' }}>No image selected</div>
          )}
        </div>
      </div>
    </div>
  )
}
