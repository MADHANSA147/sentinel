import { useState, useEffect, useRef, useCallback } from 'react'
import * as d3 from 'd3'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ── Helpers ───────────────────────────────────────────────────────────────

function scoreColor(score) {
  if (score >= 70) return '#ef4444'
  if (score >= 40) return '#f59e0b'
  return '#10b981'
}

function scoreLabel(score) {
  if (score >= 70) return 'HIGH'
  if (score >= 40) return 'MEDIUM'
  return 'LOW'
}

// ── CASES ────────────────────────────────────────────────────────────────

const PRESET_CASES = [
  { id: 'case-dataset1', label: 'Dataset 1 — Clean Win', tag: 'Gap detection demo', file: 'whatsapp_conversation_40_messages.json' },
  { id: 'case-dataset2', label: 'Dataset 2 — False Positive', tag: 'Exculpatory context demo', file: 'whatsapp_synthetic_35_messages.json' },
  { id: 'case-dataset3', label: 'Dataset 3 — Sparse Data', tag: 'Ingestion resilience demo', file: 'corrupted_whatsapp_30_messages.json' },
  { id: 'case-dataset4', label: 'Dataset 4 — Network Test', tag: 'PageRank + Betweenness demo', file: 'network_test_45_messages.json' },
]

// ── Priority Board ────────────────────────────────────────────────────────

function PersonCard({ person, onHITL }) {
  const [expanded, setExpanded] = useState(false)
  const color = scoreColor(person.score)

  return (
    <div
      className={`person-card ${expanded ? 'expanded' : ''}`}
      style={{ '--score-color': color }}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="person-card-header">
        <div className="person-avatar">
          {person.person_id?.slice(-3)}
        </div>
        <div className="person-info">
          <div className="person-id">{person.person_id}</div>
          <div className="person-role">{person.role_tag} · {scoreLabel(person.score)}</div>
        </div>
        <div className="score-badge" style={{ background: color }}>
          {person.score}
        </div>
      </div>
      <div className="score-bar">
        <div className="score-bar-fill" style={{ width: `${person.score}%`, background: color }} />
      </div>

      {expanded && (
        <div className="justification-tree">
          <h4>Justification Tree</h4>
          {person.justification_tree?.length > 0 ? (
            person.justification_tree.map((item, i) => (
              <div key={i} className="indicator-row">
                <div>
                  <div className="indicator-name">{item.indicator.replace(/_/g, ' ')}</div>
                  {item.ncmec_ref && <div className="indicator-ref">📋 {item.ncmec_ref}</div>}
                  {item.iso27037_ref && <div className="indicator-ref">📁 {item.iso27037_ref}</div>}
                </div>
                <div className="indicator-weight">+{item.weight}</div>
              </div>
            ))
          ) : (
            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>No indicators — low risk baseline</div>
          )}
          <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.6rem' }}>
            <button className="btn-approve" onClick={e => { e.stopPropagation(); onHITL(person.person_id, 'TIMELINE_GAP', 'approve') }}>✓ Approve</button>
            <button className="btn-reject" onClick={e => { e.stopPropagation(); onHITL(person.person_id, 'TIMELINE_GAP', 'reject') }}>✗ Reject Flag</button>
          </div>
        </div>
      )}
    </div>
  )
}

function PriorityBoard({ data, onHITL }) {
  if (!data?.priority_board?.length) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🔍</div>
        <div>Select a case and run the pipeline</div>
      </div>
    )
  }

  return (
    <div className="priority-board">
      {data.priority_board.map(person => (
        <PersonCard key={person.person_id} person={person} onHITL={onHITL} />
      ))}
    </div>
  )
}

// ── Alert List ────────────────────────────────────────────────────────────

function AlertList({ alerts, onHITL }) {
  if (!alerts?.length) {
    return (
      <div className="empty-state">
        <div className="empty-icon">✅</div>
        <div>No active gap alerts</div>
      </div>
    )
  }

  return (
    <div className="alert-list">
      {alerts.map((alert, i) => (
        <div key={i} className={`alert-card ${alert.suppressed ? 'suppressed' : ''}`}>
          <div className="alert-header">
            <span className="alert-icon">{alert.suppressed ? '🟢' : '⚠️'}</span>
            <div>
              <div className="alert-title">
                {alert.suppressed ? 'Suppressed (Benign Context)' : 'TIMELINE GAP ALERT'}
              </div>
              <div className="alert-pair">{alert.pair_key}</div>
            </div>
          </div>
          <div className="alert-gap">
            {alert.gap_hours?.toFixed(1)}h silence · {alert.before} → {alert.after}
          </div>
          {alert.suppression_reason && (
            <div style={{ fontSize: '0.7rem', color: 'var(--color-success)', marginTop: '0.3rem' }}>
              Context: "{alert.suppression_reason?.substring(0, 80)}..."
            </div>
          )}
          {!alert.suppressed && (
            <div className="hitl-buttons">
              <button className="btn-approve" onClick={() => onHITL(alert.pair_key?.split('->')[0], 'TIMELINE_GAP', 'approve')}>
                ✓ Approve Lead
              </button>
              <button className="btn-reject" onClick={() => onHITL(alert.pair_key?.split('->')[0], 'TIMELINE_GAP', 'reject')}>
                ✗ Reject Flag
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// ── Live Case Map (D3 force-directed) ────────────────────────────────────

function LiveCaseMap({ nodes, edges }) {
  const svgRef = useRef(null)
  const tooltipRef = useRef(null)

  useEffect(() => {
    if (!nodes?.length || !svgRef.current) return

    const container = svgRef.current.parentElement
    const W = container.clientWidth || 800
    const H = container.clientHeight || 600

    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', W).attr('height', H)

    // Background gradient
    const defs = svg.append('defs')
    const grad = defs.append('radialGradient').attr('id', 'bg-grad')
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#111827')
    grad.append('stop').attr('offset', '100%').attr('stop-color', '#0a0e1a')
    svg.append('rect').attr('width', W).attr('height', H).attr('fill', 'url(#bg-grad)')

    // Arrow marker for directed edges
    defs.append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -4 8 8')
      .attr('refX', 16).attr('refY', 0)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', '#2a3450')

    defs.append('marker')
      .attr('id', 'arrow-gap')
      .attr('viewBox', '0 -4 8 8')
      .attr('refX', 16).attr('refY', 0)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', '#ef4444')

    // Top-50 cutoff, but gap-alert nodes always render
    const gapNodeIds = new Set(edges?.filter(e => e.type === 'TIMELINE_GAP').flatMap(e => [e.source, e.target]) || [])
    const topNodes = nodes
      .sort((a, b) => b.score - a.score)
      .filter((n, i) => i < 50 || gapNodeIds.has(n.id))

    const nodeIds = new Set(topNodes.map(n => n.id))

    const simLinks = (edges || [])
      .filter(e => nodeIds.has(e.source) && nodeIds.has(e.target))
      .map(e => ({ ...e, source: e.source, target: e.target }))

    const simulation = d3.forceSimulation(topNodes)
      .force('link', d3.forceLink(simLinks).id(d => d.id).distance(80).strength(0.5))
      .force('charge', d3.forceManyBody().strength(-220))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collision', d3.forceCollide().radius(d => Math.max(16, d.score / 4) + 8))

    const g = svg.append('g')

    // Zoom
    svg.call(d3.zoom().scaleExtent([0.3, 3]).on('zoom', e => g.attr('transform', e.transform)))

    // Edges
    const link = g.append('g').selectAll('line').data(simLinks).enter().append('line')
      .attr('stroke', d => d.type === 'TIMELINE_GAP' ? '#ef4444' : '#2a3450')
      .attr('stroke-width', d => d.type === 'TIMELINE_GAP' ? 2 : 1)
      .attr('stroke-dasharray', d => d.type === 'TIMELINE_GAP' ? '6,3' : 'none')
      .attr('stroke-opacity', 0.7)
      .attr('marker-end', d => d.type === 'TIMELINE_GAP' ? 'url(#arrow-gap)' : 'url(#arrow)')

    // Nodes
    const node = g.append('g').selectAll('g').data(topNodes).enter().append('g')
      .attr('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y })
        .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null })
      )

    // Glow filter
    const filter = defs.append('filter').attr('id', 'glow')
    filter.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'coloredBlur')
    const feMerge = filter.append('feMerge')
    feMerge.append('feMergeNode').attr('in', 'coloredBlur')
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic')

    node.append('circle')
      .attr('r', d => Math.max(12, Math.min(d.score / 4, 28)))
      .attr('fill', d => scoreColor(d.score))
      .attr('fill-opacity', 0.85)
      .attr('stroke', d => scoreColor(d.score))
      .attr('stroke-width', 1.5)
      .attr('filter', d => d.score >= 60 ? 'url(#glow)' : 'none')

    node.append('text')
      .text(d => d.id?.slice(-3))
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', '9px')
      .attr('font-weight', '700')
      .attr('fill', 'white')
      .attr('pointer-events', 'none')

    // Tooltip
    const tooltip = d3.select(tooltipRef.current)
    node
      .on('mouseover', (e, d) => {
        tooltip.style('opacity', 1).html(
          `<strong>${d.id}</strong><br/>Score: ${d.score} · ${d.role || ''}<br/>` +
          `PageRank: ${d.pagerank?.toFixed(4)} · Betweenness: ${d.betweenness?.toFixed(4)}`
        )
      })
      .on('mousemove', e => {
        tooltip.style('left', (e.offsetX + 12) + 'px').style('top', (e.offsetY - 10) + 'px')
      })
      .on('mouseout', () => tooltip.style('opacity', 0))

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    return () => simulation.stop()
  }, [nodes, edges])

  return (
    <>
      <svg ref={svgRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      <div ref={tooltipRef} className="tooltip" style={{ opacity: 0, position: 'absolute' }} />
    </>
  )
}

// ── Theories Panel ────────────────────────────────────────────────────────

function TheoriesPanel({ theories }) {
  if (!theories?.length) return null
  return (
    <div className="theories-panel">
      <div className="theories-title">⚖️ Case Simulation — Competing Theories</div>
      {theories.map((t, i) => (
        <div key={i} className="theory-item">
          <div className="theory-bar-row">
            <span className="theory-pct">{t.likelihood}%</span>
            <div className="theory-bar-bg">
              <div className="theory-bar-fill" style={{ width: `${t.likelihood}%` }} />
            </div>
          </div>
          <div className="theory-text">{t.theory}</div>
        </div>
      ))}
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────

export default function App() {
  const [selectedCase, setSelectedCase] = useState(null)
  const [activeTab, setActiveTab] = useState('board')
  const [dashboard, setDashboard] = useState(null)
  const [running, setRunning] = useState(false)
  const [hitlFeedback, setHitlFeedback] = useState(null)

  const runPipeline = useCallback(async (caseId) => {
    setRunning(true)
    setDashboard(null)
    try {
      await fetch(`${API}/api/v1/run/${caseId}`, { method: 'POST' })
      const res = await fetch(`${API}/api/v1/dashboard/${caseId}`)
      const data = await res.json()
      setDashboard(data)
    } catch (e) {
      console.error('Pipeline error:', e)
    } finally {
      setRunning(false)
    }
  }, [])

  const selectCase = useCallback((c) => {
    setSelectedCase(c)
    runPipeline(c.id)
  }, [runPipeline])

  const handleHITL = useCallback(async (personId, indicator, action) => {
    if (!selectedCase) return
    try {
      const res = await fetch(`${API}/api/v1/hitl/${selectedCase.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ person_id: personId, indicator, action }),
      })
      const data = await res.json()
      setHitlFeedback(data)
      // Refresh dashboard
      const dash = await fetch(`${API}/api/v1/dashboard/${selectedCase.id}`)
      setDashboard(await dash.json())
    } catch (e) {
      console.error('HITL error:', e)
    }
  }, [selectedCase])

  const downloadCourtPack = useCallback(async () => {
    if (!selectedCase) return
    const res = await fetch(`${API}/api/v1/export/court-pack/${selectedCase.id}`)
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `court_pack_${selectedCase.id}.pdf`; a.click()
  }, [selectedCase])

  const downloadBundle = useCallback(async () => {
    if (!selectedCase) return
    const res = await fetch(`${API}/api/v1/export/audit-bundle/${selectedCase.id}`)
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `audit_bundle_${selectedCase.id}.json`; a.click()
  }, [selectedCase])

  return (
    <div className="app-shell">
      {/* ── Navbar ─────────────────────────────────────────────────────── */}
      <nav className="navbar">
        <div className="navbar-logo">
          <div className="sentinel-icon">🛡</div>
          <div>
            <div className="navbar-title">SENTINEL</div>
            <div className="navbar-sub">Investigation Support Platform</div>
          </div>
        </div>
        <div className="navbar-spacer" />
        {hitlFeedback && (
          <div style={{ fontSize: '0.75rem', color: 'var(--color-success)', marginRight: '1rem' }}>
            ✓ {hitlFeedback.person_id}: score → {hitlFeedback.new_score}
          </div>
        )}
        <div className="navbar-status">
          <div className="status-dot" />
          Synthetic data only — MVP demo
        </div>
      </nav>

      {/* ── Main ──────────────────────────────────────────────────────── */}
      <div className="main-content">
        {/* ── Sidebar ─────────────────────────────────────────────────── */}
        <aside className="sidebar">
          {/* Case Selector */}
          <div className="case-selector">
            <h3>Select Demo Case</h3>
            {PRESET_CASES.map(c => (
              <button
                key={c.id}
                className={`case-btn ${selectedCase?.id === c.id ? 'active' : ''}`}
                onClick={() => selectCase(c)}
              >
                {c.label}
                <span className="case-tag">{c.tag}</span>
              </button>
            ))}
          </div>

          {/* Tabs */}
          <div className="tab-bar">
            {[
              { id: 'board', label: '🏆 Priority' },
              { id: 'alerts', label: '⚠️ Alerts' },
            ].map(t => (
              <button
                key={t.id}
                className={`tab-btn ${activeTab === t.id ? 'active' : ''}`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="section-header">
            {activeTab === 'board' ? '📊 Suspects by Risk Score' : '🔔 Gap Alerts'}
          </div>

          {activeTab === 'board' ? (
            <PriorityBoard data={dashboard} onHITL={handleHITL} />
          ) : (
            <AlertList alerts={dashboard?.alert_list} onHITL={handleHITL} />
          )}

          {/* Export buttons */}
          {selectedCase && !running && (
            <div style={{ padding: '0.5rem 0.5rem 1rem' }}>
              <button className="court-pack-btn" onClick={downloadCourtPack}>
                📄 Generate Court Pack PDF
              </button>
              <button
                className="court-pack-btn"
                style={{ background: 'var(--color-surface-2)', border: '1px solid var(--color-border)', marginTop: '0.4rem' }}
                onClick={downloadBundle}
              >
                📦 Download Audit Bundle
              </button>
            </div>
          )}
        </aside>

        {/* ── Map Panel ───────────────────────────────────────────────── */}
        <div className="map-panel">
          {running && (
            <div className="pipeline-overlay">
              <div className="pipeline-spinner" />
              <div className="pipeline-label">Running SENTINEL pipeline…</div>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>
                Identity → Timeline → Baseline → Network → Roles → Words → Gaps → Risk Score
              </div>
            </div>
          )}

          <LiveCaseMap
            nodes={dashboard?.graph?.nodes}
            edges={dashboard?.graph?.gap_edges}
          />

          {/* Map controls */}
          <div className="map-controls">
            {selectedCase && (
              <button className="map-btn primary" onClick={() => runPipeline(selectedCase.id)}>
                ↻ Re-run Pipeline
              </button>
            )}
          </div>

          {/* Theories */}
          {dashboard?.theories?.length > 0 && (
            <TheoriesPanel theories={dashboard.theories} />
          )}

          {/* Legend */}
          <div className="map-legend">
            <div className="legend-title">Live Case Map</div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: '#ef4444' }} />
              High Risk (≥70)
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: '#f59e0b' }} />
              Medium Risk (40–69)
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: '#10b981' }} />
              Low Risk (&lt;40)
            </div>
            <div className="legend-item">
              <div className="legend-line" style={{ background: '#ef4444', borderTop: '2px dashed #ef4444', height: 0 }} />
              Timeline Gap
            </div>
            <div style={{ fontSize: '0.6rem', color: 'var(--color-text-muted)', marginTop: '0.4rem' }}>
              Top-50 nodes · Gap nodes always shown<br />
              Drag nodes · Scroll to zoom
            </div>
          </div>

          {/* Empty state */}
          {!running && !dashboard && (
            <div style={{
              position: 'absolute', inset: 0, display: 'flex',
              flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              gap: '0.75rem', color: 'var(--color-text-muted)'
            }}>
              <div style={{ fontSize: '3rem', opacity: 0.3 }}>🛡</div>
              <div style={{ fontSize: '1rem', fontWeight: 600 }}>Select a demo case to begin</div>
              <div style={{ fontSize: '0.8rem' }}>The full pipeline will run automatically</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
