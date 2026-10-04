import { useEffect, useMemo, useState } from 'react'

export interface PointTopic {
  id: number
  weight: number
  snippet: string
}

export interface Point {
  id: string
  x: number
  y: number
  name: string | null
  city: string | null
  country: string | null
  url: string
  captured_at: string
  excerpt: string
  topics: PointTopic[]
}

export interface Topic {
  id: number
  label: string
  pages: number
}

export interface ClusterData {
  run_id: string
  topics: Topic[]
  points: Point[]
}

const SIZE = 640
const PAD = 30
const NEUTRAL: [number, number, number] = [107, 114, 128]

function hue(topic: number) {
  return (topic * 137.5) % 360
}

function topicColor(topic: number) {
  return `hsl(${hue(topic)} 75% 65%)`
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const k = (n: number) => (n + h / 30) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  return [f(0) * 255, f(8) * 255, f(4) * 255]
}

/** A point's colour is the weighted mix of its topics' colours: no hard cluster boundary. */
function blend(topics: PointTopic[]) {
  const total = topics.reduce((sum, t) => sum + t.weight, 0)
  if (!total) return `rgb(${NEUTRAL.join(',')})`
  const mix = [0, 0, 0]
  for (const t of topics) {
    const rgb = hslToRgb(hue(t.id), 0.75, 0.65)
    for (let i = 0; i < 3; i++) mix[i] += (rgb[i] * t.weight) / total
  }
  return `rgb(${mix.map(Math.round).join(',')})`
}

export function ClusterView({ data }: { data: ClusterData }) {
  const [selected, setSelected] = useState<Point | null>(null)
  const [active, setActive] = useState<number | null>(null)

  const scaled = useMemo(() => {
    const xs = data.points.map((p) => p.x)
    const ys = data.points.map((p) => p.y)
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)]
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)]
    const span = Math.max(x1 - x0, y1 - y0) || 1
    const inner = SIZE - 2 * PAD
    return data.points.map((p) => ({
      p,
      cx: PAD + ((p.x - x0) / span) * inner,
      cy: PAD + ((p.y - y0) / span) * inner,
    }))
  }, [data])

  // A soft glow per topic: centred on the weighted mean of its pages, sized by their spread.
  const glows = useMemo(
    () =>
      data.topics.map((t) => {
        const members = scaled
          .map(({ p, cx, cy }) => ({ w: p.topics.find((x) => x.id === t.id)?.weight ?? 0, cx, cy }))
          .filter((m) => m.w > 0)
        const total = members.reduce((s, m) => s + m.w, 0) || 1
        const mx = members.reduce((s, m) => s + m.w * m.cx, 0) / total
        const my = members.reduce((s, m) => s + m.w * m.cy, 0) / total
        const variance =
          members.reduce((s, m) => s + m.w * ((m.cx - mx) ** 2 + (m.cy - my) ** 2), 0) / total
        return { id: t.id, cx: mx, cy: my, r: Math.max(40, Math.sqrt(variance) * 1.6) }
      }),
    [data, scaled],
  )

  const topicLabel = (id: number) => data.topics.find((t) => t.id === id)?.label ?? `topic ${id}`
  const weightOn = (p: Point) =>
    active === null ? 1 : (p.topics.find((t) => t.id === active)?.weight ?? 0)

  return (
    <div className="cluster-view">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="Pages positioned by similarity">
        <defs>
          {data.topics.map((t) => (
            <radialGradient key={t.id} id={`glow-${t.id}`}>
              <stop offset="0%" stopColor={topicColor(t.id)} stopOpacity={0.35} />
              <stop offset="100%" stopColor={topicColor(t.id)} stopOpacity={0} />
            </radialGradient>
          ))}
        </defs>
        {glows.map((g) => (
          <circle
            key={g.id}
            cx={g.cx}
            cy={g.cy}
            r={g.r}
            fill={`url(#glow-${g.id})`}
            opacity={active === null || active === g.id ? 1 : 0.1}
            pointerEvents="none"
          />
        ))}
        {scaled.map(({ p, cx, cy }) => {
          const w = weightOn(p)
          return (
            <circle
              key={p.id}
              cx={cx}
              cy={cy}
              r={selected?.id === p.id ? 8 : 4 + 2 * w}
              fill={blend(p.topics)}
              opacity={active === null ? 0.9 : 0.1 + 0.9 * w}
              stroke={selected?.id === p.id ? '#fff' : 'none'}
              tabIndex={0}
              aria-label={p.name ?? p.url}
              onClick={() => setSelected(p)}
              onKeyDown={(e) => e.key === 'Enter' && setSelected(p)}
            />
          )
        })}
      </svg>
      <aside>
        <ul className="legend">
          {data.topics.map((t) => (
            <li key={t.id}>
              <button
                aria-pressed={active === t.id}
                onClick={() => setActive(active === t.id ? null : t.id)}
              >
                <span className="swatch" style={{ background: topicColor(t.id) }} /> {t.label}{' '}
                <span className="meta">({t.pages})</span>
              </button>
            </li>
          ))}
        </ul>
        {selected ? (
          <article className="postcard">
            <h2>{selected.name ?? selected.url}</h2>
            <p className="meta">
              {[selected.city, selected.country].filter(Boolean).join(', ')} · captured{' '}
              {selected.captured_at}
            </p>
            <p className="tags">
              {selected.topics.map((t) => (
                <button
                  key={t.id}
                  className="tag"
                  style={{ borderColor: topicColor(t.id) }}
                  aria-pressed={active === t.id}
                  onClick={() => setActive(active === t.id ? null : t.id)}
                >
                  {topicLabel(t.id)} · {Math.round(t.weight * 100)}%
                </button>
              ))}
            </p>
            <blockquote>
              {(active !== null && selected.topics.find((t) => t.id === active)?.snippet) ||
                selected.excerpt}
            </blockquote>
            <a href={selected.url} target="_blank" rel="noreferrer">
              {selected.url}
            </a>
          </article>
        ) : (
          <p className="meta">Select a point, or a topic to light up every page that touches it.</p>
        )}
      </aside>
    </div>
  )
}

export default function ClusterMap() {
  const [data, setData] = useState<ClusterData | null>(null)
  const [missing, setMissing] = useState(false)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/points.json`)
      .then((r) => (r.ok ? (r.json() as Promise<ClusterData>) : Promise.reject(new Error())))
      .then((d) => (d.points.length ? setData(d) : setMissing(true)))
      .catch(() => setMissing(true))
  }, [])

  if (data) return <ClusterView data={data} />
  return missing ? (
    <p className="meta">
      No data yet. Run <code>ntn embed &amp;&amp; ntn cluster</code> in <code>pipeline/</code>.
    </p>
  ) : null
}
