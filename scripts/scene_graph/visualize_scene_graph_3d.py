#!/usr/bin/env python3
"""Create a standalone interactive 3D scene graph viewer from oracle JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from oracle_scene_graph_utils import make_jsonable, read_jsonl


RELATION_COLORS = {
    "left_of": "#71717a",
    "right_of": "#71717a",
    "front_of": "#71717a",
    "behind": "#71717a",
    "above": "#71717a",
    "below": "#71717a",
    "near": "#3b82f6",
    "next_to": "#3b82f6",
    "between": "#6b7280",
    "on": "#22c55e",
    "inside": "#f59e0b",
    "touching": "#ef4444",
    "grasped_by": "#a855f7",
    "grasping": "#a855f7",
}


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Oracle LIBERO 3D Scene Graph</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      height: 100vh;
      overflow: hidden;
      background: #f4f1eb;
      color: #18181b;
    }}
    #app {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      height: 100vh;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
      background:
        linear-gradient(180deg, rgba(255,255,255,.72), rgba(242,238,229,.84)),
        repeating-linear-gradient(0deg, rgba(0,0,0,.05), rgba(0,0,0,.05) 1px, transparent 1px, transparent 40px),
        repeating-linear-gradient(90deg, rgba(0,0,0,.05), rgba(0,0,0,.05) 1px, transparent 1px, transparent 40px);
      cursor: grab;
    }}
    canvas.dragging {{ cursor: grabbing; }}
    aside {{
      border-left: 1px solid #d6d3cd;
      background: #fbfaf7;
      padding: 14px;
      overflow: auto;
    }}
    h1 {{
      font-size: 16px;
      line-height: 1.25;
      margin: 0 0 12px;
    }}
    label {{
      display: block;
      font-size: 12px;
      font-weight: 650;
      color: #52525b;
      margin: 12px 0 5px;
    }}
    select, input[type="range"], button {{
      width: 100%;
      box-sizing: border-box;
    }}
    select, button {{
      border: 1px solid #c8c4bc;
      border-radius: 6px;
      background: white;
      min-height: 32px;
      padding: 5px 8px;
      color: #18181b;
    }}
    button {{ margin-top: 10px; }}
    .row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 7px 0;
      font-size: 13px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
      border: 1px solid rgba(0,0,0,.18);
      flex: 0 0 auto;
    }}
    .stats {{
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid #dedbd5;
      font-size: 12px;
      color: #52525b;
      line-height: 1.55;
      white-space: pre-wrap;
    }}
    @media (max-width: 820px) {{
      #app {{ grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) auto; }}
      aside {{ border-left: 0; border-top: 1px solid #d6d3cd; max-height: 42vh; }}
    }}
  </style>
</head>
<body>
<div id="app">
  <canvas id="scene"></canvas>
  <aside>
    <h1>Oracle LIBERO 3D Scene Graph</h1>
    <div class="stats" style="margin-top:0; padding-top:0; border-top:0;">
      Drag: orbit<br>
      Shift+drag: pan<br>
      Wheel: zoom<br>
      Timestep: select a graph frame
    </div>
    <label for="recordSelect">Timestep</label>
    <select id="recordSelect"></select>
    <label for="scaleRange">Graph Scale</label>
    <input id="scaleRange" type="range" min="120" max="900" value="430" />
    <button id="resetView" type="button">Reset View</button>
    <label>Relations</label>
    <div id="relationToggles"></div>
    <div id="stats" class="stats"></div>
  </aside>
</div>
<script>
const RECORDS = __RECORDS_JSON__;
const RELATION_COLORS = __RELATION_COLORS_JSON__;

const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const recordSelect = document.getElementById("recordSelect");
const relationToggles = document.getElementById("relationToggles");
const scaleRange = document.getElementById("scaleRange");
const stats = document.getElementById("stats");
let yaw = -0.72;
let pitch = 0.82;
let panX = 0;
let panY = 0;
let dragging = false;
let last = null;
const enabledRels = new Set(Object.keys(RELATION_COLORS));

function resize() {{
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function optionLabel(record, index) {{
  return `#${{index}} episode=${{record.episode_id}} t=${{record.timestep}} nodes=${{record.nodes.length}} edges=${{record.edges.length}}`;
}}

function initControls() {{
  RECORDS.forEach((record, index) => {{
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = optionLabel(record, index);
    recordSelect.appendChild(option);
  }});
  Object.entries(RELATION_COLORS).forEach(([rel, color]) => {{
    const row = document.createElement("label");
    row.className = "row";
    row.innerHTML = `<input type="checkbox" checked data-rel="${{rel}}"><span class="swatch" style="background:${{color}}"></span><span>${{rel}}</span>`;
    relationToggles.appendChild(row);
  }});
  relationToggles.addEventListener("change", (event) => {{
    const rel = event.target.dataset.rel;
    if (!rel) return;
    if (event.target.checked) enabledRels.add(rel);
    else enabledRels.delete(rel);
    draw();
  }});
  recordSelect.addEventListener("change", draw);
  scaleRange.addEventListener("input", draw);
  document.getElementById("resetView").addEventListener("click", () => {{
    yaw = -0.72;
    pitch = 0.82;
    panX = 0;
    panY = 0;
    scaleRange.value = "430";
    draw();
  }});
}}

function currentRecord() {{
  return RECORDS[Math.max(0, Number(recordSelect.value || 0))];
}}

function nodeMap(record) {{
  const out = new Map();
  record.nodes.forEach((node) => {{
    if (Array.isArray(node.pos_world) && node.pos_world.length >= 3) out.set(node.id, node);
  }});
  return out;
}}

function bounds(nodes) {{
  const pts = nodes.map(n => n.pos_world);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  pts.forEach(p => {{
    for (let i = 0; i < 3; i++) {{
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }}
  }});
  const center = min.map((v, i) => (v + max[i]) / 2);
  return {{ min, max, center }};
}}

function rotatePoint(p, center) {{
  let x = p[0] - center[0];
  let y = p[1] - center[1];
  let z = p[2] - center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * y;
  const y1 = -sy * x + cy * y;
  const z1 = z;
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  return [x1, y2, z2];
}}

function project(p, center, w, h) {{
  const scale = Number(scaleRange.value);
  const r = rotatePoint(p, center);
  const depth = 2.4 + r[2];
  const perspective = 1.0 / Math.max(0.25, depth);
  return {{
    x: w / 2 + panX + r[0] * scale * perspective,
    y: h / 2 + panY - r[1] * scale * perspective,
    z: r[2],
    perspective
  }};
}}

function edgeTargets(edge) {{
  return Array.isArray(edge.dst) ? edge.dst : [edge.dst];
}}

function drawGrid(w, h) {{
  ctx.strokeStyle = "rgba(24,24,27,.16)";
  ctx.lineWidth = 1;
  for (let x = -10; x <= 10; x++) {{
    ctx.beginPath();
    ctx.moveTo(w / 2 + panX + x * 32, h / 2 + panY - 300);
    ctx.lineTo(w / 2 + panX + x * 32, h / 2 + panY + 300);
    ctx.stroke();
  }}
  for (let y = -8; y <= 8; y++) {{
    ctx.beginPath();
    ctx.moveTo(w / 2 + panX - 360, h / 2 + panY + y * 32);
    ctx.lineTo(w / 2 + panX + 360, h / 2 + panY + y * 32);
    ctx.stroke();
  }}
}}

function draw() {{
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  drawGrid(w, h);
  const record = currentRecord();
  if (!record) return;
  const map = nodeMap(record);
  const nodes = Array.from(map.values());
  if (!nodes.length) return;
  const b = bounds(nodes);
  const projected = new Map(nodes.map(n => [n.id, project(n.pos_world, b.center, w, h)]));

  const edgeDraws = [];
  record.edges.forEach(edge => {{
    if (!enabledRels.has(edge.rel)) return;
    const src = projected.get(edge.src);
    if (!src) return;
    edgeTargets(edge).forEach(dstId => {{
      const dst = projected.get(dstId);
      if (!dst) return;
      edgeDraws.push({{ edge, src, dst, depth: (src.z + dst.z) / 2 }});
    }});
  }});
  edgeDraws.sort((a, b) => a.depth - b.depth);
  edgeDraws.forEach(item => {{
    ctx.strokeStyle = RELATION_COLORS[item.edge.rel] || "#71717a";
    ctx.globalAlpha = item.edge.rel === "next_to" ? 0.46 : 0.72;
    ctx.lineWidth = item.edge.rel === "touching" || item.edge.rel === "grasping" ? 3 : 1.4;
    ctx.beginPath();
    ctx.moveTo(item.src.x, item.src.y);
    ctx.lineTo(item.dst.x, item.dst.y);
    ctx.stroke();
  }});
  ctx.globalAlpha = 1;

  nodes
    .map(node => ({{ node, p: projected.get(node.id) }}))
    .sort((a, b) => a.p.z - b.p.z)
    .forEach(({ node, p }) => {{
      const isRobot = node.type === "robot";
      const radius = Math.max(4, (isRobot ? 8 : 6) * p.perspective * 1.8);
      ctx.fillStyle = isRobot ? "#a855f7" : "#111827";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.font = "12px ui-sans-serif, system-ui";
      const label = node.id.length > 36 ? node.id.slice(0, 35) + "…" : node.id;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(255,255,255,.86)";
      ctx.fillRect(p.x + radius + 4, p.y - 9, tw + 8, 18);
      ctx.fillStyle = "#18181b";
      ctx.fillText(label, p.x + radius + 8, p.y + 4);
    }});

  const relationCounts = record.edges.reduce((acc, edge) => {{
    acc[edge.rel] = (acc[edge.rel] || 0) + 1;
    return acc;
  }}, {{}});
  stats.textContent =
    `suite: ${{record.suite}}\\n` +
    `task: ${{record.task_id}}\\n` +
    `episode: ${{record.episode_id}}\\n` +
    `timestep: ${{record.timestep}}\\n` +
    `nodes: ${{record.nodes.length}}\\n` +
    `edges: ${{record.edges.length}}\\n` +
    `relations: ${{Object.entries(relationCounts).map(([k,v]) => `${{k}}=${{v}}`).join(", ")}}\\n\\n` +
    "Drag: orbit\\nShift+drag: pan\\nWheel: zoom";
}}

canvas.addEventListener("pointerdown", event => {{
  dragging = true;
  last = [event.clientX, event.clientY];
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointermove", event => {{
  if (!dragging || !last) return;
  const dx = event.clientX - last[0];
  const dy = event.clientY - last[1];
  last = [event.clientX, event.clientY];
  if (event.shiftKey) {{
    panX += dx;
    panY += dy;
  }} else {{
    yaw += dx * 0.008;
    pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
  }}
  draw();
}});
canvas.addEventListener("pointerup", event => {{
  dragging = false;
  last = null;
  canvas.classList.remove("dragging");
  try {{ canvas.releasePointerCapture(event.pointerId); }} catch (_e) {{}}
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  const next = Math.max(120, Math.min(900, Number(scaleRange.value) - event.deltaY * 0.6));
  scaleRange.value = String(next);
  draw();
}}, {{ passive: false }});
window.addEventListener("resize", resize);
initControls();
resize();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-dir", type=Path, default=Path("outputs/scene_graph_probe/graphs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_probe/scene_graph_3d"))
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--output-name", default="index.html")
    return parser.parse_args()


def load_records(graphs_dir: Path, max_records: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(graphs_dir.glob("episode_*.jsonl")):
        for record in read_jsonl(path):
            records.append(record)
            if len(records) >= max_records:
                return records
    return records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.graphs_dir, args.max_records)
    html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = html.replace("__RECORDS_JSON__", json.dumps(make_jsonable(records), separators=(",", ":")))
    html = html.replace("__RELATION_COLORS_JSON__", json.dumps(RELATION_COLORS, sort_keys=True))
    out_path = args.output_dir / args.output_name
    out_path.write_text(html, encoding="utf-8")
    summary = {
        "viewer": str(out_path),
        "records": len(records),
        "graphs_dir": str(args.graphs_dir),
    }
    (args.output_dir / "viewer_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
