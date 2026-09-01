"""Self-contained Verathos operator dashboard for the mesh worker pool.

Served by the pool manager at ``GET /`` and ``GET /operator``. One file, no
external assets, with a public fleet view and authenticated management through
the pool owner's Bittensor wallet. Same-origin, so no CORS.

Palette: blue is the interactive/brand accent; green is reserved for the
"verified / serving" state (that green checkmark is the product). The live
"pool map" places each machine by its round-trip latency to the coordinator
so a slow WAN worker is visible at a glance.
"""

from __future__ import annotations

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64' fill='none' stroke='%236c95eb' stroke-width='4'%3E%3Ccircle cx='32' cy='32' r='16' stroke-dasharray='22 11'/%3E%3Ccircle cx='32' cy='32' r='24' stroke-dasharray='34 12'/%3E%3C/svg%3E" />
<title>Verathos Operator</title>
<style>
  :root {
    --bg: #1e1e1e; --panel: #252526; --panel2: #2d2d2d; --line: #474747;
    --txt: #ededed; --muted: #a3a3a3; --muted2: #737373;
    --accent: #5681dc; --accent-hi: #6c95eb; --accent-ink: #ffffff;
    --serving: #35d07f; --driving: #34c6ec; --assigned: #f2b03c; --error: #f0555f;
    --radius: 10px;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; max-width:100%; overflow-x:hidden; }
  body {
    background: var(--bg);
    color: var(--txt); font-family: var(--sans); font-size: 14px; line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  a { color: var(--accent); }
  .mono { font-family: var(--mono); }
  .muted { color: var(--muted); }

  header { display: flex; align-items: center; gap: 12px; }
  .logo { display: flex; align-items: center; gap: 10px; font-weight: 650; letter-spacing: .2px; }
  .brand-mark { width: 25px; height: 25px; color: var(--accent-hi); filter: drop-shadow(0 4px 10px rgba(79,143,247,.28)); }
  .pool-chip { font-family: var(--mono); font-size: 12px; color: var(--muted);
    padding: 4px 10px; border: 1px solid var(--line); border-radius: 999px; }
  .spacer { flex: 1; }
  .header-actions { display:flex; align-items:center; gap:8px; }
  .access-badge { font-size:11.5px; font-weight:600; padding:4px 9px; border-radius:999px;
    color:var(--muted); background:var(--panel2); border:1px solid var(--line); white-space:nowrap; }
  .access-badge.owner { color:var(--serving); border-color:rgba(53,208,127,.3); background:rgba(53,208,127,.08); }
  .access-badge.viewer { color:var(--assigned); border-color:rgba(242,176,60,.3); background:rgba(242,176,60,.08); }
  .conn { display: flex; align-items: center; gap: 7px; font-size: 12.5px; color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted2); }
  .dot.live { background: var(--serving); box-shadow: 0 0 0 3px rgba(53,208,127,.18); }
  .dot.down { background: var(--error); box-shadow: 0 0 0 3px rgba(240,85,95,.16); }

  button {
    font-family: var(--sans); font-size: 13px; font-weight: 560; color: var(--txt);
    background: var(--panel2); border: 1px solid var(--line); padding: 8px 13px;
    border-radius: 9px; cursor: pointer; transition: .12s;
  }
  button:hover { border-color: #5a5a5a; background: #333333; }
  button:disabled { opacity: .45; cursor: default; }
  button.primary { color: var(--accent-ink); background: var(--accent); border-color: transparent;
    box-shadow: 0 2px 14px rgba(79,143,247,.24); }
  button.primary:hover { background: var(--accent-hi); }
  button.ghost-danger { color: var(--error); }
  button.ghost-danger:hover { border-color: rgba(240,85,95,.5); background: rgba(240,85,95,.08); }
  button.sm { padding: 5px 10px; font-size: 12px; }

  main { width: 100%; max-width: 1240px; margin: 0 auto; padding: 28px 30px 80px; }
  @media (max-width: 560px){ main { padding: 16px 13px 60px; } }

  .access-panel { margin-bottom:16px; padding:13px 16px; display:flex; align-items:center; gap:18px; }
  .access-copy { min-width:0; flex:1; }
  .access-title { font-weight:620; }
  .access-detail { color:var(--muted); font-size:12.5px; margin-top:2px; }
  .identity { font-family:var(--mono); font-size:11.5px; color:#b9c2cf; max-width:330px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  @media (max-width: 620px){ .access-panel { align-items:flex-start; flex-direction:column; gap:8px; } }

  .overview { display: grid; grid-template-columns: 1fr 300px; gap: 14px; margin-bottom: 24px; }
  @media (max-width: 900px){ .overview { grid-template-columns: 1fr; } }

  .mapwrap { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
    position: relative; overflow: hidden; min-height: 328px; }
  .mapwrap canvas { display: block; width: 100%; height: 300px; margin-top: 28px; }
  .maplabel { position: absolute; top: 12px; left: 14px; font-size: 11.5px; text-transform: uppercase;
    letter-spacing: .09em; color: var(--muted); }
  .legend { position: absolute; bottom: 11px; left: 14px; display: flex; gap: 13px; font-size: 11.5px; color: var(--muted); }
  .legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; vertical-align: 0; }

  .stats { display: grid; grid-template-rows: repeat(4, 1fr); gap: 10px; }
  @media (max-width: 900px){ .stats { grid-template-columns: repeat(4,1fr); grid-template-rows: none; } }
  @media (max-width: 560px){ .stats { grid-template-columns: repeat(2,1fr); } }
  .stat { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 12px 15px; display: flex; align-items: baseline; justify-content: space-between; }
  .stat .n { font-size: 24px; font-weight: 660; letter-spacing: -.5px; font-variant-numeric: tabular-nums; }
  .stat .l { font-size: 11.5px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }

  section { margin-bottom: 26px; }
  .sec-head { display: flex; align-items: baseline; gap: 12px; margin: 0 2px 12px; }
  .sec-head h2 { font-size: 14.5px; font-weight: 620; margin: 0; letter-spacing: .2px; }
  .sec-head .hint { color: var(--muted); font-size: 12.5px; }
  .sec-head .spacer { flex: 1; }

  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
  .empty { padding: 32px 18px; text-align: center; color: var(--muted); font-size: 13px; }

  /* min(266px,100%) lets a card shrink to the viewport on narrow phones
     instead of overflowing off-screen (the "can't see the full card" bug). */
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(266px, 100%), 1fr)); gap: 12px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 15px 16px; display: flex; flex-direction: column; gap: 11px; }
  .card .top { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
  .gpu { display: flex; align-items: flex-start; gap: 9px; min-width: 0; flex: 1; }
  .gpu .ic { width: 30px; height: 30px; border-radius: 8px; background: var(--panel2); border: 1px solid var(--line);
    display: grid; place-items: center; color: var(--accent); flex: none; margin-top: 1px; }
  .gpu .nm { font-weight: 580; line-height: 1.25; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  /* cap the badge column so a long GPU name truncates instead of colliding
     with the status pills; pills stack vertically if the name is short. */
  .badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 5px; flex: none; max-width: 45%; }
  .gpu .id { font-family: var(--mono); font-size: 11.5px; color: var(--muted2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .meta { display: flex; gap: 16px; font-size: 12.5px; }
  .meta .k { color: var(--muted); } .meta .v { font-family: var(--mono); }

  .pill { font-size: 11.5px; font-weight: 600; padding: 3px 9px; border-radius: 999px; white-space: nowrap;
    border: 1px solid transparent; letter-spacing: .2px; }
  .think { color: var(--muted2); font-size: 12px; font-style: italic; white-space: pre-wrap;
    border-left: 2px solid var(--line); padding: 2px 0 2px 9px; margin-bottom: 7px; }
  .thinklbl { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted);
    white-space: nowrap; cursor: pointer; user-select: none; }
  .pill.idle    { color: var(--muted); background: #303030; border-color: var(--line); }
  .pill.assigned{ color: var(--assigned); background: rgba(242,176,60,.10); border-color: rgba(242,176,60,.28); }
  .pill.driving, .pill.joining { color: var(--driving); background: rgba(52,198,236,.10); border-color: rgba(52,198,236,.28); }
  .pill.serving { color: var(--serving); background: rgba(53,208,127,.10); border-color: rgba(53,208,127,.3); }
  .pill.error, .pill.stale { color: var(--error); background: rgba(240,85,95,.10); border-color: rgba(240,85,95,.3); }
  .pill .bd { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:5px; vertical-align:1px; background: currentColor; }
  .pill.serving .bd, .pill.driving .bd { animation: pulse 1.6s ease-in-out infinite; }
  .vscore { display:inline-flex; align-items:baseline; gap:5px; padding:3px 8px; border:1px solid rgba(108,149,235,.28); border-radius:7px;
    color:var(--accent-hi); background:rgba(86,129,220,.08); font-family:var(--mono); font-size:11.5px; white-space:nowrap; }
  .vscore .vl { color:var(--muted); font-family:var(--sans); font-size:9px; font-weight:650; letter-spacing:.07em; text-transform:uppercase; }
  .vscore .vsample { color:var(--muted); font-size:9px; font-weight:500; padding-left:5px; border-left:1px solid var(--line); }
  .vscore .vprov { position:relative; display:inline-grid; place-items:center; width:14px; height:14px; border:1px solid var(--line); border-radius:50%;
    color:var(--muted); font-family:var(--sans); font-size:9px; font-style:normal; cursor:help; outline:none; }
  .vscore .vprov-pop { display:none; position:absolute; z-index:20; right:-8px; top:19px; width:min(390px,70vw); padding:8px 10px;
    border:1px solid var(--line); border-radius:7px; color:var(--txt); background:var(--panel); box-shadow:0 10px 30px rgba(0,0,0,.35);
    font-family:var(--mono); font-size:10px; line-height:1.55; white-space:normal; overflow-wrap:anywhere; }
  .vscore .vprov:hover .vprov-pop, .vscore .vprov:focus .vprov-pop { display:block; }
  .vscore.warning { color:var(--assigned); border-color:rgba(242,176,60,.3); background:rgba(242,176,60,.08); }
  .vscore.blocked { color:var(--error); border-color:rgba(240,85,95,.3); background:rgba(240,85,95,.08); }
  .vscore.unavailable { color:var(--muted); border-color:var(--line); background:var(--panel2); font-family:var(--sans); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  @media (prefers-reduced-motion: reduce){ .pill .bd { animation: none !important; } }
  .pbar { margin-top:5px; height:4px; width:150px; max-width:100%; background:#303030; border-radius:3px; overflow:hidden; }
  .pbi { height:100%; background: var(--driving); border-radius:3px; transition: width .5s ease; }
  .pbi.indet { width:35%; animation: indet 1.3s ease-in-out infinite; }
  @keyframes indet { 0%{margin-left:-35%} 100%{margin-left:100%} }
  @media (prefers-reduced-motion: reduce){ .pbi.indet { animation:none; width:100%; opacity:.4; } }

  .launch { display: grid; grid-template-columns: 300px 1fr; gap: 16px; }
  @media (max-width: 820px){ .launch { grid-template-columns: 1fr; } .launch-side { order: -1; } }
  .field label { display:block; font-size: 11.5px; text-transform: uppercase; letter-spacing:.08em; color: var(--muted); margin-bottom: 7px; }
  select { width: 100%; font-family: var(--sans); font-size: 14px; color: var(--txt);
    background: var(--panel2); border: 1px solid var(--line); border-radius: 9px; padding: 10px 12px; cursor: pointer; }
  .toggle { display: inline-flex; align-items: center; gap: 7px; font-size: 12.5px; color: var(--muted); cursor: pointer; user-select: none; margin-top: 14px; }
  .toggle input { accent-color: var(--accent); }

  .rec { display: flex; flex-direction: column; gap: 9px; }
  .rec-row { display: flex; align-items: center; gap: 12px; padding: 12px 14px;
    background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; }
  .rec-row.best { border-color: rgba(79,143,247,.45); box-shadow: inset 0 0 0 1px rgba(79,143,247,.1); }
  .rec-row .badge { font-size: 10.5px; font-weight: 700; color: var(--accent); letter-spacing: .06em; }
  .rec-row .who { flex: 1; min-width: 0; }
  .rec-row .who .members { font-family: var(--mono); font-size: 12.5px; }
  .rec-row .who .sub { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
  .drv { color: var(--driving); }
  .rtt-good { color: var(--serving); } .rtt-mid { color: var(--assigned); } .rtt-bad { color: var(--error); }

  .adv { display: none; flex-direction: column; gap: 10px; }
  .adv.on { display: flex; }
  .wsel { display: flex; flex-wrap: wrap; gap: 8px; }
  .wchip { display: inline-flex; align-items: center; gap: 7px; padding: 7px 11px; border: 1px solid var(--line);
    border-radius: 9px; background: var(--panel2); font-size: 12.5px; cursor: pointer; }
  .wchip.sel { border-color: var(--accent); color: var(--accent); }
  .wchip.busy { opacity: .4; cursor: not-allowed; }
  .adv-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }

  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 12px 16px; border-bottom: 1px solid var(--line); font-size: 13px; }
  th { color: var(--muted); font-weight: 560; font-size: 11.5px; text-transform: uppercase; letter-spacing: .07em; }
  tr.mrow:last-of-type td { border-bottom: none; }
  td .mk { font-family: var(--mono); color: var(--muted); font-size: 12px; }

  .chatcell { padding: 0 16px 16px; }
  .chat { background: var(--bg); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
  .chat .apibar { display:flex; align-items:center; gap:10px; font-size:12px; color: var(--muted); margin-bottom: 12px; flex-wrap: wrap; }
  .chat .apibar code { font-family: var(--mono); color: #b9c2cf; background: var(--panel2); padding: 3px 8px; border-radius: 6px; border: 1px solid var(--line); }
  .chat .apibar .lock { color: var(--assigned); }
  /* The meshes table lives in an overflow:hidden panel; give it its own
     horizontal scroll so wide rows are reachable on mobile instead of clipped. */
  #meshes { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .chat .io { display: flex; gap: 9px; }
  .chat .io input { flex: 1; min-width: 0; }
  /* Mobile: reflow the meshes table into labelled stacked blocks so every
     field AND the Test/Stop actions are visible without horizontal scroll. */
  @media (max-width: 620px){
    #meshes { overflow-x: visible; }
    #meshes thead { position: absolute; left: -9999px; }
    #meshes tr.mrow { display: block; border: 1px solid var(--line); border-radius: 10px; margin-bottom: 10px; padding: 3px 2px; }
    #meshes td { display: flex; justify-content: space-between; align-items: center; gap: 12px; border: none !important; padding: 7px 12px; text-align: left !important; white-space: normal !important; }
    #meshes td::before { content: attr(data-label); color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; flex: none; }
    #meshes td.actions { justify-content: flex-end; padding-top: 11px; }
    #meshes td.actions::before, #meshes td.chatcell::before { content: none; }
    #meshes td.chatcell { display: block; padding: 6px; }
    #meshes td .mk { word-break: break-all; text-align: right; }
    #meshes .sm { padding: 8px 14px; font-size: 13px; }
  }
  .chat input[type=text] { flex: 1; }
  .chat .out { margin-top: 12px; font-size: 13.5px; line-height: 1.55; white-space: pre-wrap; }
  .chat .verline { margin-top: 10px; display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--muted); }
  .verbadge { display:inline-flex; align-items:center; gap:6px; color: var(--serving); font-weight: 600; }
  .verbadge svg { width: 13px; height: 13px; }

  .overlay { position: fixed; inset: 0; background: rgba(6,8,11,.72); backdrop-filter: blur(4px);
    display: grid; place-items: center; z-index: 30; padding: 20px; }
  .hidden { display: none !important; }
  .dialog { width: 100%; max-width: 520px; background: var(--panel); border: 1px solid var(--line);
    border-radius: 16px; padding: 26px; box-shadow: 0 30px 90px rgba(0,0,0,.5); }
  .dialog h3 { margin: 0 0 6px; font-size: 17px; }
  .dialog p { margin: 0 0 18px; color: var(--muted); font-size: 13px; }
  input[type=text] { width: 100%; font-family: var(--mono); font-size: 13px; color: var(--txt);
    background: var(--bg); border: 1px solid var(--line); border-radius: 9px; padding: 12px 13px; }
  input[type=text]:focus, select:focus, button:focus-visible, .wchip:focus-visible { outline: 2px solid rgba(79,143,247,.55); outline-offset: 1px; }
  /* iOS Safari auto-zooms on focus when an input's font-size is under 16px —
     the "zoom in on the text input" annoyance. 16px on touch screens kills it. */
  @media (max-width: 820px){ input[type=text], select { font-size: 16px; } }
  .codebox { position: relative; background: var(--bg); border: 1px solid var(--line); border-radius: 10px;
    padding: 13px 15px; font-family: var(--mono); font-size: 12px; color: #cfd6df; word-break: break-all; }
  .copy { position: absolute; top: 8px; right: 8px; }
  .row-end { display: flex; justify-content: flex-end; gap: 10px; margin-top: 18px; }
  .steps { counter-reset: s; margin: 0 0 16px; padding: 0; list-style: none; }
  .steps li { counter-increment: s; position: relative; padding: 0 0 14px 30px; color: var(--muted); font-size: 13px; }
  .steps li::before { content: counter(s); position: absolute; left: 0; top: -1px; width: 20px; height: 20px;
    border-radius: 50%; background: var(--panel2); border: 1px solid var(--line); color: var(--txt);
    font-size: 11px; font-weight: 700; display: grid; place-items: center; }
  .steps li b { color: var(--txt); font-weight: 580; }
  .account-list { display:flex; flex-direction:column; gap:8px; max-height:280px; overflow:auto; }
  .account { display:flex; align-items:center; gap:11px; padding:11px 12px; border:1px solid var(--line);
    border-radius:10px; background:var(--panel2); cursor:pointer; }
  .account:hover { border-color:var(--accent); }
  .account-dot { width:30px; height:30px; border-radius:50%; flex:none;
    background:radial-gradient(circle at 35% 30%, var(--accent-hi), #173465 64%, #0a1220); }
  .account-name { font-weight:600; }
  .account-address { color:var(--muted); font-family:var(--mono); font-size:11px; }
  .dialog-note { padding:10px 12px; border:1px solid rgba(79,143,247,.25); border-radius:9px;
    color:var(--muted); background:rgba(79,143,247,.06); font-size:12px; margin:12px 0; }

  #toast { position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%) translateY(20px);
    background: var(--panel2); border: 1px solid var(--line); color: var(--txt); padding: 11px 16px;
    border-radius: 10px; font-size: 13px; opacity: 0; transition: .22s; box-shadow: 0 10px 40px rgba(0,0,0,.4);
    z-index: 40; pointer-events: none; }
  #toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  #toast.err { border-color: rgba(240,85,95,.5); }

  /* Operator shell follows the same restrained layout and neutral surface
     system as verathos-app. The topology remains a real operational view,
     rather than decorative dashboard art. */
  .app-shell { min-height:100dvh; }
  .sidebar { position:fixed; inset:0 auto 0 0; width:236px; z-index:12; display:flex; flex-direction:column;
    background:#232323; border-right:1px solid var(--line); }
  .side-brand { display:flex; align-items:center; gap:11px; height:68px; padding:0 18px; border-bottom:1px solid var(--line); }
  .side-brand .brand-mark { width:30px; height:30px; filter:none; }
  .brand-copy { display:flex; flex-direction:column; min-width:0; }
  .brand-copy strong { font-size:14px; font-weight:650; letter-spacing:-.01em; }
  .brand-copy span { color:var(--muted); font-size:11.5px; }
  .side-label { padding:22px 18px 8px; color:var(--muted2); font-size:10.5px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
  .nav { display:flex; flex-direction:column; gap:3px; padding:0 10px; }
  .nav button { width:100%; display:flex; align-items:center; gap:11px; padding:9px 10px; border:0; border-radius:7px;
    color:var(--muted); background:transparent; font-size:13px; font-weight:520; text-align:left; box-shadow:none; }
  .nav button:hover { color:var(--txt); background:#303030; }
  .nav button.active { color:var(--accent-hi); background:rgba(86,129,220,.13); }
  .nav svg { width:17px; height:17px; flex:none; }
  .side-foot { margin-top:auto; padding:15px 14px; border-top:1px solid var(--line); }
  .side-foot .pool-chip { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-top:7px; }
  .side-foot-label { color:var(--muted2); font-size:10.5px; text-transform:uppercase; letter-spacing:.08em; }
  .workspace { min-height:100dvh; margin-left:236px; }
  .topbar { height:68px; position:sticky; top:0; z-index:10; padding:0 30px; border-bottom:1px solid var(--line);
    background:rgba(30,30,30,.9); backdrop-filter:blur(12px); }
  .topbar-copy { min-width:0; }
  .topbar h1 { margin:0; font-size:15px; font-weight:630; letter-spacing:-.01em; }
  .topbar p { margin:1px 0 0; color:var(--muted); font-size:11.5px; }
  .mobile-brand { display:none; align-items:center; gap:8px; font-weight:650; }
  .mobile-brand .brand-mark { width:24px; height:24px; }
  .page { display:none; }
  .page.active { display:block; animation:pagein .16s ease-out; }
  @keyframes pagein { from { opacity:.45; transform:translateY(3px); } to { opacity:1; transform:none; } }
  @media (prefers-reduced-motion:reduce){ .page.active { animation:none; } }
  .page-heading { display:flex; align-items:flex-start; gap:18px; margin-bottom:20px; }
  .page-heading h2 { margin:0; font-size:22px; font-weight:660; letter-spacing:-.025em; }
  .page-heading p { margin:4px 0 0; color:var(--muted); font-size:13px; }
  .page-heading > div:first-child { min-width:0; }
  .page-heading .spacer { flex:1; }
  .stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:18px; }
  .stat-grid .stat { min-height:92px; align-items:flex-start; flex-direction:column-reverse; justify-content:space-between; }
  .stat-grid .stat .n { font-size:26px; }
  .page-heading-actions { display:flex; gap:8px; margin-left:auto; }
  .topology-panel { margin-bottom:14px; }
  .topology-head { min-height:58px; display:flex; align-items:center; gap:14px; padding:11px 14px; border-bottom:1px solid var(--line); }
  .topology-title { min-width:0; flex:1; }
  .topology-title h3 { margin:0; font-size:14px; font-weight:620; }
  .topology-title p { margin:2px 0 0; color:var(--muted); font-size:11.5px; }
  .topology-summary { color:var(--muted); font-family:var(--mono); font-size:10.5px; white-space:nowrap; }
  .topology-switch { display:flex; padding:2px; border:1px solid var(--line); border-radius:8px; background:var(--bg); }
  .topology-switch button { padding:4px 8px; border:0; border-radius:5px; color:var(--muted); background:transparent; font-size:11px; box-shadow:none; }
  .topology-switch button.active { color:var(--txt); background:var(--panel2); }
  .topology-legend { display:flex; align-items:center; gap:12px; padding:7px 14px; border-bottom:1px solid var(--line); color:var(--muted); font-size:10.5px; }
  .topology-legend span { display:inline-flex; align-items:center; gap:5px; }
  .topology-legend i { width:14px; height:2px; border-radius:2px; background:var(--serving); }
  .topology-legend i.driver { height:3px; background:var(--accent-hi); }
  .topology-legend i.starting { background:var(--driving); }
  .topology-legend i.offline { background:var(--error); }
  .topology-view { position:relative; min-height:220px; max-height:520px; overflow:auto; background:
    radial-gradient(circle at 50% 45%, rgba(86,129,220,.055), transparent 42%),
    linear-gradient(rgba(255,255,255,.012) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.012) 1px, transparent 1px);
    background-size:auto,24px 24px,24px 24px; }
  .topology-view svg { display:block; width:100%; min-width:680px; height:auto; }
  .topology-edge { fill:none; stroke:rgba(163,163,163,.24); stroke-width:1.5; transition:opacity .15s,stroke-width .15s; }
  .topology-edge.serving { stroke:rgba(53,208,127,.54); }
  .topology-edge.driving, .topology-edge.joining, .topology-edge.assigned { stroke:rgba(52,198,236,.52); }
  .topology-edge.error, .topology-edge.stale { stroke:rgba(240,85,95,.56); }
  .topology-edge.driver { stroke:rgba(108,149,235,.82); stroke-width:2.5; }
  .topology-node { cursor:pointer; outline:none; }
  .topology-node rect { fill:#292929; stroke:#484848; stroke-width:1; transition:stroke .15s,fill .15s,opacity .15s; }
  .topology-node:hover rect, .topology-node:focus rect, .topology-node.focus rect { fill:#303030; stroke:var(--accent-hi); stroke-width:1.5; }
  .topology-node text { pointer-events:none; }
  .topology-label { fill:var(--txt); font-size:12px; font-weight:590; }
  .topology-sub { fill:var(--muted); font-family:var(--mono); font-size:9.5px; }
  .topology-column { fill:var(--muted); font-size:9.5px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
  .topology-view.has-focus .topology-edge:not(.focus), .topology-view.has-focus .topology-node:not(.focus) { opacity:.12; }
  .topology-view.has-focus .topology-edge.focus { opacity:1; stroke-width:3; }
  .topology-packet { fill:var(--serving); filter:drop-shadow(0 0 3px rgba(53,208,127,.8)); }
  .topology-packet.driver { fill:var(--accent-hi); }
  @media (prefers-reduced-motion:reduce){ .topology-packet { display:none; } }
  .topology-empty { min-height:220px; display:grid; place-items:center; padding:28px; color:var(--muted); font-size:12.5px; text-align:center; }
  .topology-matrix-wrap { overflow:auto; max-height:430px; }
  .topology-matrix { min-width:100%; width:max-content; border-collapse:separate; border-spacing:0; }
  .topology-matrix th, .topology-matrix td { height:52px; min-width:116px; padding:7px 12px; border:0; border-right:1px solid rgba(71,71,71,.65); border-bottom:1px solid rgba(71,71,71,.65); text-align:center; }
  .topology-matrix th:first-child, .topology-matrix td:first-child { position:sticky; left:0; z-index:2; min-width:215px; max-width:215px; background:var(--panel); text-align:left; }
  .topology-matrix thead th { position:sticky; top:0; z-index:3; background:#292929; }
  .topology-matrix thead th:first-child { z-index:4; }
  .matrix-node { width:100%; padding:0; border:0; background:transparent; box-shadow:none; text-align:left; }
  .matrix-node:hover { background:transparent; }
  .matrix-node .name { display:block; overflow:hidden; color:var(--txt); font-size:11.5px; font-weight:590; text-overflow:ellipsis; white-space:nowrap; }
  .matrix-node .sub { display:block; overflow:hidden; margin-top:1px; color:var(--muted); font-family:var(--mono); font-size:9.5px; text-overflow:ellipsis; white-space:nowrap; }
  .matrix-cell { width:28px; height:28px; padding:0; border:1px solid rgba(53,208,127,.35); border-radius:8px; color:var(--serving); background:rgba(53,208,127,.08); box-shadow:none; }
  .matrix-cell:hover { border-color:var(--serving); background:rgba(53,208,127,.16); }
  .matrix-cell.driver { color:var(--accent-hi); border-color:rgba(108,149,235,.55); background:rgba(86,129,220,.12); }
  .matrix-cell.driving, .matrix-cell.joining, .matrix-cell.assigned { color:var(--driving); border-color:rgba(52,198,236,.4); background:rgba(52,198,236,.08); }
  .matrix-cell.error, .matrix-cell.stale { color:var(--error); border-color:rgba(240,85,95,.4); background:rgba(240,85,95,.08); }
  .matrix-none { color:#505050; }
  .resource-overview { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; }
  .resource-panel { min-width:0; overflow:hidden; }
  .resource-head { min-height:52px; display:flex; align-items:center; gap:10px; padding:11px 14px; border-bottom:1px solid var(--line); }
  .resource-head h3 { margin:0; font-size:14px; font-weight:620; }
  .resource-head .count { color:var(--muted); font-size:11.5px; }
  .resource-head button { margin-left:auto; padding:5px 9px; font-size:11.5px; }
  .resource-list { max-height:390px; overflow:auto; }
  .resource-row { min-height:66px; display:flex; align-items:center; gap:11px; padding:10px 14px; border-bottom:1px solid var(--line); }
  .resource-row:last-child { border-bottom:0; }
  .resource-icon { width:31px; height:31px; flex:none; display:grid; place-items:center; color:var(--accent-hi); background:var(--panel2); border:1px solid var(--line); border-radius:8px; }
  .resource-icon svg { width:15px; height:15px; }
  .resource-main { min-width:0; flex:1; }
  .resource-name { overflow:hidden; color:var(--txt); font-size:12.5px; font-weight:590; text-overflow:ellipsis; white-space:nowrap; }
  .resource-sub { overflow:hidden; margin-top:2px; color:var(--muted); font-family:var(--mono); font-size:10.5px; text-overflow:ellipsis; white-space:nowrap; }
  .resource-metrics { display:flex; align-items:center; justify-content:flex-end; gap:10px; color:var(--muted); font-size:11px; white-space:nowrap; }
  .resource-metrics b { color:var(--txt); font-family:var(--mono); font-size:11px; font-weight:520; }
  .resource-actions { display:flex; align-items:center; gap:6px; flex:none; }
  .resource-empty { padding:28px 16px; color:var(--muted); font-size:12.5px; text-align:center; }
  .machines-layout { display:grid; grid-template-columns:minmax(0,1fr) 290px; gap:14px; margin-bottom:22px; }
  .machines-layout > * { min-width:0; }
  .machines-layout .stats { grid-template-columns:1fr; grid-template-rows:repeat(4,1fr); }
  .machine-section .sec-head { margin-top:2px; }
  .machine-section .grid { grid-template-columns:repeat(auto-fill,minmax(min(310px,100%),1fr)); }
  .management-note { margin-bottom:18px; }
  .enrollment-state { display:flex; align-items:center; gap:9px; margin-top:12px; padding:10px 12px; border-radius:8px;
    border:1px solid var(--line); color:var(--muted); background:var(--panel2); font-size:12px; }
  .enrollment-state.connected { color:var(--serving); border-color:rgba(53,208,127,.3); background:rgba(53,208,127,.07); }
  .security-note { margin-top:10px; color:var(--muted); font-size:11.5px; }

  @media (max-width:980px){
    .machines-layout { grid-template-columns:1fr; }
    .machines-layout .stats { grid-template-columns:repeat(4,1fr); grid-template-rows:none; }
    .resource-overview { grid-template-columns:1fr; }
  }
  @media (max-width:760px){
    .sidebar { position:sticky; top:0; width:auto; height:auto; flex-direction:row; align-items:center; overflow:hidden; }
    .side-brand { height:54px; border:0; padding:0 12px; }
    .side-brand .brand-copy, .side-label, .side-foot { display:none; }
    .side-brand .brand-mark { width:25px; height:25px; }
    .nav { flex:1; flex-direction:row; gap:2px; padding:6px 8px 6px 0; overflow-x:auto; }
    .nav button { width:auto; min-width:max-content; padding:8px 10px; }
    .workspace { width:100%; min-width:0; margin-left:0; overflow-x:hidden; }
    .topbar { height:58px; padding:0 14px; }
    .topbar-copy { display:none; }
    .mobile-brand { display:flex; }
    main { padding:20px 14px 70px; }
    .stat-grid { grid-template-columns:repeat(2,1fr); }
    .header-actions { margin-left:auto; }
    .access-badge, .conn { display:none; }
    .page-heading h2 { font-size:20px; }
  }
  @media (max-width:520px){
    .nav button { font-size:0; gap:0; padding:9px 12px; }
    .nav button svg { width:18px; height:18px; }
    .header-actions #signoutButton { display:none !important; }
    .stat-grid .stat { min-height:82px; }
    .machines-layout .stats { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .machines-layout .stat { min-width:0; }
    .page-heading p { overflow-wrap:anywhere; }
    .page-heading { flex-wrap:wrap; }
    .topology-head { align-items:flex-start; flex-wrap:wrap; gap:8px; }
    .topology-title { flex-basis:100%; }
    .topology-summary { flex:1; white-space:normal; }
    .topology-legend { overflow-x:auto; }
    .topology-matrix th:first-child, .topology-matrix td:first-child { min-width:155px; max-width:155px; }
    .page-heading .spacer { display:none; }
    .page-heading > button { width:100%; }
    .page-heading-actions { width:100%; margin-left:0; }
    .page-heading-actions button { flex:1; }
    .resource-metrics { gap:7px; }
    .resource-row { padding:10px 11px; }
  }
</style>
</head>
<body>

<div id="walletModal" class="overlay hidden">
  <div class="dialog">
    <div id="walletChoose">
      <h3>Connect wallet</h3>
      <p id="walletHint">Choose a shared Bittensor account. You will sign a single-use login challenge.</p>
      <div id="accountList" class="account-list"></div>
      <div class="dialog-note">A signature proves control of the selected account. Pool ownership is the account bound during pool setup; this does not change on-chain ownership.</div>
      <div class="row-end"><button onclick="closeWallet()">Cancel</button></div>
    </div>
    <div id="walletSign" class="hidden">
      <h3>Verify account</h3>
      <p>Review and sign this Verathos Operator challenge for <span id="signAccount" class="mono"></span>.</p>
      <div id="challengeOut" class="codebox"></div>
      <div class="row-end"><button onclick="backToAccounts()">Back</button><button id="signButton" class="primary" onclick="signChallenge()">Sign challenge</button></div>
    </div>
  </div>
</div>

<div id="addModal" class="overlay hidden">
  <div class="dialog">
    <h3>Add a machine</h3>
    <p>Run one enrollment command from a Verathos checkout on the machine. The worker connects back to this coordinator.</p>
    <ol class="steps">
      <li>Choose a stable <b>machine ID</b> you will recognize later.</li>
      <li>Securely copy <b>pool-token.txt</b> to <span class="mono">~/.verathos/pool-token.txt</span> on that machine and set mode <span class="mono">0600</span>.</li>
      <li>Run the command below from the <b>Verathos source directory</b>.</li>
      <li>Keep this window open until the machine reports <b>connected</b>.</li>
    </ol>
    <div class="field" style="margin-bottom:10px"><label>Machine ID</label><input id="workerId" type="text" placeholder="gpu-lab-01" oninput="updateJoinCommand()" /></div>
    <div class="field" style="margin-bottom:12px"><label>Advertised host <span style="text-transform:none;letter-spacing:0">(optional)</span></label><input id="advertiseHost" type="text" placeholder="10.0.0.24" oninput="updateJoinCommand()" /></div>
    <div id="validatorDriverFields" class="hidden" style="margin-bottom:12px">
      <label class="toggle"><input id="coordinatorCapable" type="checkbox" onchange="toggleCoordinatorFields()" /> This machine can host the mesh coordinator</label>
      <div id="coordinatorFields" class="hidden" style="margin-top:10px">
        <div class="security-note" style="margin-bottom:10px">A coordinator-capable machine needs the registered miner wallet and the local validator allowlist. These are names and paths only; no wallet seed is placed in the command.</div>
        <div class="field" style="margin-bottom:10px"><label>Wallet name</label><input id="coordinatorWallet" type="text" placeholder="test_miner96" oninput="updateJoinCommand()" /></div>
        <div class="field" style="margin-bottom:10px"><label>Hotkey name</label><input id="coordinatorHotkey" type="text" placeholder="default" oninput="updateJoinCommand()" /></div>
        <div class="field"><label>Validator allowlist path</label><input id="validatorAllowlistPath" type="text" value="~/.verathos/validator-allowlist.json" oninput="updateJoinCommand()" /></div>
      </div>
    </div>
    <div class="field"><label>Secure enrollment command</label></div>
    <div class="codebox"><span id="commandOut"></span><button class="sm copy" onclick="copyCommand()">Copy</button></div>
    <div class="security-note">The command contains only a token-file path. Never place the raw pool token in shell history or process arguments.</div>
    <div id="enrollmentState" class="enrollment-state"><span class="dot"></span><span>Waiting for the command to run…</span></div>
    <div class="row-end"><button onclick="closeAdd()">Close</button></div>
  </div>
</div>

<div class="app-shell">
  <aside class="sidebar">
    <div class="side-brand">
      <svg class="brand-mark" viewBox="0 0 64 64" fill="none" aria-label="Verathos"><circle cx="32" cy="32" r="16" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-dasharray="22.3 11.2"/><circle cx="32" cy="32" r="21" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-dasharray="33 11" transform="rotate(20 32 32)"/><circle cx="32" cy="32" r="26" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-dasharray="45.79 8.67" transform="rotate(40 32 32)"/></svg>
      <div class="brand-copy"><strong>Verathos</strong><span>Operator</span></div>
    </div>
    <div class="side-label">Workspace</div>
    <nav class="nav" aria-label="Operator sections">
      <button class="active" data-page="overview" onclick="goPage('overview')" title="Overview"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>Overview</button>
      <button data-page="machines" onclick="goPage('machines')" title="Machines"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="5" width="18" height="6" rx="2"/><rect x="3" y="13" width="18" height="6" rx="2"/><path d="M7 8h.01M7 16h.01M11 8h7M11 16h7"/></svg>Machines</button>
      <button data-page="launch" onclick="goPage('launch')" title="Launch"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l3 6 6 3-6 3-3 6-3-6-6-3 6-3 3-6z"/></svg>Launch</button>
      <button data-page="meshes" onclick="goPage('meshes')" title="Meshes"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="5" cy="12" r="2"/><circle cx="12" cy="5" r="2"/><circle cx="19" cy="12" r="2"/><circle cx="12" cy="19" r="2"/><path d="M6.5 10.5l4-4M13.5 6.5l4 4M17.5 13.5l-4 4M10.5 17.5l-4-4"/></svg>Meshes</button>
    </nav>
    <div class="side-foot">
      <div class="side-foot-label">Coordinator pool</div>
      <span id="poolChip" class="pool-chip mono">pool —</span>
    </div>
  </aside>

  <div class="workspace">
    <header class="topbar">
      <div class="mobile-brand"><span>Verathos Operator</span></div>
      <div class="topbar-copy"><h1 id="pageTitle">Overview</h1><p id="pageSubtitle">Pool health and next actions</p></div>
      <div class="spacer"></div>
      <div class="conn"><span id="connDot" class="dot"></span><span id="connText">connecting…</span></div>
      <div class="header-actions">
        <span id="accessBadge" class="access-badge">public view</span>
        <button id="walletButton" class="primary" onclick="connectWallet()">Connect wallet</button>
        <button id="signoutButton" class="sm hidden" onclick="signOut()">Sign out</button>
      </div>
    </header>

    <main>
      <section id="page-overview" class="page active">
        <div class="page-heading"><div><h2>Pool overview</h2><p>Every machine and mesh, compacted into one operational view.</p></div><div class="page-heading-actions"><button onclick="showAdd()">+ Add machine</button><button class="primary" onclick="goPage('launch')">Launch model</button></div></div>
        <div class="panel access-panel management-note">
          <div class="access-copy"><div id="accessTitle" class="access-title">Public fleet view</div><div id="accessDetail" class="access-detail">Connect the owner wallet to manage machines and meshes.</div></div>
          <div><div class="muted" style="font-size:10.5px;text-transform:uppercase;letter-spacing:.08em">Pool owner</div><div id="ownerIdentity" class="identity">not bound</div></div>
        </div>
        <div class="stat-grid">
          <div class="stat"><span class="l">Online machines</span><span id="stMachines" class="n">—</span></div>
          <div class="stat"><span class="l">Available</span><span id="stIdle" class="n">—</span></div>
          <div class="stat"><span class="l">Serving meshes</span><span id="stMeshes" class="n">—</span></div>
          <div class="stat"><span class="l">Online VRAM</span><span id="stVram" class="n">—</span></div>
        </div>
        <div class="panel topology-panel">
          <div class="topology-head">
            <div class="topology-title"><h3>Live allocation</h3><p id="topologyHint">Every line is one machine-to-mesh membership.</p></div>
            <span id="topologySummary" class="topology-summary">waiting for pool state…</span>
            <div class="topology-switch" aria-label="Allocation visualization">
              <button id="topologyFlowButton" onclick="setTopologyMode('flow')">Flow</button>
              <button id="topologyMatrixButton" onclick="setTopologyMode('matrix')">Matrix</button>
            </div>
          </div>
          <div class="topology-legend"><span><i class="driver"></i>driver</span><span><i></i>serving member</span><span><i class="starting"></i>starting</span><span><i class="offline"></i>error / missing</span></div>
          <div id="topologyView" class="topology-view"><div class="topology-empty">Waiting for pool state…</div></div>
        </div>
        <div class="resource-overview">
          <div class="panel resource-panel">
            <div class="resource-head"><h3>Machines</h3><span id="overviewMachineCount" class="count">—</span><button onclick="goPage('machines')">Details</button></div>
            <div id="overviewMachines" class="resource-list"><div class="resource-empty">Waiting for pool state…</div></div>
          </div>
          <div class="panel resource-panel">
            <div class="resource-head"><h3>Meshes</h3><span id="overviewMeshCount" class="count">—</span><button onclick="goPage('meshes')">Details</button></div>
            <div id="overviewMeshes" class="resource-list"><div class="resource-empty">Waiting for pool state…</div></div>
          </div>
        </div>
      </section>

      <section id="page-machines" class="page">
        <div class="page-heading"><div><h2>Machines</h2><p>Enroll GPUs, inspect capacity, and spot slow or offline links.</p></div><div class="spacer"></div><button id="addButton" class="primary" onclick="showAdd()">+ Add machine</button></div>
        <div class="machines-layout">
          <div class="mapwrap">
            <span class="maplabel">Latency to coordinator</span>
            <canvas id="map"></canvas>
            <div class="legend"><span><i style="background:var(--serving)"></i>serving</span><span><i style="background:var(--driving)"></i>starting</span><span><i style="background:var(--assigned)"></i>assigned</span><span><i style="background:var(--muted)"></i>idle</span></div>
          </div>
          <div class="stats">
            <div class="stat"><span class="l">Online</span><span id="machineOnline" class="n">—</span></div>
            <div class="stat"><span class="l">Offline</span><span id="machineOffline" class="n">—</span></div>
            <div class="stat"><span class="l">Available</span><span id="machineIdle" class="n">—</span></div>
            <div class="stat"><span class="l">Total VRAM</span><span id="machineVram" class="n">—</span></div>
          </div>
        </div>
        <div class="machine-section"><div class="sec-head"><h2>Fleet</h2><span class="hint">workers connected to this coordinator</span></div><div id="workers" class="grid"></div></div>
      </section>

      <section id="page-launch" class="page">
        <div class="page-heading"><div><h2>Launch a model</h2><p>Choose a model; Verathos ranks placements by capacity and link latency.</p></div></div>
        <div class="panel" style="padding:18px">
          <div class="launch">
            <div class="launch-side"><div class="field"><label>Model</label><select id="modelSel" onchange="loadRecs()"></select></div><label class="toggle"><input id="advToggle" type="checkbox" onchange="toggleAdv()" /> Advanced — choose machines manually</label></div>
            <div><div id="recWrap"><div class="field"><label>Recommended placement</label></div><div id="recs" class="rec"></div></div>
              <div id="advBox" class="adv"><div class="field"><label>Select machines for the pipeline</label></div><div id="wsel" class="wsel"></div><div class="adv-row"><div class="field" style="margin:0"><label>Driver (runs the coordinator)</label><select id="driverSel" style="width:220px"></select></div><button class="primary" style="align-self:flex-end" onclick="launchAdvanced()">Launch here</button></div></div>
            </div>
          </div>
        </div>
      </section>

      <section id="page-meshes" class="page">
        <div class="page-heading"><div><h2>Meshes</h2><p>Follow model startup, test verified inference, and stop workloads.</p></div></div>
        <div class="panel"><div id="meshes"></div></div>
      </section>
    </main>
  </div>
</div>

<div id="toast"></div>

<script>
const LS_SESSION = "verathos.operator.session";
let SESSION = null;
let last = { workers: {}, meshes: {} };
let recModel = "";
let openChat = null;      // mesh_key with chat expanded
let advSel = new Set();
let walletAccounts = [];
let pendingChallenge = null;
let topologyMode = localStorage.getItem("verathos.topology.mode") || "";

function toast(msg, err){ const el=document.getElementById("toast"); el.textContent=msg; el.className="show"+(err?" err":""); clearTimeout(el._t); el._t=setTimeout(()=>el.className="",2800); }
function show(id){ document.getElementById(id).classList.remove("hidden"); }
function hide(id){ document.getElementById(id).classList.add("hidden"); }
function esc(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
const SAFE_STATUS_CLASSES = new Set(["idle","assigned","serving","joining","driving","fetching","error","stale"]);
function statusClass(value){ const s=String(value||"idle"); return SAFE_STATUS_CLASSES.has(s)?s:"error"; }

const PAGE_META={
  overview:["Overview","Pool health and next actions"],
  machines:["Machines","Fleet capacity and network latency"],
  launch:["Launch","Model placement and startup"],
  meshes:["Meshes","Running verified inference workloads"]
};
function goPage(page, writeHistory){
  if(!PAGE_META[page]) page="overview";
  document.querySelectorAll(".page").forEach(el=>el.classList.toggle("active",el.id==="page-"+page));
  document.querySelectorAll(".nav button").forEach(el=>{ const on=el.dataset.page===page; el.classList.toggle("active",on); if(on) el.setAttribute("aria-current","page"); else el.removeAttribute("aria-current"); });
  document.getElementById("pageTitle").textContent=PAGE_META[page][0]; document.getElementById("pageSubtitle").textContent=PAGE_META[page][1];
  if(writeHistory!==false && location.hash!=="#"+page) history.pushState({page},"","#"+page);
  if(page==="machines") setTimeout(sizeMap,0);
  window.scrollTo({top:0,behavior:"auto"});
}
window.addEventListener("popstate",()=>goPage(location.hash.slice(1)||"overview",false));

function authBody(body){
  const out=Object.assign({},body||{});
  if(SESSION&&SESSION.token) out.session=SESSION.token;
  return out;
}
function canManage(){ return !!(SESSION&&SESSION.is_owner); }
function shortAccount(a){ const s=String(a||""); return s.length>18?s.slice(0,8)+"…"+s.slice(-7):s; }

function redactValidatorScores(meshes){
  const scrubbed={};
  Object.entries(meshes).forEach(([key,mesh])=>{
    scrubbed[key]=Object.assign({},mesh||{}, {validator_score:{
      available:false, source:"validator_ema",
      reason:"owner access is required to view validator scores"
    }});
  });
  return scrubbed;
}
function scrubCachedValidatorScores(){
  last=Object.assign({},last||{}, {meshes:redactValidatorScores((last&&last.meshes)||{})});
}
function renderCachedPublicState(){
  const workers=(last&&last.workers)||{}, meshes=(last&&last.meshes)||{};
  renderStats(workers,meshes); renderOverview(workers,meshes); renderWorkers(workers);
  renderModels(workers,(last&&last.models)||{},(last&&last.serving_mode)||"");
  renderMeshes(meshes);
}
function invalidateSession(){
  SESSION=null; localStorage.removeItem(LS_SESSION); scrubCachedValidatorScores();
  if(CHAT_ABORT) CHAT_ABORT.abort(); openChat=null;
  updateAccess(); renderCachedPublicState();
}

async function api(route, body, options){
  const payload = options&&options.auth===false ? Object.assign({},body||{}) : authBody(body);
  const r = await fetch(route, { method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(payload) });
  const j = await r.json().catch(()=>({}));
  if (!r.ok){
    if(r.status===403 && SESSION&&SESSION.is_owner) invalidateSession();
    const e=new Error(j.error || ("HTTP "+r.status)); e.status=r.status; throw e;
  }
  return j;
}

function updateAccess(){
  const owner=String(last.owner_account||""); const badge=document.getElementById("accessBadge");
  let title="Public fleet view", detail="Connect the owner wallet to manage machines and meshes.", kind="";
  if(SESSION&&SESSION.is_owner){ title="Owner wallet connected"; detail="Machine onboarding, model launches, mesh tests, and stops are enabled."; kind="owner"; }
  else if(SESSION){ title="Wallet connected · read-only"; detail="This account is not the pool owner. Fleet state and recommendations remain visible."; kind="viewer"; }
  else if(!owner){ title="Owner setup required"; detail="Complete owner setup on the coordinator before using management actions."; }
  badge.className="access-badge "+kind; badge.textContent=kind==="owner"?"management access":kind==="viewer"?"read-only wallet":"public view";
  document.getElementById("accessTitle").textContent=title; document.getElementById("accessDetail").textContent=detail;
  document.getElementById("ownerIdentity").textContent=owner||"not bound";
  document.getElementById("walletButton").textContent=SESSION?shortAccount(SESSION.account):"Connect wallet";
  document.getElementById("signoutButton").classList.toggle("hidden",!SESSION);
  const adv=document.getElementById("advToggle"); adv.disabled=!canManage();
  if(!canManage()&&adv.checked){ adv.checked=false; toggleAdv(); }
}

function showAdd(){
  if(!requireManagement("add a machine")) return;
  if(!document.getElementById("workerId").value) document.getElementById("workerId").value="gpu-"+Math.random().toString(36).slice(2,7);
  const validator=last.serving_mode==="subnet";
  document.getElementById("validatorDriverFields").classList.toggle("hidden",!validator);
  if(validator){
    const hasDriver=Object.values(last.workers||{}).some(w=>{const c=(w.capability||{});return !!(c.subnet_driver_ready??c.validator_driver_ready);});
    document.getElementById("coordinatorCapable").checked=!hasDriver;
  }
  toggleCoordinatorFields();
  updateJoinCommand(); updateEnrollmentState(); show("addModal");
}
function shellArg(v){ return "'"+String(v).replaceAll("'", "'\"'\"'")+"'"; }
function toggleCoordinatorFields(){
  const enabled=last.serving_mode==="subnet"&&document.getElementById("coordinatorCapable").checked;
  document.getElementById("coordinatorFields").classList.toggle("hidden",!enabled);
  updateJoinCommand();
}
function updateJoinCommand(){
  const id=document.getElementById("workerId").value.trim(); const host=document.getElementById("advertiseHost").value.trim();
  let cmd='bash scripts/join_pool.sh --token-file "$HOME/.verathos/pool-token.txt"';
  if(id) cmd+=" --worker-id "+shellArg(id); if(host) cmd+=" --advertise-host "+shellArg(host);
  let incomplete=false;
  if(last.serving_mode==="subnet"){
    if(document.getElementById("coordinatorCapable").checked){
      const wallet=document.getElementById("coordinatorWallet").value.trim();
      const hotkey=document.getElementById("coordinatorHotkey").value.trim();
      const allowlist=document.getElementById("validatorAllowlistPath").value.trim();
      incomplete=!wallet||!hotkey||!allowlist;
      cmd+=" --wallet-name "+shellArg(wallet||"<wallet-name>");
      cmd+=" --wallet-hotkey "+shellArg(hotkey||"<hotkey-name>");
      cmd+=" --validator-allowlist-path "+shellArg(allowlist||"<allowlist-path>");
    }else cmd+=" --member-only";
  }
  const out=document.getElementById("commandOut"); out.textContent=cmd; out.dataset.incomplete=incomplete?"1":"0";
}
function updateEnrollmentState(){
  const el=document.getElementById("enrollmentState"), id=document.getElementById("workerId").value.trim();
  // The installer starts one unit per GPU named <id>-gpu<i>, so match the
  // exact id or any of its per-GPU units.
  const workers=last.workers||{};
  const match=Object.keys(workers).find(k=>k===id||k.startsWith(id+"-gpu"));
  const w=match?workers[match]:undefined;
  if(w&&!w.stale){ const ready=last.serving_mode!=="subnet"||!!(((w.capability||{}).subnet_driver_ready)??((w.capability||{}).validator_driver_ready)); el.className="enrollment-state connected"; el.innerHTML='<span class="dot live"></span><span>Machine connected. '+(ready?'It can host or join a mesh.':'It is ready as a pipeline stage; coordinator wallet setup is not present.')+'</span>'; }
  else { el.className="enrollment-state"; el.innerHTML='<span class="dot"></span><span>Waiting for the command to run…</span>'; }
}
function closeAdd(){ document.getElementById("commandOut").textContent=""; hide("addModal"); }
function copyCommand(){ const out=document.getElementById("commandOut"); if(out.dataset.incomplete==="1"){ toast("Fill in the coordinator wallet, hotkey, and allowlist path first",true); return; } const cmd=out.textContent; navigator.clipboard.writeText(cmd).then(()=>toast("Enrollment command copied")); }

function strToHex(str){ const b=new TextEncoder().encode(str); let h="0x"; b.forEach(x=>h+=x.toString(16).padStart(2,"0")); return h; }
async function injectedWallets(){
  const start=Date.now(); while(Date.now()-start<2500){ if(window.injectedWeb3&&Object.keys(window.injectedWeb3).length) break; await new Promise(r=>setTimeout(r,100)); }
  const enabled=[];
  for(const [name,provider] of Object.entries(window.injectedWeb3||{})){
    try{ const ext=typeof provider.connect==="function"?await provider.connect("Verathos Operator"):typeof provider.enable==="function"?await provider.enable("Verathos Operator"):null;
      if(ext){ ext.__walletName=name; enabled.push(ext); } }catch(e){}
  }
  return enabled;
}
async function connectWallet(){
  if(!window.isSecureContext){ toast("Wallet access requires HTTPS or localhost",true); return; }
  if(!last.owner_account){ toast("Owner setup must be completed on the coordinator first",true); return; }
  try{
    const exts=await injectedWallets(); if(!exts.length) throw new Error("No Substrate wallet connected. Unlock Talisman or another wallet and approve this site.");
    const all=[]; for(const ext of exts){ try{ const rows=await ext.accounts.get(); rows.forEach(a=>all.push(Object.assign({},a,{__ext:ext,__source:ext.__walletName}))); }catch(e){} }
    const seen=new Set(); walletAccounts=all.filter(a=>(!a.type||a.type==="sr25519"||a.type==="ed25519")&&!seen.has(a.address)&&seen.add(a.address));
    if(!walletAccounts.length) throw new Error("No Bittensor accounts are shared with this site.");
    document.getElementById("walletHint").textContent=last.owner_account?"Choose an account to sign in. Only the bound owner can manage this pool.":"Choose the account that will own this pool.";
    document.getElementById("accountList").innerHTML=walletAccounts.map((a,i)=>`<div class="account" data-account-index="${i}" onclick="startVerify(Number(this.dataset.accountIndex))"><span class="account-dot"></span><div style="min-width:0"><div class="account-name">${esc(a.name||"Account")}</div><div class="account-address">${esc(shortAccount(a.address))} · ${esc(a.__source||"wallet")}</div></div></div>`).join("");
    backToAccounts(); show("walletModal");
  }catch(e){ toast(e.message||String(e),true); }
}
async function startVerify(index){
  const account=walletAccounts[index]; if(!account) return;
  try{ const ch=await api("/v1/auth/challenge",{account:account.address},{auth:false}); pendingChallenge={account,message:ch.message,nonce:ch.nonce};
    document.getElementById("signAccount").textContent=shortAccount(account.address); document.getElementById("challengeOut").textContent=ch.message;
    document.getElementById("walletChoose").classList.add("hidden"); document.getElementById("walletSign").classList.remove("hidden");
  }catch(e){ toast(e.message,true); }
}
async function signChallenge(){
  if(!pendingChallenge) return; const button=document.getElementById("signButton"); button.disabled=true;
  try{
    const signer=pendingChallenge.account.__ext.signer; if(!signer||typeof signer.signRaw!=="function") throw new Error("This wallet does not provide Substrate raw signing.");
    const signed=await signer.signRaw({address:pendingChallenge.account.address,data:strToHex(pendingChallenge.message),type:"bytes"});
    const res=await api("/v1/auth/verify",{account:pendingChallenge.account.address,nonce:pendingChallenge.nonce,signature:signed.signature});
    SESSION={token:res.token,account:res.account,is_owner:!!res.is_owner}; localStorage.setItem(LS_SESSION,JSON.stringify(SESSION));
    closeWallet(); updateAccess(); await refresh(); loadRecs(); toast(SESSION.is_owner?"Owner wallet verified":"Wallet verified · read-only");
  }catch(e){ toast("Sign-in failed: "+(e.message||e),true); }
  finally{ button.disabled=false; }
}
function backToAccounts(){ pendingChallenge=null; document.getElementById("walletChoose").classList.remove("hidden"); document.getElementById("walletSign").classList.add("hidden"); }
function closeWallet(){ pendingChallenge=null; hide("walletModal"); }
async function signOut(){
  const token=SESSION&&SESSION.token; invalidateSession(); loadRecs();
  if(token) api("/v1/auth/logout",{session:token},{auth:false}).catch(()=>{}); toast("Signed out");
}
function setConn(ok){ document.getElementById("connDot").className="dot "+(ok?"live":"down"); document.getElementById("connText").textContent = ok?"live":"disconnected"; }
function requireManagement(action){
  if(canManage()) return true;
  if(!last.owner_account){ toast("Owner setup must be completed on the coordinator first",true); return false; }
  const task=action||"manage this pool";
  toast(SESSION?"This wallet is read-only. Connect the pool owner wallet to "+task:"Connect the owner wallet to "+task,true);
  connectWallet(); return false;
}

// Thresholds tuned for MESH inference, where each decode token pays the link
// RTT: <=50ms same DC/region (near-local), <=150ms cross-region (works, adds
// up per token), >150ms cross-continent/poor (single-stream noticeably slow).
function fmtRtt(ms){ if(ms==null) return {t:"—",c:""}; const v=Math.round(ms); return {t:v+" ms", c: v<=50?"rtt-good":v<=150?"rtt-mid":"rtt-bad"}; }
function statusColor(st){ return ({serving:"#35d07f",driving:"#34c6ec",joining:"#34c6ec",assigned:"#f2b03c",error:"#f0555f",stale:"#f0555f"})[st] || "#7c8593"; }
function gpuIcon(){ return '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="3" y="6" width="18" height="12" rx="2"/><rect x="7" y="10" width="6" height="4" rx="1"/><path d="M17 10v4"/></svg>'; }
function checkIcon(){ return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M20 6L9 17l-5-5"/></svg>'; }

/* ---- pool map (canvas: radial by latency) ---- */
const cv = document.getElementById("map"); const ctx = cv.getContext("2d");
let dpr = Math.max(1, window.devicePixelRatio||1), W=0, H=0;
function sizeMap(){ const r=cv.getBoundingClientRect(); if(r.width<1||r.height<1) return; W=r.width; H=r.height; cv.width=W*dpr; cv.height=H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); }
window.addEventListener("resize", sizeMap);
function hashAngle(id){ let h=0; for(let i=0;i<id.length;i++) h=(h*31 + id.charCodeAt(i))>>>0; return (h % 3600)/3600 * Math.PI*2; }
function rttRadius(ms, maxR){ const inner=44; if(ms==null) return maxR*0.82; const t=Math.min(1, ms/200); return inner + t*(maxR-inner); }
function mapNodeRadius(w){ return 6 + Math.min(9, ((w.capability||{}).vram_gb||0)/12); }
function mapNodeLabel(id){
  const m=String(id).match(/^([a-z0-9]+)-.*?-(\d{2})$/i);
  if(m) return m[1]+" "+m[2];
  return id.length>14?id.slice(0,13)+"…":id;
}
function clampMapLabel(label,width,height){
  label.x=Math.max(label.width/2+7,Math.min(width-label.width/2-7,label.x));
  // Reserve the top strip for the map title and outer ring captions.
  label.y=Math.max(38,Math.min(height-9,label.y));
}
function relaxMapLabels(labels,width,height,cx,cy){
  const labelHeight=15,pad=5;
  for(let it=0;it<80;it++){
    let moved=false;
    labels.forEach(label=>clampMapLabel(label,width,height));
    for(let i=0;i<labels.length;i++) for(let j=i+1;j<labels.length;j++){
      const a=labels[i],b=labels[j],dx=b.x-a.x,dy=b.y-a.y;
      const overlapX=(a.width+b.width)/2+pad-Math.abs(dx);
      const overlapY=labelHeight+pad-Math.abs(dy);
      if(overlapX<=0||overlapY<=0) continue;
      if(overlapY<=overlapX){
        const direction=Math.abs(dy)>.01?Math.sign(dy):(i<j?1:-1);
        const shift=overlapY/2+.35; a.y-=direction*shift; b.y+=direction*shift;
      }else{
        const direction=Math.abs(dx)>.01?Math.sign(dx):(i<j?1:-1);
        const shift=overlapX/2+.35; a.x-=direction*shift; b.x+=direction*shift;
      }
      moved=true;
    }
    // Keep the coordinator label readable as well.
    labels.forEach((label,index)=>{
      const overlapX=label.width/2+52-Math.abs(label.x-cx);
      const overlapY=12+10-Math.abs(label.y-(cy+20));
      if(overlapX<=0||overlapY<=0) return;
      const dx=label.x-cx,dy=label.y-(cy+20);
      if(overlapY<=overlapX){
        const direction=Math.abs(dy)>.01?Math.sign(dy):(index%2?1:-1);
        label.y+=direction*(overlapY+.5);
      }else{
        const direction=Math.abs(dx)>.01?Math.sign(dx):(index%2?1:-1);
        label.x+=direction*(overlapX+.5);
      }
      moved=true;
    });
    labels.forEach(label=>clampMapLabel(label,width,height));
    if(!moved) break;
  }
}
function wrappedAngle(delta){ return Math.atan2(Math.sin(delta),Math.cos(delta)); }
function separateMapAngles(ids,ang,radial,marker){
  // Stable IDs can hash to nearly the same angle (for example gpu-01..gpu-04).
  // Resolve those collisions tangentially while leaving every node on its
  // latency radius. This keeps co-located pods individually visible without
  // pretending they have different coordinator RTTs.
  for(let it=0;it<96;it++){
    let moved=false;
    for(let i=0;i<ids.length;i++) for(let j=i+1;j<ids.length;j++){
      const a=ids[i],b=ids[j],ra=radial[a],rb=radial[b];
      const minDistance=marker[a]+marker[b]+10;
      if(Math.abs(ra-rb)>=minDistance) continue;
      const denom=2*ra*rb;
      if(denom<=0) continue;
      const cosine=Math.max(-1,Math.min(1,(ra*ra+rb*rb-minDistance*minDistance)/denom));
      const minSeparation=Math.acos(cosine);
      const delta=wrappedAngle(ang[b]-ang[a]), separation=Math.abs(delta);
      if(separation+0.0001>=minSeparation) continue;
      const direction=separation>0.00001?Math.sign(delta):(a<b?1:-1);
      const shift=(minSeparation-separation)*0.26;
      ang[a]-=direction*shift; ang[b]+=direction*shift; moved=true;
    }
    if(!moved) break;
  }
}
let reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
function drawMap(ts){
  if(!W){ sizeMap(); if(!W){ requestAnimationFrame(drawMap); return; } }
  ctx.clearRect(0,0,W,H);
  const cx=W/2, cy=H/2, maxR=Math.min(W,H)/2 - 26;
  // latency rings
  ctx.font = "10px ui-monospace, monospace"; ctx.textAlign="left";
  [[50,"50ms"],[100,"100ms"],[150,"150ms"]].forEach(([ms,lbl])=>{
    const r=rttRadius(ms,maxR); ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2);
    ctx.strokeStyle="rgba(120,133,147,.13)"; ctx.lineWidth=1; ctx.stroke();
    ctx.fillStyle="rgba(120,133,147,.35)"; ctx.fillText(lbl, cx+2, cy-r-3);
  });
  const ws = last.workers||{}; const ids=Object.keys(ws);
  // Angle: start from a stable hash, then if workers have measured mutual RTT,
  // relax angles so network-close machines cluster (low RTT attracts, high RTT
  // repels). Radius always = latency to the coordinator. No peer data (e.g.
  // firewalled boxes that can't probe each other) => pure hash placement.
  const ang={}; ids.forEach(id=>ang[id]=hashAngle(id));
  const hasPeer = ids.some(id=>Object.keys((ws[id].peer_rtt_ms)||{}).length);
  if(hasPeer && ids.length>1){
    for(let it=0; it<60; it++){ const d={}; ids.forEach(id=>d[id]=0);
      for(let i=0;i<ids.length;i++) for(let j=i+1;j<ids.length;j++){ const A=ids[i],B=ids[j];
        const m=((ws[A].peer_rtt_ms||{})[B]) ?? ((ws[B].peer_rtt_ms||{})[A]); if(m==null) continue;
        let da=ang[B]-ang[A]; da=Math.atan2(Math.sin(da),Math.cos(da));   // wrap to [-pi,pi]
        const want=0.35 + Math.min(1,m/160)*(Math.PI-0.35);               // low rtt -> small gap
        const f=(Math.abs(da)-want)*0.04*Math.sign(da);                   // pull toward want
        d[A]+=f; d[B]-=f;
      }
      ids.forEach(id=>ang[id]+=d[id]);
    }
  }
  const radial={},marker={};
  ids.forEach(id=>{ const w=ws[id],rtt=(w.rtt_ms||{}).manager;
    radial[id]=rttRadius(w.stale?null:rtt,maxR); marker[id]=mapNodeRadius(w); });
  separateMapAngles(ids,ang,radial,marker);
  const pos={}; const driverIds=new Set();
  ids.forEach(id=>{ const w=ws[id],a=ang[id],r=radial[id];
    pos[id]={x:cx+Math.cos(a)*r,y:cy+Math.sin(a)*r,w,a}; });
  const center={x:cx,y:cy};
  // Inference flows as a round trip: coordinator -> driver -> stage 2 -> ... ->
  // last stage, then the result returns back along the same chain. One packet
  // ping-pongs the whole path to show that each decode step is a round trip.
  Object.values(last.meshes||{}).forEach(m=>{
    if(m.status==="stopped") return; if(!pos[m.driver]) return; driverIds.add(m.driver);
    const chain=[center, pos[m.driver]];
    (m.members||[]).forEach(mid=>{ if(mid!==m.driver && pos[mid]) chain.push(pos[mid]); });
    const serving = m.status==="serving";
    const col = serving ? "rgba(53,208,127,.5)" : "rgba(52,198,236,.45)";
    for(let i=0;i<chain.length-1;i++){ ctx.beginPath(); ctx.moveTo(chain[i].x,chain[i].y); ctx.lineTo(chain[i+1].x,chain[i+1].y);
      ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.stroke(); }
    if(!reduced && serving && chain.length>1){
      const segs=chain.length-1; const period=1500*segs;                       // out then back
      let ph=((ts||0)%(period*2))/period; const back=ph>1; if(back) ph=2-ph;    // ping-pong
      const f=ph*segs; const si=Math.min(segs-1, Math.floor(f)); const t=f-si;
      const A=chain[si], B=chain[si+1]; const px=A.x+(B.x-A.x)*t, py=A.y+(B.y-A.y)*t;
      ctx.beginPath(); ctx.arc(px,py,2.8,0,Math.PI*2); ctx.fillStyle=back?"#a9e8ff":"#8ef0bf"; ctx.fill();
    }
  });
  // center = coordinator (control + request entry)
  ctx.beginPath(); ctx.arc(cx,cy,7,0,Math.PI*2); ctx.fillStyle="#2a63cf"; ctx.fill();
  ctx.strokeStyle="rgba(79,143,247,.7)"; ctx.lineWidth=2; ctx.stroke();
  ctx.fillStyle="rgba(148,160,175,.72)"; ctx.textAlign="center"; ctx.font="10px ui-sans-serif, system-ui"; ctx.fillText("coordinator", cx, cy+20);
  // nodes
  ctx.font="11px ui-monospace, monospace";
  const labels=[];
  ids.forEach(id=>{ const {x,y,w,a}=pos[id]; const st=w.stale?"stale":(w.status||"idle"); const col=statusColor(st);
    const isDrv=driverIds.has(id); const rad=marker[id];
    if(!reduced && (st==="serving"||st==="driving")){ const pulse=1+0.18*Math.sin((ts||0)/380 + hashAngle(id));
      ctx.beginPath(); ctx.arc(x,y,rad*1.9*pulse,0,Math.PI*2); ctx.fillStyle=col; ctx.globalAlpha=0.10; ctx.fill(); ctx.globalAlpha=1; }
    ctx.beginPath(); ctx.arc(x,y,rad,0,Math.PI*2); ctx.fillStyle=col; ctx.fill();
    ctx.strokeStyle= isDrv ? "#4f8ff7" : "rgba(9,11,16,.9)"; ctx.lineWidth= isDrv ? 2.6 : 2; ctx.stroke();
    const ux=Math.cos(a),uy=Math.sin(a),offset=rad+10;
    const text=(isDrv?"◆ ":"")+mapNodeLabel(id),width=Math.ceil(ctx.measureText(text).width);
    let lx=x+ux*offset;
    if(ux>.24) lx+=width/2+3; else if(ux<-.24) lx-=width/2+3;
    labels.push({x:lx,y:y+uy*offset+4,text,width,anchorX:x,anchorY:y});
  });
  relaxMapLabels(labels,W,H,cx,cy);
  labels.forEach(label=>{
    ctx.beginPath(); ctx.moveTo(label.anchorX,label.anchorY); ctx.lineTo(label.x,label.y-4);
    ctx.strokeStyle="rgba(148,160,175,.25)"; ctx.lineWidth=1; ctx.stroke();
    ctx.fillStyle="rgba(37,37,38,.88)";
    ctx.fillRect(label.x-label.width/2-3,label.y-11,label.width+6,15);
    ctx.fillStyle="rgba(233,236,241,.86)"; ctx.textAlign="center";
    ctx.fillText(label.text,label.x,label.y);
  });
  requestAnimationFrame(drawMap);
}

/* ---- renders ---- */
function driverSet(meshes){ const s=new Set(); Object.values(meshes||{}).forEach(m=>{ if(m.status!=="stopped"&&m.driver) s.add(m.driver); }); return s; }
function workerPhase(id,w,meshes){
  const myMesh=Object.values(meshes||{}).find(m=>m.status!=="stopped"&&(m.driver===id||(m.members||[]).includes(id)));
  let raw,st;
  if(w.stale){ raw="offline"; st="stale"; }
  else if(myMesh&&myMesh.status==="serving"){ raw="serving"; st="serving"; }
  else if(myMesh&&myMesh.status&&myMesh.status!=="serving"){ raw=(w.status&&w.status!=="idle")?w.status:myMesh.status; st=raw.split(/[ (]/)[0]; }
  else { raw=w.status||"idle"; st=raw.split(/[ (]/)[0]; }
  return {raw,st,label:w.stale?"offline":st};
}
function renderWorkers(workers){
  const ids=Object.keys(workers); const el=document.getElementById("workers"); const drivers=driverSet(last.meshes);
  if(!ids.length){ el.innerHTML='<div class="panel empty">No machines yet. Use <b style="color:var(--txt)">Add machine</b> above to enroll your first worker.</div>'; return; }
  const meshes=last.meshes||{};
  el.innerHTML = ids.map(id=>{ const w=workers[id], cap=w.capability||{};
    // Derive the card status from the mesh this worker actually belongs to,
    // not the worker's transient self-report: a driver of a SERVING mesh
    // must read "serving" (+ driver badge), not a stuck "driving
    // (warming proofs 98%)". The verbose self-report stays in the tooltip.
    const {raw,st,label}=workerPhase(id,w,meshes); const cls=statusClass(st); const rtt=fmtRtt((w.rtt_ms||{}).manager);
    const drv = drivers.has(id) ? '<span class="pill driving" title="runs the mesh coordinator">◆ driver</span>' : '';
    const memberOnly = cap.member_only ? '<span class="pill assigned" title="serves pipeline stages; cannot host the coordinator">member-only</span>' : '';
    const subnetRole = last.serving_mode==="subnet"&&!cap.member_only ? ((cap.subnet_driver_ready??cap.validator_driver_ready)?'<span class="pill serving" title="fresh validator allowlist; can drive (no wallet needed)">coordinator-ready</span>':'<span class="pill assigned" title="no fresh validator allowlist yet">stage-only</span>') : '';
    const removeBtn = w.stale ? `<button class="ghost-danger sm" style="align-self:flex-start" data-worker-id="${esc(id)}" onclick="removeWorker(this.dataset.workerId)">Remove</button>` : "";
    return `<div class="card"><div class="top"><div class="gpu"><div class="ic">${gpuIcon()}</div><div style="min-width:0;flex:1"><div class="nm">${esc(cap.gpu_name||"GPU")}</div><div class="id">${esc(id)}</div></div></div><div class="badges"><span class="pill ${cls}" title="${esc(raw)}"><span class="bd"></span>${esc(label)}</span>${drv}${memberOnly}${subnetRole}</div></div>
      <div class="meta"><div><span class="k">VRAM</span> <span class="v">${cap.vram_gb?esc(cap.vram_gb)+" GB":"—"}</span></div><div><span class="k">RTT</span> <span class="v ${rtt.c}">${rtt.t}</span></div><div><span class="k">Mesh</span> <span class="v">${w.mesh?esc(w.mesh):"—"}</span></div></div>${removeBtn}</div>`;
  }).join("");
}
function modelCatalog(workers, registry, servingMode){ const m=new Map();
  if(servingMode!=="subnet") Object.values(workers).forEach(w=>(w.catalog||[]).forEach(c=>{ if(c.model_id) m.set(c.model_id, Math.max(m.get(c.model_id)||0, c.model_bytes||0)); }));
  // Validator pools expose only management-pinned chain/model anchors. A raw
  // worker catalog entry is not launchable until register-model binds it.
  Object.entries(registry||{}).forEach(([id,r])=>{ if(r.launch_ready!==false&&!m.has(id)) m.set(id, r.model_bytes||0); }); return m; }
function renderModels(workers, registry, servingMode){ const cat=modelCatalog(workers, registry, servingMode); const sel=document.getElementById("modelSel"); const prev=sel.value; const opts=[...cat.keys()];
  const empty=servingMode==="subnet"?"no chain-bound models registered":"no models in worker catalogs";
  sel.innerHTML = opts.length ? opts.map(m=>`<option value="${esc(m)}">${esc(m)}</option>`).join("") : `<option value="">${empty}</option>`;
  if(opts.includes(prev)) sel.value=prev;
  if(sel.value!==recModel){ loadRecs(); } if(document.getElementById("advToggle").checked) renderAdv();
}
async function loadRecs(){ const model=document.getElementById("modelSel").value; recModel=model; const box=document.getElementById("recs");
  if(!model){ box.innerHTML='<div class="muted" style="font-size:13px">Connect a machine that has a model.</div>'; return; }
  try { const {suggestions, reasons}=await api("/v1/pool/recommend",{model_id:model},{auth:false});
    if(!suggestions.length){
      const rs=Object.entries(reasons||{}).map(([id,why])=>`<div style="margin-top:4px"><span class="mono" style="font-size:12px">${esc(id)}</span> <span class="muted">— ${esc(why)}</span></div>`).join("");
      box.innerHTML=`<div style="font-size:13px"><div class="muted">No machine set can host this model right now:</div>${rs}</div>`; return; }
    const linkChip = (s)=>{ if(s.workers.length<2) return '<span class="rtt-good">local</span>';
      const cls = s.link_class||""; const c = (cls==="local"||cls==="lan")?"rtt-good":(cls==="regional")?"rtt-mid":"rtt-bad";
      return `<span class="${c}" title="worker-to-worker round trip; every generated token crosses this link">${Math.round(s.max_rtt_ms)} ms · ${esc(cls)}</span>`; };
    const rows = suggestions.slice(0,6).map((s,i)=>{ const kind=s.workers.length>1?"pipeline":"single";
      const dl = s.fetch ? ` · <span class="drv" title="the driver will download the model first (auto), with progress">↓ downloads ${s.download_gb||"?"} GB</span>` : "";
      const btn = s.fetch ? "Download &amp; launch" : "Launch";
      const warn = s.warn ? `<div class="sub" style="color:#f2b03c;margin-top:2px" title="placement works but the link caps decode speed">⚠ ${esc(s.warn)}</div>` : "";
      const placement=esc(JSON.stringify({workers:s.workers,driver:s.driver,model}));
      return `<div class="rec-row ${i===0?'best':''}">${i===0?'<span class="badge">BEST</span>':''}<div class="who"><div class="members">${s.workers.map(esc).join('  +  ')}</div><div class="sub">${kind} · driver <span class="drv">${esc(s.driver)}</span> · link ${linkChip(s)}${dl}</div>${warn}</div><button class="primary sm" data-placement="${placement}" onclick="launchPlacement(this)">${btn}</button></div>`;
    }).join("");
    // Machines NOT in any suggestion: say why, so a "missing" box never
    // reads as a bug (no disk for the download, VRAM too small solo, busy).
    const placed = new Set(suggestions.flatMap(s=>s.workers));
    const left = Object.entries(reasons||{}).filter(([id])=>!placed.has(id));
    const why = left.length ? `<div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(124,133,147,.25)"><div class="muted" style="font-size:12px;margin-bottom:2px">Not placeable right now</div>${left.map(([id,r])=>`<div style="margin-top:3px;font-size:12px"><span class="mono">${esc(id)}</span> <span class="muted">— ${esc(r)}</span></div>`).join("")}</div>` : "";
    box.innerHTML = rows + why;
  } catch(e){ box.innerHTML=`<div class="muted" style="font-size:13px">${esc(e.message)}</div>`; }
}
function toggleAdv(){ const on=document.getElementById("advToggle").checked;
  document.getElementById("advBox").classList.toggle("on", on); document.getElementById("recWrap").style.display = on?"none":"block";
  if(on) renderAdv();
}
function renderAdv(){ const workers=last.workers||{}; const wsel=document.getElementById("wsel");
  wsel.innerHTML = Object.keys(workers).map(id=>{ const w=workers[id]; const busy=!canManage()||w.stale||((w.status||"idle")!=="idle"); const sel=advSel.has(id);
    return `<div class="wchip ${sel?'sel':''} ${busy?'busy':''}" data-worker-id="${esc(id)}" ${busy?'':`onclick="toggleWorker(this.dataset.workerId)"`}>${sel?checkIconMini():''}${esc(id)} · ${esc((w.capability||{}).vram_gb||0)}GB</div>`;
  }).join("") || '<span class="muted" style="font-size:13px">No machines.</span>';
  // Stage-only workers can contribute layers but cannot host the coordinator.
  const ds=document.getElementById("driverSel"); ds.innerHTML=[...advSel].filter(id=>{ const cap=((workers[id]||{}).capability||{}); return !cap.member_only&&(last.serving_mode!=="subnet"||(cap.subnet_driver_ready??cap.validator_driver_ready)); }).map(id=>`<option value="${esc(id)}">${esc(id)}</option>`).join("") || '<option value="">no coordinator-ready machine</option>';
}
function checkIconMini(){ return '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M20 6L9 17l-5-5"/></svg>'; }
function toggleWorker(id){ if(!requireManagement()) return; if(advSel.has(id)) advSel.delete(id); else advSel.add(id); renderAdv(); }
function launchPlacement(el){ const p=JSON.parse(el.dataset.placement||"{}"); doLaunch(p.workers||[],p.driver||"",p.model||""); }
async function launchAdvanced(){ const model=document.getElementById("modelSel").value; const workers=[...advSel]; const driver=document.getElementById("driverSel").value;
  if(!requireManagement("launch a model")) return;
  if(!workers.length){ toast("Select at least one machine", true); return; }
  doLaunch(workers, driver||workers[0], model);
}
async function doLaunch(workers, driver, model){ if(!requireManagement("launch a model")) return; try { const r=await api("/v1/pool/launch",{model_id:model,workers,driver}); toast("Launching "+model+" → "+r.mesh_key); advSel.clear(); refresh(true); } catch(e){ toast(e.message,true); } }
async function removeWorker(id){ if(!requireManagement("remove a machine")) return; if(!confirm("Remove offline machine \""+id+"\" from the pool?")) return;
  try { await api("/v1/pool/remove-worker",{worker_id:id}); toast("Removed "+id); refresh(true); } catch(e){ toast(e.message,true); } }
async function stopMesh(key){ if(!requireManagement("stop a mesh")) return; try { await api("/v1/pool/stop",{mesh_key:key}); toast("Stopping "+key); if(openChat===key){ if(CHAT_ABORT) CHAT_ABORT.abort(); openChat=null; } refresh(true); } catch(e){ toast(e.message,true); } }
function thinkPref(){ return localStorage.getItem("mesh_thinking")==="1"; }
let CHAT_ABORT=null;
let CHAT_SENDING=false;
function openMeshTest(key){
  if(!requireManagement("test a mesh")) return;
  if(CHAT_ABORT) CHAT_ABORT.abort();
  openChat=key; renderMeshes(last.meshes||{}); goPage("meshes");
  setTimeout(()=>{ const input=document.getElementById("ci_"+key); if(input) input.focus(); },0);
}
function toggleChat(key){ if(openChat===key){ if(CHAT_ABORT) CHAT_ABORT.abort(); openChat=null; renderMeshes(last.meshes||{}); } else openMeshTest(key); }
function sseData(frame){
  const lines=frame.replace(/\r/g,"").split("\n"), data=[];
  for(const line of lines) if(line.startsWith("data:")) data.push(line.slice(5).replace(/^ /,""));
  if(!data.length) return null;
  try{ return JSON.parse(data.join("\n")); }
  catch(e){ throw new Error("mesh returned an invalid stream event"); }
}

async function sendChat(key){ if(CHAT_SENDING) return; if(!requireManagement("test a mesh")) return; const inp=document.getElementById("ci_"+key); const out=document.getElementById("co_"+key); const send=document.getElementById("cb_"+key); const msg=inp.value.trim(); if(!msg) return;
  inp.value="";
  out.innerHTML=`<div class="think" id="ck_${esc(key)}" style="display:none"></div><div class="out" id="cs_${esc(key)}"></div><div class="verline" id="cv_${esc(key)}"><span class="muted">streaming from the mesh…</span></div>`;
  const outDiv=document.getElementById("cs_"+key), verDiv=document.getElementById("cv_"+key), thinkDiv=document.getElementById("ck_"+key);
  const controller=new AbortController();
  CHAT_ABORT=controller; CHAT_SENDING=true; inp.disabled=true; if(send) send.disabled=true;
  let thinkTxt="", sawFinal=false, sawTerminal=false;
  const t0=performance.now(); let tFirst=null, tLast=null, content="", nTok=0;
  function renderStreamEvent(ev){
    if(ev.type==="delta"){ if(tFirst===null){ tFirst=performance.now(); } tLast=performance.now(); content+=String(ev.delta||""); nTok++; outDiv.textContent=content; }
    else if(ev.type==="thinking"){ if(tFirst===null){ tFirst=performance.now(); } thinkTxt+=String(ev.thinking||""); thinkDiv.style.display="block"; thinkDiv.textContent="thinking… "+thinkTxt.slice(-400); }
    else if(ev.type==="done"){
      sawFinal=true; sawTerminal=true;
      if(typeof ev.content==="string") outDiv.textContent=ev.content||"(no content)";
      if(thinkTxt) thinkDiv.textContent="thought for a while ("+thinkTxt.length+" chars)";
      // tok/s must reflect GENERATION only: the clock stops at the last
      // token (tLast), not at the done event — otherwise the proving
      // phase deflates the number. Prefer the engine's measured rate.
      const total=(performance.now()-t0)/1000;
      const gen=(tFirst&&tLast&&tLast>tFirst)?(tLast-tFirst)/1000:0;
      const reportedTokens=Number((ev.usage||{}).completion_tokens);
      const tk=Number.isFinite(reportedTokens)&&reportedTokens>=0?Math.trunc(reportedTokens):nTok;
      const engineTps=Number(ev.engine_tps);
      const tps=Number.isFinite(engineTps)?engineTps.toFixed(0):(gen>0?(tk/gen).toFixed(0):"–");
      const proofS=Math.max(0,total-(tLast?(tLast-t0)/1000:0));
      const proofStages=Number(ev.proof_stages), expectedStages=Number(ev.expected_stage_count);
      const fullyVerified=ev.verified===true&&ev.receipt_verified===true
        &&Number.isInteger(proofStages)&&Number.isInteger(expectedStages)
        &&expectedStages>0&&proofStages===expectedStages;
      const ver=ev.error?`<span style="color:var(--error)">${esc(ev.error)}</span>`
        : fullyVerified?`<span class="verbadge">${checkIcon()} coordinator verified · ${proofStages} stages · ${Number(ev.receipts)||0} proofs</span>`
        : (ev.deferred||ev.deferred_obligation)?`<span class="verbadge" style="color:var(--accent)" title="witnesses committed before the beacon; audit resolves off the hot path">${checkIcon()} proof committed · deferred audit</span>`
        : '<span style="color:var(--error)">unverified</span>';
      verDiv.innerHTML=`${ver}<span>${tk} tok · ${tps} tok/s · proof ${proofS.toFixed(1)}s · ${total.toFixed(1)}s total</span>`;
      return true;
    }else if(ev.type==="phase"){
      verDiv.innerHTML='<span class="muted">response complete — generating + verifying proof… (up to ~1 min on large models)</span>';
    }else if(ev.type==="error"){
      sawTerminal=true;
      verDiv.innerHTML=`<span style="color:var(--error)">${esc(ev.error||"stream error")}</span>`;
      return true;
    }
    return false;
  }
  try {
    const resp = await fetch("/v1/pool/chat-stream", { method:"POST", headers:{"Content-Type":"application/json"},
      signal:controller.signal,
      body: JSON.stringify(authBody({mesh_key:key, prompt:msg, max_tokens:256,
        thinking: !!(document.getElementById("ct_"+key)||{}).checked})) });
    if(!resp.ok){ const j=await resp.json().catch(()=>({}));
      if(resp.status===403&&SESSION&&SESSION.is_owner) invalidateSession();
      throw new Error(j.error||("HTTP "+resp.status)); }
    if(!resp.body) throw new Error("mesh stream is unavailable in this browser");
    const reader=resp.body.getReader(), dec=new TextDecoder(); let buf="";
    stream: for(;;){
      const {done,value}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true}); let match;
      while((match=buf.match(/\r?\n\r?\n/))){
        const frame=buf.slice(0,match.index); buf=buf.slice(match.index+match[0].length);
        const ev=sseData(frame); if(!ev) continue;
        if(renderStreamEvent(ev)){ try{ await reader.cancel(); }catch(e){} break stream; }
      }
    }
    buf+=dec.decode();
    if(!sawTerminal&&buf.trim()){ const ev=sseData(buf); if(ev) renderStreamEvent(ev); }
    if(!sawTerminal&&!controller.signal.aborted) throw new Error("mesh stream ended before the proof-bearing final event");
  } catch(e){
    verDiv.innerHTML=`<span style="color:var(--error)">${esc(controller.signal.aborted?"request cancelled":(e.message||"mesh stream failed"))}</span>`;
  } finally {
    if(CHAT_ABORT===controller){
      CHAT_ABORT=null; CHAT_SENDING=false; inp.disabled=false; if(send) send.disabled=false;
      if(openChat===key) inp.focus();
    }
  }
}

// Turn a raw mesh status into a human phase + a progress hint, so an operator
// watching a mesh come up sees WHAT it's doing (downloading 42%, loading the
// model, assembling the pipeline, verifying) and a bar, not a bare word.
function meshPhase(m, workers){
  const st=m.status||"—";
  const drv=(workers||{})[m.driver]||{};
  const dstat=String(drv.status||"");
  if(st==="error")   return {label:"error", cls:"error", err:m.error||""};
  if(st==="serving"&&m.routing_ready===false){
    if(m.driver_stale) return {label:"driver offline", cls:"error", err:"the machine running this mesh stopped responding (crashed or reclaimed); stop the mesh or restart its worker"};
    const n=(m.stale_members||[]).length;
    return {label:n===1?"member offline":n+" members offline", cls:"error", err:"one or more pipeline stages stopped responding; restore every member before routing traffic"};
  }
  if(st==="serving") return {label:"serving", cls:"serving"};
  if(st==="stopped") return {label:"stopped", cls:"idle"};
  if(st==="fetching"){
    const mm=dstat.match(/fetching\s+(\d+)%/); const pct=mm?+mm[1]:null;
    if(dstat.indexOf("verifying")>=0) return {label:"verifying download", cls:"assigned", indet:true};
    return {label: pct!=null?("downloading model "+pct+"%"):"downloading model", cls:"assigned", pct:pct};
  }
  if(st==="driving"){
    const wm=dstat.match(/warming proofs\s*(\d+)?%?/);
    if(wm) return {label: wm[1]?("preparing proofs "+wm[1]+"%"):"preparing proofs…", cls:"driving", pct: wm[1]?+wm[1]:null, indet: !wm[1]};
    return {label:"loading model…", cls:"driving", indet:true};
  }
  if(st==="joining")  return {label:"assembling + verifying…", cls:"driving", indet:true};
  return {label:st, cls:"assigned", indet:true};
}
function progressBar(ph){
  if(ph.cls==="serving"||ph.cls==="error"||ph.cls==="idle") return "";
  const inner = ph.indet
    ? '<div class="pbi indet"></div>'
    : `<div class="pbi" style="width:${Math.max(3,Math.min(100,ph.pct||3))}%"></div>`;
  return `<div class="pbar" title="${esc(ph.label)}">${inner}</div>`;
}
function validatorScoreBadge(m){
  const s=m.validator_score||{};
  if(!s.available){
    const reason=s.reason||"No exact coordinator/model mapping in validator state";
    const label=reason.indexOf("owner access")>=0?"owner only":"not scored";
    return `<span class="vscore unavailable" title="${esc(reason)}">${label}</span>`;
  }
  const score=Number(s.score);
  if(!Number.isFinite(score)) return '<span class="vscore unavailable">not scored</span>';
  const value=Math.abs(score)>=100?score.toFixed(1):Math.abs(score)>=1?score.toFixed(3):score.toFixed(4);
  const latest=s.latest_score||{};
  const completed=Number(latest.ema_at_completion);
  const completedValue=Number.isFinite(completed)?(Math.abs(completed)>=100?completed.toFixed(1):Math.abs(completed)>=1?completed.toFixed(3):completed.toFixed(4)):"—";
  const details=["Current coordinator/model slot EMA",`model index ${s.model_index}`,`validator state epoch ${s.epoch_number}`];
  if(Number.isInteger(latest.score_epoch)) details.push(`latest completed sample epoch ${latest.score_epoch}`);
  if(latest.scored_mesh_id) details.push(`scored mesh ${latest.scored_mesh_id}`);
  if(Number.isInteger(latest.chain_id)&&Number.isInteger(latest.netuid)) details.push(`chain ${latest.chain_id}, netuid ${latest.netuid}`);
  if(Number.isInteger(latest.scored_snapshot_generation)) details.push(`scored snapshot generation ${latest.scored_snapshot_generation}`);
  const snapshotHash=String(latest.scored_verification_snapshot_hash||"");
  if(/^[0-9a-f]{64}$/.test(snapshotHash)) details.push(`scored snapshot ${snapshotHash}`);
  if(s.current_topology_matches_latest_score===true) details.push("latest sample matches current topology");
  else details.push("slot EMA carried across topology; latest sample provenance is shown separately");
  if(s.score_adjusted_since_completion) details.push("current slot EMA was adjusted after the completed sample");
  if(Number.isInteger(s.scored_epochs)) details.push(`${s.scored_epochs} scored epoch${s.scored_epochs===1?"":"s"}`);
  if(s.uid!=null) details.splice(1,0,`UID ${s.uid}`);
  const age=s.age_seconds==null?NaN:Number(s.age_seconds);
  if(Number.isFinite(age)) details.push(`validator state ${age<60?Math.round(age)+"s":Math.round(age/60)+"m"} old`);
  if(s.stale) details.push("validator state is stale");
  if(s.probation) details.push("probation");
  if(s.blacklisted) details.push("blacklisted");
  const cls=s.blacklisted?" blocked":(s.probation||s.stale)?" warning":"";
  const label=s.stale?"slot EMA · stale":s.score_adjusted_since_completion?"slot EMA · adjusted":s.current_topology_matches_latest_score===false?"slot EMA · carried":"slot EMA";
  const sample=Number.isInteger(latest.score_epoch)?`<span class="vsample">sample e${latest.score_epoch} ${completedValue}</span>`:"";
  const exactProvenance=Number.isInteger(latest.chain_id)&&Number.isInteger(latest.netuid)
    &&Number.isInteger(latest.scored_snapshot_generation)&&/^[0-9a-f]{64}$/.test(snapshotHash);
  const provenanceText=exactProvenance
    ? `chain ${latest.chain_id} · netuid ${latest.netuid} · generation ${latest.scored_snapshot_generation} · snapshot ${snapshotHash}`:"";
  const provenance=exactProvenance
    ? `<span class="vprov" tabindex="0" aria-label="Score provenance: ${esc(provenanceText)}">i<span class="vprov-pop">${esc(provenanceText)}</span></span>`:"";
  return `<span class="vscore${cls}" title="${esc(details.join(" · "))}"><span class="vl">${label}</span>${value}${sample}${provenance}</span>`;
}
function renderMeshes(meshes){
  const keys=Object.keys(meshes); const el=document.getElementById("meshes");
  const workers=(last&&last.workers)||{};
  if(!keys.length){ el.innerHTML='<div class="empty">No meshes running. Launch a model above.</div>'; return; }
  el.innerHTML = `<table><thead><tr><th>Model</th><th>Validator score</th><th>Members</th><th>Driver</th><th>Status</th><th>Mesh</th><th></th></tr></thead><tbody>` +
    keys.map(k=>{ const m=meshes[k]; const st=m.status||"—"; const ph=meshPhase(m,workers); const isServing=st==="serving"&&m.routing_ready!==false; const manage=canManage();
      const pill=`<span class="pill ${ph.cls}"${ph.err?` title="${esc(ph.err)}"`:""}><span class="bd"></span>${esc(ph.label)}</span>${progressBar(ph)}`;
      let row = `<tr class="mrow mesh-main-row" data-mesh-key="${esc(k)}"><td data-label="Model">${esc(m.model_id||"—")}</td><td data-label="Validator score">${validatorScoreBadge(m)}</td><td class="mono" data-label="Members" style="font-size:12.5px">${(m.members||[]).map(esc).join(", ")}</td><td class="mono drv" data-label="Driver" style="font-size:12.5px">${esc(m.driver||"—")}</td><td data-label="Status">${pill}</td><td data-label="Mesh"><span class="mk">${esc(k)}</span></td><td class="actions" data-label="" style="text-align:right;white-space:nowrap">${isServing?`<button class="sm" data-mesh-key="${esc(k)}" onclick="toggleChat(this.dataset.meshKey)">${openChat===k?"Close":"Test"}</button> `:""}${st!=="stopped"?`<button class="ghost-danger sm" data-mesh-key="${esc(k)}" onclick="stopMesh(this.dataset.meshKey)">Stop</button>`:""}</td></tr>`;
      if(openChat===k && isServing && manage){
        row += `<tr class="mrow mesh-chat-row" data-mesh-key="${esc(k)}"><td colspan="7" class="chatcell"><div class="chat">
          <div class="apibar">Operator test · routed through the pool manager to the mesh driver</div>
          <div class="io"><input id="ci_${esc(k)}" data-mesh-key="${esc(k)}" type="text" placeholder="Ask the mesh something…" onkeydown="if(event.key==='Enter')sendChat(this.dataset.meshKey)" /><label class="thinklbl" title="Optional reasoning mode. Off by default so operator health checks finish quickly."><input type="checkbox" id="ct_${esc(k)}" ${thinkPref()?"checked":""} onchange="localStorage.setItem('mesh_thinking', this.checked?'1':'0')"/> thinking</label><button class="primary" id="cb_${esc(k)}" data-mesh-key="${esc(k)}" onclick="sendChat(this.dataset.meshKey)">Send</button></div>
          <div id="co_${esc(k)}"></div></div></td></tr>`;
      }
      return row;
    }).join("") + `</tbody></table>`;
}
function renderMeshesPreservingChat(meshes){
  const key=openChat, el=document.getElementById("meshes");
  if(!key){ renderMeshes(meshes); return; }
  const chatRow=[...el.querySelectorAll("tr.mesh-chat-row")].find(row=>row.dataset.meshKey===key);
  if(!chatRow){ renderMeshes(meshes); return; }
  const focused=document.activeElement, restoreFocus=focused&&chatRow.contains(focused);
  const selection=restoreFocus&&typeof focused.selectionStart==="number"
    ? {start:focused.selectionStart,end:focused.selectionEnd,direction:focused.selectionDirection}:null;
  chatRow.remove();
  renderMeshes(meshes);
  const replacement=[...el.querySelectorAll("tr.mesh-chat-row")].find(row=>row.dataset.meshKey===key);
  const mainRow=[...el.querySelectorAll("tr.mesh-main-row")].find(row=>row.dataset.meshKey===key);
  if(replacement) replacement.replaceWith(chatRow);
  else if(mainRow) mainRow.after(chatRow);
  else { openChat=null; return; }
  const mesh=(meshes||{})[key]||{}, routable=mesh.status==="serving"&&mesh.routing_ready!==false;
  const input=chatRow.querySelector("input[type=text]"), send=chatRow.querySelector("button.primary");
  if(!CHAT_SENDING){
    if(input){ input.disabled=!routable; input.title=routable?"":"This mesh is not currently routable"; }
    if(send) send.disabled=!routable;
  }
  if(restoreFocus&&focused&&document.contains(focused)){
    focused.focus();
    if(selection&&typeof focused.setSelectionRange==="function") focused.setSelectionRange(selection.start,selection.end,selection.direction);
  }
}
function renderStats(workers, meshes){
  const ws=Object.values(workers), online=ws.filter(w=>!w.stale), offline=ws.length-online.length;
  const idle=online.filter(w=>(w.status||"idle")==="idle").length, serving=Object.values(meshes).filter(m=>m.status==="serving"&&m.routing_ready!==false).length;
  const vram=online.reduce((a,w)=>a+(+(w.capability||{}).vram_gb||0),0), vramText=vram?vram+" GB":"—";
  document.getElementById("stMachines").textContent=online.length; document.getElementById("stIdle").textContent=idle;
  document.getElementById("stMeshes").textContent=serving; document.getElementById("stVram").textContent=vramText;
  document.getElementById("machineOnline").textContent=online.length; document.getElementById("machineOffline").textContent=offline;
  document.getElementById("machineIdle").textContent=idle; document.getElementById("machineVram").textContent=vramText;
}
function topologyData(workers, meshes){
  const rank={serving:0,driving:1,joining:1,fetching:1,error:2,stopped:3};
  const meshKeys=Object.keys(meshes).filter(k=>(meshes[k].status||"")!=="stopped")
    .sort((a,b)=>(rank[meshes[a].status]??2)-(rank[meshes[b].status]??2)||a.localeCompare(b));
  const machineSet=new Set(Object.keys(workers)); const links=[];
  meshKeys.forEach(mesh=>{
    const m=meshes[mesh], members=new Set(m.members||[]); if(m.driver) members.add(m.driver);
    members.forEach(machine=>{ machineSet.add(machine); links.push({machine,mesh,driver:machine===m.driver,phase:meshPhase(m,workers)}); });
  });
  const machineIds=[...machineSet].sort((a,b)=>{
    const ar=!workers[a]?2:workers[a].stale?1:0, br=!workers[b]?2:workers[b].stale?1:0;
    return ar-br||a.localeCompare(b);
  });
  return {machineIds,meshKeys,links};
}
function setTopologyMode(mode){
  topologyMode=mode; localStorage.setItem("verathos.topology.mode",mode);
  renderTopology(last.workers||{},last.meshes||{});
}
function activateTopology(el){
  if(el.dataset.nodeKind==="machine"){ goPage("machines"); return; }
  if(el.dataset.testable==="1") openMeshTest(el.dataset.key); else goPage("meshes");
}
function topologyKeydown(event,el){
  if(event.key==="Enter"||event.key===" "){ event.preventDefault(); activateTopology(el); }
}
function highlightTopology(el,on){
  const root=document.getElementById("topologyView");
  if(!on){ root.classList.remove("has-focus"); root.querySelectorAll(".focus").forEach(n=>n.classList.remove("focus")); return; }
  const kind=el.dataset.nodeKind, key=el.dataset.key, machines=new Set(), meshes=new Set();
  if(kind==="machine") machines.add(key); else meshes.add(key);
  root.querySelectorAll(".topology-edge").forEach(edge=>{
    const hit=kind==="machine"?edge.dataset.machine===key:edge.dataset.mesh===key;
    edge.classList.toggle("focus",hit); if(hit){ machines.add(edge.dataset.machine); meshes.add(edge.dataset.mesh); }
  });
  root.querySelectorAll(".topology-packet").forEach(packet=>packet.classList.toggle("focus",
    kind==="machine"?packet.dataset.machine===key:packet.dataset.mesh===key));
  root.querySelectorAll(".topology-node").forEach(node=>node.classList.toggle("focus",
    node.dataset.nodeKind==="machine"?machines.has(node.dataset.key):meshes.has(node.dataset.key)));
  root.classList.add("has-focus");
}
function topologyNodeAttrs(kind,key,testable){
  return `data-node-kind="${kind}" data-key="${esc(key)}"${kind==="machine"?` data-machine="${esc(key)}"`:` data-mesh="${esc(key)}" data-testable="${testable?1:0}"`} onclick="activateTopology(this)" onkeydown="topologyKeydown(event,this)"`;
}
function clipLabel(value,n){ const s=String(value||""); return s.length>n?s.slice(0,n-1)+"…":s; }
function renderTopologyFlow(workers,meshes,data){
  const {machineIds,meshKeys,links}=data, width=1000, rowH=58, top=52, nodeH=42;
  const rows=Math.max(machineIds.length,meshKeys.length,1), height=Math.max(230,top+rows*rowH+16);
  const machineY={},meshY={};
  machineIds.forEach((id,i)=>machineY[id]=top+(i+(rows-machineIds.length)/2)*rowH);
  meshKeys.forEach((key,i)=>meshY[key]=top+(i+(rows-meshKeys.length)/2)*rowH);
  const edges=links.map((link,i)=>{
    const y1=machineY[link.machine]+nodeH/2,y2=meshY[link.mesh]+nodeH/2;
    const d=`M286 ${y1} C430 ${y1},570 ${y2},714 ${y2}`;
    const cls=`${link.phase.cls}${link.driver?" driver":""}`, attrs=`data-machine="${esc(link.machine)}" data-mesh="${esc(link.mesh)}"`;
    const packet=link.phase.cls==="serving"?`<circle r="2.8" class="topology-packet${link.driver?" driver":""}" ${attrs}><animateMotion dur="${1.8+(i%4)*.18}s" repeatCount="indefinite" path="${d}" /></circle>`:"";
    return `<path class="topology-edge ${cls}" ${attrs} d="${d}"/>${packet}`;
  }).join("");
  const machineNodes=machineIds.map(id=>{
    const w=workers[id],cap=(w||{}).capability||{}, y=machineY[id], phase=w?workerPhase(id,w,meshes):{st:"stale",label:"missing"};
    const title=w?`${cap.gpu_name||"GPU"} · ${id} · ${phase.label}`:`${id} · referenced by a mesh but not reporting`;
    return `<g class="topology-node" tabindex="0" role="button" aria-label="${esc(title)}. Open machine details" ${topologyNodeAttrs("machine",id,false)} onmouseenter="highlightTopology(this,true)" onmouseleave="highlightTopology(this,false)" onfocus="highlightTopology(this,true)" onblur="highlightTopology(this,false)"><title>${esc(title)}</title><rect x="36" y="${y}" width="250" height="${nodeH}" rx="9"/><circle cx="53" cy="${y+15}" r="4" fill="${statusColor(phase.st)}"/><text class="topology-label" x="64" y="${y+18}">${esc(clipLabel(cap.gpu_name||(w?"GPU":"Missing machine"),28))}</text><text class="topology-sub" x="53" y="${y+33}">${esc(clipLabel(id,34))}${w&&cap.vram_gb?` · ${esc(cap.vram_gb)} GB`:""}</text></g>`;
  }).join("");
  const meshNodes=meshKeys.map(key=>{
    const m=meshes[key], ph=meshPhase(m,workers), y=meshY[key], testable=ph.cls==="serving"&&m.routing_ready!==false;
    const title=`${m.model_id||"Unknown model"} · ${key} · ${ph.label}`;
    return `<g class="topology-node" tabindex="0" role="button" aria-label="${esc(title)}. ${testable?"Test mesh":"Open mesh details"}" ${topologyNodeAttrs("mesh",key,testable)} onmouseenter="highlightTopology(this,true)" onmouseleave="highlightTopology(this,false)" onfocus="highlightTopology(this,true)" onblur="highlightTopology(this,false)"><title>${esc(title)}</title><rect x="714" y="${y}" width="250" height="${nodeH}" rx="9"/><circle cx="731" cy="${y+15}" r="4" fill="${statusColor(ph.cls)}"/><text class="topology-label" x="742" y="${y+18}">${esc(clipLabel(m.model_id||"Unknown model",29))}</text><text class="topology-sub" x="731" y="${y+33}">${esc(clipLabel(key,24))} · ${esc(ph.label)}</text></g>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Machine to mesh allocation graph"><text class="topology-column" x="36" y="29">MACHINES</text><text class="topology-column" x="714" y="29">ACTIVE MESHES</text>${edges}${machineNodes}${meshNodes}</svg>`;
}
function renderTopologyMatrix(workers,meshes,data){
  const {machineIds,meshKeys,links}=data, byPair=new Map(links.map(link=>[link.machine+"\u0000"+link.mesh,link]));
  if(!meshKeys.length) return '<div class="topology-empty">No active meshes yet. Available machines will connect here after a model is launched.</div>';
  const head=meshKeys.map(key=>{ const m=meshes[key],ph=meshPhase(m,workers),testable=ph.cls==="serving"&&m.routing_ready!==false;
    return `<th><button class="matrix-node" title="${esc(m.model_id||key)}" ${topologyNodeAttrs("mesh",key,testable)}><span class="name">${esc(clipLabel(m.model_id||"Unknown model",18))}</span><span class="sub">${esc(clipLabel(key,16))} · ${esc(ph.label)}</span></button></th>`; }).join("");
  const rows=machineIds.map(id=>{ const w=workers[id],cap=(w||{}).capability||{},phase=w?workerPhase(id,w,meshes):{st:"stale",label:"missing"};
    const machine=`<button class="matrix-node" ${topologyNodeAttrs("machine",id,false)}><span class="name">${esc(cap.gpu_name||(w?"GPU":"Missing machine"))}</span><span class="sub">${esc(id)} · ${esc(phase.label)}</span></button>`;
    const cells=meshKeys.map(key=>{ const link=byPair.get(id+"\u0000"+key); if(!link) return '<td class="matrix-none">·</td>';
      const m=meshes[key],testable=link.phase.cls==="serving"&&m.routing_ready!==false,role=link.driver?"driver":"member";
      return `<td><button class="matrix-cell ${link.phase.cls}${link.driver?" driver":""}" title="${esc(id)} is ${role} of ${esc(key)}" aria-label="${esc(id)} is ${role} of ${esc(key)}" ${topologyNodeAttrs("mesh",key,testable)}>${link.driver?"◆":"●"}</button></td>`; }).join("");
    return `<tr><th>${machine}</th>${cells}</tr>`; }).join("");
  return `<div class="topology-matrix-wrap"><table class="topology-matrix"><thead><tr><th>Machine / mesh</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
}
function renderTopology(workers,meshes){
  const data=topologyData(workers,meshes), dense=data.machineIds.length>8||data.meshKeys.length>6||data.links.length>24;
  let mode=topologyMode||((matchMedia("(max-width: 620px)").matches||dense)?"matrix":"flow");
  if(!data.meshKeys.length) mode="flow";
  document.getElementById("topologyFlowButton").classList.toggle("active",mode==="flow");
  document.getElementById("topologyMatrixButton").classList.toggle("active",mode==="matrix");
  document.getElementById("topologyFlowButton").setAttribute("aria-pressed",mode==="flow");
  document.getElementById("topologyMatrixButton").setAttribute("aria-pressed",mode==="matrix");
  document.getElementById("topologySummary").textContent=data.machineIds.length+" machine"+(data.machineIds.length===1?"":"s")+" · "+data.meshKeys.length+" active mesh"+(data.meshKeys.length===1?"":"es");
  document.getElementById("topologyHint").textContent=mode==="flow"?"Each machine runs one active mesh in this version. Hover a node to isolate its allocation.":"Lit cells are allocations; diamonds identify mesh drivers.";
  const view=document.getElementById("topologyView");
  const signature=JSON.stringify({mode,machines:data.machineIds.map(id=>{ const w=workers[id]||{}; return [id,w.status,!!w.stale,(w.capability||{}).gpu_name,(w.capability||{}).vram_gb]; }),meshes:data.meshKeys.map(key=>{ const m=meshes[key]; return [key,m.status,m.routing_ready!==false,[...(m.stale_members||[])].sort(),m.model_id,m.driver,[...(m.members||[])].sort()]; })});
  // The overview refreshes every two seconds. Keep an unchanged SVG alive so
  // packet motion, keyboard focus, hover isolation, and matrix scroll do not
  // jump back to their starting positions on every poll.
  if(view.dataset.signature===signature) return;
  const oldMatrix=view.querySelector(".topology-matrix-wrap"), scrollLeft=oldMatrix?oldMatrix.scrollLeft:0, scrollTop=oldMatrix?oldMatrix.scrollTop:0;
  view.dataset.signature=signature;
  if(!data.machineIds.length&&!data.meshKeys.length){ view.innerHTML='<div class="topology-empty">Add a machine to start building the pool.</div>'; return; }
  view.innerHTML=mode==="matrix"?renderTopologyMatrix(workers,meshes,data):renderTopologyFlow(workers,meshes,data);
  const newMatrix=view.querySelector(".topology-matrix-wrap"); if(newMatrix){ newMatrix.scrollLeft=scrollLeft; newMatrix.scrollTop=scrollTop; }
}
function renderOverview(workers, meshes){
  renderTopology(workers,meshes);
  const machineIds=Object.keys(workers).sort((a,b)=>Number(!!workers[a].stale)-Number(!!workers[b].stale)||a.localeCompare(b));
  const online=machineIds.filter(id=>!workers[id].stale), drivers=driverSet(meshes);
  document.getElementById("overviewMachineCount").textContent=online.length+" online · "+machineIds.length+" total";
  document.getElementById("overviewMachines").innerHTML=machineIds.length?machineIds.map(id=>{
    const w=workers[id],cap=w.capability||{},phase=workerPhase(id,w,meshes),rtt=fmtRtt((w.rtt_ms||{}).manager);
    const sub=id+(drivers.has(id)?" · driver":"");
    return `<div class="resource-row"><div class="resource-icon">${gpuIcon()}</div><div class="resource-main"><div class="resource-name">${esc(cap.gpu_name||"GPU")}</div><div class="resource-sub">${esc(sub)}</div></div><div class="resource-metrics"><span><b>${cap.vram_gb?esc(cap.vram_gb)+" GB":"—"}</b></span><span class="${rtt.c}">${rtt.t}</span></div><span class="pill ${statusClass(phase.st)}"><span class="bd"></span>${esc(phase.label)}</span></div>`;
  }).join(""):'<div class="resource-empty">No machines yet. Add one to begin.</div>';

  const meshKeys=Object.keys(meshes).sort((a,b)=>{ const rank={serving:0,driving:1,joining:1,fetching:1,error:2,stopped:3}; return (rank[meshes[a].status]??2)-(rank[meshes[b].status]??2)||a.localeCompare(b); });
  const serving=meshKeys.filter(k=>meshes[k].status==="serving"&&meshes[k].routing_ready!==false).length;
  document.getElementById("overviewMeshCount").textContent=serving+" serving · "+meshKeys.length+" total";
  document.getElementById("overviewMeshes").innerHTML=meshKeys.length?meshKeys.map(k=>{
    const m=meshes[k],ph=meshPhase(m,workers),members=(m.members||[]).length;
    const icon='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="5" cy="12" r="2"/><circle cx="19" cy="12" r="2"/><path d="M7 12h10"/></svg>';
    const test=ph.cls==="serving"?`<button class="sm" data-mesh-key="${esc(k)}" onclick="openMeshTest(this.dataset.meshKey)">Test</button>`:"";
    return `<div class="resource-row"><div class="resource-icon">${icon}</div><div class="resource-main"><div class="resource-name">${esc(m.model_id||"Unknown model")}</div><div class="resource-sub">${esc(k)} · ${members} machine${members===1?"":"s"} · driver ${esc(m.driver||"—")}</div></div><div class="resource-actions">${chainPill(k,m)}${validatorScoreBadge(m)}<span class="pill ${ph.cls}"><span class="bd"></span>${esc(ph.label)}</span>${test}</div></div>`;
  }).join(""):'<div class="resource-empty">No meshes yet. Launch a model when capacity is ready.</div>';
}

function leaseText(expiresAt){
  const s=Number(expiresAt)-Date.now()/1000;
  if(!isFinite(s)) return "";
  if(s<=0) return "lease expired";
  if(s<3600) return "lease "+Math.floor(s/60)+"m";
  if(s<172800) return "lease "+Math.round(s/3600)+"h";
  return "lease "+Math.round(s/86400)+"d";
}

function chainPill(k,m){
  // On-chain standing per mesh: registered index + lease countdown, or the
  // measurement lane. Dev pools show nothing (everything is local there).
  if(!last||last.serving_mode!=="subnet") return "";
  if(m.model_index==null) return '<span class="pill assigned" title="serving locally; not registered on the subnet">local only</span>';
  // Per-model registration map; the single-slot field remains for
  // pre-multi-model managers.
  const regs=last.mesh_registrations||{};
  const reg=Object.values(regs).find(r=>String((r||{}).mesh_key||"")===String(k))||(last.mesh_registration)||{};
  const lease=String(reg.mesh_key||"")===String(k)&&reg.expires_at?" · "+leaseText(reg.expires_at):"";
  return `<span class="pill serving" title="registered on the subnet at this model index">on-chain #${esc(m.model_index)}${esc(lease)}</span>`;
}

let poolSig = "";
async function refresh(){
  try { const response=await api("/v1/operator/overview",{});
    // A sign-out can race an already in-flight owner request. Never let that
    // late private response repopulate the public cache or DOM.
    const s=canManage()?response:Object.assign({},response,{meshes:redactValidatorScores(response.meshes||{})});
    setConn(true); const staged=s.validator_binding&&s.validator_binding.epoch;
    const netLbl=({test:"testnet",finney:"mainnet"})[s.subtensor_network]||s.subtensor_network||"";
    const modeLbl=s.serving_mode==="subnet"?("subnet"+(netLbl?" · "+netLbl:"")):(s.serving_mode==="dev"?"local only":"");
    const walletLbl=s.wallet_name?" · "+s.wallet_name+"/"+(s.wallet_hotkey||"default"):"";
    document.getElementById("poolChip").textContent=(s.pool_id||"pool —")+(modeLbl?" · "+modeLbl:"")+walletLbl+(staged!=null?" · next e"+staged:""); last=s; updateAccess();
    renderStats(s.workers||{}, s.meshes||{}); renderOverview(s.workers||{}, s.meshes||{}); renderWorkers(s.workers||{}); renderModels(s.workers||{}, s.models||{}, s.serving_mode||"");
    // Placement suggestions depend on worker availability, not just the
    // selected model: when a mesh stops (or a worker joins/leaves/frees),
    // the freed machines must show up without a page reload.
    const sig = Object.entries(s.workers||{}).map(([id,w])=>id+":"+(w.stale?"x":w.status)).sort().join("|")
      + "#" + Object.keys(s.meshes||{}).sort().join(",");
    if(sig !== poolSig){ poolSig = sig; loadRecs(); }
    // Rebuild the live health/score rows while retaining the exact chat DOM;
    // streamed output, a draft prompt, focus, and AbortController references survive.
    if(openChat) renderMeshesPreservingChat(s.meshes||{}); else renderMeshes(s.meshes||{});
    if(!document.getElementById("addModal").classList.contains("hidden")) updateEnrollmentState();
  } catch(e){ setConn(false); }
}

document.getElementById("walletModal").addEventListener("click",e=>{ if(e.target.id==="walletModal") closeWallet(); });
document.getElementById("addModal").addEventListener("click",e=>{ if(e.target.id==="addModal") closeAdd(); });
document.addEventListener("keydown",e=>{ if(e.key==="Escape"){ closeWallet(); closeAdd(); } });

(async function init(){
  const savedSession=localStorage.getItem(LS_SESSION);
  if(savedSession){ try{ const parsed=JSON.parse(savedSession); SESSION={token:parsed.token,account:parsed.account,is_owner:false}; }catch(e){ localStorage.removeItem(LS_SESSION); } }
  goPage(location.hash.slice(1)||"overview",false); sizeMap(); requestAnimationFrame(drawMap);
  if(SESSION){
    try{ const s=await api("/v1/auth/session",{session:SESSION.token},{auth:false});
      if(!s.signed_in) throw new Error("expired"); SESSION.account=s.account; SESSION.is_owner=!!s.is_owner; localStorage.setItem(LS_SESSION,JSON.stringify(SESSION));
    }catch(e){ invalidateSession(); }
  }
  updateAccess(); await refresh(); setInterval(refresh, 2000);
})();
</script>
</body>
</html>
"""
