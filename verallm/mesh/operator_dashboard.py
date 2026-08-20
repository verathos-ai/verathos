"""Alternate Verathos operator dashboard prototype.

A miner-facing control surface for a whole fleet: log in with the Bittensor
wallet that owns the pool (single-use challenge -> Talisman ``signRaw`` ->
SR25519 verify on the manager), then view fleet health and manage meshes.
Viewing a pool is public (read-only); every management action needs the
verified owner session (or the raw pool token, which the CLI/tools use).

The pool manager currently serves ``pool_dashboard.py``. Keep this alternate
page aligned with the same API contract so it cannot reintroduce unsafe model
launch choices if it is wired back in later.
"""

from __future__ import annotations

OPERATOR_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Verathos Operator</title>
<style>
:root{
  --bg:#ededee; --fg:#26282e; --card:#fdfdfd; --card2:#f4f4f5; --border:#cfd0d4;
  --muted:#5b5f6b; --primary:#3466d6; --primary-soft:rgba(52,102,214,.12);
  --ok:#16a34a; --warn:#b45309; --err:#dc2626; --chip:#e7e8ea; --sidebar:#e6e6e8;
  --glow:rgba(52,102,214,.10);
}
@media (prefers-color-scheme: dark){:root{
  --bg:#1e1e1e; --fg:#e8e8e8; --card:#252526; --card2:#2d2d2d; --border:#474747;
  --muted:#a0a0a0; --primary:#4a7fe0; --primary-soft:rgba(74,127,224,.14);
  --ok:#3fb950; --warn:#f2b03c; --err:#f85149; --chip:#333333; --sidebar:#202020;
  --glow:rgba(74,127,224,.09);
}}
:root[data-theme="dark"]{
  --bg:#1e1e1e; --fg:#e8e8e8; --card:#252526; --card2:#2d2d2d; --border:#474747;
  --muted:#a0a0a0; --primary:#4a7fe0; --primary-soft:rgba(74,127,224,.14);
  --ok:#3fb950; --warn:#f2b03c; --err:#f85149; --chip:#333333; --sidebar:#202020;
  --glow:rgba(74,127,224,.09);
}
:root[data-theme="light"]{
  --bg:#ededee; --fg:#26282e; --card:#fdfdfd; --card2:#f4f4f5; --border:#cfd0d4;
  --muted:#5b5f6b; --primary:#3466d6; --primary-soft:rgba(52,102,214,.12);
  --ok:#16a34a; --warn:#b45309; --err:#dc2626; --chip:#e7e8ea; --sidebar:#e6e6e8;
  --glow:rgba(52,102,214,.10);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;font-size:14px;line-height:1.5}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
a{color:var(--primary);text-decoration:none;cursor:pointer}
button{font:inherit;cursor:pointer}
.hidden{display:none!important}

.top{position:sticky;top:0;z-index:50;height:56px;display:flex;align-items:center;gap:12px;
  padding:0 20px;border-bottom:1px solid var(--border);
  background:color-mix(in srgb, var(--bg) 82%, transparent);backdrop-filter:blur(12px)}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:-.01em}
.brand small{color:var(--muted);font-weight:500;margin-left:2px}
.netpill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--border);
  background:var(--card);border-radius:999px;padding:3px 10px;font-size:10px;color:var(--muted)}
.netpill .dot{width:6px;height:6px;border-radius:50%;background:var(--ok);animation:pulse 2s infinite}
@keyframes pulse{50%{opacity:.4}}
@media (prefers-reduced-motion: reduce){.netpill .dot,.fv-links line{animation:none!important}}
.top .grow{flex:1}
.search{display:flex;align-items:center;gap:8px;border:1px solid var(--border);background:var(--card);
  border-radius:8px;padding:6px 10px;min-width:280px;color:var(--muted)}
.search input{border:0;outline:0;background:transparent;color:var(--fg);flex:1;font-size:12.5px}
.btn{display:inline-flex;align-items:center;gap:8px;border-radius:8px;border:1px solid transparent;
  height:34px;padding:0 14px;font-weight:500;font-size:13px;transition:opacity .15s,background .15s}
.btn.primary{background:var(--primary);color:#fff}
.btn.primary:hover{opacity:.9}
.btn.outline{border-color:var(--border);background:transparent;color:var(--fg)}
.btn.outline:hover{background:var(--card2)}
.btn.sm{height:28px;padding:0 10px;font-size:12px;border-radius:7px}
.btn.danger{background:transparent;border-color:var(--border);color:var(--err)}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn:focus-visible{outline:2px solid var(--primary);outline-offset:2px}

.shell{display:flex;min-height:calc(100vh - 56px)}
.side{width:232px;flex:0 0 232px;background:var(--sidebar);border-right:1px solid var(--border);
  padding:14px 10px;display:flex;flex-direction:column;gap:2px}
.side .lbl{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);padding:10px 10px 4px}
.nav{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;color:var(--muted);
  border:0;background:transparent;width:100%;text-align:left;font-size:13px}
.nav:hover{background:var(--card2);color:var(--fg)}
.nav.on{background:var(--primary-soft);color:var(--primary);font-weight:600}
.nav svg{flex:0 0 16px}
.side .foot{margin-top:auto;padding:10px;font-size:11px;color:var(--muted);word-break:break-all}
main{flex:1;padding:22px 26px;max-width:1200px}

.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;
  transition:border-color .3s, box-shadow .3s}
.card:hover{border-color:color-mix(in srgb, var(--primary) 40%, transparent);box-shadow:0 0 30px var(--glow)}
.grid{display:grid;gap:14px}
.grid.c4{grid-template-columns:repeat(4,1fr)}
.grid.c3{grid-template-columns:repeat(3,1fr)}
.grid.c2{grid-template-columns:repeat(2,1fr)}
@media (max-width:1100px){.grid.c4{grid-template-columns:repeat(2,1fr)}.grid.c3{grid-template-columns:1fr 1fr}}
h2{font-size:15px;font-weight:600;margin:0 0 4px}
h3{font-size:13.5px;font-weight:600;margin:0}
.sub{color:var(--muted);font-size:12px}
.kpi .v{font-size:22px;font-weight:700;letter-spacing:-.01em}
.kpi .lbl{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.sec{margin-bottom:22px}
.sec-head{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:10px}
.pgtitle{margin:0 0 2px;font-size:19px;font-weight:700;letter-spacing:-.01em}
.pgsub{color:var(--muted);font-size:12.5px;margin-bottom:18px}

.chip{display:inline-flex;align-items:center;gap:5px;border-radius:999px;padding:2px 9px;
  font-size:11px;font-weight:500;background:var(--chip);color:var(--muted)}
.chip .d{width:6px;height:6px;border-radius:50%}
.chip.ok{color:var(--ok)} .chip.ok .d{background:var(--ok)}
.chip.warn{color:var(--warn)} .chip.warn .d{background:var(--warn)}
.chip.err{color:var(--err)} .chip.err .d{background:var(--err)}
.chip.info{color:var(--primary)} .chip.info .d{background:var(--primary)}
.tag{font-size:10.5px;border:1px solid var(--border);border-radius:6px;padding:1px 7px;color:var(--muted)}

.mesh-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
.score{display:flex;flex-direction:column;align-items:flex-end}
.score .n{font-size:19px;font-weight:700;color:var(--primary)}
.score .lbl{font-size:9.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.meter{height:5px;border-radius:3px;background:var(--chip);overflow:hidden;margin-top:9px}
.meter i{display:block;height:100%;border-radius:3px;background:var(--primary)}
.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.kv{display:flex;justify-content:space-between;font-size:12px;padding:3px 0}
.kv b{font-weight:600}.kv .k{color:var(--muted)}
.actions{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
.spark{display:flex;align-items:flex-end;gap:2px;height:56px}
.spark i{width:100%;border-radius:2px 2px 0 0;background:color-mix(in srgb, var(--primary) 45%, transparent)}
.spark i.hot{background:var(--primary)}

.fleetviz{position:relative;height:210px;margin:8px 8px 0;border-radius:10px;
  background:radial-gradient(ellipse at 50% 130%, var(--primary-soft), transparent 65%),
             repeating-radial-gradient(circle at 50% 50%, transparent 0 39px, color-mix(in srgb, var(--border) 40%, transparent) 39px 40px)}
.fv-links{position:absolute;inset:0}
.fv-links line{stroke:color-mix(in srgb, var(--primary) 55%, transparent);stroke-width:1.5;stroke-dasharray:5 5;animation:dash 1.4s linear infinite}
@keyframes dash{to{stroke-dashoffset:-10}}
.fnode{position:absolute;border-radius:50%;transform:translate(-50%,-50%);display:flex;flex-direction:column;
  align-items:center;justify-content:center;font-size:11px;line-height:1.15;text-align:center;cursor:default;transition:transform .2s}
.fnode:hover{transform:translate(-50%,-50%) scale(1.08)}
.fnode b{font-weight:600}.fnode span{font-size:9px;opacity:.75}
.fnode.serving,.fnode.driving{background:color-mix(in srgb, var(--primary) 22%, var(--card));border:1.5px solid var(--primary);
  box-shadow:0 0 22px color-mix(in srgb, var(--primary) 35%, transparent);color:var(--fg)}
.fnode.idle{background:var(--card);border:1.5px dashed var(--muted);color:var(--muted)}
.fnode.offline{background:color-mix(in srgb, var(--err) 10%, var(--card));border:1.5px solid var(--err);color:var(--err);opacity:.85}
.fleetstrip{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;padding:12px 16px}
@media (max-width:800px){.fleetstrip{grid-template-columns:repeat(3,1fr)}}
.fleetstrip div{display:flex;flex-direction:column;align-items:center;border:1px solid var(--border);
  border-radius:9px;padding:7px 4px;background:var(--card2)}
.fleetstrip b{font-size:15px;font-weight:700}
.fleetstrip span{font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}

table{width:100%;border-collapse:collapse;font-size:12.5px}
th{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);text-align:left;padding:6px 10px;border-bottom:1px solid var(--border)}
td{padding:8px 10px;border-bottom:1px solid color-mix(in srgb, var(--border) 55%, transparent)}
.linkchip{font-size:11px;font-weight:600}
.l-good{color:var(--ok)}.l-mid{color:var(--warn)}.l-bad{color:var(--err)}
.code{background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:10px 12px;font-size:11.5px;overflow-x:auto;white-space:pre-wrap;word-break:break-all}

.banner{border:1px solid var(--border);border-left:3px solid var(--warn);background:var(--card);
  border-radius:10px;padding:10px 14px;font-size:12.5px;margin-bottom:16px;display:flex;gap:10px;align-items:center}
.banner.blue{border-left-color:var(--primary)}
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:center;justify-content:center;z-index:90}
.overlay.on{display:flex}
.modal{width:400px;max-width:92vw;background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px}
.acct{display:flex;align-items:center;gap:10px;border:1px solid var(--border);border-radius:10px;padding:10px;margin-top:8px;background:var(--card2);cursor:pointer}
.acct:hover{border-color:var(--primary)}
.avatar{width:30px;height:30px;border-radius:50%;flex:0 0 30px;background:conic-gradient(from 40deg,#4a7fe0,#7dd3fc,#3fb950,#4a7fe0)}
.toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:10px 16px;font-size:13px;box-shadow:0 8px 30px rgba(0,0,0,.3);z-index:99;display:none}
.toast.on{display:block}
.mgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.mcard{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:13px 14px;transition:border-color .2s,box-shadow .2s}
.mcard:hover{border-color:color-mix(in srgb,var(--primary) 35%,transparent)}
.mcard.m-on{border-left:3px solid var(--primary)}
.mcard.m-idle{border-left:3px solid var(--muted)}
.mcard.m-off{border-left:3px solid var(--err);opacity:.72}
.mrow{display:flex;justify-content:space-between;align-items:center;gap:8px}
.mname{display:flex;align-items:center;gap:6px;font-size:13.5px}
.mgpu{font-size:11.5px;color:var(--muted);margin-top:6px}
.mserve{font-size:12.5px;margin-top:8px}
.mbar{height:5px;border-radius:3px;background:var(--chip);overflow:hidden;margin-top:10px}
.mbar i{display:block;height:100%;background:var(--primary);border-radius:3px}
.mcard.m-off .mbar i{background:var(--err)}
.chatlog{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:12px;min-height:120px;max-height:340px;overflow-y:auto;font-size:13px;white-space:pre-wrap;line-height:1.55}
.chatlog .you{color:var(--muted)}
.vbadge{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:600;border-radius:999px;padding:3px 10px;margin-top:10px}
.vbadge.ok{color:var(--ok);background:color-mix(in srgb,var(--ok) 12%,transparent)}
.vbadge.no{color:var(--err);background:color-mix(in srgb,var(--err) 12%,transparent)}
.chatin{display:flex;gap:8px;margin-top:10px}
.chatin input{flex:1;background:var(--card2);color:var(--fg);border:1px solid var(--border);border-radius:8px;padding:9px 11px;font-size:13px;outline:none}
.chatin input:focus{border-color:var(--primary)}
.latmap{width:100%;height:240px;display:block;border-radius:10px;background:var(--card2)}
.maplegend{display:flex;gap:14px;font-size:11px;color:var(--muted);padding:6px 2px 0}
.maplegend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
</style>
</head>
<body>

<div class="top">
  <div class="brand">
    <svg width="24" height="24" viewBox="0 0 64 64" fill="none" stroke="var(--primary)" stroke-width="3" stroke-linecap="round">
      <circle cx="32" cy="32" r="16" stroke-dasharray="22.3 11.2"/>
      <circle cx="32" cy="32" r="21" stroke-dasharray="33 11" transform="rotate(20 32 32)"/>
      <circle cx="32" cy="32" r="26" stroke-dasharray="45.79 8.67" transform="rotate(40 32 32)"/>
    </svg>
    Verathos <small>Operator</small>
  </div>
  <div class="netpill"><span class="dot"></span> <span id="netName">mesh pool</span></div>
  <div class="grow"></div>
  <div class="search">
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>
    <input id="hkSearch" placeholder="Search a hotkey to view its pool (read-only)…" />
  </div>
  <button class="btn outline sm" id="themeBtn" title="Toggle theme">&#9680;</button>
  <button class="btn primary" id="walletBtn">Connect wallet</button>
  <div id="walletBadge" class="hidden row" style="gap:8px">
    <span class="chip ok" id="ownerChip"><span class="d"></span> verified</span>
    <span class="mono sub" id="ownerAddr"></span>
    <button class="btn outline sm" id="signoutBtn">Sign out</button>
  </div>
</div>

<div class="shell">
  <div class="side">
    <div class="lbl">Pool</div>
    <button class="nav on" data-page="overview"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="8" height="8" rx="2"/><rect x="13" y="3" width="8" height="8" rx="2"/><rect x="3" y="13" width="8" height="8" rx="2"/><rect x="13" y="13" width="8" height="8" rx="2"/></svg> Overview</button>
    <button class="nav" data-page="meshes"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="5" cy="12" r="2.4"/><circle cx="19" cy="6" r="2.4"/><circle cx="19" cy="18" r="2.4"/><path d="M7.2 11l9.4-4.2M7.2 13l9.4 4.2"/></svg> Meshes</button>
    <button class="nav" data-page="machines"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="6" width="18" height="12" rx="2"/><rect x="7" y="10" width="6" height="4" rx="1"/><path d="M17 10v4"/></svg> Machines</button>
    <button class="nav" data-page="launch"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 19l4-4M12 15l-3-3 8.5-8.5a2.1 2.1 0 013 3L12 15z"/><path d="M15 6l3 3"/></svg> Launch model</button>
    <div class="lbl">Account</div>
    <button class="nav" data-page="settings"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 00-.1-1.2l2-1.6-2-3.4-2.4 1a7 7 0 00-2-1.2L14 3h-4l-.4 2.6a7 7 0 00-2 1.2l-2.4-1-2 3.4 2 1.6A7 7 0 005 12c0 .4 0 .8.1 1.2l-2 1.6 2 3.4 2.4-1a7 7 0 002 1.2L10 21h4l.4-2.6a7 7 0 002-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2z"/></svg> Settings</button>
    <div class="foot">Pool <span class="mono" id="poolIdFoot">…</span></div>
  </div>

  <main>
    <div class="banner blue hidden" id="visitorHint"><span class="chip info"><span class="d"></span> viewer</span> Connect your Bittensor wallet (coldkey) to manage your fleet.</div>
    <div class="banner hidden" id="roBanner"><span class="chip warn"><span class="d"></span> read-only</span> <span id="roText">Signed in, but not the owner of this pool — viewing only.</span></div>

    <section id="pg-overview"></section>
    <section id="pg-meshes" class="hidden"></section>
    <section id="pg-machines" class="hidden"></section>
    <section id="pg-launch" class="hidden"></section>
    <section id="pg-settings" class="hidden"></section>
  </main>
</div>

<div class="overlay" id="walletModal"><div class="modal">
  <div id="wStep1">
    <h2>Connect wallet</h2>
    <div class="sub" id="wHint" style="margin-bottom:8px">Sign with your Bittensor wallet (coldkey) to log in.</div>
    <div id="acctList"></div>
    <div class="sub" style="margin-top:10px;padding-top:8px;border-top:1px solid var(--border)">
      Signing proves you control this account — nothing is sent on-chain and it costs nothing. You stay logged in on this browser until you sign out.
    </div>
  </div>
  <div id="wStep2" class="hidden">
    <h2>Verify wallet</h2>
    <div class="sub" style="margin:8px 0 12px">Sign this one-time challenge to prove you control <span class="mono" id="signHk"></span>, then we check on-chain that it owns a registered UID. Nothing goes on-chain and it costs nothing.</div>
    <div class="code" id="challengeBox" style="margin-bottom:12px"></div>
    <button class="btn primary" id="doSign" style="width:100%">Sign with wallet</button>
    <div class="sub" style="text-align:center;margin-top:8px"><a id="wCancel">Cancel</a></div>
  </div>
</div></div>

<div class="overlay" id="chatModal"><div class="modal" style="width:560px">
  <div class="row" style="justify-content:space-between"><h2 style="margin:0">Test chat</h2><a id="chatClose">close</a></div>
  <div class="sub" id="chatMesh" style="margin:2px 0 10px"></div>
  <div class="chatlog" id="chatLog"><span class="sub">Send a prompt to verify this mesh answers with a real, proof-checked response.</span></div>
  <div id="chatBadge"></div>
  <div class="chatin"><input id="chatInput" placeholder="Ask the mesh something…" value="In one sentence, why is the sky blue?"><button class="btn primary" id="chatSend">Send</button></div>
</div></div>
<div class="toast" id="toast"></div>
<div id="wdbg" style="position:fixed;left:10px;bottom:10px;z-index:100;max-width:60vw;font:11px ui-monospace,monospace;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:6px 9px;color:var(--muted)"></div>

<script>
const API = ""; // same origin as the manager
const LS_SESSION = "verathos_operator_session";   // coldkey/wallet: persists
const SS_SESSION = "verathos_operator_session_hk";  // hotkey: this browser session only
let STATE = {overview:null, session:null, page:"overview", recModel:""};

function esc(s){ return String(s==null?"":s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function shortHk(a){ if(!a) return "—"; return a.length>12 ? a.slice(0,6)+"…"+a.slice(-4) : a; }
function toast(m){ const t=document.getElementById("toast"); t.textContent=m; t.classList.add("on"); clearTimeout(t._t); t._t=setTimeout(()=>t.classList.remove("on"),2600); }
async function api(route, body, opts){
  const r = await fetch(API+route, {method:(opts&&opts.method)||"POST", headers:{"Content-Type":"application/json"},
    body: body===undefined?undefined:JSON.stringify(body)});
  const txt = await r.text(); let j={}; try{ j=JSON.parse(txt); }catch(e){ j={error:txt}; }
  if(!r.ok) throw new Error(j.error || ("HTTP "+r.status));
  return j;
}
function withSession(body){ const b=Object.assign({}, body||{}); if(STATE.session) b.session=STATE.session.token; return b; }
function isOwner(){ return !!(STATE.session && STATE.session.is_owner); }

/* ---------- data ---------- */
async function loadOverview(){
  try{
    const o = await api("/v1/operator/overview", {}, {method:"POST"});
    STATE.overview = o;
    document.getElementById("poolIdFoot").textContent = o.pool_id||"—";
    document.getElementById("netName").textContent = (o.pool_id||"mesh pool");
    render();
  }catch(e){ document.getElementById("pg-overview").innerHTML = `<div class="card">Could not load pool: ${esc(e.message)}</div>`; }
}

/* ---------- derived fleet metrics ---------- */
function fleet(){
  const o=STATE.overview||{workers:{},meshes:{}};
  const workers=Object.entries(o.workers||{}).map(([id,w])=>({id, ...w, cap:w.capability||{}}));
  const meshes=Object.entries(o.meshes||{}).filter(([,m])=>m.status!=="stopped").map(([key,m])=>({key,...m}));
  const now=Date.now()/1000;
  const live = w => !w.stale;  // trust the server-computed stale flag (client clock may be skewed)
  let serving=0, idle=0, offline=0, vram=0, vramUse=0;
  workers.forEach(w=>{ const on=live(w); const st=(w.status||"").split(" ")[0];
    vram += (+w.cap.vram_gb||0);
    if(!on){ offline++; }
    else if(st==="serving"||st==="driving"||st==="joining"){ serving++; vramUse+=(+w.cap.vram_gb||0); }
    else { idle++; } });
  const modelsReady = new Set();
  if(o.serving_mode!=="subnet") workers.forEach(w=>(w.catalog||[]).forEach(c=>c.model_id&&modelsReady.add(c.model_id)));
  Object.entries(o.models||{}).forEach(([m,r])=>{ if(r.launch_ready!==false) modelsReady.add(m); });
  return {workers,meshes,serving,idle,offline,vram,vramUse,modelsReady:modelsReady.size,live};
}

/* ---------- pages ---------- */
function render(){
  const owner=isOwner(), signed=!!STATE.session;
  document.getElementById("visitorHint").classList.toggle("hidden", signed);
  document.getElementById("roBanner").classList.toggle("hidden", !(signed && !owner));
  const p=STATE.page;
  ["overview","meshes","machines","launch","settings"].forEach(x=>document.getElementById("pg-"+x).classList.toggle("hidden", x!==p));
  document.querySelectorAll(".nav").forEach(n=>n.classList.toggle("on", n.dataset.page===p));
  if(p==="overview") renderOverview();
  if(p==="meshes") renderMeshes();
  if(p==="machines") renderMachines();
  if(p==="launch") renderLaunch();
  if(p==="settings") renderSettings();
}

function meterMesh(m){ const st=(m.status||"").split(" ")[0]; return st==="serving"?96:(st==="driving"||st==="joining")?55:st==="error"?20:40; }
function meshChip(m){ const st=(m.status||"").split(" ")[0];
  if(st==="serving") return '<span class="chip ok"><span class="d"></span> serving</span>';
  if(st==="error") return '<span class="chip err"><span class="d"></span> error</span>';
  if(st==="driving"||st==="joining"||st==="fetching") return '<span class="chip info"><span class="d"></span> '+esc(m.status)+'</span>';
  return '<span class="chip"><span class="d"></span> '+esc(st||"—")+'</span>'; }

function workerMesh(wid){
  const f=fleet();
  return f.meshes.find(m=> m.driver===wid || (m.members||[]).includes(wid)) || null;
}
function machineCard(w){
  const f=fleet(); const on=f.live(w); const st=on?(w.status||"idle").split(" ")[0]:"offline";
  const mesh=on?workerMesh(w.id):null;
  const cls = st==="offline"?"m-off":(st==="serving"||st==="driving"||st==="joining")?"m-on":"m-idle";
  const chip = st==="offline"?'<span class="chip err"><span class="d"></span> offline</span>'
             : (st==="serving")?'<span class="chip ok"><span class="d"></span> serving</span>'
             : (st==="idle")?'<span class="chip"><span class="d"></span> idle</span>'
             : '<span class="chip info"><span class="d"></span> '+esc(w.status)+'</span>';
  const vram=+w.cap.vram_gb||0;
  return `<div class="mcard ${cls}">
    <div class="mrow"><div class="mname">${gpuIcon()} <b>${esc(w.id)}</b></div>${chip}</div>
    <div class="mgpu">${esc(w.cap.gpu_name||"GPU")} · ${vram} GB</div>
    <div class="mserve">${ mesh ? ('serving <b>'+esc(mesh.model_id||mesh.key)+'</b>'+((mesh.model_id||'').match(/moe|a3b/i)?' <span class="tag">MoE</span>':'')) : (st==="offline"?'—':'idle, ready') }</div>
    <div class="mbar"><i style="width:${on?(mesh?96:40):8}%"></i></div>
  </div>`;
}
function gpuIcon(){ return '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" style="vertical-align:-2px"><rect x="3" y="6" width="18" height="12" rx="2"/><rect x="7" y="10" width="6" height="4" rx="1"/><path d="M17 10v4"/></svg>'; }

function renderOverview(){
  const f=fleet(), o=STATE.overview;
  const util = f.vram ? Math.round(100*f.vramUse/f.vram) : 0;
  const sc=STATE.score; const nUids = sc && sc.uids ? sc.uids.length : 0;
  const scoreKpi = sc ? `${sc.score} <span style="font-size:13px;color:var(--muted);font-weight:500">· inc ${sc.incentive}</span>` : "—";
  const uidLine = sc ? (`UID ${sc.uid}` + (nUids>1?` <span class="tag">+${nUids-1} more</span>`:'')) : 'UID resolves once chain lookup is wired';
  document.getElementById("pg-overview").innerHTML = `
    <div class="pgtitle">Overview</div>
    <div class="pgsub">Pool <span class="mono">${esc(o.pool_id)}</span> · owner ${o.owner_account?('<span class="mono">'+esc(shortHk(o.owner_account))+'</span>'):'<span class="sub">unclaimed</span>'}</div>
    <div class="grid c4 sec">
      <div class="card kpi"><div class="lbl">Score${sc?' · Incentive':''}</div><div class="v">${scoreKpi}</div><div class="sub">${uidLine}</div></div>
      <div class="card kpi"><div class="lbl">Serving</div><div class="v">${f.serving} <span style="font-size:12px;font-weight:500;color:var(--muted)">/ ${f.workers.length}</span></div><div class="sub">${f.idle} idle · ${f.offline} offline</div></div>
      <div class="card kpi"><div class="lbl">VRAM in use</div><div class="v">${f.vramUse} <span style="font-size:12px;font-weight:500;color:var(--muted)">/ ${f.vram} GB</span></div><div class="meter" style="margin-top:8px"><i style="width:${util}%"></i></div></div>
      <div class="card kpi"><div class="lbl">Models ready</div><div class="v">${f.modelsReady}</div><div class="sub">across the fleet</div></div>
    </div>
    <div class="sec">
      <div class="sec-head"><h2>Fleet</h2><span class="sub">${f.workers.length} machine${f.workers.length===1?'':'s'} in this pool</span></div>
      <div class="mgrid">${ f.workers.length? f.workers.map(machineCard).join("") : '<div class="card sub">No machines joined yet — add one from <a data-nav="machines">Machines</a>.</div>' }</div>
    </div>
    <div class="sec"><div class="sec-head"><h2>Meshes</h2><a data-nav="meshes">manage →</a></div>
      <div class="grid c3">${ f.meshes.length? f.meshes.map(m=>`
        <div class="card"><div class="mesh-top"><div><h3>${esc(m.model_id||m.key)}</h3><div class="sub mono">${esc(m.driver||"")}</div></div>
          ${meshChip(m)}</div>
          <div class="actions"><button class="btn outline sm act-test" data-mesh="${esc(m.key)}">Test chat</button></div></div>`).join("")
        : '<div class="sub">No meshes running. Go to <a data-nav="launch">Launch model</a>.</div>' }</div>
    </div>`;
  document.querySelectorAll(".act-test").forEach(b=>b.onclick=()=>openTestChat(b.dataset.mesh));
}

function renderMeshes(){
  const f=fleet();
  document.getElementById("pg-meshes").innerHTML = `
    <div class="pgtitle">Meshes</div>
    <div class="pgsub">Every mesh in your pool — health and controls.</div>
    <div class="grid c2">${ f.meshes.length? f.meshes.map(m=>`
      <div class="card"><div class="mesh-top"><div><h3>${esc(m.model_id||m.key)}</h3><div class="sub mono">${esc(m.key)} · driver ${esc(m.driver||"")}</div></div>
        <div class="score"><span class="n">${meterMesh(m)}</span><span class="lbl">health</span></div></div>
        <div style="margin-top:10px">
          <div class="kv"><span class="k">Status</span>${meshChip(m)}</div>
          <div class="kv"><span class="k">Workers</span><b>${esc((m.members||[m.driver]).filter(Boolean).join(", "))}</b></div>
          ${m.error?`<div class="kv"><span class="k">Error</span><span style="color:var(--err)">${esc(m.error)}</span></div>`:''}
        </div>
        <div class="actions">
          <button class="btn outline sm act-test" data-mesh="${esc(m.key)}" ${isOwner()?'':'disabled title="verify owner wallet to manage"'}>Test chat</button>
          <button class="btn danger sm act-stop" data-mesh="${esc(m.key)}" ${isOwner()?'':'disabled title="verify owner wallet to manage"'}>Stop</button>
        </div></div>`).join("")
      : '<div class="card sub">No meshes running yet.</div>' }</div>`;
  document.querySelectorAll(".act-stop").forEach(b=>b.onclick=()=>doStop(b.dataset.mesh));
  document.querySelectorAll(".act-test").forEach(b=>b.onclick=()=>openTestChat(b.dataset.mesh));
}

function renderMachines(){
  const f=fleet();
  const rows=f.workers.map(w=>{ const on=f.live(w); const link=w.rtt_ms&&w.rtt_ms.manager!=null?linkClass(w.rtt_ms.manager):{t:"—",c:""};
    return `<tr><td class="mono">${esc(w.id)}</td><td>${esc(w.cap.gpu_name||"?")}</td>
      <td>${on?meshChip({status:w.status}):'<span class="chip err"><span class="d"></span> offline</span>'}</td>
      <td>${esc(+w.cap.vram_gb||0)} GB</td><td>${w.cap.free_disk_gb!=null?Math.round(w.cap.free_disk_gb)+" GB":"—"}</td>
      <td><span class="linkchip ${link.c}">${link.t}</span></td></tr>`; }).join("");
  document.getElementById("pg-machines").innerHTML = `
    <div class="sec-head"><div><div class="pgtitle">Machines</div><div class="pgsub">Workers joined to your pool.</div></div>
      <button class="btn primary" id="addBtn" ${isOwner()?'':'disabled title="verify owner wallet to manage"'}>+ Add machine</button></div>
    <div class="card hidden" id="addBox" style="margin-bottom:14px">
      <h3>Add a machine</h3>
      <div class="sub" style="margin:4px 0 8px">Run this on any GPU box — it detects the GPU, builds the matching backend automatically, and joins your pool:</div>
      <div class="code" id="joinCmd">verathos mesh pool worker --pool-token-file ~/.verathos/pool-token.txt --worker-id my-gpu</div>
      <div class="sub" style="margin-top:6px">Securely copy the owner-only token file to the machine and set mode <span class="mono">0600</span>.</div>
    </div>
    <div class="card" style="padding:0"><table><tr><th>Machine</th><th>GPU</th><th>Status</th><th>VRAM</th><th>Disk free</th><th>Link</th></tr>${rows||'<tr><td colspan="6" class="sub" style="padding:14px">No machines joined.</td></tr>'}</table></div>`;
  const ab=document.getElementById("addBtn"); if(ab) ab.onclick=()=>document.getElementById("addBox").classList.toggle("hidden");
}

function linkClass(ms){ if(ms==null) return {t:"—",c:""}; const v=Math.round(ms);
  if(v<2) return {t:"local",c:"l-good"}; if(v<10) return {t:v+" ms · lan",c:"l-good"};
  if(v<40) return {t:v+" ms · regional",c:"l-mid"}; return {t:v+" ms · far",c:"l-bad"}; }

function renderLaunch(){
  const o=STATE.overview||{};
  const models = Object.entries(o.models||{}).filter(([,r])=>r.launch_ready!==false).map(([m])=>m);
  if(o.serving_mode!=="subnet"){
    const local = new Set(); fleet().workers.forEach(w=>(w.catalog||[]).forEach(c=>c.model_id&&local.add(c.model_id)));
    local.forEach(m=>{ if(!models.includes(m)) models.push(m); });
  }
  const opts = models.map(m=>`<option value="${esc(m)}">${esc(m)}</option>`).join("");
  const empty = o.serving_mode==="subnet" ? "no chain-bound models registered" : "no models registered";
  document.getElementById("pg-launch").innerHTML = `
    <div class="pgtitle">Launch a model</div>
    <div class="pgsub">Placements ranked by expected speed — singles first, then link quality.</div>
    ${isOwner()?'':'<div class="banner"><span class="chip warn"><span class="d"></span> read-only</span> Verify the owner wallet to launch.</div>'}
    <div class="card" style="max-width:760px">
      <div class="row"><b>Model</b>
        <select id="launchModel" style="flex:1;background:var(--card2);color:var(--fg);border:1px solid var(--border);border-radius:8px;padding:7px 10px">${opts||('<option>'+empty+'</option>')}</select>
        <button class="btn outline sm" id="recBtn">Rank placements</button></div>
      <canvas class="latmap" id="latmapLaunch" style="margin-top:12px;height:200px"></canvas>
      <div class="sub" style="margin-top:4px">Placement quality follows these links: every generated token crosses the driver↔member link.</div>
      <div id="recBox" style="margin-top:12px"><div class="sub">Pick a model and rank placements.</div></div>
    </div>`;
  const rb=document.getElementById("recBtn"); if(rb) rb.onclick=loadRecs;
}
async function loadRecs(){
  const model=document.getElementById("launchModel").value; STATE.recModel=model;
  const box=document.getElementById("recBox"); box.innerHTML='<div class="sub">Ranking…</div>';
  try{
    const {suggestions, reasons}=await api("/v1/pool/recommend", withSession({model_id:model}));
    const chip=s=>{ if(s.workers.length<2) return '<span class="l-good linkchip">local</span>';
      const c=(s.link_class==="local"||s.link_class==="lan")?"l-good":(s.link_class==="regional"?"l-mid":"l-bad");
      return `<span class="linkchip ${c}">${Math.round(s.max_rtt_ms)} ms · ${esc(s.link_class||"")}</span>`; };
    const rows=(suggestions||[]).slice(0,6).map((s,i)=>{ const kind=s.workers.length>1?"pipeline":"single";
      const dl=s.fetch?` · ↓ downloads ${s.download_gb||"?"} GB`:"";
      const warn=s.warn?`<div class="sub" style="color:var(--warn);margin-top:3px">⚠ ${esc(s.warn)}</div>`:"";
      return `<div class="card" style="padding:12px;${i===0?'border-color:color-mix(in srgb,var(--primary) 45%,transparent)':''}">
        <div class="row" style="justify-content:space-between"><div>${i===0?'<span class="chip info"><span class="d"></span> best</span> ':''}<b>${s.workers.map(esc).join('  +  ')}</b>
        <div class="sub" style="margin-top:3px">${kind} · driver <b>${esc(s.driver)}</b> · ${chip(s)}${dl}</div>${warn}</div>
        <button class="btn ${i===0?'primary':'outline'} sm do-launch" data-w='${esc(JSON.stringify(s.workers))}' data-d="${esc(s.driver)}" ${isOwner()?'':'disabled'}>${s.fetch?'Download &amp; launch':'Launch'}</button></div></div>`;
    }).join("");
    const placed=new Set((suggestions||[]).flatMap(s=>s.workers));
    const left=Object.entries(reasons||{}).filter(([id])=>!placed.has(id));
    const why=left.length?`<div style="margin-top:10px;padding-top:8px;border-top:1px solid var(--border)"><div class="sub" style="text-transform:uppercase;font-size:10px;letter-spacing:.06em;margin-bottom:3px">Not placeable right now</div>${left.map(([id,r])=>`<div class="kv"><span class="mono">${esc(id)}</span><span class="sub">${esc(r)}</span></div>`).join("")}</div>`:"";
    const topPair = (suggestions||[]).find(s=>s.workers.length>1);
    MAP_HIGHLIGHT = topPair ? topPair.workers.slice(0,2) : null;
    box.innerHTML = (rows||'<div class="sub">No feasible placement for this model right now.</div>') + why;
    box.querySelectorAll(".do-launch").forEach(b=>b.onclick=()=>doLaunch(JSON.parse(b.dataset.w), b.dataset.d, model));
  }catch(e){ box.innerHTML=`<div class="sub">${esc(e.message)}</div>`; }
}

function renderSettings(){
  const o=STATE.overview;
  document.getElementById("pg-settings").innerHTML = `
    <div class="pgtitle">Settings</div>
    <div class="pgsub">Pool identity and access.</div>
    <div class="card" style="max-width:600px">
      <div class="kv"><span class="k">Pool</span><span class="mono">${esc(o.pool_id)}</span></div>
      <div class="kv"><span class="k">Owner account (coldkey)</span>${o.owner_account?('<span class="mono">'+esc(o.owner_account)+'</span>'):'<span class="sub">unclaimed — the first verified wallet claims it</span>'}</div>
      <div class="kv"><span class="k">Your session</span>${STATE.session?('<span>'+esc(shortHk(STATE.session.account))+' · '+(isOwner()?'owner':'viewer')+'</span>'):'<span class="sub">not signed in</span>'}</div>
      <div class="kv"><span class="k">Add-machine token</span><span class="sub">shown in your worker\'s config; keep it private</span></div>
    </div>`;
}

/* ---------- actions ---------- */
async function doStop(key){ if(!confirm("Stop mesh "+key+"?")) return;
  try{ await api("/v1/pool/stop", withSession({mesh_key:key})); toast("Stopping "+key); setTimeout(loadOverview,600); }
  catch(e){ toast(e.message); } }
async function doLaunch(workers, driver, model){
  try{ toast("Launching "+model+"…"); await api("/v1/pool/launch", withSession({model_id:model, workers, driver})); setTimeout(loadOverview,800); go("overview"); }
  catch(e){ toast(e.message); } }
let CHAT_MESH=null;
let CHAT_ABORT=null;
let CHAT_SENDING=false;
function openTestChat(key){
  if(CHAT_ABORT) CHAT_ABORT.abort();
  CHAT_MESH=key;
  document.getElementById("chatMesh").innerHTML='mesh <span class="mono">'+esc(key)+'</span>';
  document.getElementById("chatLog").innerHTML='<span class="sub">Send a prompt to verify this mesh answers with a real, proof-checked response.</span>';
  document.getElementById("chatBadge").innerHTML='';
  document.getElementById("chatModal").classList.add("on");
  setTimeout(()=>document.getElementById("chatInput").focus(),50);
}
function closeChat(){
  document.getElementById("chatModal").classList.remove("on");
  CHAT_MESH=null;
  if(CHAT_ABORT) CHAT_ABORT.abort();
}
function sseData(frame){
  const lines=frame.replace(/\r/g,"").split("\n");
  const data=[];
  for(const line of lines){
    if(line.startsWith("data:")) data.push(line.slice(5).replace(/^ /,""));
  }
  if(!data.length) return null; // comments/keepalives are valid SSE frames
  try{ return JSON.parse(data.join("\n")); }
  catch(e){ throw new Error("mesh returned an invalid stream event"); }
}
async function sendTestChat(){
  if(CHAT_SENDING) return;
  const inp=document.getElementById("chatInput"); const prompt=inp.value.trim(); if(!prompt||!CHAT_MESH) return;
  const meshKey=CHAT_MESH;
  const log=document.getElementById("chatLog"), badge=document.getElementById("chatBadge");
  log.innerHTML='<div class="you">you: '+esc(prompt)+'</div><div id="chatThinking" class="sub" style="display:none;margin-top:8px"></div><div id="chatAnswer" style="margin-top:8px"></div><div id="chatPhase" class="sub" style="margin-top:8px">connecting to the mesh…</div>';
  const answer=document.getElementById("chatAnswer"), phase=document.getElementById("chatPhase"), thinking=document.getElementById("chatThinking");
  badge.innerHTML='';
  const send=document.getElementById("chatSend"), oldLabel=send.textContent;
  const controller=new AbortController();
  CHAT_ABORT=controller; CHAT_SENDING=true; send.disabled=true; inp.disabled=true; send.textContent="Streaming…";
  let content="", thought="", sawFinal=false, sawTerminal=false;
  function renderStreamEvent(ev){
    if(ev.type==="delta"){
      const delta=String(ev.delta||""); content+=delta; answer.append(document.createTextNode(delta));
      phase.textContent="streaming from the mesh…";
    }else if(ev.type==="thinking"){
      thought+=String(ev.thinking||""); thinking.style.display="block"; thinking.textContent="thinking… "+thought.slice(-400);
    }else if(ev.type==="phase"){
      phase.textContent=ev.phase==="proving"?"response complete — generating + verifying proof…":String(ev.phase||"processing…");
    }else if(ev.type==="done"){
      sawFinal=true; sawTerminal=true;
      if(typeof ev.content==="string"){ content=ev.content; answer.textContent=content||"(no content)"; }
      if(thought) thinking.textContent="reasoning complete";
      const details=[];
      if(ev.engine_tps!=null&&Number.isFinite(Number(ev.engine_tps))) details.push(Math.round(Number(ev.engine_tps))+" tok/s");
      if(ev.receipts!=null&&Number.isFinite(Number(ev.receipts))){ const n=Number(ev.receipts); details.push(n+" receipt"+(n===1?"":"s")); }
      const proofStages=Number(ev.proof_stages), expectedStages=Number(ev.expected_stage_count);
      const verified=ev.verified===true&&ev.receipt_verified===true
        &&Number.isInteger(proofStages)&&Number.isInteger(expectedStages)
        &&expectedStages>0&&proofStages===expectedStages;
      badge.innerHTML='<span class="vbadge '+(verified?'ok':'no')+'">'+(verified?'✓ coordinator verified':'✗ not verified')+(details.length?' · '+details.join(' · '):'')+'</span>';
      phase.textContent=ev.error?String(ev.error):"proof-bearing final received";
      if(ev.error) phase.style.color="var(--err)";
      return true;
    }else if(ev.type==="error"){
      sawTerminal=true; phase.textContent=String(ev.error||"mesh stream failed"); phase.style.color="var(--err)";
      return true;
    }
    log.scrollTop=log.scrollHeight;
    return false;
  }
  try{
    const resp=await fetch(API+"/v1/pool/chat-stream", {
      method:"POST", headers:{"Content-Type":"application/json"}, signal:controller.signal,
      body:JSON.stringify(withSession({mesh_key:meshKey, prompt, max_tokens:256, thinking:false}))
    });
    if(!resp.ok){
      const txt=await resp.text(); let body={}; try{ body=JSON.parse(txt); }catch(e){}
      throw new Error(body.error||txt||("HTTP "+resp.status));
    }
    if(!resp.body) throw new Error("mesh stream is unavailable in this browser");
    const reader=resp.body.getReader(), decoder=new TextDecoder(); let buffer="";
    stream: for(;;){
      const {done,value}=await reader.read();
      if(done) break;
      buffer+=decoder.decode(value,{stream:true});
      let match;
      while((match=buffer.match(/\r?\n\r?\n/))){
        const frame=buffer.slice(0,match.index); buffer=buffer.slice(match.index+match[0].length);
        const ev=sseData(frame); if(!ev) continue;
        if(renderStreamEvent(ev)){
          await reader.cancel();
          break stream;
        }
      }
    }
    buffer+=decoder.decode();
    // SSE dispatches a final unterminated event at EOF too. The manager sends
    // delimiters, but accepting this keeps a proxy from losing the proof final.
    if(!sawTerminal && buffer.trim()){
      const ev=sseData(buffer); if(ev) renderStreamEvent(ev);
    }
    if(!sawTerminal && !controller.signal.aborted){
      throw new Error("mesh stream ended before the proof-bearing final event");
    }
    log.scrollTop=log.scrollHeight;
  }catch(e){
    if(controller.signal.aborted){ phase.textContent="request cancelled"; }
    else { phase.textContent=e.message||"mesh stream failed"; phase.style.color="var(--err)"; }
  }finally{
    if(CHAT_ABORT===controller){
      CHAT_ABORT=null; CHAT_SENDING=false; send.disabled=false; inp.disabled=false; send.textContent=oldLabel;
      if(document.getElementById("chatModal").classList.contains("on")) inp.focus();
    }
  }
}

/* ---------- wallet login: faithful @polkadot/extension-dapp web3Enable ---------- */
function strToHex(str){ const b=new TextEncoder().encode(str); let h="0x"; b.forEach(x=>h+=x.toString(16).padStart(2,"0")); return h; }
async function waitForInjected(ms){
  const t0=Date.now();
  while(Date.now()-t0<ms){ const w3=window.injectedWeb3; if(w3&&Object.keys(w3).length) return w3; await new Promise(r=>setTimeout(r,100)); }
  return window.injectedWeb3||null;
}
// Mirrors web3Enable: for each injected wallet try connect() (new standard) or
// enable() (legacy). Returns enabled injectors, each tagged with its name.
async function web3EnableAll(origin){
  const w3=await waitForInjected(3000);
  if(!w3||!Object.keys(w3).length) return [];
  const out=[];
  for(const [name,prov] of Object.entries(w3)){
    try{
      const ext = typeof prov.connect==="function" ? await prov.connect(origin)
                : typeof prov.enable==="function"  ? await prov.enable(origin) : null;
      if(ext){ ext.__name=name; out.push(ext); }
    }catch(e){ console.warn("[wallet] enable failed for",name,e); }
  }
  return out;
}
async function web3AllAccounts(exts){
  let accts=[];
  for(const ext of exts){
    try{ const a=await ext.accounts.get(); accts=accts.concat(a.map(x=>Object.assign({},x,{__ext:ext,__source:ext.__name}))); }catch(e){}
  }
  return accts;
}
function wdbg(m){ const el=document.getElementById("wdbg"); if(el) el.textContent="wallet: "+m; console.log("[wallet]",m); }
(function(){ // report injection state on load + as it appears
  let n=0; const iv=setInterval(()=>{ const w=window.injectedWeb3; const ks=w?Object.keys(w):[];
    wdbg("secureContext="+window.isSecureContext+" injectedWeb3=["+ks.join(",")+"]"); if(ks.length||++n>25) clearInterval(iv); }, 200);
})();
async function connectWallet(){
  const w=window.injectedWeb3; wdbg("click: secureContext="+window.isSecureContext+" injected=["+(w?Object.keys(w).join(","):"")+"]");
  let exts;
  try{ exts = await web3EnableAll("Verathos Operator"); }
  catch(e){ toast("Wallet error: "+(e.message||e)); return; }
  wdbg("enabled "+exts.length+" wallet(s): "+exts.map(e=>e.__name).join(","));
  if(!exts.length){
    const present = window.injectedWeb3?Object.keys(window.injectedWeb3):[];
    toast(present.length ? ("Approve the connection in "+present.join("/")+" (check the extension popup)")
                         : "No Substrate wallet detected. Open Talisman, unlock it, and click Connect again.");
    return;
  }
  let accts=[]; try{ accts = await web3AllAccounts(exts); }catch(e){}
  // Substrate accounts only (skip ethereum-type entries)
  accts = accts.filter(a=> !a.type || a.type==="sr25519" || a.type==="ed25519");
  if(!accts.length){ toast("Wallet connected but no accounts are shared — enable an account for this site in the extension."); return; }
  document.getElementById("wHint").textContent = (STATE.overview&&STATE.overview.owner_account) ? "Choose the wallet account to sign in with." : "The first verified wallet claims this pool.";
  document.getElementById("acctList").innerHTML = accts.map((a,i)=>`
    <div class="acct" data-i="${i}"><div class="avatar" style="filter:hue-rotate(${(i*70)%360}deg)"></div>
      <div><b>${esc(a.name||"account")}</b><div class="sub mono">${esc(shortHk(a.address))} · ${esc(a.__source||"")}</div></div></div>`).join("");
  window.__vAccts=accts;
  document.querySelectorAll(".acct").forEach(el=>el.onclick=()=>startVerify(accts[+el.dataset.i]));
  document.getElementById("wStep1").classList.remove("hidden");
  document.getElementById("wStep2").classList.add("hidden");
  document.getElementById("walletModal").classList.add("on");
}
let PENDING=null; let LOGIN_MODE="wallet";
async function startVerify(acct){
  try{
    const ch=await api("/v1/auth/challenge", {account:acct.address});
    PENDING={acct, message:ch.message, nonce:ch.nonce};
    document.getElementById("signHk").textContent=shortHk(acct.address);
    document.getElementById("challengeBox").textContent=ch.message;
    document.getElementById("wStep1").classList.add("hidden");
    document.getElementById("wStep2").classList.remove("hidden");
  }catch(e){ toast(e.message); }
}
async function doSign(){
  if(!PENDING||!PENDING.acct.__ext){ toast("reconnect wallet"); return; }
  try{
    const {signature}=await PENDING.acct.__ext.signer.signRaw({
      address:PENDING.acct.address, data:strToHex(PENDING.message), type:"bytes"});
    const body={account:PENDING.acct.address, nonce:PENDING.nonce, signature, mode:LOGIN_MODE};
    const claim=new URLSearchParams(location.search).get("claim"); if(claim) body.pool_secret=claim;
    const res=await api("/v1/auth/verify", body);
    STATE.session={token:res.token, account:res.account, mode:res.mode||LOGIN_MODE};
    if(STATE.session.mode==="hotkey"){ sessionStorage.setItem(SS_SESSION,JSON.stringify(STATE.session)); localStorage.removeItem(LS_SESSION); }
    else { localStorage.setItem(LS_SESSION,JSON.stringify(STATE.session)); sessionStorage.removeItem(SS_SESSION); }
    STATE.session.is_owner = !!res.is_owner;
    closeWallet(); applySession(); toast(res.is_owner?"Verified — you own this pool":"Signed in (viewer)");
    await loadOverview(); resolveScore();
  }catch(e){ toast("Sign failed: "+(e.message||e)); }
}
function applySession(){
  const s=STATE.session;
  document.getElementById("walletBtn").classList.toggle("hidden", !!s);
  document.getElementById("walletBadge").classList.toggle("hidden", !s);
  if(s){ document.getElementById("ownerAddr").textContent=shortHk(s.account);
    const own=isOwner(); const chip=document.getElementById("ownerChip");
    chip.className="chip "+(own?"ok":"warn"); chip.innerHTML='<span class="d"></span> '+(own?("owner"+(s.mode==="hotkey"?" · hotkey (this session)":"")):"viewer"); }
  render();
}
function signOut(){ STATE.session=null; localStorage.removeItem(LS_SESSION); sessionStorage.removeItem(SS_SESSION); applySession(); toast("Signed out"); }
function closeWallet(){ document.getElementById("walletModal").classList.remove("on"); PENDING=null; }

/* ---------- score/UID (best-effort; manager fills when it can) ---------- */
async function resolveScore(){
  try{ const s=await api("/v1/operator/score", withSession({})); if(s&&s.uid!=null){ STATE.score=s; render(); } }catch(e){}
}

/* ---------- latency map (radial, animated signals like dashboard v1) ---------- */
const MAPS = {};   // canvasId -> {raf}
let MAP_HIGHLIGHT = null; // [workerA, workerB] links to emphasize (launch page)
function hashAngle(id){ let h=0; for(let i=0;i<id.length;i++) h=(h*31+id.charCodeAt(i))>>>0; return (h%3600)/3600*Math.PI*2; }
function rttRadius(ms, maxR){ const inner=40; if(ms==null||isNaN(ms)) return maxR*0.82; const t=Math.min(1, ms/200); return inner+t*(maxR-inner); }
function cssVar(n){ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function drawLatMap(canvasId){
  const cv=document.getElementById(canvasId); if(!cv || cv.offsetParent===null) return;
  const dpr=Math.max(1,window.devicePixelRatio||1);
  const r=cv.getBoundingClientRect(); if(!r.width) return;
  cv.width=r.width*dpr; cv.height=r.height*dpr;
  const ctx=cv.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  const W=r.width,H=r.height, cx=W/2, cy=H/2, maxR=Math.min(W,H)/2-26;
  const f=fleet(); const t=performance.now()/1000;
  const cPrimary=cssVar("--primary")||"#4a7fe0", cMuted=cssVar("--muted")||"#a0a0a0",
        cErr=cssVar("--err")||"#f85149", cBorder=cssVar("--border")||"#474747", cFg=cssVar("--fg")||"#e8e8e8";
  ctx.clearRect(0,0,W,H);
  // rtt rings
  ctx.strokeStyle=cBorder; ctx.globalAlpha=.35; ctx.setLineDash([3,5]);
  [50,100,200].forEach(ms=>{ ctx.beginPath(); ctx.arc(cx,cy,rttRadius(ms,maxR),0,Math.PI*2); ctx.stroke();
    ctx.globalAlpha=.55; ctx.fillStyle=cMuted; ctx.font="9px sans-serif";
    ctx.fillText(ms+"ms", cx+rttRadius(ms,maxR)-16, cy-4); ctx.globalAlpha=.35; });
  ctx.setLineDash([]); ctx.globalAlpha=1;
  // node positions
  const pos={};
  f.workers.forEach(w=>{ const a=hashAngle(w.id); const rr=rttRadius(w.rtt_ms&&w.rtt_ms.manager, maxR);
    pos[w.id]={x:cx+Math.cos(a)*rr, y:cy+Math.sin(a)*rr, w}; });
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  // edges: coordinator->worker (+ traveling pulse), worker<->worker peers
  Object.values(pos).forEach(n=>{
    const on=f.live(n.w); const col = on? cPrimary : cErr;
    ctx.strokeStyle=col; ctx.globalAlpha= on? .35 : .18; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(n.x,n.y); ctx.stroke(); ctx.globalAlpha=1;
    if(on && !reduced){ // signal pulse traveling coordinator <-> worker
      const ph=(t*0.5 + hashAngle(n.w.id))%1;
      const px=cx+(n.x-cx)*ph, py=cy+(n.y-cy)*ph;
      ctx.fillStyle=col; ctx.beginPath(); ctx.arc(px,py,2.2,0,Math.PI*2); ctx.fill();
    }
  });
  // peer links with rtt labels
  const seen=new Set();
  f.workers.forEach(a=>{ const pr=a.peer_rtt_ms||{};
    Object.entries(pr).forEach(([bid,ms])=>{
      const key=[a.id,bid].sort().join("|"); if(seen.has(key)||!pos[bid]||!pos[a.id]) return; seen.add(key);
      const A=pos[a.id], B=pos[bid];
      const hl = MAP_HIGHLIGHT && MAP_HIGHLIGHT.includes(a.id) && MAP_HIGHLIGHT.includes(bid);
      const cls = ms<10? cPrimary : ms<40? (cssVar("--warn")||"#f2b03c") : cErr;
      ctx.strokeStyle=cls; ctx.lineWidth= hl? 2.5 : 1; ctx.globalAlpha= hl? .95 : .45;
      ctx.setLineDash(hl? [] : [4,4]);
      ctx.beginPath(); ctx.moveTo(A.x,A.y); ctx.lineTo(B.x,B.y); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha=1;
      const mx=(A.x+B.x)/2, my=(A.y+B.y)/2;
      ctx.fillStyle=cls; ctx.font=(hl?"bold ":"")+"9.5px sans-serif";
      ctx.fillText(Math.round(ms)+"ms", mx+4, my-3);
      if(hl && !reduced){ const ph=(t*0.8)%1; const px=A.x+(B.x-A.x)*ph, py=A.y+(B.y-A.y)*ph;
        ctx.fillStyle=cls; ctx.beginPath(); ctx.arc(px,py,2.6,0,Math.PI*2); ctx.fill(); }
    });
  });
  // coordinator node
  ctx.fillStyle=cPrimary; ctx.beginPath(); ctx.arc(cx,cy,7,0,Math.PI*2); ctx.fill();
  ctx.strokeStyle=cPrimary; ctx.globalAlpha=.4; ctx.beginPath(); ctx.arc(cx,cy,11,0,Math.PI*2); ctx.stroke(); ctx.globalAlpha=1;
  ctx.fillStyle=cMuted; ctx.font="9.5px sans-serif"; ctx.textAlign="center";
  ctx.fillText("coordinator", cx, cy+22); ctx.textAlign="left";
  // worker nodes
  Object.values(pos).forEach(n=>{ const on=f.live(n.w); const st=(n.w.status||"").split(" ")[0];
    const col = !on? cErr : (st==="serving"||st==="driving"||st==="joining")? cPrimary : cMuted;
    ctx.fillStyle=col; ctx.beginPath(); ctx.arc(n.x,n.y,5,0,Math.PI*2); ctx.fill();
    ctx.fillStyle=cFg; ctx.font="10px sans-serif"; ctx.fillText(n.w.id, n.x+8, n.y+3);
  });
}
function animateMaps(){
  drawLatMap("latmapOverview"); drawLatMap("latmapLaunch");
  requestAnimationFrame(animateMaps);
}
if(!matchMedia("(prefers-reduced-motion: reduce)").matches) requestAnimationFrame(animateMaps);
else setInterval(()=>{drawLatMap("latmapOverview");drawLatMap("latmapLaunch");}, 3000);

/* ---------- nav + boot ---------- */
function go(page){ STATE.page=page; render(); }
document.querySelectorAll(".nav").forEach(n=>n.onclick=()=>go(n.dataset.page));
document.addEventListener("click", e=>{ const t=e.target.closest("[data-nav]"); if(t){ e.preventDefault(); go(t.dataset.nav); } });
document.getElementById("walletBtn").onclick=connectWallet;
document.getElementById("signoutBtn").onclick=signOut;
document.getElementById("doSign").onclick=doSign;
document.getElementById("wCancel").onclick=closeWallet;
document.getElementById("walletModal").onclick=e=>{ if(e.target.id==="walletModal") closeWallet(); };
document.getElementById("chatClose").onclick=closeChat;
document.getElementById("chatSend").onclick=sendTestChat;
document.getElementById("chatInput").addEventListener("keydown",e=>{ if(e.key==="Enter") sendTestChat(); });
document.getElementById("chatModal").onclick=e=>{ if(e.target.id==="chatModal") closeChat(); };
document.getElementById("themeBtn").onclick=()=>{ const r=document.documentElement, cur=r.getAttribute("data-theme");
  const dark = cur? cur==="dark" : matchMedia("(prefers-color-scheme: dark)").matches; r.setAttribute("data-theme", dark?"light":"dark"); };

(function boot(){
  try{ const raw=sessionStorage.getItem(SS_SESSION)||localStorage.getItem(LS_SESSION); if(raw){ STATE.session=JSON.parse(raw); } }catch(e){}
  applySession();
  (async()=>{
    if(STATE.session){
      try{ const s=await api("/v1/auth/session", {session:STATE.session.token});
        if(!s.signed_in){ STATE.session=null; localStorage.removeItem(LS_SESSION); sessionStorage.removeItem(SS_SESSION); }
        else { STATE.session.is_owner=!!s.is_owner; STATE.session.account=s.account; }
      }catch(e){}
      applySession();
    }
    await loadOverview(); if(STATE.session) resolveScore();
  })();
  setInterval(loadOverview, 15000);
})();
</script>
</body>
</html>
"""
