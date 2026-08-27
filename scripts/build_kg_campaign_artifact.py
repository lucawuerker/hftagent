"""Build the KG-campaign evaluation artifact HTML from overview/ data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CAMP = REPO / "data/kg_campaign"
OV = CAMP / "overview"

base = json.loads((OV / "artifact_data.json").read_text())

if (OV / "local_results.csv").exists():
    local = pd.read_csv(OV / "local_results.csv")
    local["ics"] = local["ics"].apply(json.loads)
else:
    local = pd.DataFrame(columns=["label", "method", "n_factors",
                                  "blockmean", "blockstd", "hit", "ics"])

data = dict(base)
data["local"] = local.to_dict("records")

# 20x20 correlation of the per-run lasso block-IC vectors (10 WF blocks)
vecs = {}
for b in base["worker_blocks"]:
    if b["scope"] == "run" and b["method"] == "lasso":
        vecs[b["run"]] = b["ics"]
for r in data["local"]:
    if r["label"].startswith("run") and r["method"] == "lasso":
        vecs[int(r["label"][3:])] = r["ics"]
runs_v = sorted(vecs)
V = np.array([vecs[r] for r in runs_v], dtype=float)
C = np.corrcoef(V)
data["block_corr"] = {"labels": runs_v, "m": np.round(C, 4).tolist(),
                      "vecs": {r: [round(x, 4) for x in vecs[r]]
                               for r in runs_v}}

nl = pd.DataFrame(data["nonlinear"])
data["nl_summary"] = (nl.groupby("model")["ic"]
                      .agg(["mean", "std"]).round(4).to_dict("index"))

html = """<title>KG-Breadth-Kampagne — Auswertung der 20 Läufe</title>
<style>
:root{
  --paper:#fafbfc; --card:#ffffff; --ink:#16181d; --ink-2:#5b6270;
  --ink-3:#8a91a0; --line:#e4e7ec; --accent:#2a78d6; --accent-2:#eb6834;
  --accent-3:#1baf7a; --neutral:#52514e; --good:#1baf7a;
}
@media (prefers-color-scheme: dark){:root{
  --paper:#15171c; --card:#1c1f26; --ink:#eceef2; --ink-2:#a7adba;
  --ink-3:#767d8c; --line:#2b2f38; --accent:#3987e5; --accent-2:#d95926;
  --accent-3:#199e70; --neutral:#c3c2b7;
}}
:root[data-theme="dark"]{
  --paper:#15171c; --card:#1c1f26; --ink:#eceef2; --ink-2:#a7adba;
  --ink-3:#767d8c; --line:#2b2f38; --accent:#3987e5; --accent-2:#d95926;
  --accent-3:#199e70; --neutral:#c3c2b7;
}
:root[data-theme="light"]{
  --paper:#fafbfc; --card:#ffffff; --ink:#16181d; --ink-2:#5b6270;
  --ink-3:#8a91a0; --line:#e4e7ec; --accent:#2a78d6; --accent-2:#eb6834;
  --accent-3:#1baf7a; --neutral:#52514e;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;
  font:15px/1.55 "Avenir Next","Segoe UI",system-ui,sans-serif}
main{max-width:1060px;margin:0 auto;padding:40px 22px 80px}
.mono{font-family:"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
header.top{border-bottom:1px solid var(--line);padding-bottom:22px;
  margin-bottom:30px}
.eyebrow{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 8px}
h1{font-size:29px;line-height:1.2;margin:0 0 10px;text-wrap:balance;
  font-weight:600}
.sub{color:var(--ink-2);max-width:72ch;margin:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px;margin:26px 0 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;
  padding:12px 14px}
.kpi .v{font-size:21px;font-weight:600}
.kpi .l{font-size:12px;color:var(--ink-2);margin-top:2px}
h2{font-size:19px;margin:44px 0 6px;font-weight:600}
h2 .no{color:var(--ink-3);font-weight:500;margin-right:8px}
.lead{color:var(--ink-2);max-width:78ch;margin:0 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:18px;margin:12px 0;overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;
  color:var(--ink-2);margin:0 0 8px}
.legend .sw{display:inline-block;width:11px;height:11px;border-radius:3px;
  margin-right:5px;vertical-align:-1px}
svg text{fill:var(--ink-2);font-size:11px;
  font-family:"SF Mono",Menlo,monospace}
svg .ttl{fill:var(--ink);font-size:13px;
  font-family:"Avenir Next","Segoe UI",sans-serif;font-weight:600}
.grid line{stroke:var(--line);stroke-width:1}
.axisline{stroke:var(--line)}
#tip{position:fixed;pointer-events:none;background:var(--card);
  border:1px solid var(--line);border-radius:6px;padding:7px 10px;
  font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.13);opacity:0;
  z-index:10;max-width:280px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{color:var(--ink-2);font-weight:600;text-align:right;padding:7px 10px;
  border-bottom:1.5px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right}
.findings li{margin:0 0 10px;max-width:82ch}
.note{font-size:12.5px;color:var(--ink-3);max-width:82ch}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:840px){.two{grid-template-columns:1fr}}
</style>
<main>
<header class="top">
<p class="eyebrow">QuantFundAgent · Thesis-Finale · 2026-08-16 → 08-18</p>
<h1>KG-Breadth-Kampagne: 20 Seeding-Läufe auf dem lebenden Knowledge-Graph</h1>
<p class="sub">20 sequenzielle, reine Seeding-Läufe (GPT-5.6 Terra, 12 Ideen ×
8 Mechanismus-Gruppen, keine Evolution) gegen den <em>mitwachsenden</em>
Knowledge-Graph: jeder Lauf linkt seine Faktoren zurück, der nächste seedet
gegen den erweiterten Graph. Bewertung im Standard-Walk-Forward (10
prequentiale 126-Bar-Blöcke ab 2021-07-20, expandierender Refit, gepoolter
per-underlying IC).</p>
<div class="kpis">
<div class="kpi"><div class="v mono">__KPI_FACTORS__</div><div class="l">neue Faktoren (20 Läufe)</div></div>
<div class="kpi"><div class="v mono">2 522</div><div class="l">kumulatives Buch inkl. 619 Alt-Faktoren</div></div>
<div class="kpi"><div class="v mono">$210.85</div><div class="l">LLM-Kosten gesamt (~$10.50/Lauf)</div></div>
<div class="kpi"><div class="v mono">__KPI_CUMIC__</div><div class="l">WF-IC des Gesamtbuchs (ridge, linear)</div></div>
<div class="kpi"><div class="v mono">0.086</div><div class="l">WF-IC nichtlinear (TabM-NN / RF, Clean-Pool)</div></div>
</div>
</header>

<h2><span class="no">1</span>Buchwachstum und das lineare Plateau</h2>
<p class="lead">Links das kumulative Buch nach jedem Lauf; rechts der
Walk-Forward-IC des kumulativen Buchs unter dem linearen Referenz-Kombinierer
(Gram-Ridge, α=10⁴). Der zentrale Befund: <strong>ab ~Lauf 5 ist der lineare
IC ein Plateau bei ~0.066–0.070</strong> — mehr Faktoren verbessern das
linear kombinierte Buch nicht mehr.</p>
<div class="two">
<div class="card"><div id="c_growth"></div></div>
<div class="card"><div id="c_cumic"></div></div>
</div>

<h2><span class="no">2</span>Qualität der einzelnen Run-Bücher</h2>
<p class="lead">Jeder Lauf einzeln als eigenes ~80–96-Faktoren-Buch im selben
WF-Protokoll (ridge und LassoCV, 10 Block-Refits). Die Run-Bücher sind
bemerkenswert homogen — kein Lauf fällt ab, Lasso liegt fast überall über
Ridge.</p>
<div class="card"><div id="c_runs"></div></div>

<h2><span class="no">3</span>Korrelation der Läufe: Lasso-Block-IC-Verläufe</h2>
<p class="lead">Jeder Lauf hat als eigenes Buch unter LassoCV zehn
Walk-Forward-Block-ICs (Refit je Block). Gezeigt ist die Korrelation dieser
10-Punkte-Verläufe zwischen allen 20 Läufen — sie misst, wie gleichläufig
die Run-Bücher über die Marktregime 2021–2026 performen. Hinweis: 10
Beobachtungen je Paar, und alle Läufe teilen dieselben Blöcke — ein großer
Teil der Korrelation ist gemeinsames Regime, nicht Buch-Ähnlichkeit.</p>
<div class="card"><div id="h_block"></div></div>

<h2><span class="no">4</span>Nichtlineare Kombinierer auf dem Gesamtbuch</h2>
<p class="lead">Auf dem rho&lt;0.9-bereinigten Clean-Pool (1 632 Faktoren)
wurden drei nichtlineare Kombinierer im identischen WF gefahren
(2026-08-17/18): TabM-MLP-Ensemble (k=8), Random Forest (LightGBM-rf-Modus)
und LightGBM (λ₂=5N). <strong>NN und RF durchbrechen das lineare Plateau
(0.086 vs. 0.066)</strong>; LightGBM bleibt auf Ridge-Niveau.</p>
<div class="card"><div id="c_nl"></div></div>

<h2><span class="no">5</span>Datentabelle</h2>
<div class="card"><table class="mono" id="t_runs"></table>
<p class="note" style="margin-bottom:0">ridge/lasso = Mittel der 10
WF-Block-ICs des einzelnen Run-Buchs (Läufe 19–20 lokal nachgerechnet,
identisches Protokoll).</p></div>

<h2><span class="no">6</span>Befunde</h2>
<ul class="findings">
<li><strong>Der lebende Graph verhindert Kollaps, aber lineare Kombination
sättigt.</strong> Jeder Lauf liefert ein eigenständig gutes Buch
(ridge ~0.05–0.08), doch der kumulative lineare IC bleibt ab ~700 Faktoren
flach — zusätzliche Faktoren sind linear redundant zum Bestand.</li>
<li><strong>__FINDING_CORR__</strong></li>
<li><strong>Nichtlinearität hebt das Plateau:</strong> TabM-NN und RF holen
aus demselben Pool ~30 % mehr WF-IC (0.086 vs. 0.066) bei 10/10 positiven
Blöcken; das NN mit der halben Block-Streuung des RF. LightGBM profitiert
trotz N-skalierter Regulierung nicht.</li>
<li><strong>Kosten-Seite:</strong> ~$10.50 pro ~95-Faktoren-Buch; die gesamte
Kampagne (1 903 Faktoren) kostete $210.85 — weniger als ein einzelner
L4-Evolutionslauf ($250), bei bereits gleichwertigem linearen Buch-IC und
klarem Mehrwert unter nichtlinearer Kombination.</li>
</ul>
<p class="note">Protokoll-Notizen: WF-Blöcke = die prequentialen Fenster der
Terra-WF-Leiter; alle ICs gepoolt per-underlying (Konvention 2026-08-05).
Faktor-Korrelationen auf dem Dev-Fenster (&lt; 2021-07-20). Läufe 19/20 und
cum 18–20 lokal berechnet, da der Server-Worker noch aufholt; identischer
Code-Pfad. Artefakte: <span class="mono">data/kg_campaign/overview/</span>,
<span class="mono">data/comparisons/kg_nonlinear_combiners/</span>.</p>
</main>
<div id="tip"></div>
<script>
const D = __DATA__;
const css = v => getComputedStyle(document.documentElement)
  .getPropertyValue(v).trim();
const tip = document.getElementById('tip');
function showTip(ev, html){ tip.innerHTML = html; tip.style.opacity = 1;
  const x = Math.min(ev.clientX + 14, innerWidth - 300);
  tip.style.left = x + 'px'; tip.style.top = (ev.clientY + 14) + 'px'; }
function hideTip(){ tip.style.opacity = 0; }
const NS = 'http://www.w3.org/2000/svg';
function el(tag, at, parent){ const e = document.createElementNS(NS, tag);
  for (const k in at) e.setAttribute(k, at[k]);
  if (parent) parent.appendChild(e); return e; }

function chart(id, w, h, m){ const svg = el('svg',
  {viewBox:`0 0 ${w} ${h}`, width:'100%'},
  document.getElementById(id)); return svg; }

function linscale(d0, d1, r0, r1){ return v => r0 + (v - d0)/(d1 - d0)*(r1 - r0); }

function axis(svg, x0, x1, y0, y1, ymin, ymax, ticks){
  for (const t of ticks){ const y = y0 + (y1-y0)*(t-ymin)/(ymax-ymin);
    el('line',{x1:x0,x2:x1,y1:y,y2:y,class:'grid',stroke:'var(--line)'},svg);
    el('text',{x:x0-6,y:y+3.5,'text-anchor':'end'},svg).textContent =
      t.toFixed(t % 1 ? 2 : 0); } }

// ---- 1a growth ------------------------------------------------------------
(function(){
  const w=480,h=300,L=52,R=14,T=36,B=40;
  const svg=chart('c_growth',w,h);
  el('text',{x:L,y:18,class:'ttl'},svg).textContent='Kumulatives Buch (Faktoren)';
  const cumN=[]; let acc=D.alt_book;
  for(const r of D.runs){ acc+=r.persisted; cumN.push(acc); }
  const X=linscale(1,20,L,w-R), Y=linscale(0,2600,h-B,T);
  axis(svg,L,w-R,h-B,T,0,2600,[0,650,1300,1950,2600]);
  let p='';
  cumN.forEach((v,i)=>{p+=(i?'L':'M')+X(i+1)+' '+Y(v)+' ';});
  el('path',{d:p+'L'+X(20)+' '+Y(0)+' L'+X(1)+' '+Y(0)+' Z',
    fill:css('--accent'),opacity:.14},svg);
  el('path',{d:p,fill:'none',stroke:css('--accent'),'stroke-width':2},svg);
  el('line',{x1:L,x2:w-R,y1:Y(D.alt_book),y2:Y(D.alt_book),
    stroke:css('--neutral'),'stroke-dasharray':'4 3'},svg);
  el('text',{x:w-R,y:Y(D.alt_book)-5,'text-anchor':'end'},svg)
    .textContent='Alt-Buch 619';
  cumN.forEach((v,i)=>{ const c=el('circle',{cx:X(i+1),cy:Y(v),r:7,
    fill:'transparent'},svg);
    c.addEventListener('mousemove',ev=>showTip(ev,
      `<b>nach Lauf ${i+1}</b><br>${v} Faktoren (+${D.runs[i].persisted})`+
      `<br>Kosten Lauf: $${D.runs[i].cost.toFixed(2)}`));
    c.addEventListener('mouseleave',hideTip);
    el('circle',{cx:X(i+1),cy:Y(v),r:2.5,fill:css('--accent'),
      'pointer-events':'none'},svg); });
  for(let i=1;i<=20;i+=(i==1?4:5)) el('text',{x:X(i),y:h-B+16,
    'text-anchor':'middle'},svg).textContent=i;
  el('text',{x:(L+w-R)/2,y:h-6,'text-anchor':'middle'},svg)
    .textContent='Lauf';
})();

// ---- 1b cum ridge IC ------------------------------------------------------
(function(){
  const w=480,h=300,L=52,R=14,T=36,B=40;
  const svg=chart('c_cumic',w,h);
  el('text',{x:L,y:18,class:'ttl'},svg)
    .textContent='WF-IC kumulatives Buch (ridge)';
  const rows=D.worker.filter(r=>r.scope=='cum'&&r.method=='ridge')
    .map(r=>({run:r.run,v:r.blockmean,n:r.n_factors,src:'lagias'}));
  for(const l of D.local.filter(r=>r.label.startsWith('cum'))){
    const run=parseInt(l.label.slice(3));
    if(!rows.some(r=>r.run==run)) rows.push({run,v:l.blockmean,
      n:l.n_factors,src:'lokal'});}
  rows.sort((a,b)=>a.run-b.run);
  const X=linscale(1,20,L,w-R), Y=linscale(0,0.10,h-B,T);
  axis(svg,L,w-R,h-B,T,0,0.10,[0,0.02,0.04,0.06,0.08,0.10]);
  el('rect',{x:L,y:Y(0.070),width:w-R-L,height:Y(0.066)-Y(0.070),
    fill:css('--neutral'),opacity:.13},svg);
  el('text',{x:w-R,y:Y(0.071)-4,'text-anchor':'end'},svg)
    .textContent='Plateau ~0.066–0.070';
  let p=''; rows.forEach((r,i)=>{p+=(i?'L':'M')+X(r.run)+' '+Y(r.v)+' ';});
  el('path',{d:p,fill:'none',stroke:css('--accent'),'stroke-width':2},svg);
  rows.forEach(r=>{ const c=el('circle',{cx:X(r.run),cy:Y(r.v),r:8,
    fill:'transparent'},svg);
    c.addEventListener('mousemove',ev=>showTip(ev,
      `<b>Buch nach Lauf ${r.run}</b> (${r.n} F.)<br>ridge Block-Mean-IC `+
      `${r.v.toFixed(4)}<br><i>${r.src}</i>`));
    c.addEventListener('mouseleave',hideTip);
    el('circle',{cx:X(r.run),cy:Y(r.v),r:3,fill:css('--accent'),
      'pointer-events':'none'},svg);});
  for(let i=1;i<=20;i+=(i==1?4:5)) el('text',{x:X(i),y:h-B+16,
    'text-anchor':'middle'},svg).textContent=i;
  el('text',{x:(L+w-R)/2,y:h-6,'text-anchor':'middle'},svg)
    .textContent='Buchstand nach Lauf';
})();

// ---- 2 per-run books ------------------------------------------------------
(function(){
  const w=1000,h=320,L=52,R=14,T=40,B=42;
  const svg=chart('c_runs',w,h);
  el('text',{x:L,y:18,class:'ttl'},svg)
    .textContent='Run-Bücher einzeln: WF-Block-Mean-IC (ridge vs. LassoCV)';
  const runs=[];
  for(let n=1;n<=20;n++){
    const rr=D.worker.find(r=>r.scope=='run'&&r.method=='ridge'&&r.run==n);
    const rl=D.worker.find(r=>r.scope=='run'&&r.method=='lasso'&&r.run==n);
    const lr=D.local.find(r=>r.label=='run'+n&&r.method=='ridge');
    const ll=D.local.find(r=>r.label=='run'+n&&r.method=='lasso');
    runs.push({run:n,ridge:rr?rr.blockmean:(lr?lr.blockmean:null),
      lasso:rl?rl.blockmean:(ll?ll.blockmean:null),
      n:rr?rr.n_factors:(lr?lr.n_factors:null),
      src:rr?'lagias':'lokal'});}
  const X=linscale(0,20,L,w-R), Y=linscale(0,0.10,h-B,T);
  axis(svg,L,w-R,h-B,T,0,0.10,[0,0.02,0.04,0.06,0.08,0.10]);
  const bw=X(1)-X(0);
  runs.forEach(r=>{
    const x=X(r.run-1)+bw/2;
    const g=el('g',{},svg);
    if(r.ridge!=null) el('rect',{x:x-bw*0.33,y:Y(r.ridge),width:bw*0.30,
      height:Y(0)-Y(r.ridge),fill:css('--neutral'),rx:2},g);
    if(r.lasso!=null) el('rect',{x:x+bw*0.03,y:Y(r.lasso),width:bw*0.30,
      height:Y(0)-Y(r.lasso),fill:css('--accent'),rx:2},g);
    const hit=el('rect',{x:x-bw*0.45,y:T,width:bw*0.9,height:h-B-T,
      fill:'transparent'},g);
    hit.addEventListener('mousemove',ev=>showTip(ev,
      `<b>Lauf ${r.run}</b> (${r.n||'?'} F., ${r.src})<br>`+
      `ridge ${r.ridge?r.ridge.toFixed(4):'—'} · `+
      `lasso ${r.lasso?r.lasso.toFixed(4):'—'}`));
    hit.addEventListener('mouseleave',hideTip);
    el('text',{x:x,y:h-B+16,'text-anchor':'middle'},svg).textContent=r.run;});
  el('text',{x:(L+w-R)/2,y:h-6,'text-anchor':'middle'},svg)
    .textContent='Lauf';
  const lg=document.createElement('div'); lg.className='legend';
  lg.innerHTML=`<span><span class="sw" style="background:var(--neutral)"></span>ridge</span>
  <span><span class="sw" style="background:var(--accent)"></span>LassoCV</span>`;
  document.getElementById('c_runs').prepend(lg);
})();

// ---- heatmaps -------------------------------------------------------------
function heat(id, H, title, vmax, fmt){
  const n=H.labels.length, cell=34, L=64, T=54, w=L+n*cell+20, h=T+n*cell+16;
  const svg=chart(id,w,h);
  el('text',{x:L,y:20,class:'ttl'},svg).textContent=title;
  const ramp=['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#1c5cab','#0d366b'];
  const col=v=>{const t=Math.max(0,Math.min(1,v/vmax));
    const i=Math.min(ramp.length-1,Math.floor(t*(ramp.length-1)));
    return ramp[i];};
  const lab=x=>x==0?'Alt':x;
  H.labels.forEach((la,i)=>{
    el('text',{x:L-6,y:T+i*cell+cell/2+4,'text-anchor':'end'},svg)
      .textContent=lab(la);
    el('text',{x:L+i*cell+cell/2,y:T-8,'text-anchor':'middle'},svg)
      .textContent=lab(la);
    H.labels.forEach((lb,j)=>{
      const v=H.m[i][j];
      const r=el('rect',{x:L+j*cell+1,y:T+i*cell+1,width:cell-2,
        height:cell-2,rx:3,fill:col(Math.abs(v))},svg);
      r.addEventListener('mousemove',ev=>showTip(ev,
        `<b>${lab(la)} × ${lab(lb)}</b><br>${fmt}: ${v.toFixed(3)}`));
      r.addEventListener('mouseleave',hideTip);
      if(Math.abs(v)>=vmax*0.55)
        el('text',{x:L+j*cell+cell/2,y:T+i*cell+cell/2+4,
          'text-anchor':'middle',fill:'#fff',
          style:'fill:#fff;font-size:9px','pointer-events':'none'},svg)
          .textContent=v.toFixed(2).replace('0.','.');
    });});
}
// block-IC-vector correlation: diverging scale (corr can be negative)
(function(){
  const H=D.block_corr, n=H.labels.length, cell=38, L=58, T=54,
    w=L+n*cell+20, h=T+n*cell+16;
  const svg=chart('h_block',w,h);
  el('text',{x:L,y:20,class:'ttl'},svg).textContent=
    'Korrelation der Lasso-Block-IC-Verläufe (10 WF-Blöcke, alle Läufe)';
  const neg=['#0d366b','#1c5cab','#3987e5','#6da7ec','#9ec5f4'];
  const pos=['#f6c9a8','#f0a370','#eb6834','#c94d1e','#8f3512'];
  const mid='#e8e8e4';
  const col=v=>{ if(Math.abs(v)<0.1) return mid;
    const ramp=v<0?neg:pos; const t=Math.min(1,(Math.abs(v)-0.1)/0.9);
    return ramp[Math.min(4,Math.floor(t*5))]; };
  H.labels.forEach((la,i)=>{
    el('text',{x:L-6,y:T+i*cell+cell/2+4,'text-anchor':'end'},svg)
      .textContent=la;
    el('text',{x:L+i*cell+cell/2,y:T-8,'text-anchor':'middle'},svg)
      .textContent=la;
    H.labels.forEach((lb,j)=>{
      const v=H.m[i][j];
      const r=el('rect',{x:L+j*cell+1,y:T+i*cell+1,width:cell-2,
        height:cell-2,rx:3,fill:col(v)},svg);
      r.addEventListener('mousemove',ev=>showTip(ev,
        `<b>Lauf ${la} × Lauf ${lb}</b><br>ρ(Block-ICs) = ${v.toFixed(3)}`+
        `<br><span style="color:var(--ink-3)">ICs ${la}: [${H.vecs[la].map(x=>x.toFixed(2)).join(', ')}]</span>`));
      r.addEventListener('mouseleave',hideTip);
      if(i!=j && Math.abs(v)>=0.45)
        el('text',{x:L+j*cell+cell/2,y:T+i*cell+cell/2+4,
          'text-anchor':'middle',
          style:'fill:'+(Math.abs(v)>0.65?'#fff':'var(--ink)')+';font-size:9px',
          'pointer-events':'none'},svg)
          .textContent=v.toFixed(2).replace('0.','.').replace('-0.','-.');
    });});
  const lg=document.createElement('div'); lg.className='legend';
  lg.innerHTML='<span><span class="sw" style="background:#3987e5"></span>negativ</span>'+
    '<span><span class="sw" style="background:#e8e8e4"></span>|ρ|<0.1</span>'+
    '<span><span class="sw" style="background:#eb6834"></span>positiv</span>';
  document.getElementById('h_block').prepend(lg);
})();

// ---- 5 nonlinear ---------------------------------------------------------
(function(){
  const w=1000,h=330,L=52,R=150,T=40,B=42;
  const svg=chart('c_nl',w,h);
  el('text',{x:L,y:18,class:'ttl'},svg)
    .textContent='Nichtlineare Kombinierer, Per-Block-IC (Clean-Pool 1 632 F.)';
  const models={nn:{c:css('--accent'),n:'TabM-NN (k=8)'},
    rf:{c:css('--accent-2'),n:'Random Forest'},
    lightgbm:{c:css('--accent-3'),n:'LightGBM (λ₂=5N)'}};
  const RIDGE=[0.0565,0.1384,0.0899,0.1093,0.0093,0.0389,0.0323,0.0684,
    0.0633,0.0511];
  const X=linscale(0,9,L,w-R), Y=linscale(-0.01,0.21,h-B,T);
  axis(svg,L,w-R,h-B,T,-0.01,0.21,[0,0.05,0.10,0.15,0.20]);
  let pr=''; RIDGE.forEach((v,i)=>{pr+=(i?'L':'M')+X(i)+' '+Y(v)+' ';});
  el('path',{d:pr,fill:'none',stroke:css('--neutral'),'stroke-width':1.6,
    'stroke-dasharray':'5 4'},svg);
  el('text',{x:X(9)+8,y:Y(RIDGE[9])+4},svg).textContent='Ridge (voll)';
  const ends=[];
  for(const m in models){
    const rows=D.nonlinear.filter(r=>r.model==m).sort((a,b)=>a.gen-b.gen);
    let p=''; rows.forEach((r,i)=>{p+=(i?'L':'M')+X(i)+' '+Y(r.ic)+' ';});
    el('path',{d:p,fill:'none',stroke:models[m].c,'stroke-width':2.2},svg);
    rows.forEach((r,i)=>{const c=el('circle',{cx:X(i),cy:Y(r.ic),r:8,
      fill:'transparent'},svg);
      c.addEventListener('mousemove',ev=>showTip(ev,
        `<b>${models[m].n}</b><br>Block ${r.gen-10} (${r.start} → ${r.end})`+
        `<br>IC ${r.ic.toFixed(4)}`));
      c.addEventListener('mouseleave',hideTip);
      el('circle',{cx:X(i),cy:Y(r.ic),r:3.2,fill:models[m].c,
        'pointer-events':'none'},svg);});
    ends.push({y:Y(rows[9].ic),c:models[m].c,
      n:models[m].n.split(' (')[0],
      s:D.nl_summary[m]});}
  ends.sort((a,b)=>a.y-b.y);
  let last=-1e9;
  for(const e of ends){ const y=Math.max(e.y,last+15); last=y;
    el('text',{x:X(9)+8,y:y+4,style:`fill:${e.c}`},svg)
      .textContent=`${e.n} ø${e.s.mean.toFixed(3)}`;}
  for(let i=0;i<10;i++) el('text',{x:X(i),y:h-B+16,
    'text-anchor':'middle'},svg).textContent=i+1;
  el('text',{x:(L+w-R-140)/2,y:h-6,'text-anchor':'middle'},svg)
    .textContent='Walk-Forward-Block (126 Bars, Refit ab 2021-07-20)';
})();

// ---- table ---------------------------------------------------------------
(function(){
  const t=document.getElementById('t_runs');
  let rows='<tr><th>Lauf</th><th>Faktoren</th><th>Kosten $</th>'+
    '<th>ridge IC</th><th>lasso IC</th></tr>';
  for(let n=1;n<=20;n++){
    const s=D.runs[n-1];
    const rr=D.worker.find(r=>r.scope=='run'&&r.method=='ridge'&&r.run==n)
      ||D.local.find(r=>r.label=='run'+n&&r.method=='ridge');
    const rl=D.worker.find(r=>r.scope=='run'&&r.method=='lasso'&&r.run==n)
      ||D.local.find(r=>r.label=='run'+n&&r.method=='lasso');
    rows+=`<tr><td>KG${String(n).padStart(2,'0')}</td>`+
      `<td>${s.persisted}</td><td>${s.cost.toFixed(2)}</td>`+
      `<td>${rr?rr.blockmean.toFixed(4):'—'}</td>`+
      `<td>${rl?rl.blockmean.toFixed(4):'—'}</td></tr>`;}
  t.innerHTML=rows;
})();
</script>
"""

off = np.array(data["block_corr"]["m"])
iu = np.triu_indices(len(data["block_corr"]["labels"]), k=1)
offv = off[iu]
finding = (f"Die Performance-Verläufe der Run-Bücher sind hochgradig "
           f"gleichläufig: mittlere Paar-Korrelation der Lasso-Block-IC-"
           f"Verläufe ρ = {offv.mean():.2f} "
           f"({(offv > 0.5).mean():.0%} der Paare über 0.5, nur "
           f"{(offv < 0).mean():.0%} negativ). Die Läufe finden verschiedene "
           f"Formeln, deren kombinierte Bücher aber in denselben Blöcken "
           f"gewinnen und verlieren — konsistent damit, dass der lineare "
           f"kumulative Kombinierer sättigt.")
html_finding = finding
n_new = sum(r["persisted"] for r in data["runs"])
cum_last = [r for r in data["local"]
            if r["label"] == "cum20" and r["method"] == "ridge"]
if cum_last:
    cum_ic = cum_last[0]["blockmean"]
else:
    wc = [r for r in data["worker"]
          if r["scope"] == "cum" and r["method"] == "ridge"]
    cum_ic = sorted(wc, key=lambda r: r["run"])[-1]["blockmean"]
html = html.replace("<strong>__FINDING_CORR__</strong>",
                    "<strong>Gleichläufige Performance trotz verschiedener "
                    "Formeln.</strong> " + html_finding)
html = html.replace("__KPI_FACTORS__", f"{n_new:,}".replace(",", " "))
html = html.replace("__KPI_CUMIC__", f"{cum_ic:.3f}")
html = html.replace("__DATA__", json.dumps(data))
out = OV / "kg_campaign_report.html"
out.write_text(html)
print("written", out, len(html) // 1024, "KB")
