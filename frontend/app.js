/* ============================================================================
 * HAWK frontend — FastAPI 백엔드 연동 클라이언트
 * 백엔드(/api/*)를 호출해 분석 결과를 받아 SVG 대시보드로 렌더링.
 * ========================================================================= */
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const COLORS={robustz:'#3ae6d6',iqr:'#ffb13c',iforest:'#9d7bff',forecast:'#c9ff46',levelshift:'#ff5d7a',matrixprofile:'#ff9be0'};
const DET_NAMES={robustz:'Robust Z-Score',iqr:'IQR Fence',iforest:'Isolation Forest',forecast:'Forecast Residual',levelshift:'Level Shift',matrixprofile:'Matrix Profile'};
const DET_FAMILY={robustz:'통계',iqr:'통계',iforest:'ML',forecast:'예측',levelshift:'변화점',matrixprofile:'SOTA'};

// API base 결정:
//  1) window.HAWK_API 전역변수가 있으면 그것 (배포 시 index.html에서 지정 가능)
//  2) URL 파라미터 ?api=https://... 가 있으면 그것
//  3) file:// 로 열었으면 로컬 백엔드(8000)
//  4) 그 외엔 같은 오리진(백엔드가 프론트를 서빙하는 단일배포 — 권장)
const API = (function(){
  if(typeof window!=='undefined' && window.HAWK_API) return window.HAWK_API;
  const p=new URLSearchParams(location.search).get('api');
  if(p) return p.replace(/\/$/,'');
  if(location.protocol==='file:') return 'http://localhost:8000';
  return '';
})();

const STATE={
  data:null,           // 백엔드 analyze_all 응답
  activeVar:null,
  activeDetectors:new Set(Object.keys(COLORS)),
  prevSignature:null, drift:null,
};

/* ---------------- 백엔드 헬스 체크 ---------------- */
async function checkHealth(){
  try{
    const r=await fetch(API+'/api/health');
    if(!r.ok) throw 0;
    const h=await r.json();
    $('#apiPill').textContent='API '+h.version+(h.darts_available?' · darts':'');
    $('#apiPill').classList.add('ok');
    STATE.keySet=!!h.openai_key_set;
  }catch(e){
    $('#apiPill').textContent='API 오프라인';
    $('#statusDot').className='dot err';
    $('#statusTxt').textContent='백엔드 미연결';
  }
}

/* ---------------- API 호출 ---------------- */
function showError(msg){
  const e=$('#errbox'); e.textContent='⚠ '+msg; e.classList.add('show');
  setTimeout(()=>e.classList.remove('show'),8000);
}
function setLoading(on){
  $('#loading').classList.toggle('show',on);
  if(on){ $('#main').classList.remove('show'); }
}

async function callAnalyze(formData){
  setLoading(true); $('#errbox').classList.remove('show');
  try{
    const r=await fetch(API+'/api/analyze',{method:'POST',body:formData});
    if(!r.ok){ const j=await r.json().catch(()=>({detail:'서버 오류'})); throw new Error(j.detail||('HTTP '+r.status)); }
    const data=await r.json();
    onData(data);
  }catch(e){
    setLoading(false);
    showError('분석 실패: '+e.message+(API?'':' (백엔드가 실행 중인지 확인하세요)'));
  }
}
async function callDemo(){
  setLoading(true); $('#errbox').classList.remove('show');
  const mode=$('#evalMode').value;
  try{
    const r=await fetch(API+`/api/demo?eval_mode=${mode}`);
    if(!r.ok) throw new Error('HTTP '+r.status);
    onData(await r.json());
  }catch(e){
    setLoading(false);
    showError('데모 분석 실패: '+e.message+(API?'':' (백엔드 미실행: uvicorn app.main:app)'));
  }
}

function onData(data){
  STATE.data=data;
  STATE.activeVar=data.active_var;
  // 드리프트
  const sig=data.signature;
  if(STATE.prevSignature){
    const pct=(a,b)=>b!==0?((a-b)/Math.abs(b)*100):0;
    STATE.drift={mean:pct(sig.mean,STATE.prevSignature.mean),std:pct(sig.std,STATE.prevSignature.std),
      shifted:Math.abs(pct(sig.mean,STATE.prevSignature.mean))>15||Math.abs(pct(sig.std,STATE.prevSignature.std))>25};
  } else STATE.drift=null;
  STATE.prevSignature=sig;
  setLoading(false);
  $('#controls').classList.add('show');
  $('#main').classList.add('show');
  renderAll();
}

/* ---------------- 렌더 ---------------- */
function curResult(){ return STATE.data.results[STATE.activeVar]; }

function renderAll(){
  const R=curResult(), meta=STATE.data.meta;
  const n=R.n;
  $('#statusDot').className='dot live';
  $('#statusTxt').textContent='탐지 완료';
  $('#hSeries').textContent=meta.value_cols.length;
  $('#hPoints').textContent=n;
  $('#footMeta').textContent=`${STATE.data.source} · ${n}×${meta.value_cols.length} · ${new Date().toLocaleTimeString('ko')}`;
  $('#dataSummary').textContent=`${STATE.data.source} · ${n} points · ${meta.value_cols.length} vars · 시간축 ${meta.time_col?'자동감지':'인덱스'}`;
  $('#seriesTitle').textContent=STATE.activeVar.toUpperCase();
  renderVarButtons(); renderDetToggles();
  renderKPIs(R,n); renderTrustNote(R); renderTripleMetrics(R);
  renderMainChart(R); renderHeat(R); renderVotes(R);
  renderConfusionTabs(R); renderPR(R); renderCompareTable(R);
  renderNSweep(R); renderWeights(R); renderHoldout(R);
  renderVarChart(); renderFeed(R);
  if(!$('.subtab.on')) switchSub('overview');
}

// 활성 탐지기 기준 합의 votes 재계산 (클라이언트)
function computeVotes(R){
  const keys=[...STATE.activeDetectors].filter(k=>R.detectors[k]);
  const n=R.n; const votes=new Int16Array(n);
  for(const k of keys){
    const sc=R.detectors[k].score, th=R.thresholds[k];
    for(let i=0;i<n;i++) if(sc[i]>=th) votes[i]++;
  }
  return {votes, nDet:keys.length};
}

function renderVarButtons(){
  const box=$('#varBtns');box.innerHTML='';
  for(const col of STATE.data.meta.value_cols){
    const b=document.createElement('button');
    b.className='varbtn'+(col===STATE.activeVar?' on':'');
    b.textContent=col;
    b.onclick=()=>{STATE.activeVar=col;renderAll();};
    box.appendChild(b);
  }
}
function renderDetToggles(){
  const box=$('#detToggles');box.innerHTML='';
  for(const key of Object.keys(COLORS)){
    const c=document.createElement('div');
    c.className='chip'+(STATE.activeDetectors.has(key)?' on':'');
    c.style.setProperty('--c',COLORS[key]);
    c.innerHTML=`<span class="sw"></span>${DET_NAMES[key]}`;
    c.onclick=()=>{
      if(STATE.activeDetectors.has(key)){ if(STATE.activeDetectors.size>1) STATE.activeDetectors.delete(key); }
      else STATE.activeDetectors.add(key);
      renderAll();
    };
    box.appendChild(c);
  }
}

function evalConsensus(R,N){
  const {votes}=computeVotes(R);
  const n=R.n;
  if(!R.inject) return {votes, evalM:null};
  const flags=new Int8Array(n); for(let i=0;i<n;i++) flags[i]=votes[i]>=N?1:0;
  return {votes, flags};
}

// 클라이언트 평가 (3지표) — 백엔드와 동일 로직
function evaluatePW(labels,flags){
  let tp=0,fp=0,fn=0,tn=0;
  for(let i=0;i<labels.length;i++){
    if(labels[i]&&flags[i])tp++; else if(!labels[i]&&flags[i])fp++;
    else if(labels[i]&&!flags[i])fn++; else tn++;
  }
  const p=tp+fp?tp/(tp+fp):0,r=tp+fn?tp/(tp+fn):0;
  return {tp,fp,fn,tn,precision:p,recall:r,f1:p+r?2*p*r/(p+r):0};
}
function evaluatePA(labels,flags,events){
  const pred=Int8Array.from(flags);
  for(const ev of events){let hit=false;const end=Math.min(labels.length,ev.pos+ev.len);
    for(let i=ev.pos;i<end;i++)if(flags[i]){hit=true;break;}
    if(hit)for(let i=ev.pos;i<end;i++)pred[i]=1;}
  return evaluatePW(labels,pred);
}
function evaluateAff(labels,flags,scale=20){
  const n=labels.length,gt=[],pr=[];
  for(let i=0;i<n;i++){if(labels[i])gt.push(i);if(flags[i])pr.push(i);}
  if(!gt.length||!pr.length)return{precision:0,recall:0,f1:0};
  const near=(x,arr)=>{let b=Infinity;for(const a of arr){const d=Math.abs(x-a);if(d<b)b=d;}return b;};
  let ps=0;for(const x of pr)ps+=1/(1+near(x,gt)/n*scale);const p=ps/pr.length;
  let rs=0;for(const x of gt)rs+=1/(1+near(x,pr)/n*scale);const r=rs/gt.length;
  return{precision:p,recall:r,f1:p+r?2*p*r/(p+r):0};
}

function renderKPIs(R,n){
  const N=parseInt($('#consensusN').value);
  const {votes}=computeVotes(R);
  let anom=0; for(const v of votes) if(v>=N) anom++;
  const rate=anom/n*100;
  let evalM=null;
  if(R.inject){const flags=new Int8Array(n);for(let i=0;i<n;i++)flags[i]=votes[i]>=N?1:0;
    evalM=evaluatePA(R.inject.labels,flags,R.inject.events);}
  const cards=[
    {lab:'탐지된 이상',val:anom,delta:`전체의 ${rate.toFixed(1)}%`,c:'var(--acid)'},
    {lab:'합의 F1 (PA)',val:evalM?evalM.f1.toFixed(2):'—',delta:evalM?`P ${evalM.precision.toFixed(2)} · R ${evalM.recall.toFixed(2)}`:'평가 비활성',c:'var(--cyan)'},
    {lab:'활성 탐지기',val:STATE.activeDetectors.size,delta:`${Object.keys(COLORS).length}개 중 선택`,c:'var(--violet)'},
    {lab:'드리프트',val:STATE.drift?(STATE.drift.shifted?'⚠ 변화':'안정'):'기준',
      delta:STATE.drift?`평균 ${STATE.drift.mean>=0?'+':''}${STATE.drift.mean.toFixed(0)}% · 산포 ${STATE.drift.std>=0?'+':''}${STATE.drift.std.toFixed(0)}%`:'첫 분석',
      c:STATE.drift&&STATE.drift.shifted?'var(--rose)':'var(--amber)'},
  ];
  $('#kpis').innerHTML=cards.map(c=>`<div class="kpi" style="--c:${c.c}">
    <div class="lab">${c.lab}</div><div class="val">${c.val}</div><div class="delta"><b>${c.delta}</b></div></div>`).join('');
  const tn=$('#tripN'); if(tn) tn.textContent=N;
}

function renderTrustNote(R){
  const el=$('#trustNote');
  const N=parseInt($('#consensusN').value);
  if(!R.inject){ el.style.display='block'; el.innerHTML='평가 모드가 꺼져 있어 탐지만 수행합니다. 정밀도·재현율을 보려면 <b>합성주입 자동평가</b>를 켜세요.'; return; }
  const {votes}=computeVotes(R);const n=R.n;
  const flags=new Int8Array(n);for(let i=0;i<n;i++)flags[i]=votes[i]>=N?1:0;
  const m=evaluatePA(R.inject.labels,flags,R.inject.events);
  const verdict=m.f1>=0.8?'✓ 신뢰할 만함':m.f1>=0.6?'△ 보통':'⚠ 주의 — 임계 조정 권장';
  el.style.display='block';
  el.innerHTML=`이 데이터에 <b>${R.inject.events.length}개</b>의 합성 이상을 주입하고 탐지기가 잡는지 검증했습니다.
    현재 합의(≥${N}표) 기준 point-adjusted F1 = <b>${m.f1.toFixed(2)}</b> → <b>${verdict}</b>`;
}

function renderTripleMetrics(R){
  const box=$('#tripleMetrics');
  if(!R.inject){ box.innerHTML='<div style="grid-column:1/4;padding:24px;text-align:center;color:var(--dim);font-family:var(--mono);font-size:12px">평가 모드를 켜면 3개 지표가 표시됩니다.</div>'; $('#metricCritique').textContent=''; return; }
  const N=parseInt($('#consensusN').value);
  const {votes}=computeVotes(R);const n=R.n;
  const flags=new Int8Array(n);for(let i=0;i<n;i++)flags[i]=votes[i]>=N?1:0;
  const pw=evaluatePW(R.inject.labels,flags), pa=evaluatePA(R.inject.labels,flags,R.inject.events), aff=evaluateAff(R.inject.labels,flags);
  const cards=[
    {lab:'Point-wise F1',m:pw,ref:'보정 없는 정직한 기준선',warn:'가장 보수적. 구간 이상을 점 단위로 엄격 평가.',flag:false},
    {lab:'Point-adjusted F1',m:pa,ref:'Xu et al. 2018 · 구간 보정',warn:'⚠ Kim et al.(2022): 무작위 점수도 부풀려질 수 있음. 단독 신뢰 금물.',flag:true},
    {lab:'Affiliation F1',m:aff,ref:'Huet et al. 2022 · 시간거리',warn:'파라미터 없음. 예측–정답 시간적 근접도 기반.',flag:false},
  ];
  box.innerHTML=cards.map(c=>`<div class="metricbox${c.flag?' flag':''}">
    <div class="mlab">${c.lab}</div><div class="mf1">${c.m.f1.toFixed(3)}</div>
    <div class="mpr">P ${c.m.precision.toFixed(3)} · R ${c.m.recall.toFixed(3)}</div>
    <div class="${c.flag?'warn':'mpr'}" style="${c.flag?'':'color:var(--dimmer)'}">${c.warn}</div></div>`).join('');
  const gap=pa.f1-pw.f1;
  $('#metricCritique').innerHTML=`동일한 탐지 결과인데 지표마다 F1이 다릅니다 — Point-wise <b>${pw.f1.toFixed(2)}</b>, Adjusted <b>${pa.f1.toFixed(2)}</b>, Affiliation <b>${aff.f1.toFixed(2)}</b>.
    Point-adjusted가 point-wise보다 <b>${gap>=0?'+':''}${gap.toFixed(2)}</b> 높습니다. Kim et al.(2022)은 이 보정이 무작위 예측조차 고득점시킬 수 있다고 지적했습니다 — 그래서 <b>세 지표를 함께</b> 봐야 합니다.`;
  $('#tripN').textContent=N;
}

/* ---- SVG helpers ---- */
function svgOpen(w,h,fixed){return `<svg viewBox="0 0 ${w} ${h}"${fixed?'':' preserveAspectRatio="none"'}>`;}
function sY(v,mn,mx,top,bot){return mx===mn?(top+bot)/2:bot-(v-mn)/(mx-mn)*(bot-top);}

function renderMainChart(R){
  const s=R.target, n=R.n;
  const W=1000,H=300,padL=46,padR=12,padT=14,padB=24;
  let mn=Infinity,mx=-Infinity;for(const v of s){if(v<mn)mn=v;if(v>mx)mx=v;}
  const pad=(mx-mn)*0.08||1; mn-=pad;mx+=pad;
  const x=i=>padL+(i/(n-1))*(W-padL-padR), y=v=>sY(v,mn,mx,padT,H-padB);
  let svg=svgOpen(W,H);
  for(let g=0;g<=4;g++){const yy=padT+(g/4)*(H-padT-padB);
    svg+=`<line class="gridline" x1="${padL}" y1="${yy}" x2="${W-padR}" y2="${yy}"/>`;
    svg+=`<text class="axis-t" x="${padL-6}" y="${yy+3}" text-anchor="end">${(mx-(g/4)*(mx-mn)).toFixed(1)}</text>`;}
  if(R.inject){const lab=R.inject.labels;let i=0;while(i<n){if(lab[i]){let j=i;while(j<n&&lab[j])j++;
    svg+=`<rect x="${x(i)}" y="${padT}" width="${Math.max(1.5,x(j-1)-x(i))}" height="${H-padT-padB}" fill="rgba(255,93,122,.08)"/>`;i=j;}else i++;}}
  const pred=R.preds&&R.preds.forecast;
  if(pred&&STATE.activeDetectors.has('forecast')){let d='';for(let i=0;i<n;i++){if(pred[i]==null)continue;d+=(d?'L':'M')+x(i).toFixed(1)+' '+y(pred[i]).toFixed(1)+' ';}
    svg+=`<path d="${d}" fill="none" stroke="var(--violet)" stroke-width="1" opacity=".55" stroke-dasharray="3 3"/>`;}
  let d='';for(let i=0;i<n;i++)d+=(i?'L':'M')+x(i).toFixed(1)+' '+y(s[i]).toFixed(1)+' ';
  svg+=`<path d="${d}" fill="none" stroke="var(--cyan)" stroke-width="1.4"/>`;
  const N=parseInt($('#consensusN').value);const {votes}=computeVotes(R);
  for(let i=0;i<n;i++)if(votes[i]>=N){const r=3+votes[i]*0.7;
    svg+=`<circle cx="${x(i).toFixed(1)}" cy="${y(s[i]).toFixed(1)}" r="${r}" fill="var(--acid)" opacity=".85"/>`;
    svg+=`<circle cx="${x(i).toFixed(1)}" cy="${y(s[i]).toFixed(1)}" r="${r}" fill="none" stroke="var(--acid)" stroke-width="1" opacity=".4"><animate attributeName="r" from="${r}" to="${r+5}" dur="1.5s" repeatCount="indefinite"/><animate attributeName="opacity" from=".4" to="0" dur="1.5s" repeatCount="indefinite"/></circle>`;}
  if(R.inject)for(let i=0;i<n;i++)if(R.inject.labels[i])svg+=`<circle cx="${x(i).toFixed(1)}" cy="${y(s[i]).toFixed(1)}" r="2" fill="var(--rose)"/>`;
  svg+='</svg>';
  $('#mainChart').innerHTML=svg; $('#mainChart').querySelector('svg').style.height='300px';
}

function renderHeat(R){
  const keys=[...STATE.activeDetectors].filter(k=>R.detectors[k]);
  const n=R.n,W=1000,rowH=28,H=keys.length*rowH+10,labW=140;
  let svg=svgOpen(W,H,true);
  keys.forEach((k,ri)=>{const yTop=6+ri*rowH;
    // 색칩
    svg+=`<rect x="6" y="${yTop+(rowH-4)/2-4}" width="9" height="9" rx="2" fill="${COLORS[k]}"/>`;
    // 탐지기 이름 (밝게)
    svg+=`<text x="20" y="${yTop+rowH/2+1}" fill="var(--ink2)" font-family="var(--mono)" font-size="11">${DET_NAMES[k]}</text>`;
    const sc=R.detectors[k].score,cw=(W-labW)/n;
    // 행 배경 트랙 (빈 구간도 행 위치를 인지하게)
    svg+=`<rect x="${labW}" y="${yTop}" width="${W-labW}" height="${rowH-6}" fill="var(--grid)" opacity="0.4" rx="2"/>`;
    for(let i=0;i<n;i++){const a=sc[i];if(a<0.05)continue;
      svg+=`<rect x="${labW+i*cw}" y="${yTop}" width="${Math.max(1,cw+0.5)}" height="${rowH-6}" fill="${COLORS[k]}" opacity="${(0.15+a*0.85).toFixed(2)}"/>`;}});
  svg+='</svg>'; $('#heatChart').innerHTML=svg; $('#heatChart').querySelector('svg').style.height=H+'px';
}

function renderVotes(R){
  const {votes,nDet}=computeVotes(R);
  const hist=new Array(nDet+1).fill(0);for(const v of votes)hist[v]++;
  const W=400,H=210,padB=34,padL=12,padT=18,mx=Math.max(...hist),bw=(W-padL*2)/(nDet+1);
  let svg=svgOpen(W,H,true);
  hist.forEach((c,i)=>{
    const h=mx?(c/mx)*(H-padB-padT):0;
    // 0표=회색(정상), N이 클수록 파랑→진한 파랑 (신뢰도 직관화)
    const col=i===0?'var(--dimmer)':'var(--accent)';
    const op=i===0?0.5:(0.45+i/(nDet)*0.55);
    const bx=padL+i*bw+5, bwid=bw-10;
    // 배경 트랙
    svg+=`<rect x="${bx}" y="${padT}" width="${bwid}" height="${H-padB-padT}" fill="var(--grid)" rx="3" opacity="0.5"/>`;
    // 값 막대
    if(h>0)svg+=`<rect x="${bx}" y="${H-padB-h}" width="${bwid}" height="${h}" fill="${col}" opacity="${op.toFixed(2)}" rx="3"/>`;
    // x축 라벨
    svg+=`<text x="${padL+i*bw+bw/2}" y="${H-padB+15}" text-anchor="middle" fill="var(--dim)" font-family="var(--mono)" font-size="10.5">${i}표</text>`;
    // 값 라벨
    if(c>0)svg+=`<text x="${padL+i*bw+bw/2}" y="${H-padB-h-6}" text-anchor="middle" fill="${i===0?'var(--dim)':'var(--ink)'}" font-family="var(--mono)" font-size="11" font-weight="500">${c}</text>`;
  });
  svg+='</svg>'; $('#votesChart').innerHTML=svg; $('#votesChart').querySelector('svg').style.height=H+'px';
}

let cmActive='_consensus';
function renderConfusionTabs(R){
  const tabs=$('#cmTabs');tabs.innerHTML='';
  const items=[['_consensus','합의'],...[...STATE.activeDetectors].filter(k=>R.detectors[k]).map(k=>[k,DET_NAMES[k].split(' ')[0]])];
  items.forEach(([k,lab])=>{const b=document.createElement('button');b.textContent=lab;b.className=(k===cmActive?'on':'');
    b.onclick=()=>{cmActive=k;renderConfusion(R);renderConfusionTabs(R);};tabs.appendChild(b);});
  renderConfusion(R);
}
function renderConfusion(R){
  if(!R.inject){$('#confusion').innerHTML='<div style="grid-column:1/4;padding:30px;text-align:center;color:var(--dim);font-family:var(--mono);font-size:12px">평가 모드를 켜면 표시됩니다.</div>';$('#cmNote').textContent='';return;}
  const n=R.n;let flags;
  if(cmActive==='_consensus'){const N=parseInt($('#consensusN').value);const {votes}=computeVotes(R);flags=new Int8Array(n);for(let i=0;i<n;i++)flags[i]=votes[i]>=N?1:0;}
  else{flags=new Int8Array(n);const sc=R.detectors[cmActive].score,th=R.thresholds[cmActive];for(let i=0;i<n;i++)flags[i]=sc[i]>=th?1:0;}
  const m=evaluatePA(R.inject.labels,flags,R.inject.events);
  $('#confusion').innerHTML=`<div class="hd"></div><div class="hd">예측: 정상</div><div class="hd">예측: 이상</div>
    <div class="hd">실제 이상</div><div class="cell fn"><div class="big">${m.fn}</div><div class="lab">FN · 놓침</div></div><div class="cell tp"><div class="big">${m.tp}</div><div class="lab">TP · 정탐</div></div>
    <div class="hd">실제 정상</div><div class="cell tn"><div class="big">${m.tn}</div><div class="lab">TN</div></div><div class="cell fp"><div class="big">${m.fp}</div><div class="lab">FP · 오탐</div></div>`;
  const nm=cmActive==='_consensus'?`합의(≥${$('#consensusN').value}표)`:DET_NAMES[cmActive];
  $('#cmNote').innerHTML=`<b>${nm}</b> · Precision ${m.precision.toFixed(3)} · Recall ${m.recall.toFixed(3)} · F1 ${m.f1.toFixed(3)} <span style="color:var(--dimmer)">(point-adjust)</span>`;
}

function renderPR(R){
  if(!R.inject){$('#prChart').innerHTML='<div style="padding:40px;text-align:center;color:var(--dim);font-family:var(--mono);font-size:12px">평가 모드 필요</div>';return;}
  const W=400,H=240,padL=42,padB=34,padT=12,padR=12;
  const x=v=>padL+v*(W-padL-padR),y=v=>(H-padB)-v*(H-padB-padT);
  let svg=svgOpen(W,H,true);
  for(let g=0;g<=4;g++){const gv=g/4;
    svg+=`<line class="gridline" x1="${padL}" y1="${y(gv)}" x2="${W-padR}" y2="${y(gv)}"/>`;
    svg+=`<text class="axis-t" x="${padL-5}" y="${y(gv)+3}" text-anchor="end">${gv.toFixed(1)}</text>`;
    svg+=`<text class="axis-t" x="${x(gv)}" y="${H-padB+14}" text-anchor="middle">${gv.toFixed(1)}</text>`;}
  svg+=`<text class="axis-t" x="${W/2}" y="${H-4}" text-anchor="middle">Recall →</text>`;
  for(const k of STATE.activeDetectors){const roc=R.detectors[k]&&R.detectors[k].roc;if(!roc)continue;
    let d='';roc.forEach((p,i)=>{d+=(i?'L':'M')+x(p.recall).toFixed(1)+' '+y(p.precision).toFixed(1)+' ';});
    svg+=`<path d="${d}" fill="none" stroke="${COLORS[k]}" stroke-width="1.4" opacity=".8"/>`;
    const b=R.detectors[k].best;if(b&&b.recall!=null)svg+=`<text x="${x(b.recall)}" y="${y(b.precision)-6}" fill="${COLORS[k]}" font-family="var(--mono)" font-size="13" text-anchor="middle">★</text>`;}
  svg+='</svg>'; $('#prChart').innerHTML=svg; $('#prChart').querySelector('svg').style.height=H+'px';
}

// 합의 N 튜닝 곡선 — 백엔드 consensus_sweep 사용
function renderNSweep(R){
  const box=$('#nSweepChart'), note=$('#nSweepNote'), hint=$('#bestNHint');
  if(!R.inject||!R.consensus_sweep){
    box.innerHTML='<div style="padding:40px;text-align:center;color:var(--dim);font-family:var(--mono);font-size:12px">평가 모드를 켜면 표시됩니다.</div>';
    note.textContent=''; hint.textContent='—'; return;
  }
  const sw=R.consensus_sweep, curve=sw.curve, bestN=sw.best.n;
  const curN=parseInt($('#consensusN').value);
  const W=400,H=240,padL=42,padB=40,padT=14,padR=14;
  const nDet=curve.length;
  const x=i=>padL+(nDet<=1?0:i/(nDet-1))*(W-padL-padR);
  const y=v=>(H-padB)-v*(H-padB-padT);
  let svg=svgOpen(W,H,true);
  for(let g=0;g<=4;g++){const gv=g/4;
    svg+=`<line class="gridline" x1="${padL}" y1="${y(gv)}" x2="${W-padR}" y2="${y(gv)}"/>`;
    svg+=`<text class="axis-t" x="${padL-5}" y="${y(gv)+3}" text-anchor="end">${gv.toFixed(1)}</text>`;}
  // F1 곡선
  let d='';curve.forEach((p,i)=>{d+=(i?'L':'M')+x(i).toFixed(1)+' '+y(p.f1).toFixed(1)+' ';});
  svg+=`<path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2"/>`;
  // 점 + N 라벨
  curve.forEach((p,i)=>{
    const isBest=p.n===bestN, isCur=p.n===curN;
    const c=isBest?'var(--green)':isCur?'var(--amber)':'var(--accent)';
    const rad=isBest||isCur?5:3.2;
    svg+=`<circle cx="${x(i)}" cy="${y(p.f1)}" r="${rad}" fill="${c}"/>`;
    svg+=`<text class="axis-t" x="${x(i)}" y="${H-padB+15}" text-anchor="middle" fill="${isBest?'var(--green)':isCur?'var(--amber)':'var(--dim)'}">${p.n}표</text>`;
    if(isBest)svg+=`<text x="${x(i)}" y="${y(p.f1)-10}" text-anchor="middle" fill="var(--green)" font-family="var(--mono)" font-size="9">${p.f1.toFixed(2)}</text>`;
  });
  svg+=`<text class="axis-t" x="${W/2}" y="${H-4}" text-anchor="middle">합의 임계 N →</text>`;
  svg+='</svg>'; box.innerHTML=svg; box.querySelector('svg').style.height=H+'px';
  hint.innerHTML=`추천 <b style="color:var(--green)">${bestN}표</b> · 현재 <b style="color:var(--amber)">${curN}표</b>`;
  const diff=bestN!==curN;
  note.innerHTML=`이 데이터의 최적 합의 임계는 <b>${bestN}표</b>(F1 ${sw.best.f1.toFixed(3)})입니다.`
    +(diff?` 현재 설정은 ${curN}표이므로, 합의 임계를 <b>${bestN}표</b>로 바꾸면 균형이 더 좋아집니다.`:` 현재 설정과 일치합니다.`);
}

// 탐지기 가중치 막대
function renderWeights(R){
  const box=$('#weightChart');
  if(!R.weights){box.innerHTML='<div style="padding:40px;text-align:center;color:var(--dim);font-family:var(--mono);font-size:12px">평가 모드를 켜면 표시됩니다.</div>';return;}
  const entries=Object.entries(R.weights)
    .filter(([k])=>R.detectors[k])
    .sort((a,b)=>b[1]-a[1]);
  const maxW=Math.max(...entries.map(e=>e[1]),1);
  let html='';
  for(const [k,w] of entries){
    const pct=(w/maxW*100).toFixed(0);
    const above=w>=1;
    html+=`<div style="display:flex;align-items:center;gap:12px;margin:9px 0">
      <div style="width:120px;font-size:12px;color:var(--ink2);display:flex;align-items:center;gap:8px">
        <span style="width:8px;height:8px;border-radius:2px;background:${COLORS[k]}"></span>${DET_NAMES[k].split(' ')[0]}</div>
      <div style="flex:1;height:14px;background:var(--grid);border-radius:3px;overflow:hidden">
        <div style="height:100%;width:${pct}%;background:${COLORS[k]};opacity:${above?1:.5};border-radius:3px"></div></div>
      <div style="width:46px;text-align:right;font-family:var(--mono);font-size:11.5px;color:${above?'var(--ink)':'var(--dim)'}">${w.toFixed(2)}</div>
    </div>`;
  }
  box.innerHTML=html;
}

// Train/Test 일반화 갭 — 백엔드 holdout 사용
function renderHoldout(R){
  const box=$('#holdoutChart'), note=$('#holdoutNote'), split=$('#holdoutSplit');
  if(!R.holdout){
    box.innerHTML='<div style="padding:40px;text-align:center;color:var(--dim);font-family:var(--mono);font-size:12px">평가 모드를 켜면 표시됩니다.</div>';
    note.textContent=''; split.textContent='—'; return;
  }
  const H=R.holdout, det=H.detectors;
  split.innerHTML=`train ${H.split.n_train} · test ${H.split.n_test} (50/50 시간 분할)`;
  // 탐지기별 train→test 막대 (정렬: test F1 내림차순)
  const rows=Object.keys(det).filter(k=>R.detectors[k])
    .map(k=>({k, ...det[k]}))
    .sort((a,b)=>b.test.f1-a.test.f1);
  const W=1000, rowH=46, padTop=30, H2=rows.length*rowH+padTop+10, labW=150, barL=labW, barW=W-labW-180;
  let svg=svgOpen(W,H2,true);
  // 헤더
  svg+=`<text x="${barL}" y="18" fill="var(--dim)" font-family="var(--mono)" font-size="10">0.0</text>`;
  svg+=`<text x="${barL+barW}" y="18" fill="var(--dim)" font-family="var(--mono)" font-size="10" text-anchor="end">1.0</text>`;
  svg+=`<text x="${barL+barW+12}" y="18" fill="var(--dim)" font-family="var(--mono)" font-size="10">train→test · gap</text>`;
  rows.forEach((r,i)=>{
    const yy=padTop+i*rowH;
    const xtr=barL+r.train.f1*barW, xte=barL+r.test.f1*barW;
    const col=COLORS[r.k];
    const bigGap=Math.abs(r.gap)>=0.25;
    // 라벨
    svg+=`<rect x="6" y="${yy+10}" width="9" height="9" rx="2" fill="${col}"/>`;
    svg+=`<text x="20" y="${yy+18}" fill="var(--ink2)" font-family="var(--mono)" font-size="11">${DET_NAMES[r.k].split(' ')[0]}</text>`;
    // 트랙
    svg+=`<line x1="${barL}" y1="${yy+15}" x2="${barL+barW}" y2="${yy+15}" stroke="var(--grid)" stroke-width="6" stroke-linecap="round"/>`;
    // train→test 연결선
    svg+=`<line x1="${xtr}" y1="${yy+15}" x2="${xte}" y2="${yy+15}" stroke="${bigGap?'var(--rose)':col}" stroke-width="2" opacity="0.5"/>`;
    // train 점 (빈 원), test 점 (채운 원)
    svg+=`<circle cx="${xtr}" cy="${yy+15}" r="5" fill="var(--bg)" stroke="${col}" stroke-width="2"/>`;
    svg+=`<circle cx="${xte}" cy="${yy+15}" r="5" fill="${col}"/>`;
    // 수치
    svg+=`<text x="${barL+barW+12}" y="${yy+18}" fill="${bigGap?'var(--rose)':'var(--dim)'}" font-family="var(--mono)" font-size="11">${r.train.f1.toFixed(2)}→${r.test.f1.toFixed(2)} · ${r.gap>=0?'+':''}${r.gap.toFixed(2)}</text>`;
  });
  svg+='</svg>'; box.innerHTML=svg; box.querySelector('svg').style.height=H2+'px';
  // 합의 결과 요약 + 과적합 경고
  const c=H.consensus;
  const worst=rows.reduce((a,b)=>Math.abs(b.gap)>Math.abs(a.gap)?b:a,rows[0]);
  let msg=`합의(≥${c.n}표)는 train F1 <b>${c.train.f1.toFixed(2)}</b> → test F1 <b>${c.test.f1.toFixed(2)}</b> (gap ${c.gap>=0?'+':''}${c.gap.toFixed(2)})입니다. `;
  if(Math.abs(worst.gap)>=0.25){
    msg+=`<b style="color:var(--rose)">${DET_NAMES[worst.k].split(' ')[0]}</b>의 gap이 ${worst.gap>=0?'+':''}${worst.gap.toFixed(2)}로 커서, 이 탐지기는 train 구간에 과적합된 신호입니다. 합의 투표가 이런 단일 탐지기 편향을 완화합니다.`;
  } else {
    msg+=`모든 탐지기의 gap이 작아 일반화가 안정적입니다. ○ train · ● test.`;
  }
  note.innerHTML=msg;
}

// 실제 라벨 벤치마크 실행 + 렌더
async function runBenchmark(){
  const btn=$('#benchBtn'), body=$('#benchBody');
  btn.disabled=true; btn.classList.add('loading-spin');
  body.innerHTML='<div class="insight-placeholder"><div class="ico">▦</div><div>실제 라벨 데이터에서 6개 탐지기를 평가하는 중…</div></div>';
  try{
    const r=await fetch(API+'/api/benchmark');
    if(!r.ok) throw new Error('HTTP '+r.status);
    const d=await r.json();
    body.innerHTML=renderBenchHTML(d);
  }catch(e){
    body.innerHTML=`<div class="insight-error">벤치마크 실패<div class="err-code">${e.message}</div></div>`;
  }finally{
    btn.disabled=false; btn.classList.remove('loading-spin'); btn.textContent='벤치마크 실행';
  }
}

function renderBenchHTML(d){
  // 분석 탭 합성 F1과 비교용: 현재 데이터 합의 PA F1
  let synthF1=null;
  if(STATE.data){const R=STATE.data.results[STATE.activeVar];
    if(R&&R.inject){const N=parseInt($('#consensusN').value);const{votes}=computeVotes(R);
      const fl=new Int8Array(R.n);for(let i=0;i<R.n;i++)fl[i]=votes[i]>=N?1:0;
      synthF1=evaluatePA(R.inject.labels,fl,R.inject.events).f1;}}
  let html='';
  if(synthF1!=null){
    html+=`<div class="trust-bar" style="margin:0 0 18px">참고 · 분석 탭의 <b>합성</b> 합의 F1은 <b>${synthF1.toFixed(2)}</b>입니다. 아래 <b>실제 라벨</b> 점수와 비교하세요 — 실제가 낮다면 합성 평가가 낙관적이었다는 뜻입니다.</div>`;
  }
  for(const [key,ds] of Object.entries(d.datasets)){
    const cb=ds.consensus_best;
    const top=Object.entries(ds.detectors).sort((a,b)=>b[1].f1-a[1].f1);
    html+=`<div class="panel" style="margin-top:14px">
      <div class="panel-h">
        <div class="t">${ds.title}</div>
        <div class="hint">${ds.ref} · n=${ds.n} · 이상 ${(ds.anomaly_ratio*100).toFixed(1)}%</div>
      </div>
      <div style="font-size:13px;color:var(--ink2);margin-bottom:14px">${ds.desc}</div>
      <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:16px">
        <span style="font-size:12px;color:var(--dim)">합의 최적 (실제 라벨)</span>
        <span style="font-size:24px;font-weight:700;color:var(--accent)">F1 ${cb.f1.toFixed(2)}</span>
        <span style="font-family:var(--mono);font-size:12px;color:var(--dim)">≥${cb.n}표 · P ${cb.precision.toFixed(2)} · R ${cb.recall.toFixed(2)}</span>
      </div>`;
    // 탐지기 막대
    for(const [dk,dv] of top){
      const pct=(dv.f1*100).toFixed(0);
      html+=`<div style="display:flex;align-items:center;gap:12px;margin:7px 0">
        <div style="width:130px;font-size:12px;color:var(--ink2);display:flex;align-items:center;gap:8px">
          <span style="width:8px;height:8px;border-radius:2px;background:${COLORS[dk]}"></span>${DET_NAMES[dk].split(' ')[0]}</div>
        <div style="flex:1;height:13px;background:var(--grid);border-radius:3px;overflow:hidden">
          <div style="height:100%;width:${pct}%;background:${COLORS[dk]};border-radius:3px"></div></div>
        <div style="width:42px;text-align:right;font-family:var(--mono);font-size:11.5px;color:var(--ink2)">${dv.f1.toFixed(2)}</div>
      </div>`;
    }
    html+=`</div>`;
  }
  html+=`<div class="note" style="margin-top:16px">실제 라벨 데이터에서는 drift·점진 변화가 많은 시나리오(장비 온도)일수록 F1이 낮습니다. 합성 평가만으로는 보이지 않던 <b>실제 난이도</b>가 드러납니다.</div>`;
  return html;
}
const benchBtn=document.querySelector('#benchBtn'); if(benchBtn) benchBtn.onclick=runBenchmark;

function renderCompareTable(R){
  let html=`<thead><tr><th>탐지기</th><th>계열</th><th>F1</th><th>Precision</th><th>Recall</th><th>임계값</th><th>성능</th></tr></thead><tbody>`;
  const rows=Object.keys(COLORS).map(k=>{const b=R.detectors[k].best;return {k,f1:b&&b.f1!=null?b.f1:-1,b};});
  rows.sort((a,b)=>b.f1-a.f1);
  for(const {k,b} of rows){const f1=b&&b.f1!=null?b.f1:null;const active=STATE.activeDetectors.has(k);
    html+=`<tr style="opacity:${active?1:.4}">
      <td><div class="nm" style="--c:${COLORS[k]}"><span class="d"></span>${DET_NAMES[k]}</div></td>
      <td><span class="fam">${DET_FAMILY[k]}</span></td>
      <td>${f1!=null?f1.toFixed(3):'—'}</td><td>${b&&b.precision!=null?b.precision.toFixed(3):'—'}</td>
      <td>${b&&b.recall!=null?b.recall.toFixed(3):'—'}</td><td>${b?b.th.toFixed(2):'—'}</td>
      <td><span class="bar" style="--c:${COLORS[k]};width:${f1!=null?(f1*120).toFixed(0):0}px"></span></td></tr>`;}
  html+='</tbody>'; $('#compareTable').innerHTML=html;
}

function renderVarChart(){
  const N=parseInt($('#consensusN').value);
  const data=STATE.data.meta.value_cols.map(c=>{const R=STATE.data.results[c];const {votes}=computeVotes(R);
    let a=0;for(const v of votes)if(v>=N)a++;return {col:c,rate:a/R.n*100,count:a};})
    .sort((p,q)=>q.rate-p.rate);
  const mx=Math.max(...data.map(d=>d.rate),1),W=1000,barH=38,H=data.length*barH+12,labW=130,trackW=W-labW-120;
  let svg=svgOpen(W,H,true);
  data.forEach((d,i)=>{const yy=10+i*barH,w=Math.max(3,(d.rate/mx)*trackW),active=d.col===STATE.activeVar;
    const col=active?'var(--accent)':'var(--cyan)';
    // 라벨
    svg+=`<text x="6" y="${yy+(barH-12)/2+5}" fill="${active?'var(--accent)':'var(--ink2)'}" font-family="var(--mono)" font-size="12" font-weight="${active?'600':'400'}">${d.col}</text>`;
    // 배경 트랙
    svg+=`<rect x="${labW}" y="${yy+4}" width="${trackW}" height="${barH-14}" fill="var(--grid)" rx="4"/>`;
    // 값 막대
    svg+=`<rect x="${labW}" y="${yy+4}" width="${w}" height="${barH-14}" fill="${col}" opacity="${active?'1':'0.7'}" rx="4"/>`;
    // 수치
    svg+=`<text x="${labW+trackW+10}" y="${yy+(barH-12)/2+5}" fill="${active?'var(--ink)':'var(--dim)'}" font-family="var(--mono)" font-size="12">${d.rate.toFixed(1)}% · ${d.count}건</text>`;});
  svg+='</svg>'; $('#varChart').innerHTML=svg; $('#varChart').querySelector('svg').style.height=H+'px';
}

/* ---- 근거 텍스트: 백엔드 /api/explain 호출 ---- */
async function renderFeed(R){
  const N=parseInt($('#consensusN').value);
  try{
    const fd=new FormData();fd.append('var',STATE.activeVar);fd.append('consensus_n',N);
    const r=await fetch(API+'/api/explain',{method:'POST',body:fd});
    if(!r.ok)throw 0;
    const {explanations}=await r.json();
    if(!explanations.length){$('#feed').innerHTML='<div style="padding:24px;color:var(--dim);font-family:var(--mono);font-size:12px">현재 합의 기준에서 탐지된 이상 구간이 없습니다.</div>';return;}
    $('#feed').innerHTML=explanations.map(e=>{
      const sev=e.severity==='심각'?'hi':e.severity==='주의'?'md':'lo';
      return `<div class="evt"><div class="ts">${e.time}</div>
        <div class="desc">${e.type}. <b>${e.n_agree}개</b> 탐지기 동의 (${e.agree.join(', ')}). 관측값 <b>${e.value}</b>.</div>
        <div class="sev ${sev}">${e.severity} · ${e.votes}표</div></div>`;}).join('');
  }catch(e){ $('#feed').innerHTML='<div style="padding:24px;color:var(--dim);font-family:var(--mono);font-size:12px">근거 생성 실패 (백엔드 연결 확인).</div>'; }
}

/* ---- 방법론 ---- */
const METHODOLOGY=[
  {key:'robustz',ref:'Iglewicz & Hoaglin (1993)',desc:'중앙값과 MAD(중앙값 절대편차)로 각 점의 표준화 편차를 계산. 평균 대신 중앙값을 써서 이상치 자체가 임계값을 오염시키는 문제를 막는다.',why:'가장 단순하면서도 강건. 점 이상의 1차 방어선.',formula:`<span class="var">S<sub>i</sub></span> <span class="op">=</span> <span class="frac"><span class="num">|x<sub>i</sub> − median(X)|</span><span class="den">1.4826 · MAD</span></span>`},
  {key:'iqr',ref:'Tukey (1977)',desc:'사분위 범위(IQR=Q3−Q1)의 1.5배 울타리 밖을 이상으로 본다. 박스플롯 원리.',why:'분포 가정이 없고 비대칭 분포에 강해 임의 CSV에 안전.',formula:`<span class="dim">이상 ⟺</span> x<sub>i</sub> <span class="op">&lt;</span> Q<sub>1</sub>−1.5·IQR <span class="dim">또는</span> x<sub>i</sub> <span class="op">&gt;</span> Q<sub>3</sub>+1.5·IQR`},
  {key:'iforest',ref:'Liu, Ting & Zhou (2008), ICDM',desc:'무작위 분할 트리에서 한 점이 고립되기까지의 평균 경로길이를 측정. 이상치는 적은 분할로 고립된다. scikit-learn 정식 구현 + 슬라이딩 윈도우 특징.',why:'다변량·맥락 이상에 강한 ML 표준.',formula:`<span class="var">s</span>(x) <span class="op">=</span> 2<sup>−E[h(x)]/c(n)</sup>`},
  {key:'forecast',ref:'Box & Jenkins (1970) · 수업 회귀모형',desc:'과거 p개 값으로 다음 값을 자기회귀(AR) 선형예측하고, 잔차가 크면 이상. 정규방정식+릿지로 계수 추정.',why:'수업에서 배운 예측을 이상탐지로 직접 연결. 시간 구조를 명시적으로 사용.',formula:`<span class="var">ŷ<sub>t</sub></span> <span class="op">=</span> c + Σ<sub>k=1</sub><sup>p</sup> φ<sub>k</sub>·y<sub>t−k</sub><span class="dim">,&nbsp;</span>r<sub>t</sub>=|y<sub>t</sub>−ŷ<sub>t</sub>|/(1.4826·MAD)`},
  {key:'levelshift',ref:'Page (1954), CUSUM',desc:'시점 좌우 윈도우의 평균 차이를 풀드 표준편차로 표준화. 레벨이 통째로 바뀌는 체제 전환을 포착.',why:'다른 탐지기가 놓치는 구조적 변화점 담당.',formula:`<span class="var">S<sub>i</sub></span> <span class="op">=</span> <span class="frac"><span class="num">|μ<sub>right</sub> − μ<sub>left</sub>|</span><span class="den">√((σ²<sub>L</sub>+σ²<sub>R</sub>)/2)</span></span>`},
  {key:'matrixprofile',ref:'Yeh et al. (2016), ICDM',desc:'길이 m 모든 서브시퀀스에 대해 자기 자신을 제외한 최근접 이웃까지의 z-정규화 거리를 계산. 거리가 큰 구간(discord)이 패턴 이상.',why:'시계열 이상탐지 SOTA baseline. 파라미터가 m 하나뿐이고 도메인 가정이 없다.',formula:`<span class="var">P<sub>i</sub></span> <span class="op">=</span> min<sub>|i−j|&gt;m/2</sub> ‖ẑ(T<sub>i,m</sub>) − ẑ(T<sub>j,m</sub>)‖<sub>2</sub>`},
];
const EVALDOCS=[
  {name:'Point-wise F1',ref:'표준 분류 지표',desc:'각 시점을 독립적으로 정상/이상 분류로 보고 TP·FP·FN을 센다. 가장 보수적이고 해석이 명확하나, 구간 이상의 위치 오차에 가혹하다.',formula:`P=<span class="frac"><span class="num">TP</span><span class="den">TP+FP</span></span> R=<span class="frac"><span class="num">TP</span><span class="den">TP+FN</span></span> F1=<span class="frac"><span class="num">2PR</span><span class="den">P+R</span></span>`},
  {name:'Point-adjusted F1',ref:'Xu et al. (2018) · 비판: Kim et al. (2022)',desc:'정답 구간 안에서 하나라도 탐지하면 구간 전체를 정탐으로 인정. 널리 쓰이나 Kim et al.(2022)은 무작위·자명한 예측조차 높은 F1을 만든다고 증명했다. 본 도구는 이 지표를 표시하되 한계를 함께 경고한다.',formula:`<span class="dim">구간 G 내 ∃ i: pred<sub>i</sub>=1 ⟹ ∀ i∈G: pred<sub>i</sub>←1</span>`},
  {name:'Affiliation F1',ref:'Huet, Navarro & Rossi (2022), KDD',desc:'예측·정답 이벤트 간 시간적 거리로 P/R을 정의. 파라미터가 없고, "거의 맞춤"을 부분 점수로 인정해 point-wise의 가혹함과 point-adjusted의 관대함 사이를 메운다.',formula:`<span class="var">aff</span> <span class="op">=</span> <span class="frac"><span class="num">1</span><span class="den">1 + λ·d(pred, gt)</span></span>`},
];
function renderMethodology(){
  $('#methodCards').innerHTML=METHODOLOGY.map(m=>`<div class="mcard" style="--c:${COLORS[m.key]}">
    <div class="mhead"><div class="mname"><span class="method-badge" style="--c:${COLORS[m.key]}"></span>${DET_NAMES[m.key]}</div>
      <div class="mfam">${DET_FAMILY[m.key]} family</div><div class="mref">${m.ref}</div></div>
    <div class="mbody"><div class="mdesc">${m.desc}</div><div class="formula">${m.formula}</div>
      <div class="mwhy" style="--c:${COLORS[m.key]}"><b>선정 근거 · </b>${m.why}</div></div></div>`).join('');
  $('#evalCards').innerHTML=EVALDOCS.map(e=>`<div class="mcard" style="--c:var(--cyan)">
    <div class="mhead"><div class="mname" style="font-size:17px">${e.name}</div><div class="mref">${e.ref}</div></div>
    <div class="mbody"><div class="mdesc">${e.desc}</div><div class="formula">${e.formula}</div></div></div>`).join('');
}

/* ---- 탭 전환 ---- */
function switchView(v){
  $$('.toptab').forEach(b=>b.classList.toggle('on',b.dataset.view===v));
  const method=v==='method';
  $('#hero').style.display=method?'none':'';
  $('#intake').style.display=method?'none':'';
  $('#controls').style.display=method?'none':(STATE.data?'':'none');
  $('#main').style.display=method?'none':(STATE.data?'block':'none');
  $('#method').classList.toggle('show',method);
  if(method)renderMethodology();
}
$$('.toptab').forEach(b=>b.onclick=()=>switchView(b.dataset.view));

/* 하위 섹션 탭 (개요 / 평가 / 근거) */
function switchSub(v){
  $$('.subtab').forEach(b=>b.classList.toggle('on',b.dataset.sub===v));
  $$('.subview').forEach(s=>s.classList.toggle('show',s.dataset.subview===v));
}
$$('.subtab').forEach(b=>b.onclick=()=>switchSub(b.dataset.sub));

/* ---- AI 인사이트 ---- */
// 아주 가벼운 마크다운 → HTML (## 헤더, **굵게**, - 리스트, `코드`)
function miniMarkdown(md){
  const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const lines=md.split('\n'); let html='',inList=false;
  for(let raw of lines){
    let line=esc(raw);
    line=line.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/`(.+?)`/g,'<code>$1</code>');
    if(/^##\s+/.test(line)){ if(inList){html+='</ul>';inList=false;} html+='<h2>'+line.replace(/^##\s+/,'')+'</h2>'; }
    else if(/^[-*]\s+/.test(line)){ if(!inList){html+='<ul>';inList=true;} html+='<li>'+line.replace(/^[-*]\s+/,'')+'</li>'; }
    else if(line.trim()===''){ if(inList){html+='</ul>';inList=false;} }
    else { if(inList){html+='</ul>';inList=false;} html+='<p>'+line+'</p>'; }
  }
  if(inList)html+='</ul>';
  return html;
}

// 프롬프트 복사 폴백 UI (키 없음 / API 오류 공통)
function renderPromptFallback(body, d, headerHtml){
  const sys=d.system_prompt||'', usr=d.user_prompt||'';
  const full=sys+'\n\n---\n\n'+usr;
  body.innerHTML=headerHtml+
    `<button class="btn solid" id="copyPromptBtn" style="margin:0 0 14px">프롬프트 복사 → ChatGPT에 붙여넣기</button>
     <div class="insight-prompt" id="promptText">${full.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</div>`;
  const cp=$('#copyPromptBtn');
  if(cp) cp.onclick=()=>{
    navigator.clipboard.writeText(full).then(()=>{cp.textContent='복사됨 ✓';
      setTimeout(()=>cp.textContent='프롬프트 복사 → ChatGPT에 붙여넣기',2000);});
  };
}

async function generateInsight(){
  const btn=$('#insightBtn'), body=$('#insightBody');
  if(!STATE.data){ showError('먼저 데이터를 분석하세요.'); return; }
  const N=parseInt($('#consensusN').value);
  btn.disabled=true; btn.classList.add('loading-spin');
  body.innerHTML='<div class="insight-placeholder"><div class="ico">◈</div><div>HAWK 수치를 정리해 AI에게 해석을 요청하는 중…</div></div>';
  try{
    const fd=new FormData(); fd.append('consensus_n',N);
    const r=await fetch(API+'/api/insight',{method:'POST',body:fd});

    // HTTP 오류(404 등) → 본문 detail을 읽어 프롬프트 폴백
    if(!r.ok){
      let detail='HTTP '+r.status;
      try{ const j=await r.json(); detail=j.detail||detail; }catch(_){}
      body.innerHTML=`<div class="insight-error">인사이트 요청이 거부되었습니다.<div class="err-code">${detail}</div></div>`;
      return;
    }

    const d=await r.json();

    if(d.ok && d.insight){
      // 정상 생성
      body.innerHTML=miniMarkdown(d.insight)+
        `<div class="insight-meta">model: ${d.model||'gpt-4o-mini'} · 탐지·계산은 HAWK, 해석만 AI가 담당</div>`;
    }
    else if(d.reason==='no_api_key'){
      // 키 없음 → 폴백 (정상 흐름)
      renderPromptFallback(body, d,
        `<div class="insight-error" style="background:rgba(240,192,96,.08);border-color:rgba(240,192,96,.3);color:var(--amber)">
          서버에 OpenAI 키가 설정되지 않았습니다. 아래 프롬프트를 복사해 ChatGPT(GPT-4o)에 붙여넣으면 동일한 인사이트를 받을 수 있습니다.
          <div class="err-code" style="color:var(--amber);opacity:.7">백엔드 .env 에 OPENAI_API_KEY 를 넣으면 자동 생성됩니다.</div>
        </div>`);
    }
    else {
      // API 호출은 됐으나 OpenAI 측 오류 → 에러 + 폴백 둘 다 제공
      const errMsg=d.error||d.reason||'알 수 없는 오류';
      renderPromptFallback(body, d,
        `<div class="insight-error">AI 호출 중 오류가 발생했습니다. 아래 프롬프트로 직접 생성할 수 있습니다.
          <div class="err-code">${String(errMsg).replace(/</g,'&lt;')}</div>
        </div>`);
    }
  }catch(e){
    body.innerHTML=`<div class="insight-error">요청 실패 — 백엔드 연결을 확인하세요.<div class="err-code">${e.message}</div></div>`;
  }finally{
    btn.disabled=false; btn.classList.remove('loading-spin'); btn.textContent='인사이트 생성';
  }
}
const insBtn=document.querySelector('#insightBtn'); if(insBtn) insBtn.onclick=generateInsight;

/* ---- 이벤트 바인딩 ---- */
$('#browseBtn').onclick=()=>$('#fileInput').click();
$('#dz').onclick=e=>{if(e.target.closest('.btn'))return;$('#fileInput').click();};
$('#fileInput').onchange=e=>{const f=e.target.files[0];if(!f)return;const fd=new FormData();fd.append('file',f);fd.append('eval_mode',$('#evalMode').value);$('#dzSub').textContent=`${f.name} 업로드 중…`;callAnalyze(fd);};
$('#demoBtn').onclick=callDemo;
$('#rerunBtn').onclick=()=>{ if(STATE.data) renderAll(); };
$('#consensusN').onchange=()=>{if(STATE.data)renderAll();};
$('#evalMode').onchange=()=>{ if(STATE.data){ /* 평가모드 바뀌면 재분석 필요 */ callDemoOrReanalyze(); } };
function callDemoOrReanalyze(){ if(STATE.data&&STATE.data.source&&STATE.data.source.startsWith('demo')) callDemo(); }
const dz=$('#dz');
['dragenter','dragover'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add('over');}));
['dragleave','drop'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove('over');}));
dz.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(!f)return;const fd=new FormData();fd.append('file',f);fd.append('eval_mode',$('#evalMode').value);callAnalyze(fd);});

checkHealth();
