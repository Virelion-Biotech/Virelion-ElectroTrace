const state={time:[],rawSignals:{},displaySignals:{},filename:'',timeCol:'time',channels:[],fs:null,duration:0,annotations:[],editingId:null,pendingSelection:null,preprocessing:{baseline:false,highpass_hz:null,lowpass_hz:null,notch_hz:null}};
const labels=['P Wave','QRS','T Wave','Pacemaker Spike','R Peak','Artifact','Abnormal Beat','Graft Activation','Arrhythmia','Custom'];
const $=id=>document.getElementById(id);

labels.forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=x;$('labelSelect').appendChild(o)});

function message(text,type='ok'){const el=$('messages');el.innerHTML=`<div class="message ${type}">${escapeHtml(text)}</div>`}
function escapeHtml(s){return String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\\':'&#92;','"':'&quot;'}[c]))}
function formatTime(x){return Number(x).toFixed(4)}
function currentChannel(){return $('channelSelect').value}
function selectedLabel(){return $('customLabel').value.trim()||$('labelSelect').value}
function syncChannels(){
  ['channelSelect','annChannel'].forEach(id=>{const el=$(id);el.innerHTML='';state.channels.forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;el.appendChild(o)})});
}
function renderMeta(){
  $('recordingMeta').innerHTML=state.filename?`<span><b>File</b><br>${escapeHtml(state.filename)}</span><span><b>Samples</b><br>${state.time.length}</span><span><b>Sampling rate</b><br>${state.fs?state.fs+' Hz':'unknown'}</span><span><b>Duration</b><br>${state.duration.toFixed(3)} s</span><span><b>Channels</b><br>${state.channels.length}</span><span><b>Recording ID</b><br>${simpleHash(state.filename+state.time.length).slice(0,12)}</span>`:'<span>No recording loaded.</span>';
}
function simpleHash(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return (h>>>0).toString(16).padStart(8,'0')}
function redraw(){
  if(!state.time.length)return;
  const traces=state.channels.map(c=>({x:state.time,y:state.displaySignals[c]??state.rawSignals[c],mode:'lines',name:c,line:{width:c===currentChannel()?1.8:1}}));
  const shapes=[],ann=[];
  state.annotations.forEach(a=>{
    if(a.type==='interval'){shapes.push({type:'rect',x0:a.start,x1:a.end,y0:0,y1:1,yref:'paper',fillcolor:a.status==='flagged'?'#d94b4b':'#f0a43b',opacity:.18,line:{width:0}})}
    else {shapes.push({type:'line',x0:a.time,x1:a.time,y0:0,y1:1,yref:'paper',line:{dash:'dash',width:1.5,color:'#c13a3a'}})}
  });
  const layout={margin:{l:55,r:20,t:25,b:45},height:560,dragmode:'select',showlegend:true,shapes,xaxis:{title:'Time (s)',rangeslider:{visible:true}},yaxis:{title:'Amplitude'},hovermode:'closest'};
  Plotly.react('plot',traces,layout,{responsive:true,displaylogo:false});
  renderAnnotations();renderQC();renderMeta();
}
async function loadFile(file){
  const fd=new FormData();fd.append('file',file);const response=await fetch('/api/analyze',{method:'POST',body:fd});const data=await response.json();
  if(!response.ok||!data.valid){message(data.error||data.errors?.join(' ')||'Validation failed.','error');return}
  state.time=data.time;state.rawSignals=data.signals;state.displaySignals=structuredClone(data.signals);state.filename=data.filename;state.timeCol=data.time_col;state.channels=data.signal_cols;state.fs=data.sampling_rate_hz;state.duration=data.duration_s;state.annotations=[];state.editingId=null;state.preprocessing={baseline:false,highpass_hz:null,lowpass_hz:null,notch_hz:null};syncChannels();renderMeta();renderPreprocess();redraw();message(data.warnings?.length?data.warnings.join(' '):'Recording loaded and validated.','ok');
}
$('fileInput').addEventListener('change',e=>e.target.files[0]&&loadFile(e.target.files[0]));
$('loadSample').addEventListener('click',async()=>{const r=await fetch('/sample_data/sample_ecg.csv');const b=await r.blob();await loadFile(new File([b],'sample_ecg.csv',{type:'text/csv'}))});
$('channelSelect').addEventListener('change',redraw);$('annChannel').addEventListener('change',()=>{});
$('annType').addEventListener('change',()=>{const interval=$('annType').value==='interval';$('startInput').disabled=!interval;$('endInput').disabled=!interval;$('timeInput').disabled=interval});

function setSelectionFromChart(){
  if(!state.pendingSelection)return;$('startInput').value=formatTime(state.pendingSelection.start);$('endInput').value=formatTime(state.pendingSelection.end);$('timeInput').value=formatTime(state.pendingSelection.start);$('selectionHint').textContent=`Selected ${formatTime(state.pendingSelection.start)}–${formatTime(state.pendingSelection.end)} s`;
}
$('plot').on?.('plotly_selected',ev=>{});
$('plot').addEventListener?.('plotly_selected',()=>{});

function addOrUpdate(){
  if(!state.fs){message('Load a recording first.','error');return}
  const type=$('annType').value,label=selectedLabel(),channel=$('annChannel').value,confidence=Number($('confidenceInput').value),notes=$('notesInput').value.trim();
  const a={label,type,channel,confidence,notes,annotator:$('annotator').value.trim(),status:'unreviewed',reviewer:'',review_notes:''};
  if(!label||!channel){message('Label and channel are required.','error');return}
  if(type==='interval'){a.start=Number($('startInput').value);a.end=Number($('endInput').value);if(!(a.end>a.start)){message('End must be greater than start.','error');return}if(a.start<0||a.end>state.duration){message('Interval must stay within recording bounds.','error');return}}
  else {a.time=Number($('timeInput').value);if(a.time<0||a.time>state.duration){message('Point must stay within recording bounds.','error');return}}
  if(!Number.isFinite(confidence)||confidence<0||confidence>1){message('Confidence must be between 0 and 1.','error');return}
  if(state.editingId){const idx=state.annotations.findIndex(x=>x.id===state.editingId);if(idx>=0){a.id=state.editingId;state.annotations[idx]={...state.annotations[idx],...a};}}
  else state.annotations.push({...a,id:crypto.randomUUID?crypto.randomUUID():simpleHash(Date.now()+Math.random())});
  state.annotations.sort((x,y)=>(x.start??x.time)-(y.start??y.time));cancelEdit();redraw();
}
$('saveAnnotation').addEventListener('click',addOrUpdate);
function cancelEdit(){state.editingId=null;$('saveAnnotation').textContent='Add annotation';$('cancelEdit').disabled=true;$('selectionHint').textContent='No selection'}
$('cancelEdit').addEventListener('click',cancelEdit);
function editAnnotation(id){const a=state.annotations.find(x=>x.id===id);if(!a)return;state.editingId=id;$('annType').value=a.type;$('annType').dispatchEvent(new Event('change'));$('labelSelect').value=labels.includes(a.label)?a.label:'Custom';$('customLabel').value=labels.includes(a.label)?'':a.label;$('annChannel').value=a.channel;$('startInput').value=a.start??'';$('endInput').value=a.end??'';$('timeInput').value=a.time??'';$('confidenceInput').value=a.confidence;$('notesInput').value=a.notes||'';$('saveAnnotation').textContent='Save changes';$('cancelEdit').disabled=false}
function setStatus(id,status){const a=state.annotations.find(x=>x.id===id);if(a){a.status=status;redraw()}}
function duplicateAnnotation(id){const a=state.annotations.find(x=>x.id===id);if(a){const copy=structuredClone(a);copy.id=crypto.randomUUID?crypto.randomUUID():simpleHash(Date.now()+Math.random());copy.status='unreviewed';copy.reviewer='';copy.review_notes='';state.annotations.push(copy);redraw()}}
function deleteAnnotation(id){state.annotations=state.annotations.filter(a=>a.id!==id);redraw()}
function renderAnnotations(){
  $('annCount').textContent=state.annotations.length;const el=$('annotationList');el.innerHTML='';if(!state.annotations.length){el.innerHTML='<div class="muted">No annotations yet.</div>';return}
  state.annotations.forEach(a=>{const loc=a.type==='interval'?`${formatTime(a.start)}–${formatTime(a.end)} s`:`${formatTime(a.time)} s`;const d=document.createElement('div');d.className=`annotation ${a.status}`;d.innerHTML=`<div><div class="ann-title">${escapeHtml(a.label)}</div><div class="ann-meta">${escapeHtml(a.channel)} · ${loc} · confidence ${Number(a.confidence).toFixed(2)} · ${escapeHtml(a.status)}</div>${a.notes?`<div class="ann-meta">${escapeHtml(a.notes)}</div>`:''}</div><div class="ann-actions"><button class="secondary" data-action="edit">Edit</button><button class="secondary" data-action="duplicate">Copy</button><button class="secondary" data-action="status">${a.status==='accepted'?'Flag':a.status==='flagged'?'Accept':'Accept'}</button><button class="secondary" data-action="delete">Delete</button></div>`;d.querySelector('[data-action=edit]').onclick=()=>editAnnotation(a.id);d.querySelector('[data-action=duplicate]').onclick=()=>duplicateAnnotation(a.id);d.querySelector('[data-action=delete]').onclick=()=>deleteAnnotation(a.id);d.querySelector('[data-action=status]').onclick=()=>setStatus(a.id,a.status==='accepted'?'flagged':'accepted');el.appendChild(d)})
}
function renderQC(){const counts={accepted:0,flagged:0,unreviewed:0};state.annotations.forEach(a=>counts[a.status]++);$('qcSummary').innerHTML=`<div class="qc-item"><strong>${counts.accepted}</strong>accepted</div><div class="qc-item"><strong>${counts.unreviewed}</strong><span class="warn">unreviewed</span></div><div class="qc-item"><strong>${counts.flagged}</strong><span class="error">flagged</span></div>`}
$('acceptAll').onclick=()=>{state.annotations.forEach(a=>a.status='accepted');redraw()};$('flagUnreviewed').onclick=()=>{state.annotations.forEach(a=>{if(a.status==='unreviewed')a.status='flagged'});redraw()};
function renderPreprocess(){ $('baseline').checked=state.preprocessing.baseline;$('highpass').value=state.preprocessing.highpass_hz??'';$('lowpass').value=state.preprocessing.lowpass_hz??'';$('notch').value=state.preprocessing.notch_hz??'' }
$('applyFilters').addEventListener('click',async()=>{if(!state.fs){message('Load a recording first.','error');return}const hp=$('highpass').value?Number($('highpass').value):null,lp=$('lowpass').value?Number($('lowpass').value):null,n=$('notch').value?Number($('notch').value):null;for(const c of state.channels){const r=await fetch('/api/filter',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sampling_rate_hz:state.fs,signal:state.rawSignals[c],baseline:$('baseline').checked,highpass_hz:hp,lowpass_hz:lp,notch_hz:n})});const d=await r.json();if(!r.ok){$('filterMessage').textContent=d.error;return}state.displaySignals[c]=d.signal}state.preprocessing={baseline:$('baseline').checked,highpass_hz:hp,lowpass_hz:lp,notch_hz:n};$('filterMessage').textContent='Display filters applied. Raw data remain unchanged.';redraw()});

$('detectPeaks').onclick=async()=>{if(!state.fs){message('Load a recording first.','error');return}const c=currentChannel();const r=await fetch('/api/detect/r-peaks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sampling_rate_hz:state.fs,signal:state.displaySignals[c]??state.rawSignals[c],min_distance_ms:250,prominence_factor:.5})});const d=await r.json();if(!r.ok){message(d.error,'error');return}d.peaks.forEach((idx,i)=>{const t=state.time[idx];state.annotations.push({id:crypto.randomUUID?crypto.randomUUID():simpleHash(Date.now()+i),label:'R Peak',type:'point',channel:c,time:t,confidence:.5,notes:'Auto-detected; review required.',annotator:$('annotator').value.trim(),status:'unreviewed',reviewer:'',review_notes:''})});state.annotations.sort((a,b)=>(a.time??a.start)-(b.time??b.start));redraw();message(`Added ${d.peaks.length} candidate R peaks. Review them before export.`,'warn')};

$('exportJson').onclick=()=>{if(!state.filename){message('Load a recording first.','error');return}const payload={schema:'electrotrace.annotation/v2',file:state.filename,metadata:{sampling_rate_hz:state.fs,duration_s:state.duration,channels:state.channels,annotator:$('annotator').value.trim(),source:$('source').value.trim(),preprocessing:state.preprocessing,exported_at:new Date().toISOString()},annotations:state.annotations};downloadBlob(JSON.stringify(payload,null,2),`${state.filename.replace(/\.csv$/i,'')}_annotations.json`,'application/json')}
$('exportCsv').onclick=()=>{if(!state.annotations.length){message('No annotations to export.','error');return}const cols=['file','id','type','label','channel','start','end','time','confidence','notes','annotator','status','reviewer','review_notes'];const rows=[cols.join(',')];state.annotations.forEach(a=>rows.push(cols.map(k=>csvCell(k==='file'?state.filename:a[k]??'')).join(',')));downloadBlob(rows.join('\n'),`${state.filename.replace(/\.csv$/i,'')}_annotations.csv`,'text/csv')}
function csvCell(v){return `"${String(v).replace(/"/g,'""')}"`};function downloadBlob(text,name,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}

$('reviewFile').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;try{const other=JSON.parse(await file.text());const b=other.annotations||[];const pointsA=state.annotations.filter(a=>a.type==='point'),pointsB=b.filter(a=>a.type==='point');let matches=0,errors=[];const used=new Set();pointsA.forEach(a=>{let best=null;pointsB.forEach(x=>{if(used.has(x.id)||x.label!==a.label)return;const err=Math.abs(x.time-a.time);if(err<=.04&&(!best||err<best.err))best={x,err}});if(best){used.add(best.x.id);matches++;errors.push(best.err)}});const rate=matches/Math.max(pointsA.length,pointsB.length,1);$('agreementResult').innerHTML=`<b>Point agreement</b><br>${matches} matches · ${(rate*100).toFixed(1)}% within 40 ms${errors.length?` · mean absolute error ${(errors.reduce((a,b)=>a+b,0)/errors.length*1000).toFixed(1)} ms`:''}`;}catch(err){$('agreementResult').textContent='Could not read annotation JSON: '+err.message}});

$('plot').on?.('plotly_click',ev=>{});
window.addEventListener('load',()=>{ $('annType').dispatchEvent(new Event('change')); });

// Plotly event handlers are attached after the plot exists.
const plotEl=$('plot');plotEl.addEventListener('click',()=>{});
const observer=new MutationObserver(()=>{
  if(plotEl.data){
    if(!plotEl.__bound){
      plotEl.on('plotly_selected',ev=>{const pts=ev?.range?.x;if(pts&&pts.length===2){state.pendingSelection={start:Math.min(...pts),end:Math.max(...pts)};setSelectionFromChart()}});
      plotEl.on('plotly_click',ev=>{const p=ev?.points?.[0];if(p){$('timeInput').value=formatTime(p.x);$('selectionHint').textContent=`Point ${formatTime(p.x)} s`}});plotEl.__bound=true;
    }
  }
});observer.observe(plotEl,{childList:true,subtree:true});
