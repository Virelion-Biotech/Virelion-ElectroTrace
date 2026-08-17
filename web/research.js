const researchState={recordingId:null,storedMeta:null};
function researchMsg(text,type='ok'){const el=$('researchMessage');if(el)el.textContent=text;message(text,type)}

$('storeRecording')?.addEventListener('click',async()=>{
  const file=$('fileInput')?.files?.[0]; if(!file)return researchMsg('Choose a recording first.','warn');
  const fd=new FormData(); fd.append('file',file);
  const r=await fetch('/api/recording',{method:'POST',body:fd}); const d=await r.json();
  if(!r.ok)return researchMsg(d.error||'Could not store recording.','error');
  researchState.recordingId=d.recording_id; researchState.storedMeta=d;
  $('storedMeta').textContent=`Stored ${d.format.toUpperCase()} · ${d.n_samples.toLocaleString()} samples · ${Number(d.duration_s).toFixed(2)} s`;
  researchMsg('Recording stored for windowed access and project tracking.','ok');
});

$('fetchWindow')?.addEventListener('click',async()=>{
  if(!researchState.recordingId)return researchMsg('Store the recording first.','warn');
  const start=Math.max(0,Number($('windowStart').value||0)); const stop=Math.max(start+1,Number($('windowStop').value||5000));
  const r=await fetch(`/api/recording/${researchState.recordingId}/window?start=${start}&stop=${stop}`); const d=await r.json();
  if(!r.ok)return researchMsg(d.error,'error');
  $('windowInfo').textContent=`Loaded samples ${d.start.toLocaleString()}–${(d.stop-1).toLocaleString()} (${(d.stop-d.start).toLocaleString()} samples) without loading the full recording into the browser.`;
});

$('saveProjectRecording')?.addEventListener('click',async()=>{
  if(!researchState.storedMeta)return researchMsg('Store the recording first.','warn');
  const payload={project:$('projectName').value.trim()||'default',recording_id:researchState.recordingId,subject_id:$('subjectId').value.trim()||'unassigned',group:$('groupName').value.trim(),visit:$('visitName').value.trim(),format:researchState.storedMeta.format,sampling_rate_hz:researchState.storedMeta.sampling_rate_hz,duration_s:researchState.storedMeta.duration_s,channels:researchState.storedMeta.channels,source_path:researchState.storedMeta.filename};
  const r=await fetch('/api/project/recording',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); const d=await r.json();
  if(!r.ok)return researchMsg(d.error,'error'); $('projectCount').textContent=`${d.recordings.length} recording(s) in project`; researchMsg('Recording registered in project metadata.','ok');
});

$('phenotypeAnalysis')?.addEventListener('click',async()=>{
  if(!state.peaks.length)return researchMsg('Detect R peaks first.','warn');
  const sig=state.displaySignals[channel()]||state.rawSignals[channel()];
  const r=await fetch('/api/phenotype',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({time:state.time,signal:sig,r_indices:state.peaks})}); const d=await r.json();
  if(!r.ok)return researchMsg(d.error,'error');
  const h=d.summary.heart_rate_bpm||{}; const q=d.summary.qrs_width_proxy_s||{};
  $('phenotypeResult').innerHTML=`<b>${d.summary.n_beats} beats</b> · HR median ${h.median==null?'—':h.median.toFixed(1)} bpm · RR median ${d.summary.rr_prev_s?.median==null?'—':d.summary.rr_prev_s.median.toFixed(3)} s · QRS proxy median ${q.median==null?'—':(q.median*1000).toFixed(1)} ms · tachy ${(100*(d.summary.tachycardia_fraction||0)).toFixed(1)}% · brady ${(100*(d.summary.bradycardia_fraction||0)).toFixed(1)}%`;
});

$('comparePhenotype')?.addEventListener('click',async()=>{
  try{
    const a=$('groupAValues').value.split(',').map(Number).filter(Number.isFinite), b=$('groupBValues').value.split(',').map(Number).filter(Number.isFinite);
    const r=await fetch('/api/statistics/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({group_a:a,group_b:b})}); const d=await r.json();
    if(!r.ok)return researchMsg(d.error,'error');
    $('statsResult').textContent=`n=${d.n_a}/${d.n_b} · means ${d.mean_a.toFixed(3)} vs ${d.mean_b.toFixed(3)} · Welch p=${d.welch_p.toExponential(3)} · Mann–Whitney p=${d.mann_whitney_p.toExponential(3)} · Cohen's d=${d.cohens_d.toFixed(3)}`;
  }catch(e){researchMsg(e.message,'error')}
});

$('runBenchmark')?.addEventListener('click',async()=>{
  try{
    const payload=JSON.parse($('benchmarkJson').value); const r=await fetch('/api/benchmark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); const d=await r.json();
    if(!r.ok)return researchMsg(d.error,'error');
    const rf=d.models.random_forest.mean, lr=d.models.logistic_regression.mean;
    $('benchmarkResult').textContent=`${d.n_samples} beats · ${d.n_subjects} subjects · RF macro-F1 ${(100*rf.macro_f1).toFixed(1)}% · RF balanced accuracy ${(100*rf.balanced_accuracy).toFixed(1)}% · LR macro-F1 ${(100*lr.macro_f1).toFixed(1)}%. Splits are subject-level GroupKFold.`;
  }catch(e){researchMsg('Benchmark JSON is invalid: '+e.message,'error')}
});

window.addEventListener('load',()=>{ if($('projectName'))$('projectName').value='default'; });
