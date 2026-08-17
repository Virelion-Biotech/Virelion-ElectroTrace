// Large-recording audit patch: CSV uses the same bounded window path as native formats.
async function loadCsv(file){
  await loadNativeWindow(file);
  message('CSV recording stored and loaded in windowed mode.','ok');
}
window.loadCsv=loadCsv;
