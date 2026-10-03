// Keyboard-first annotation helpers. Inputs/textareas are excluded so normal typing is safe.
document.addEventListener('keydown', event => {
  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    if (event.key === 'Escape') document.activeElement.blur();
    return;
  }
  const key = event.key.toLowerCase();
  const map = { q: 'QRS', p: 'P Wave', t: 'T Wave', r: 'R Peak', a: 'Artifact' };
  if (map[key]) {
    $('labelSelect').value = map[key];
    $('customLabel').value = '';
    $('selectionHint').textContent = `${map[key]} selected (keyboard)`;
    event.preventDefault();
    return;
  }
  if (key === 'enter') {
    $('saveAnnotation').click();
    event.preventDefault();
    return;
  }
  if (key === 'escape') {
    cancelEdit();
    event.preventDefault();
    return;
  }
  if (key === 'delete' && state.editingId) {
    deleteAnnotation(state.editingId);
    cancelEdit();
    event.preventDefault();
    return;
  }
  if ((key === 'arrowleft' || key === 'arrowright') && state.fs) {
    const step = 1 / state.fs;
    const field = $('annType').value === 'point' ? $('timeInput') : ($('startInput').disabled ? $('timeInput') : $('startInput'));
    if (field.value !== '') {
      const delta = key === 'arrowleft' ? -step : step;
      field.value = (Number(field.value) + delta).toFixed(4);
      field.dispatchEvent(new Event('change'));
      event.preventDefault();
    }
  }
});
