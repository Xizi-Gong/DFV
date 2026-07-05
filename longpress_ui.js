(function () {
  const el = document.getElementById('longpress-indicator');
  if (!el) return;
  const fill = document.getElementById('longpress-fill');
  const icon = document.getElementById('longpress-icon');
  const label = document.getElementById('longpress-label');
  const hint = document.getElementById('longpress-hint');
  let clearRequested = false;

  function show() {
    el.classList.add('visible');
  }

  function hide() {
    el.classList.remove('visible');
    fill.style.width = '0%';
  }

  function clearFrozen() {
    if (clearRequested) return;
    clearRequested = true;
    if (typeof window.sendCommand === 'function') {
      window.sendCommand('set_freeze_input', false);
    }
  }

  window.handleLongpressMessage = function (msg) {
    const progress = Math.max(0, Math.min(1, Number(msg.progress) || 0));
    const touching = !!msg.has_touch;
    const hasFrozen = !!msg.frozen;

    if (touching && progress > 0) {
      icon.textContent = '🔥';
      label.textContent = 'HOLD';
      hint.textContent = Math.round(progress * 100) + '%';
      fill.style.width = (progress * 100) + '%';
      show();
      return;
    }

    if (hasFrozen) {
      clearRequested = false;
      icon.textContent = '❄️';
      label.textContent = 'FROZEN';
      hint.textContent = 'tap → clear all';
      fill.style.width = '100%';
      show();
      return;
    }

    clearRequested = false;
    hide();
  };

  el.addEventListener('click', function (event) {
    event.stopPropagation();
    clearFrozen();
  });
})();

