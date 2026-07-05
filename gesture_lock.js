(function () {
  function lockStyles() {
    document.documentElement.style.touchAction = 'none';
    document.documentElement.style.overscrollBehavior = 'none';
    document.documentElement.style.overflow = 'hidden';
    if (document.body) {
      document.body.style.touchAction = 'none';
      document.body.style.overscrollBehavior = 'none';
      document.body.style.overflow = 'hidden';
    }
  }

  lockStyles();
  document.addEventListener('DOMContentLoaded', lockStyles, { once: true });

  const stop = function (event) {
    event.preventDefault();
  };

  document.addEventListener('touchmove', stop, { passive: false, capture: true });
  document.addEventListener('gesturestart', stop, { passive: false, capture: true });
  document.addEventListener('gesturechange', stop, { passive: false, capture: true });
  document.addEventListener('gestureend', stop, { passive: false, capture: true });
  document.addEventListener('wheel', stop, { passive: false, capture: true });

  document.addEventListener('keydown', function (event) {
    if (!(event.ctrlKey || event.metaKey)) return;
    if (event.key === '+' || event.key === '-' || event.key === '=' || event.key === '0') {
      event.preventDefault();
    }
  }, { capture: true });

  window.addEventListener('load', function () {
    document.querySelectorAll('canvas, #sim-stage, #app').forEach(function (element) {
      element.style.touchAction = 'none';
      element.style.overscrollBehavior = 'none';
    });
  }, { once: true });
})();

