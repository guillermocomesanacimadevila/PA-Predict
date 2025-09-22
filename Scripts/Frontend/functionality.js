(function () {
  // =========================
  // Theme (persisted + smooth)
  // =========================
  const root = document.documentElement;
  const themeToggle = document.getElementById('themeToggle');
  const THEME_KEY = 'pa_theme';
  const savedTheme = localStorage.getItem(THEME_KEY) || 'light';
  root.setAttribute('data-theme', savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const current = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.style.transition = 'background .2s ease, color .2s ease';
      root.setAttribute('data-theme', current);
      localStorage.setItem(THEME_KEY, current);
      setTimeout(() => { root.style.transition = ''; }, 250);
    });
  }

  // Keyboard shortcuts: "t" toggles theme, "/" focuses search
  document.addEventListener('keydown', (e) => {
    if (e.key === 't' && !/input|textarea/i.test(document.activeElement.tagName)) {
      themeToggle?.click();
    }
    if (e.key === '/' && !/input|textarea/i.test(document.activeElement.tagName)) {
      e.preventDefault();
      document.getElementById('searchInput')?.focus();
    }
  });

  // =========================
  // Download current report
  // =========================
  const downloadBtn = document.getElementById('downloadBtn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
      try {
        const a = document.createElement('a');
        a.href = window.location.href;
        a.download = 'pa_benchmark_report.html';
        document.body.appendChild(a);
        a.click();
        a.remove();
      } catch (e) {
        alert('Use your browser File > Save Page As… to save this report.');
      }
    });
  }

  // =========================
  // Table utils (filter + sort)
  // =========================
  const table = document.getElementById('metricsTable');
  const tbody = table ? table.querySelector('tbody') : null;
  const searchInput = document.getElementById('searchInput');
  const hideNonTop = document.getElementById('hideNonTop');

  const SORT_KEY = 'pa_table_sort'; // persist sort col/direction

  function getTopModelByLogLoss(tableEl) {
    const body = tableEl?.querySelector('tbody');
    if (!body) return null;
    const rows = Array.from(body.querySelectorAll('tr'));
    if (!rows.length) return null;

    const headers = Array.from(tableEl.querySelectorAll('thead th'));
    const logLossIdx = headers.findIndex(h => (h.getAttribute('data-key') || '').toLowerCase() === 'logloss');
    if (logLossIdx === -1) return (rows[0].getAttribute('data-model') || '').toUpperCase();

    let best = null, bestVal = Infinity;
    rows.forEach(r => {
      const cell = r.children[logLossIdx];
      const v = parseFloat(cell?.textContent) || Infinity;
      if (v < bestVal) { bestVal = v; best = (r.getAttribute('data-model') || '').toUpperCase(); }
    });
    return best;
  }

  function filterTable() {
    if (!tbody) return;
    const q = (searchInput?.value || '').trim().toUpperCase();
    const showOnlyTop = hideNonTop?.checked || false;
    const topModel = getTopModelByLogLoss(table);

    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.forEach(tr => {
      const model = (tr.getAttribute('data-model') || '').toUpperCase();
      const match = model.includes(q);
      const topKeep = !showOnlyTop || (model === topModel);
      tr.style.display = (match && topKeep) ? '' : 'none';
    });
  }

  searchInput && searchInput.addEventListener('input', filterTable);
  hideNonTop && hideNonTop.addEventListener('change', filterTable);

  function applySort(colIndex, asc) {
    if (!tbody) return;
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
      const va = parseFloat(a.children[colIndex].textContent);
      const vb = parseFloat(b.children[colIndex].textContent);
      const aNum = Number.isFinite(va);
      const bNum = Number.isFinite(vb);
      if (aNum && bNum) return asc ? va - vb : vb - va;
      const sa = a.children[colIndex].textContent.trim().toUpperCase();
      const sb = b.children[colIndex].textContent.trim().toUpperCase();
      if (sa < sb) return asc ? -1 : 1;
      if (sa > sb) return asc ? 1 : -1;
      return 0;
    });
    rows.forEach(r => tbody.appendChild(r));
    filterTable();
  }

  // Sortable headers with persistence
  if (table) {
    const headers = table.querySelectorAll('thead th.sortable');

    // Restore saved sort
    const saved = localStorage.getItem(SORT_KEY);
    if (saved) {
      try {
        const { index, asc } = JSON.parse(saved);
        const th = headers[index];
        if (th) {
          th.classList.toggle('asc', !!asc);
          applySort(index, !!asc);
        }
      } catch {}
    }

    headers.forEach((th, idx) => {
      th.addEventListener('click', () => {
        const isAsc = th.classList.toggle('asc'); // toggle indicator
        // clear others' indicators
        headers.forEach(h => { if (h !== th) h.classList.remove('asc'); });
        applySort(idx, isAsc);
        localStorage.setItem(SORT_KEY, JSON.stringify({ index: idx, asc: isAsc }));
      });
    });
  }

  // =========================
  // Lightbox for zooming images
  // =========================
  const lightbox = document.getElementById('lightbox');
  const lightboxImg = document.getElementById('lightboxImg');

  function openLightbox(src, alt) {
    if (!lightbox || !lightboxImg) return;
    lightboxImg.src = src;
    lightboxImg.alt = alt || 'Visualization';
    lightbox.style.display = 'flex';
    lightbox.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
  }
  function closeLightbox() {
    if (!lightbox || !lightboxImg) return;
    lightbox.style.display = 'none';
    lightbox.setAttribute('aria-hidden', 'true');
    lightboxImg.src = '';
    lightboxImg.alt = '';
    document.body.style.overflow = '';
  }

  document.addEventListener('click', (e) => {
    const t = e.target;
    if (t && t.classList.contains('zoomable')) {
      openLightbox(t.src, t.alt);
    } else if (t && (t.id === 'lightbox' || t.id === 'lightboxImg')) {
      closeLightbox();
    }
  });

  // Close on ESC
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeLightbox();
  });

  // Initial filter pass
  filterTable();
})();
