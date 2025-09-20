(function () {
  // =========================
  // Theme (persisted)
  // =========================
  const root = document.documentElement;
  const themeToggle = document.getElementById('themeToggle');
  const savedTheme = localStorage.getItem('pa_theme') || 'light';
  root.setAttribute('data-theme', savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const current = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', current);
      localStorage.setItem('pa_theme', current);
    });
  }

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
  // Table utils
  // =========================
  const searchInput = document.getElementById('searchInput');
  const hideNonTop = document.getElementById('hideNonTop');
  const table = document.getElementById('metricsTable');
  const tbody = table ? table.querySelector('tbody') : null;

  // Find “Top-1” model by MIN LogLoss (robust even after user sorts columns)
  function getTopModelByLogLoss(tableEl) {
    const body = tableEl?.querySelector('tbody');
    if (!body) return null;
    const rows = Array.from(body.querySelectorAll('tr'));
    if (!rows.length) return null;

    // locate LogLoss column index (fallback: first row)
    const headers = Array.from(tableEl.querySelectorAll('thead th'));
    const logLossIdx = headers.findIndex(h => (h.getAttribute('data-key') || '').toLowerCase() === 'logloss');
    if (logLossIdx === -1) {
      return (rows[0].getAttribute('data-model') || '').toUpperCase();
    }

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

  // =========================
  // Sortable headers (numeric-aware)
  // =========================
  if (table) {
    const headers = table.querySelectorAll('th.sortable');
    headers.forEach(th => {
      th.addEventListener('click', () => {
        const colIndex = Array.from(th.parentNode.children).indexOf(th);
        const asc = th.classList.toggle('asc');  // toggle direction indicator
        const rows = Array.from(tbody.querySelectorAll('tr'));

        rows.sort((a, b) => {
          const va = parseFloat(a.children[colIndex].textContent);
          const vb = parseFloat(b.children[colIndex].textContent);
          const aNum = Number.isFinite(va);
          const bNum = Number.isFinite(vb);
          if (aNum && bNum) return asc ? va - vb : vb - va;
          // fallback lexicographic
          const sa = a.children[colIndex].textContent.trim().toUpperCase();
          const sb = b.children[colIndex].textContent.trim().toUpperCase();
          if (sa < sb) return asc ? -1 : 1;
          if (sa > sb) return asc ? 1 : -1;
          return 0;
        });

        rows.forEach(r => tbody.appendChild(r));
        filterTable(); // re-apply filters after resort
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
  }
  function closeLightbox() {
    if (!lightbox || !lightboxImg) return;
    lightbox.style.display = 'none';
    lightbox.setAttribute('aria-hidden', 'true');
    lightboxImg.src = '';
    lightboxImg.alt = '';
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
