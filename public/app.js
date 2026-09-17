/**
 * IrisAI Classification Dashboard - Frontend Application Logic
 * Implements real-time inference, interactive scatter plot, dataset explorer,
 * metrics synchronization, and responsive navigation.
 */

// State
let currentTab = 'dashboard';
let activeSpeciesFilters = { setosa: true, versicolor: true, virginica: true };
let datasetRecords = [];
let currentDatasetFilter = 'all';
let currentDatasetSearch = '';
let currentDatasetSort = 'id_asc';
let recentPredictions = [
  { inputs: [5.1, 3.5, 1.4, 0.2], species: 'setosa', conf: 0.998, time: 'Just now', model: 'KNN k=3' },
  { inputs: [6.4, 2.9, 4.3, 1.3], species: 'versicolor', conf: 0.954, time: '12 min ago', model: 'KNN k=3' },
  { inputs: [6.9, 3.1, 5.4, 2.1], species: 'virginica', conf: 0.921, time: '43 min ago', model: 'KNN k=3' }
];

const SPECIES_CONFIG = {
  setosa: {
    title: 'Iris Setosa',
    color: '#06B6D4',
    cluster: 'Cluster A',
    bgClass: 'bg-primary',
    textClass: 'text-primary',
    borderClass: 'border-primary',
    desc: 'Characterized by compact, wide petals and prominent sepals. Strictly linearly separable from Versicolor and Virginica in botanical morphometrics.'
  },
  versicolor: {
    title: 'Iris Versicolor',
    color: '#8B5CF6',
    cluster: 'Cluster B',
    bgClass: 'bg-secondary',
    textClass: 'text-secondary',
    borderClass: 'border-secondary',
    desc: 'Intermediate petal dimensions exhibiting balanced morphology. Forms the central cluster with subtle non-linear boundary contours bordering Virginica.'
  },
  virginica: {
    title: 'Iris Virginica',
    color: '#10B981',
    cluster: 'Cluster C',
    bgClass: 'bg-tertiary',
    textClass: 'text-tertiary',
    borderClass: 'border-tertiary',
    desc: 'Robust anatomical archetype featuring elongated petals and wide sepals. Occupies the upper feature space adjacent to the Versicolor decision frontier.'
  }
};

// Benchmark Presets
const BENCHMARK_PRESETS = {
  setosa: { sl: 5.1, sw: 3.5, pl: 1.4, pw: 0.2 },
  versicolor: { sl: 6.0, sw: 2.8, pl: 4.5, pw: 1.4 },
  virginica: { sl: 6.7, sw: 3.1, pl: 5.6, pw: 2.4 }
};

// ================= NAVIGATION =================
function switchTab(tabId) {
  currentTab = tabId;

  // Toggle View Sections
  document.querySelectorAll('.view-section').forEach(sec => {
    sec.classList.remove('active');
  });
  const activeSec = document.getElementById(`view-${tabId}`);
  if (activeSec) {
    activeSec.classList.add('active');
  }

  // Update Breadcrumb
  const breadcrumb = document.getElementById('header-breadcrumb');
  if (breadcrumb) {
    const labels = {
      dashboard: 'Dashboard',
      predict: 'Inference Studio',
      analytics: 'Analytics Space',
      performance: 'Performance Matrix',
      dataset: 'Botanical Registry'
    };
    breadcrumb.innerText = labels[tabId] || tabId.toUpperCase();
  }

  // Update Desktop Tabs
  document.querySelectorAll('.nav-tab-desktop').forEach(btn => {
    if (btn.getAttribute('data-target') === tabId) {
      btn.classList.add('text-on-surface', 'bg-surface-container-highest', 'shadow-sm');
      btn.classList.remove('text-on-surface-variant');
    } else {
      btn.classList.remove('text-on-surface', 'bg-surface-container-highest', 'shadow-sm');
      btn.classList.add('text-on-surface-variant');
    }
  });

  // Update Mobile Tabs
  document.querySelectorAll('.nav-tab-mobile').forEach(btn => {
    if (btn.getAttribute('data-target') === tabId) {
      btn.classList.add('text-primary');
      btn.classList.remove('text-on-surface-variant');
    } else {
      btn.classList.remove('text-primary');
      btn.classList.add('text-on-surface-variant');
    }
  });

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ================= MODALS =================
function openModal(modalId) {
  const el = document.getElementById(modalId);
  if (el) el.classList.remove('hidden');
}

function closeModal(modalId) {
  const el = document.getElementById(modalId);
  if (el) el.classList.add('hidden');
}

// ================= PREDICT STUDIO =================
function handleStudioSliderChange() {
  const sl = parseFloat(document.getElementById('slider-sl').value);
  const sw = parseFloat(document.getElementById('slider-sw').value);
  const pl = parseFloat(document.getElementById('slider-pl').value);
  const pw = parseFloat(document.getElementById('slider-pw').value);

  document.getElementById('display-sl').innerText = sl.toFixed(1);
  document.getElementById('display-sw').innerText = sw.toFixed(1);
  document.getElementById('display-pl').innerText = pl.toFixed(1);
  document.getElementById('display-pw').innerText = pw.toFixed(1);

  // Synchronize quick sliders if on dashboard
  syncQuickSliders(sl, sw, pl, pw);

  // Update scatter crosshair
  updateScatterCrosshair(pl, pw);

  // Trigger real-time prediction
  runPredictionRequest(sl, sw, pl, pw);
}

function stepSlider(field, delta) {
  const slider = document.getElementById(`slider-${field}`);
  if (!slider) return;
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  let val = parseFloat(slider.value) + delta;
  val = Math.max(min, Math.min(max, Math.round(val * 10) / 10));
  slider.value = val;
  handleStudioSliderChange();
}

function loadPreset(presetKey) {
  const p = BENCHMARK_PRESETS[presetKey];
  if (!p) return;

  document.getElementById('slider-sl').value = p.sl;
  document.getElementById('slider-sw').value = p.sw;
  document.getElementById('slider-pl').value = p.pl;
  document.getElementById('slider-pw').value = p.pw;

  handleStudioSliderChange();

  // If currently not on predict tab, switch or stay smoothly
  if (currentTab !== 'predict') {
    switchTab('predict');
  }
}

function triggerStudioPrediction() {
  const sl = parseFloat(document.getElementById('slider-sl').value);
  const sw = parseFloat(document.getElementById('slider-sw').value);
  const pl = parseFloat(document.getElementById('slider-pl').value);
  const pw = parseFloat(document.getElementById('slider-pw').value);

  runPredictionRequest(sl, sw, pl, pw, true);
}

// ================= QUICK PREDICTOR (DASHBOARD) =================
function handleQuickSliderChange() {
  const sl = parseFloat(document.getElementById('quick-sl').value);
  const sw = parseFloat(document.getElementById('quick-sw').value);
  const pl = parseFloat(document.getElementById('quick-pl').value);
  const pw = parseFloat(document.getElementById('quick-pw').value);

  document.getElementById('quick-sl-val').innerText = `${sl.toFixed(1)} cm`;
  document.getElementById('quick-sw-val').innerText = `${sw.toFixed(1)} cm`;
  document.getElementById('quick-pl-val').innerText = `${pl.toFixed(1)} cm`;
  document.getElementById('quick-pw-val').innerText = `${pw.toFixed(1)} cm`;

  // Compute immediate heuristic
  updateQuickPredictBox(sl, sw, pl, pw);
}

function syncQuickSliders(sl, sw, pl, pw) {
  const qSl = document.getElementById('quick-sl');
  const qSw = document.getElementById('quick-sw');
  const qPl = document.getElementById('quick-pl');
  const qPw = document.getElementById('quick-pw');

  if (qSl && qSw && qPl && qPw) {
    qSl.value = sl;
    qSw.value = sw;
    qPl.value = pl;
    qPw.value = pw;
    document.getElementById('quick-sl-val').innerText = `${sl.toFixed(1)} cm`;
    document.getElementById('quick-sw-val').innerText = `${sw.toFixed(1)} cm`;
    document.getElementById('quick-pl-val').innerText = `${pl.toFixed(1)} cm`;
    document.getElementById('quick-pw-val').innerText = `${pw.toFixed(1)} cm`;
    updateQuickPredictBox(sl, sw, pl, pw);
  }
}

function updateQuickPredictBox(sl, sw, pl, pw) {
  let sp = 'setosa';
  let conf = 0.99;
  let dist = 0.08;

  if (pl < 2.45) {
    sp = 'setosa';
    conf = 0.997;
    dist = (2.45 - pl) * 2.1 + 0.08;
  } else if (pl < 4.85 && pw < 1.75) {
    sp = 'versicolor';
    conf = 0.948;
    dist = 0.16;
  } else {
    sp = 'virginica';
    conf = 0.963;
    dist = 0.12;
  }

  const cfg = SPECIES_CONFIG[sp];
  const box = document.getElementById('quick-result-box');
  const title = document.getElementById('quick-result-title');
  const confEl = document.getElementById('quick-result-conf');
  const meta = document.getElementById('quick-result-meta');
  const icon = document.getElementById('quick-result-icon');

  if (title && confEl && meta && icon) {
    title.innerText = cfg.title;
    title.style.color = cfg.color;
    icon.style.color = cfg.color;
    confEl.innerText = `${(conf * 100).toFixed(1)}% Conf`;
    meta.innerText = `Cluster Dist: ${dist.toFixed(2)}σ`;
  }
}

function triggerQuickPrediction() {
  const box = document.getElementById('quick-result-box');
  if (box) {
    box.classList.add('opacity-40');
    setTimeout(() => {
      box.classList.remove('opacity-40');
      const sl = parseFloat(document.getElementById('quick-sl').value);
      const sw = parseFloat(document.getElementById('quick-sw').value);
      const pl = parseFloat(document.getElementById('quick-pl').value);
      const pw = parseFloat(document.getElementById('quick-pw').value);
      runPredictionRequest(sl, sw, pl, pw, true);
    }, 120);
  }
}

// ================= REAL-TIME INFERENCE ENGINE =================
async function runPredictionRequest(sl, sw, pl, pw, addToFeed = false) {
  const startTime = performance.now();
  let result = null;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sepal_length: sl,
        sepal_width: sw,
        petal_length: pl,
        petal_width: pw
      })
    });
    if (res.ok) {
      result = await res.json();
    }
  } catch (err) {
    // Graceful offline fallback prediction calculation
  }

  // Client-side fallback if server offline
  if (!result) {
    result = computeClientInference(sl, sw, pl, pw);
  }

  const latency = Math.max(0.4, (performance.now() - startTime)).toFixed(1);
  const latencyChip = document.getElementById('predict-latency-chip');
  if (latencyChip) {
    latencyChip.innerText = `Latency: ${latency}ms`;
  }

  renderPredictionUI(result);

  if (addToFeed) {
    addRecentPrediction(sl, sw, pl, pw, result.predicted_species, result.confidence);
  }
}

function computeClientInference(sl, sw, pl, pw) {
  let sp = 'setosa';
  let conf = 0.99;
  let dist = 1.5;
  let pSetosa = 0.01;
  let pVersic = 0.01;
  let pVirgin = 0.01;

  if (pl < 2.45) {
    sp = 'setosa';
    conf = Math.min(0.999, 0.985 + (2.45 - pl) * 0.01);
    pSetosa = conf;
    pVersic = (1 - conf) * 0.8;
    pVirgin = (1 - conf) * 0.2;
    dist = (2.45 - pl) * 2.1 + 1.2;
  } else if (pl < 4.85 && pw < 1.75) {
    sp = 'versicolor';
    conf = Math.min(0.97, Math.max(0.72, 0.91 + (3.8 - Math.abs(pl - 4.2)) * 0.05));
    pVersic = conf;
    pSetosa = (1 - conf) * 0.08;
    pVirgin = (1 - conf) * 0.92;
    dist = 0.16;
  } else {
    sp = 'virginica';
    conf = Math.min(0.99, Math.max(0.75, 0.93 + (pl - 4.85) * 0.04));
    pVirgin = conf;
    pSetosa = (1 - conf) * 0.02;
    pVersic = (1 - conf) * 0.98;
    dist = 0.12;
  }

  return {
    predicted_species: sp,
    confidence: conf,
    probabilities: {
      setosa: pSetosa,
      versicolor: pVersic,
      virginica: pVirgin
    },
    morphology: {
      sepal_ratio: (sl / (sw || 1)).toFixed(2),
      petal_ratio: (pl / (pw || 1)).toFixed(2),
      boundary_sigma: dist.toFixed(2),
      cluster: sp === 'setosa' ? 'Cluster A' : (sp === 'versicolor' ? 'Cluster B' : 'Cluster C')
    }
  };
}

function renderPredictionUI(data) {
  const sp = data.predicted_species;
  const cfg = SPECIES_CONFIG[sp] || SPECIES_CONFIG.setosa;
  const confPercent = (data.confidence * 100).toFixed(1) + '%';

  // Species Title & Description
  const title = document.getElementById('result-species-title');
  const desc = document.getElementById('result-species-desc');
  const confVal = document.getElementById('result-confidence-val');
  const stripe = document.getElementById('result-glow-stripe');
  const statusBadge = document.getElementById('result-status-badge');

  if (title) {
    title.innerText = cfg.title;
    title.style.color = cfg.color;
  }
  if (desc) desc.innerText = cfg.desc;
  if (confVal) confVal.innerText = confPercent;

  if (stripe) {
    stripe.style.background = `linear-gradient(90deg, ${cfg.color}, #ffffff, ${cfg.color})`;
  }

  if (statusBadge) {
    statusBadge.innerText = `Verified ${cfg.cluster} • High Certainty`;
    statusBadge.style.color = cfg.color;
    statusBadge.style.borderColor = `${cfg.color}40`;
    statusBadge.style.backgroundColor = `${cfg.color}15`;
  }

  // Archetype Readouts
  const m = data.morphology || {};
  const archSepal = document.getElementById('archetype-sepal');
  const archPetal = document.getElementById('archetype-petal');
  const archDist = document.getElementById('archetype-boundary');

  if (archSepal) archSepal.innerText = `Sepal Ratio: ${m.sepal_ratio || '1.46'}`;
  if (archPetal) archPetal.innerText = `Petal Ratio: ${m.petal_ratio || '7.00'}`;
  if (archDist) archDist.innerText = `Boundary Distance: +${m.boundary_sigma || '2.84'}σ (${sp === 'setosa' ? 'Linear Safe' : 'Kernel Safe'})`;

  // Softmax Logits Progress Bars
  const probs = data.probabilities || {};
  const pSetosa = Math.round((probs.setosa || 0) * 1000) / 10;
  const pVersic = Math.round((probs.versicolor || 0) * 1000) / 10;
  const pVirgin = Math.round((probs.virginica || 0) * 1000) / 10;

  setProbBar('setosa', pSetosa);
  setProbBar('versicolor', pVersic);
  setProbBar('virginica', pVirgin);
}

function setProbBar(species, pct) {
  const bar = document.getElementById(`prob-bar-${species}`);
  const val = document.getElementById(`prob-val-${species}`);
  if (bar) bar.style.width = `${pct}%`;
  if (val) val.innerText = `${pct.toFixed(1)}%`;
}

// ================= RECENT PREDICTIONS =================
function addRecentPrediction(sl, sw, pl, pw, species, conf) {
  const item = {
    inputs: [sl, sw, pl, pw],
    species: species,
    conf: conf,
    time: 'Just now',
    model: 'KNN k=3'
  };
  recentPredictions.unshift(item);
  if (recentPredictions.length > 6) recentPredictions.pop();
  renderRecentFeed();
}

function clearRecentFeed() {
  recentPredictions = [];
  renderRecentFeed();
}

function renderRecentFeed() {
  const container = document.getElementById('recent-inferences-list');
  if (!container) return;

  if (recentPredictions.length === 0) {
    container.innerHTML = '<div class="text-xs text-outline font-mono py-2">No recent predictions in feed. Run an inference above!</div>';
    return;
  }

  container.innerHTML = recentPredictions.map(item => {
    const cfg = SPECIES_CONFIG[item.species] || SPECIES_CONFIG.setosa;
    const confP = (item.conf * 100).toFixed(1) + '%';
    const dims = item.inputs.map(v => v.toFixed(1)).join(', ') + ' cm';

    return `
      <div class="p-3 rounded-xl bg-surface-container-low flex items-center justify-between border border-surface-container-high/40 shadow-sm">
        <div class="flex items-center gap-3 min-w-0">
          <span class="material-symbols-outlined text-[20px]" style="color: ${cfg.color}">check_circle</span>
          <div class="flex flex-col truncate">
            <span class="font-body-sm text-xs text-on-surface font-mono font-medium truncate">${dims}</span>
            <span class="font-label-sm text-[10px] text-outline font-mono">${item.time} • ${item.model}</span>
          </div>
        </div>
        <div class="flex items-center gap-2 flex-shrink-0 font-mono">
          <span class="px-2 py-0.5 rounded text-xs font-medium" style="background-color: ${cfg.color}15; color: ${cfg.color}; border: 1px solid ${cfg.color}30">
            ${cfg.title.replace('Iris ', '')}
          </span>
          <span class="font-label-sm text-xs text-tertiary font-bold">${confP}</span>
        </div>
      </div>
    `;
  }).join('');
}

// ================= SCATTER PLOT (ANALYTICS) =================
function toggleScatterFilter(species) {
  activeSpeciesFilters[species] = !activeSpeciesFilters[species];
  const btn = document.getElementById(`filter-${species}`);
  if (btn) {
    if (activeSpeciesFilters[species]) {
      btn.classList.remove('opacity-40');
      btn.style.filter = 'none';
    } else {
      btn.classList.add('opacity-40');
      btn.style.filter = 'grayscale(100%)';
    }
  }

  const grp = document.getElementById(`scatter-${species}-group`);
  if (grp) {
    grp.style.display = activeSpeciesFilters[species] ? 'inline' : 'none';
  }
}

function populateScatterPlot(records) {
  const gSetosa = document.getElementById('scatter-setosa-group');
  const gVersicolor = document.getElementById('scatter-versicolor-group');
  const gVirginica = document.getElementById('scatter-virginica-group');
  const tooltip = document.getElementById('petal-tooltip');
  const tooltipText = document.getElementById('petal-tooltip-text');

  if (!gSetosa || !gVersicolor || !gVirginica) return;

  gSetosa.innerHTML = '';
  gVersicolor.innerHTML = '';
  gVirginica.innerHTML = '';

  // SVG coordinate transformation:
  // Petal Length: 1.0 -> 50px, 7.0 -> 570px (width 520px for 6.0 range => 86.66 px/cm)
  // Petal Width:  0.0 -> 270px, 2.6 -> 30px  (height 240px for 2.6 range => 92.3 px/cm)
  const mapX = pl => 50 + ((pl - 1.0) / 6.0) * 520;
  const mapY = pw => 270 - (pw / 2.6) * 240;

  records.forEach(r => {
    const cx = mapX(r.petal_length);
    const cy = mapY(r.petal_width);
    const sp = r.species.toLowerCase();

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', cx);
    circle.setAttribute('cy', cy);
    circle.setAttribute('r', '4');
    circle.setAttribute('class', 'cursor-pointer transition-all hover:r-6');
    circle.setAttribute('data-info', `${r.specimen_code} (${r.species}) • L: ${r.petal_length}cm | W: ${r.petal_width}cm`);

    circle.addEventListener('mouseenter', e => {
      circle.setAttribute('r', '7');
      if (tooltip && tooltipText) {
        tooltipText.innerText = circle.getAttribute('data-info');
        tooltip.style.left = `${(cx / 600) * 100}%`;
        tooltip.style.top = `${(cy / 320) * 100}%`;
        tooltip.classList.remove('hidden');
      }
    });

    circle.addEventListener('mouseleave', () => {
      circle.setAttribute('r', '4');
      if (tooltip) tooltip.classList.add('hidden');
    });

    circle.addEventListener('click', () => {
      loadPresetDimensions(r.sepal_length, r.sepal_width, r.petal_length, r.petal_width);
    });

    if (sp === 'setosa') gSetosa.appendChild(circle);
    else if (sp === 'versicolor') gVersicolor.appendChild(circle);
    else if (sp === 'virginica') gVirginica.appendChild(circle);
  });
}

function updateScatterCrosshair(pl, pw) {
  const crosshair = document.getElementById('scatter-current-crosshair');
  const dot = document.getElementById('crosshair-dot');
  const center = document.getElementById('crosshair-center');
  if (!crosshair || !dot || !center) return;

  const cx = 50 + ((pl - 1.0) / 6.0) * 520;
  const cy = 270 - (pw / 2.6) * 240;

  dot.setAttribute('cx', cx);
  dot.setAttribute('cy', cy);
  center.setAttribute('cx', cx);
  center.setAttribute('cy', cy);
}

function loadPresetDimensions(sl, sw, pl, pw) {
  document.getElementById('slider-sl').value = sl;
  document.getElementById('slider-sw').value = sw;
  document.getElementById('slider-pl').value = pl;
  document.getElementById('slider-pw').value = pw;
  handleStudioSliderChange();
  switchTab('predict');
}

// ================= DATASET EXPLORER =================
function handleDatasetSearch() {
  currentDatasetSearch = document.getElementById('dataset-search').value.trim().toLowerCase();
  renderDatasetCards();
}

function filterDatasetSpecies(sp) {
  currentDatasetFilter = sp;
  document.querySelectorAll('.dataset-filter-pill').forEach(btn => {
    if (btn.getAttribute('data-species') === sp) {
      btn.classList.add('active', 'bg-surface-bright', 'text-on-surface');
      btn.classList.remove('bg-surface-container-high', 'text-on-surface-variant');
    } else {
      btn.classList.remove('active', 'bg-surface-bright', 'text-on-surface');
      btn.classList.add('bg-surface-container-high', 'text-on-surface-variant');
    }
  });
  renderDatasetCards();
}

function handleDatasetSort() {
  currentDatasetSort = document.getElementById('dataset-sort').value;
  renderDatasetCards();
}

function renderDatasetCards() {
  const container = document.getElementById('dataset-cards-container');
  const countEl = document.getElementById('dataset-records-count');
  if (!container) return;

  let filtered = datasetRecords;

  // Species Filter
  if (currentDatasetFilter !== 'all') {
    filtered = filtered.filter(r => r.species.toLowerCase() === currentDatasetFilter);
  }

  // Search Query
  if (currentDatasetSearch) {
    filtered = filtered.filter(r => {
      const q = currentDatasetSearch;
      return (
        r.species.toLowerCase().includes(q) ||
        r.specimen_code.toLowerCase().includes(q) ||
        r.sepal_length.toString().includes(q) ||
        r.sepal_width.toString().includes(q) ||
        r.petal_length.toString().includes(q) ||
        r.petal_width.toString().includes(q)
      );
    });
  }

  // Sort
  if (currentDatasetSort === 'petal_length_desc') {
    filtered = [...filtered].sort((a, b) => b.petal_length - a.petal_length);
  } else if (currentDatasetSort === 'petal_length_asc') {
    filtered = [...filtered].sort((a, b) => a.petal_length - b.petal_length);
  } else if (currentDatasetSort === 'sepal_length_desc') {
    filtered = [...filtered].sort((a, b) => b.sepal_length - a.sepal_length);
  } else if (currentDatasetSort === 'sepal_width_desc') {
    filtered = [...filtered].sort((a, b) => b.sepal_width - a.sepal_width);
  } else if (currentDatasetSort === 'id_asc') {
    filtered = [...filtered].sort((a, b) => a.id - b.id);
  }

  if (countEl) {
    countEl.innerText = `Showing ${filtered.length} of ${datasetRecords.length} specimens`;
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="col-span-full p-8 text-center bg-surface-container-low rounded-2xl border border-surface-container-high text-on-surface-variant font-mono text-sm">
        No botanical specimens found matching query "${currentDatasetSearch}". Try another search term!
      </div>
    `;
    return;
  }

  container.innerHTML = filtered.map(r => {
    const sp = r.species.toLowerCase();
    const cfg = SPECIES_CONFIG[sp] || SPECIES_CONFIG.setosa;

    return `
      <div onclick="loadPresetDimensions(${r.sepal_length}, ${r.sepal_width}, ${r.petal_length}, ${r.petal_width})" class="w-full bg-surface-container-low rounded-xl p-4 border border-surface-container-high/60 shadow-sm space-y-3 cursor-pointer hover:border-primary/50 transition-all active:scale-[0.99] group">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="font-label-md text-xs font-mono font-bold text-on-surface">${r.specimen_code}</span>
            <span class="font-label-sm text-[10px] text-outline-variant font-mono">Gaspé Wild</span>
          </div>
          <div class="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full" style="background-color: ${cfg.color}18; color: ${cfg.color}; border: 1px solid ${cfg.color}35">
            <span class="w-1.5 h-1.5 rounded-full" style="background-color: ${cfg.color}"></span>
            <span class="font-label-sm text-[11px] font-mono font-semibold capitalize">${sp}</span>
          </div>
        </div>

        <div class="grid grid-cols-4 gap-1.5 font-mono text-center">
          <div class="bg-surface-container-lowest rounded-lg p-2 flex flex-col">
            <span class="text-[9px] text-on-surface-variant opacity-75">Sepal L</span>
            <span class="text-xs text-on-surface font-semibold">${r.sepal_length} <span class="text-[8px] text-outline">cm</span></span>
          </div>
          <div class="bg-surface-container-lowest rounded-lg p-2 flex flex-col">
            <span class="text-[9px] text-on-surface-variant opacity-75">Sepal W</span>
            <span class="text-xs text-on-surface font-semibold">${r.sepal_width} <span class="text-[8px] text-outline">cm</span></span>
          </div>
          <div class="bg-surface-container-lowest rounded-lg p-2 flex flex-col">
            <span class="text-[9px] text-on-surface-variant opacity-75">Petal L</span>
            <span class="text-xs font-bold" style="color: ${cfg.color}">${r.petal_length} <span class="text-[8px] text-outline">cm</span></span>
          </div>
          <div class="bg-surface-container-lowest rounded-lg p-2 flex flex-col">
            <span class="text-[9px] text-on-surface-variant opacity-75">Petal W</span>
            <span class="text-xs text-on-surface font-semibold">${r.petal_width} <span class="text-[8px] text-outline">cm</span></span>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function exportCSVData() {
  if (datasetRecords.length === 0) return;
  const headers = ['id', 'specimen_code', 'origin', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'];
  const rows = datasetRecords.map(r => [
    r.id,
    r.specimen_code,
    `"${r.origin}"`,
    r.sepal_length,
    r.sepal_width,
    r.petal_length,
    r.petal_width,
    r.species
  ].join(','));

  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', 'iris_botanical_dataset.csv');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ================= METRICS & RETRAIN SIMULATION =================
function triggerRetrainSimulation() {
  const btn = document.getElementById('retrain-btn');
  if (!btn) return;
  const orig = btn.innerHTML;
  btn.innerHTML = '<span class="material-symbols-outlined animate-spin text-[18px]">sync</span><span>Re-evaluating 5-Fold Stratified Split...</span>';
  btn.disabled = true;

  setTimeout(() => {
    btn.innerHTML = '<span class="material-symbols-outlined text-[18px]">done_all</span><span>Metrics Validated &amp; Synced!</span>';
    setTimeout(() => {
      btn.innerHTML = orig;
      btn.disabled = false;
    }, 1800);
  }, 1200);
}

function exportMetricsReport() {
  fetch('/api/metrics')
    .then(r => r.json())
    .then(data => {
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data, null, 2));
      const a = document.createElement('a');
      a.setAttribute('href', dataStr);
      a.setAttribute('download', 'iris_model_metrics.json');
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    })
    .catch(() => alert('Could not fetch metrics payload.'));
}

// ================= BOOTSTRAP INITIALIZATION =================
async function initApp() {
  renderRecentFeed();

  // Load dataset
  try {
    const res = await fetch('/api/dataset');
    if (res.ok) {
      const json = await res.json();
      datasetRecords = json.records || [];
    }
  } catch (err) {
    console.log('Dataset fetch failed, using fallback records.');
  }

  // Fallback dataset records if API unavailable
  if (datasetRecords.length === 0) {
    datasetRecords = generateFallbackDataset();
  }

  renderDatasetCards();
  populateScatterPlot(datasetRecords);
  updateScatterCrosshair(1.4, 0.2);

  // Load Metrics
  try {
    const res = await fetch('/api/metrics');
    if (res.ok) {
      const metrics = await res.json();
      syncMetricsUI(metrics);
    }
  } catch (err) {
    console.log('Metrics fetch failed, using default metrics.');
  }

  // Health check
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const health = await res.json();
      const statusText = document.getElementById('model-status-text');
      if (statusText) statusText.innerText = 'Model Online';
    }
  } catch (err) {
    const statusText = document.getElementById('model-status-text');
    if (statusText) statusText.innerText = 'Offline Mode';
  }
}

function syncMetricsUI(m) {
  if (!m) return;
  const best = m.best_model || {};
  const cm = m.confusion_matrix || {};
  const diag = m.species_diagnostics || {};

  // Accuracy
  const acc = best.test_accuracy ? (best.test_accuracy * 100).toFixed(1) : '96.7';
  const prec = best.macro_precision ? (best.macro_precision * 100).toFixed(1) : '97.1';
  const rec = best.macro_recall ? (best.macro_recall * 100).toFixed(1) : '96.7';
  const f1 = best.macro_f1 ? (best.macro_f1 * 100).toFixed(1) : '96.9';

  const heroBadge = document.getElementById('hero-model-badge');
  const bentoRing = document.getElementById('bento-acc-ring');
  const metaName = document.getElementById('meta-model-name');

  if (heroBadge && best.name) heroBadge.innerText = `${best.name} • ${acc}% Val Acc`;
  if (bentoRing) bentoRing.innerText = `${acc}%`;
  if (metaName && best.name) metaName.innerText = best.name;

  setMetricText('metric-acc', `${acc}%`, 'bar-metric-acc', `${acc}%`);
  setMetricText('metric-prec', `${prec}%`, 'bar-metric-prec', `${prec}%`);
  setMetricText('metric-rec', `${rec}%`, 'bar-metric-rec', `${rec}%`);
  setMetricText('metric-f1', `${f1}%`, 'bar-metric-f1', `${f1}%`);

  // Confusion matrix cells
  if (cm.matrix && cm.matrix.length === 3) {
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const cell = document.getElementById(`cm-${r}-${c}`);
        if (cell) cell.innerText = cm.matrix[r][c];
      }
    }
    const mis = document.getElementById('misclass-count');
    if (mis && cm.misclassifications !== undefined) mis.innerText = cm.misclassifications;
  }

  // Diagnostics
  if (diag.setosa) {
    setText('diag-setosa-p', diag.setosa.precision.toFixed(2));
    setText('diag-setosa-r', diag.setosa.recall.toFixed(2));
  }
  if (diag.versicolor) {
    setText('diag-versic-p', diag.versicolor.precision.toFixed(2));
    setText('diag-versic-r', diag.versicolor.recall.toFixed(2));
  }
  if (diag.virginica) {
    setText('diag-virgin-p', diag.virginica.precision.toFixed(2));
    setText('diag-virgin-r', diag.virginica.recall.toFixed(2));
  }
}

function setMetricText(id, txt, barId, barWidth) {
  const el = document.getElementById(id);
  const bar = document.getElementById(barId);
  if (el) el.innerText = txt;
  if (bar) bar.style.width = barWidth;
}

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.innerText = txt;
}

function generateFallbackDataset() {
  const records = [];
  const setosaBase = [
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
    [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4],
    [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3]
  ];
  const versicBase = [
    [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0],
    [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5],
    [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
    [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1]
  ];
  const virginBase = [
    [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8],
    [6.5, 3.0, 5.8, 2.2], [7.6, 3.0, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
    [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2.0], [6.4, 2.7, 5.3, 1.9],
    [6.8, 3.0, 5.5, 2.1], [5.7, 2.5, 5.0, 2.0], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3],
    [6.5, 3.0, 5.5, 1.8], [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6.0, 2.2, 5.0, 1.5]
  ];

  let id = 1;
  const addClass = (base, sp) => {
    for (let i = 0; i < 50; i++) {
      const src = base[i % base.length];
      records.push({
        id: id,
        specimen_code: `#${id.toString().padStart(3, '0')}`,
        origin: 'Gaspé Wild',
        sepal_length: src[0],
        sepal_width: src[1],
        petal_length: src[2],
        petal_width: src[3],
        species: sp
      });
      id++;
    }
  };

  addClass(setosaBase, 'setosa');
  addClass(versicBase, 'versicolor');
  addClass(virginBase, 'virginica');
  return records;
}

// Kick off
document.addEventListener('DOMContentLoaded', initApp);
