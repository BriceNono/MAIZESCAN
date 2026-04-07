/* ═══════════════════════════════════════════
   MaizeScan — Frontend Application Logic
   ═══════════════════════════════════════════ */

// ── State ────────────────────────────────────
let selectedFile = null;

// ── Element refs ─────────────────────────────
const fileInput     = document.getElementById('fileInput');
const dropZone      = document.getElementById('dropZone');
const uploadIdle    = document.getElementById('uploadIdle');
const uploadPreview = document.getElementById('uploadPreview');
const previewImg    = document.getElementById('previewImg');
const analyzeBtn    = document.getElementById('analyzeBtn');
const demoNotice    = document.getElementById('demoNotice');

const resultEmpty      = document.getElementById('resultEmpty');
const resultProcessing = document.getElementById('resultProcessing');
const resultOutput     = document.getElementById('resultOutput');
const processingSteps  = document.getElementById('processingSteps');

// ── Drag-and-drop ─────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropZone.addEventListener('click', () => {
  if (!selectedFile) fileInput.click();
});
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// ── File handling ─────────────────────────────
function handleFile(file) {
  const allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showToast('Please upload a JPG, PNG, or WEBP image.', 'error');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showToast('File too large. Maximum size is 16 MB.', 'error');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    uploadIdle.style.display = 'none';
    uploadPreview.style.display = 'block';
    analyzeBtn.disabled = false;
    showResultEmpty();
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  uploadIdle.style.display = 'flex';
  uploadPreview.style.display = 'none';
  analyzeBtn.disabled = true;
  showResultEmpty();
}

function resetAll() {
  resetUpload();
  showResultEmpty();
}

// ── State transitions ─────────────────────────
function showResultEmpty() {
  resultEmpty.style.display = 'flex';
  resultProcessing.style.display = 'none';
  resultOutput.style.display = 'none';
}
function showResultProcessing() {
  resultEmpty.style.display = 'none';
  resultProcessing.style.display = 'block';
  resultOutput.style.display = 'none';
}
function showResultOutput() {
  resultEmpty.style.display = 'none';
  resultProcessing.style.display = 'none';
  resultOutput.style.display = 'block';
}

// ── Processing steps animation ────────────────
const CNN_STEPS = [
  'Resizing image to 224 × 224 px…',
  'Normalising pixel values [0, 1]…',
  'VGG16 block1–block3 feature extraction…',
  'VGG16 block4–block5 deep feature pass…',
  'GlobalAveragePooling2D → 512-dim vector…',
  'Dense(512) + BatchNorm + Dropout…',
  'Dense(256) + BatchNorm + Dropout…',
  'Softmax — computing class probabilities…',
];

function runProcessingSteps() {
  return new Promise(resolve => {
    processingSteps.innerHTML = '';
    let idx = 0;

    function addStep() {
      if (idx >= CNN_STEPS.length) {
        setTimeout(resolve, 200);
        return;
      }
      const step = document.createElement('div');
      step.className = 'proc-step active';
      step.innerHTML = `
        <div class="proc-icon active"><div class="proc-spin"></div></div>
        <span>${CNN_STEPS[idx]}</span>
      `;
      processingSteps.appendChild(step);

      // Mark previous step done
      const prev = processingSteps.children[idx - 1];
      if (prev) {
        prev.className = 'proc-step done';
        prev.querySelector('.proc-icon').className = 'proc-icon done';
        prev.querySelector('.proc-icon').innerHTML = `
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M2 6L5 9L10 3" stroke="#3e7a28" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>`;
      }
      idx++;
      setTimeout(addStep, 280);
    }
    addStep();
  });
}

// ── Main analysis ─────────────────────────────
async function analyze() {
  if (!selectedFile) return;

  // Set loading state
  analyzeBtn.querySelector('.btn-analyze-text').style.display = 'none';
  analyzeBtn.querySelector('.btn-analyze-loading').style.display = 'flex';
  analyzeBtn.disabled = true;
  showResultProcessing();

  // Run step animations in parallel with the fetch
  const stepsPromise = runProcessingSteps();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const [response] = await Promise.all([
      fetch('/predict', { method: 'POST', body: formData }),
      stepsPromise
    ]);

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Prediction failed. Please try again.');
    }

    // Show demo notice if model not loaded
    if (data.demo_mode) {
      demoNotice.style.display = 'flex';
    }

    renderResult(data);
  } catch (err) {
    showResultEmpty();
    showToast(err.message, 'error');
  } finally {
    analyzeBtn.querySelector('.btn-analyze-text').style.display = 'flex';
    analyzeBtn.querySelector('.btn-analyze-loading').style.display = 'none';
    analyzeBtn.disabled = false;
  }
}

// ── Render result ─────────────────────────────
function renderResult(data) {
  // Leaf preview
  document.getElementById('resultLeafImg').src = data.image_preview;

  // Badge
  const badge = document.getElementById('verdictBadge');
  const isBad = data.severity !== 'None';
  badge.textContent = data.confidence_pct + ' confidence';
  badge.style.background = hexToRgba(data.color, .12);
  badge.style.color = data.color;
  badge.style.border = `1px solid ${hexToRgba(data.color, .25)}`;

  // Disease name
  document.getElementById('verdictDisease').textContent = data.prediction;
  document.getElementById('verdictScientific').textContent = data.scientific;
  document.getElementById('verdictSeverity').textContent = data.severity;
  document.getElementById('verdictSpread').textContent = data.spread;

  // Description
  document.getElementById('resultDescription').textContent = data.description;

  // Confidence bars
  const barsEl = document.getElementById('confidenceBars');
  barsEl.innerHTML = '';
  const maxProb = Math.max(...Object.values(data.all_probs));
  Object.entries(data.all_probs)
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, prob]) => {
      const pct = (prob * 100).toFixed(1);
      const isTop = prob === maxProb;
      const barColor = isTop ? data.color : '#c8ddb0';
      barsEl.innerHTML += `
        <div class="conf-row">
          <span class="conf-name" style="font-weight:${isTop ? 500 : 400}">${name}</span>
          <div class="conf-track">
            <div class="conf-fill" style="width:0%;background:${barColor}" data-w="${pct}"></div>
          </div>
          <span class="conf-pct">${pct}%</span>
        </div>`;
    });

  // Animate bars after render
  requestAnimationFrame(() => {
    document.querySelectorAll('.conf-fill').forEach(el => {
      el.style.width = el.dataset.w + '%';
    });
  });

  // Actions
  const actionList = document.getElementById('actionList');
  actionList.innerHTML = data.actions
    .map((action, i) => `
      <li>
        <span class="action-num">${i + 1}</span>
        <span>${action}</span>
      </li>`)
    .join('');

  // Prevention
  document.getElementById('preventionText').textContent = data.prevention;

  showResultOutput();
}

// ── Toast notifications ───────────────────────
function showToast(message, type = 'info') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  const styles = {
    position: 'fixed', bottom: '24px', left: '50%',
    transform: 'translateX(-50%)',
    background: type === 'error' ? '#c0392b' : '#2d5a1b',
    color: '#fff', padding: '12px 24px', borderRadius: '8px',
    fontSize: '14px', fontFamily: 'inherit',
    boxShadow: '0 4px 20px rgba(0,0,0,.2)',
    zIndex: 9999, maxWidth: '90vw', textAlign: 'center',
    animation: 'fadeIn .25s ease'
  };
  Object.assign(toast.style, styles);
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ── Helpers ───────────────────────────────────
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Check health on load ──────────────────────
window.addEventListener('load', async () => {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.model === 'demo_mode') {
      console.info('[MaizeScan] Running in demo mode — no model file found.');
    } else {
      console.info('[MaizeScan] Model loaded. TF version:', data.tf_version);
    }
  } catch (e) {
    console.warn('[MaizeScan] Could not reach health endpoint.');
  }
});
