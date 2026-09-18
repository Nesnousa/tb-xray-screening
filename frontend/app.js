/* TB Screening Console — vanilla JS frontend.
 *
 * No framework on purpose: this is a small enough surface (one screen,
 * three views) that React/Vite tooling would be overhead rather than
 * a benefit, and it keeps the whole frontend readable as three files.
 */
const HISTORY_KEY = "tb-screening-history-v1";
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52;

const els = {
  apiUrl: document.getElementById("apiUrl"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  dropZone: document.getElementById("dropZone"),
  fileInput: document.getElementById("fileInput"),
  gradcamToggle: document.getElementById("gradcamToggle"),
  clearBtn: document.getElementById("clearBtn"),
  resultCard: document.getElementById("resultCard"),
  previewImg: document.getElementById("previewImg"),
  heatmapImg: document.getElementById("heatmapImg"),
  heatmapSwitch: document.getElementById("heatmapSwitch"),
  heatmapCheckbox: document.getElementById("heatmapCheckbox"),
  loadingState: document.getElementById("loadingState"),
  resultState: document.getElementById("resultState"),
  errorState: document.getElementById("errorState"),
  errorText: document.getElementById("errorText"),
  borderlineNote: document.getElementById("borderlineNote"),
  verdictLabel: document.getElementById("verdictLabel"),
  verdictPill: document.getElementById("verdictPill"),
  gaugeArc: document.getElementById("gaugeArc"),
  gaugeValue: document.getElementById("gaugeValue"),
  barNormal: document.getElementById("barNormal"),
  barNormalValue: document.getElementById("barNormalValue"),
  barTB: document.getElementById("barTB"),
  barTBValue: document.getElementById("barTBValue"),
  downloadReportBtn: document.getElementById("downloadReportBtn"),
  historyList: document.getElementById("historyList"),
  clearHistoryBtn: document.getElementById("clearHistoryBtn"),
};

let lastResult = null;
let lastFileName = "";

/* ---------- navigation ---------- */
document.querySelectorAll(".side-nav__item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".side-nav__item").forEach((b) => b.classList.remove("side-nav__item--active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("view--active"));
    btn.classList.add("side-nav__item--active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("view--active");
    if (btn.dataset.view === "history") renderHistory();
  });
});

/* ---------- health check ---------- */
async function checkHealth() {
  try {
    const res = await fetch(`${els.apiUrl.value}/health`);
    const body = await res.json();
    if (body.model_loaded) {
      setStatus("ok", `online · ${body.model_version}`);
    } else {
      setStatus("down", "server up, model not loaded");
    }
  } catch {
    setStatus("down", "cannot reach API");
  }
}
function setStatus(kind, text) {
  els.statusDot.className = `status-dot status-dot--${kind}`;
  els.statusText.textContent = text;
}
checkHealth();
setInterval(checkHealth, 15000);
els.apiUrl.addEventListener("change", checkHealth);

/* ---------- upload / drag & drop ---------- */
els.dropZone.addEventListener("click", () => els.fileInput.click());
els.dropZone.addEventListener("dragover", (e) => { e.preventDefault(); els.dropZone.classList.add("drop-zone--active"); });
els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("drop-zone--active"));
els.dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  els.dropZone.classList.remove("drop-zone--active");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
els.fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});
els.clearBtn.addEventListener("click", resetUI);

async function handleFile(file) {
  if (!file.type.startsWith("image/")) return;
  lastFileName = file.name;

  els.resultCard.hidden = false;
  els.clearBtn.hidden = false;
  els.loadingState.hidden = false;
  els.resultState.hidden = true;
  els.errorState.hidden = true;
  els.borderlineNote.hidden = true;
  els.heatmapImg.hidden = true;
  els.heatmapSwitch.hidden = true;
  els.heatmapCheckbox.checked = false;

  const reader = new FileReader();
  reader.onload = (e) => { els.previewImg.src = e.target.result; };
  reader.readAsDataURL(file);

  const base = els.apiUrl.value.replace(/\/$/, "");
  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${base}/predict`, { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Server error (${res.status})`);
    }
    const data = await res.json();
    lastResult = data;
    showResult(data);
    saveToHistory(file, data);

    if (els.gradcamToggle.checked) fetchGradcam(base, file);
  } catch (err) {
    showError(err.message);
  }
}

async function fetchGradcam(base, file) {
  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${base}/predict/gradcam`, { method: "POST", body: form });
    if (!res.ok) return;
    const blob = await res.blob();
    els.heatmapImg.src = URL.createObjectURL(blob);
    els.heatmapImg.hidden = false;
    els.heatmapSwitch.hidden = false;
    els.heatmapCheckbox.checked = true;
  } catch {
    /* Grad-CAM is a nice-to-have; silently skip on failure. */
  }
}
els.heatmapCheckbox.addEventListener("change", () => {
  els.heatmapImg.hidden = !els.heatmapCheckbox.checked;
});

function showResult(data) {
  els.loadingState.hidden = true;
  els.resultState.hidden = false;

  const isTB = data.prediction === "Tuberculosis";
  els.verdictLabel.textContent = data.prediction;
  els.verdictPill.textContent = isTB ? "TB detected" : "clear";
  els.verdictPill.className = `pill ${isTB ? "pill--tb" : "pill--normal"}`;

  const offset = GAUGE_CIRCUMFERENCE - (data.confidence / 100) * GAUGE_CIRCUMFERENCE;
  els.gaugeArc.style.strokeDashoffset = offset;
  els.gaugeArc.style.stroke = isTB ? "#f87171" : "#2dd4bf";
  els.gaugeValue.textContent = `${data.confidence.toFixed(1)}%`;

  els.barNormal.style.width = `${data.probabilities.Normal}%`;
  els.barNormalValue.textContent = `${data.probabilities.Normal.toFixed(1)}%`;
  els.barTB.style.width = `${data.probabilities.Tuberculosis}%`;
  els.barTBValue.textContent = `${data.probabilities.Tuberculosis.toFixed(1)}%`;

  // The verdict is decided against a deliberately high threshold (e.g. 96%,
  // see app/config.py) so we never miss a real TB case - which means a
  // "Normal" verdict can still have a Tuberculosis probability that LOOKS
  // alarming (e.g. 91%) without crossing the cutoff. Flag that explicitly
  // instead of just showing a low "confidence" number, which reads as a
  // contradiction otherwise.
  const thresholdPct = (data.threshold_used ?? 0.5) * 100;
  const isBorderline = !isTB && data.probabilities.Tuberculosis >= 50 && data.probabilities.Tuberculosis < thresholdPct;
  if (isBorderline) {
    els.borderlineNote.hidden = false;
    els.borderlineNote.textContent =
      `Borderline case: Tuberculosis probability is ${data.probabilities.Tuberculosis.toFixed(1)}%, ` +
      `below the ${thresholdPct.toFixed(0)}% cutoff used to flag "Tuberculosis" (kept high on purpose ` +
      `to minimize missed TB cases). Still verdict "Normal", but worth a second look.`;
  } else {
    els.borderlineNote.hidden = true;
  }
}

function showError(message) {
  els.loadingState.hidden = true;
  els.resultState.hidden = true;
  els.errorState.hidden = false;
  els.errorText.textContent = message;
}

function resetUI() {
  els.resultCard.hidden = true;
  els.clearBtn.hidden = true;
  els.fileInput.value = "";
  lastResult = null;
}

/* ---------- history (localStorage) ---------- */
function saveToHistory(file, data) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const history = getHistory();
    history.unshift({
      filename: file.name,
      thumbnail: e.target.result,
      prediction: data.prediction,
      confidence: data.confidence,
      timestamp: new Date().toISOString(),
    });
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 25)));
  };
  reader.readAsDataURL(file);
}
function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}
function renderHistory() {
  const history = getHistory();
  if (history.length === 0) {
    els.historyList.innerHTML = `<p class="history-empty">No scans yet — run one from "New Scan".</p>`;
    return;
  }
  els.historyList.innerHTML = history
    .map((item) => {
      const isTB = item.prediction === "Tuberculosis";
      const date = new Date(item.timestamp).toLocaleString();
      return `
        <div class="history-item">
          <img src="${item.thumbnail}" alt="" />
          <div class="meta">
            ${item.filename}
            <span>${date}</span>
          </div>
          <span class="pill ${isTB ? "pill--tb" : "pill--normal"}">${item.prediction} · ${item.confidence.toFixed(1)}%</span>
        </div>`;
    })
    .join("");
}
els.clearHistoryBtn.addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

/* ---------- PDF report ---------- */
els.downloadReportBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();

  doc.setFontSize(16);
  doc.text("TB Screening Report", 14, 20);
  doc.setFontSize(10);
  doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 28);
  doc.text(`File: ${lastFileName}`, 14, 34);

  doc.setFontSize(13);
  doc.text(`Prediction: ${lastResult.prediction}`, 14, 48);
  doc.text(`Confidence: ${lastResult.confidence.toFixed(1)}%`, 14, 56);
  doc.text(`Normal probability: ${lastResult.probabilities.Normal.toFixed(1)}%`, 14, 64);
  doc.text(`Tuberculosis probability: ${lastResult.probabilities.Tuberculosis.toFixed(1)}%`, 14, 72);

  if (els.previewImg.src) {
    try { doc.addImage(els.previewImg.src, "JPEG", 14, 82, 80, 80); } catch { /* ignore unsupported formats */ }
  }

  doc.setFontSize(9);
  doc.text(
    "AI-generated screening result for educational/demo purposes only. Not a substitute for",
    14,
    175
  );
  doc.text("evaluation by a licensed radiologist.", 14, 180);

  doc.save(`tb-screening-report-${Date.now()}.pdf`);
});
