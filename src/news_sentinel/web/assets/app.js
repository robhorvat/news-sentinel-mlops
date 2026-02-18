const serviceStatusEl = document.getElementById("service-status");
const availableModelsEl = document.getElementById("available-models");
const incidentStatusEl = document.getElementById("incident-status");
const modelSelectEl = document.getElementById("model-select");
const headlineEl = document.getElementById("headline-input");
const predictBtnEl = document.getElementById("predict-btn");
const incidentBtnEl = document.getElementById("incident-btn");
const incidentOutputEl = document.getElementById("incident-output");
const predLabelEl = document.getElementById("pred-label");
const predModelEl = document.getElementById("pred-model");
const predConfidenceEl = document.getElementById("pred-confidence");
const scoreBarsEl = document.getElementById("score-bars");
const historyEl = document.getElementById("history");
const chipsEl = document.getElementById("example-chips");
const requestsMetricEl = document.getElementById("m-requests");
const predsMetricEl = document.getElementById("m-preds");
const errorsMetricEl = document.getElementById("m-errors");

const labelMap = {
  0: "World",
  1: "Sports",
  2: "Business",
  3: "Sci/Tech",
};

const demoExamples = [
  "Stocks rally after earnings beat analyst expectations",
  "Government officials announce new climate treaty",
  "Underdog team wins dramatic championship final",
  "AI chip startup unveils faster inference hardware",
  "Oil prices climb after supply chain disruptions",
  "Space agency confirms successful satellite deployment",
];

const history = [];

function setExamples() {
  chipsEl.innerHTML = "";
  demoExamples.forEach((text) => {
    const btn = document.createElement("button");
    btn.className = "chip";
    btn.type = "button";
    btn.textContent = text;
    btn.addEventListener("click", () => {
      headlineEl.value = text;
    });
    chipsEl.appendChild(btn);
  });
}

async function fetchHealth() {
  const res = await fetch("/healthz");
  const data = await res.json();
  serviceStatusEl.textContent = data.status;

  const models = data.available_models || [];
  availableModelsEl.textContent = models.join(", ") || "none";

  const fixedOptions = ["auto", "baseline", "textcnn"];
  const allowed = new Set(["auto", ...models]);
  for (const option of modelSelectEl.options) {
    if (fixedOptions.includes(option.value) && option.value !== "auto") {
      option.disabled = !allowed.has(option.value);
    }
  }
}

async function fetchRootStatus() {
  const res = await fetch("/");
  const data = await res.json();
  incidentStatusEl.textContent = data.incident_summary_status || "unknown";
}

function renderScores(classScores, predictedLabel) {
  scoreBarsEl.innerHTML = "";
  Object.entries(classScores)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .forEach(([labelId, score]) => {
      const scoreValue = Number(score);
      const row = document.createElement("div");
      row.className = "score-row";

      const title = document.createElement("div");
      title.className = "label";
      title.textContent = `${labelMap[labelId]} (${(scoreValue * 100).toFixed(1)}%)`;

      const track = document.createElement("div");
      track.className = "score-track";

      const fill = document.createElement("div");
      fill.className = "score-fill";
      if (labelMap[labelId] === predictedLabel) {
        fill.style.background = "linear-gradient(90deg, #17c3b2, #23a6d5)";
      }
      fill.style.width = `${Math.max(1, scoreValue * 100)}%`;

      track.appendChild(fill);
      row.appendChild(title);
      row.appendChild(track);
      scoreBarsEl.appendChild(row);
    });
}

function renderHistory() {
  historyEl.innerHTML = "";
  if (!history.length) {
    historyEl.innerHTML = '<div class="history-item">No requests yet.</div>';
    return;
  }

  history.slice(0, 6).forEach((item) => {
    const div = document.createElement("div");
    div.className = "history-item";
    div.textContent = `${item.label} (${item.model}, ${(item.conf * 100).toFixed(1)}%) - ${item.text}`;
    historyEl.appendChild(div);
  });
}

async function runPrediction() {
  const text = headlineEl.value.trim();
  if (!text) {
    headlineEl.focus();
    return;
  }

  predictBtnEl.disabled = true;
  predictBtnEl.textContent = "Running...";

  try {
    const payload = {
      text,
      model: modelSelectEl.value,
    };

    const res = await fetch("/predict", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "prediction failed");
    }

    predLabelEl.textContent = data.label_name;
    predModelEl.textContent = data.model_used;
    predConfidenceEl.textContent = `${(Number(data.confidence) * 100).toFixed(2)}%`;

    renderScores(data.class_scores, data.label_name);

    history.unshift({
      text: text.slice(0, 80),
      label: data.label_name,
      model: data.model_used,
      conf: Number(data.confidence),
    });
    renderHistory();

    await refreshMetrics();
  } catch (err) {
    predLabelEl.textContent = "Error";
    predModelEl.textContent = "-";
    predConfidenceEl.textContent = err.message;
    scoreBarsEl.innerHTML = "";
  } finally {
    predictBtnEl.disabled = false;
    predictBtnEl.textContent = "Run Prediction";
  }
}

async function runIncidentSummary() {
  const text = headlineEl.value.trim();
  if (!text) {
    headlineEl.focus();
    return;
  }

  incidentBtnEl.disabled = true;
  incidentBtnEl.textContent = "Generating...";

  try {
    const payload = {
      text,
      model: modelSelectEl.value,
    };

    const res = await fetch("/incident-summary", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "incident summary failed");
    }

    incidentOutputEl.textContent = [
      data.summary,
      "",
      `Predicted Label: ${data.predicted_label}`,
      `Model Used: ${data.model_used}`,
      `Confidence: ${(Number(data.confidence) * 100).toFixed(2)}%`,
    ].join("\n");

    await refreshMetrics();
  } catch (err) {
    incidentOutputEl.textContent = `Incident summary unavailable: ${err.message}`;
  } finally {
    incidentBtnEl.disabled = false;
    incidentBtnEl.textContent = "Generate Incident Summary";
  }
}

function sumMetricValues(metricsText, metricName, lineMustContain = null) {
  const lines = metricsText.split("\n");
  let total = 0;

  for (const line of lines) {
    if (!line.startsWith(metricName + "{")) {
      continue;
    }
    if (lineMustContain && !line.includes(lineMustContain)) {
      continue;
    }
    const value = Number(line.split(" ").pop());
    if (!Number.isNaN(value)) {
      total += value;
    }
  }
  return total;
}

async function refreshMetrics() {
  const res = await fetch("/metrics");
  const text = await res.text();

  const requests = sumMetricValues(text, "news_api_requests_total");
  const preds = sumMetricValues(text, "news_api_predictions_total");
  const errors = sumMetricValues(text, "news_api_errors_total", 'path="/predict"');

  requestsMetricEl.textContent = String(Math.round(requests));
  predsMetricEl.textContent = String(Math.round(preds));
  errorsMetricEl.textContent = String(Math.round(errors));
}

predictBtnEl.addEventListener("click", runPrediction);
incidentBtnEl.addEventListener("click", runIncidentSummary);

(async function init() {
  setExamples();
  renderHistory();
  await fetchRootStatus();
  await fetchHealth();
  await refreshMetrics();
})();
