// ============================================================
// Demonstração ao vivo do Classificador Genético (Pima Diabetes)
// Porte fiel de preparacao_de_dados.py + classificador.py para o navegador.
// A evolução roda geração a geração dentro de um loop de animação,
// permitindo "assistir" o algoritmo aprender a regra de decisão.
// ============================================================

// ---------- Utilidades numéricas e RNG semeado ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// RNG global da execução (resemeado a cada "Iniciar")
let rng = mulberry32(Date.now());

function gauss(mu, sigma) {
  // Box-Muller
  let u = 0,
    v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return (
    mu + sigma * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
  );
}

const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const sum = (arr) => arr.reduce((a, b) => a + b, 0);
const mean = (arr) => (arr.length ? sum(arr) / arr.length : 0);

function shuffle(arr, r) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ---------- Metadados das 8 features ----------
const FEATURE_NAMES = [
  "Pregnancies",
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI",
  "DiabetesPedigreeFunction",
  "Age",
];
const FEATURE_PT = [
  "Gestações",
  "Glicose",
  "Pressão arterial",
  "Espessura da pele",
  "Insulina",
  "IMC",
  "Histórico familiar",
  "Idade",
];
const N_FEATURES = 8;
const COLS_TO_IMPUTE = [1, 2, 3, 4, 5];

// ============================================================
// 1. PREPARAÇÃO DE DADOS (porte de preparacao_de_dados.py)
// ============================================================
const DATA_URL =
  "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv";

async function carregarDados() {
  const resp = await fetch(DATA_URL);
  if (!resp.ok)
    throw new Error("Falha ao baixar o dataset (HTTP " + resp.status + ")");
  const text = await resp.text();
  const X = [],
    y = [];
  for (const line of text.trim().split(/\r?\n/)) {
    const parts = line.split(",").map(Number);
    if (parts.length < 9 || parts.some(isNaN)) continue;
    X.push(parts.slice(0, 8));
    y.push(parts[8]);
  }
  return { X, y };
}

function medianaSemZeros(values) {
  const v = values.filter((x) => x !== 0 && !isNaN(x)).sort((a, b) => a - b);
  if (v.length === 0) return 0;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

function dividirEstratificado(y, testFrac, r) {
  const byClass = {};
  y.forEach((c, i) => (byClass[c] = byClass[c] || []).push(i));
  let train = [],
    test = [];
  for (const c of Object.keys(byClass)) {
    const idxs = shuffle(byClass[c], r);
    const nTest = Math.round(idxs.length * testFrac);
    test.push(...idxs.slice(0, nTest));
    train.push(...idxs.slice(nTest));
  }
  return [shuffle(train, r), shuffle(test, r)];
}

function kFoldEstratificado(yTrain, k, r) {
  const byClass = {};
  yTrain.forEach((c, i) => (byClass[c] = byClass[c] || []).push(i));
  const assign = new Array(yTrain.length);
  for (const c of Object.keys(byClass)) {
    shuffle(byClass[c], r).forEach((pos, j) => (assign[pos] = j % k));
  }
  const folds = [];
  for (let f = 0; f < k; f++) {
    const train = [],
      val = [];
    for (let i = 0; i < yTrain.length; i++) {
      if (assign[i] === f) val.push(i);
      else train.push(i);
    }
    folds.push([train, val]);
  }
  return folds;
}

function prepararDados(rawX, rawY, r) {
  // Split estratificado 80/20
  const [trainIdx, testIdx] = dividirEstratificado(rawY, 0.2, r);

  const Xtr = trainIdx.map((i) => rawX[i].slice());
  const Xte = testIdx.map((i) => rawX[i].slice());
  const yTrain = trainIdx.map((i) => rawY[i]);
  const yTest = testIdx.map((i) => rawY[i]);

  // Imputação da mediana (exclusiva do treino)
  const medians = {};
  for (const c of COLS_TO_IMPUTE) {
    medians[c] = medianaSemZeros(Xtr.map((row) => row[c]));
  }
  for (const row of Xtr)
    for (const c of COLS_TO_IMPUTE) if (row[c] === 0) row[c] = medians[c];
  for (const row of Xte)
    for (const c of COLS_TO_IMPUTE) if (row[c] === 0) row[c] = medians[c];

  // Normalização (StandardScaler, ddof=0) ajustada no treino
  const means = new Array(N_FEATURES).fill(0);
  const stds = new Array(N_FEATURES).fill(0);
  for (let c = 0; c < N_FEATURES; c++) {
    const col = Xtr.map((row) => row[c]);
    means[c] = mean(col);
    stds[c] = Math.sqrt(mean(col.map((v) => (v - means[c]) ** 2))) || 1;
  }
  const scale = (row) => row.map((v, c) => (v - means[c]) / stds[c]);
  const XtrScaled = Xtr.map(scale);
  const XteScaled = Xte.map(scale); // teste já imputado + escalonado

  // Limites para inicialização de t e folds de CV
  const tBounds = [];
  for (let c = 0; c < N_FEATURES; c++) {
    const col = XtrScaled.map((row) => row[c]);
    tBounds.push([Math.min(...col), Math.max(...col)]);
  }
  const stdsTrain = new Array(N_FEATURES).fill(1); // após scaler ~1
  const cvFolds = kFoldEstratificado(yTrain, 5, r);

  return {
    XtrScaled,
    XteScaled,
    yTrain,
    yTest,
    cvFolds,
    tBounds,
    stdsTrain,
    means,
    stds,
  };
}

// ============================================================
// 2. INDIVÍDUO E OPERADORES GENÉTICOS (porte de classificador.py)
// ============================================================
function novoIndividuo(tBounds) {
  return {
    s: Array.from({ length: N_FEATURES }, () => (rng() < 0.5 ? 0 : 1)),
    d: Array.from({ length: N_FEATURES }, () => (rng() < 0.5 ? 0 : 1)),
    t: tBounds.map((b) => b[0] + rng() * (b[1] - b[0])),
    w: Array.from({ length: N_FEATURES }, () => rng()),
    k: rng(),
    fitness: -1.0,
  };
}

function copiar(ind) {
  return {
    s: ind.s.slice(),
    d: ind.d.slice(),
    t: ind.t.slice(),
    w: ind.w.slice(),
    k: ind.k,
    fitness: ind.fitness,
  };
}

function crossover(p1, p2, crossoverRate) {
  if (rng() < crossoverRate) {
    const c1 = copiar(p1),
      c2 = copiar(p2);
    // Cruzamento de 1 ponto para genes binários (s + d concatenados)
    const bin1 = [...p1.s, ...p1.d];
    const bin2 = [...p2.s, ...p2.d];
    const cp = 1 + Math.floor(rng() * 15);
    const nb1 = bin1.slice(0, cp).concat(bin2.slice(cp));
    const nb2 = bin2.slice(0, cp).concat(bin1.slice(cp));
    c1.s = nb1.slice(0, 8);
    c1.d = nb1.slice(8, 16);
    c2.s = nb2.slice(0, 8);
    c2.d = nb2.slice(8, 16);

    // BLX-alpha para valores reais (t e w)
    const alpha = 0.5;
    for (let i = 0; i < N_FEATURES; i++) {
      let cmin = Math.min(p1.t[i], p2.t[i]),
        cmax = Math.max(p1.t[i], p2.t[i]);
      let diff = cmax - cmin;
      c1.t[i] = cmin - alpha * diff + rng() * (diff * (1 + 2 * alpha));
      c2.t[i] = cmin - alpha * diff + rng() * (diff * (1 + 2 * alpha));
      let wmin = Math.min(p1.w[i], p2.w[i]),
        wmax = Math.max(p1.w[i], p2.w[i]);
      let wdiff = wmax - wmin;
      c1.w[i] = clip(
        wmin - alpha * wdiff + rng() * (wdiff * (1 + 2 * alpha)),
        0,
        1,
      );
      c2.w[i] = clip(
        wmin - alpha * wdiff + rng() * (wdiff * (1 + 2 * alpha)),
        0,
        1,
      );
    }
    // BLX-alpha para o limiar de inferência k
    let kmin = Math.min(p1.k, p2.k),
      kmax = Math.max(p1.k, p2.k);
    let kdiff = kmax - kmin;
    c1.k = clip(kmin - alpha * kdiff + rng() * (kdiff * (1 + 2 * alpha)), 0, 1);
    c2.k = clip(kmin - alpha * kdiff + rng() * (kdiff * (1 + 2 * alpha)), 0, 1);
    return [c1, c2];
  }
  return [copiar(p1), copiar(p2)];
}

function mutate(ind, stdsTrain, mutationRate) {
  for (let i = 0; i < N_FEATURES; i++) {
    if (rng() < mutationRate) ind.s[i] = 1 - ind.s[i];
    if (rng() < mutationRate) ind.d[i] = 1 - ind.d[i];
    if (rng() < mutationRate) ind.t[i] += gauss(0, stdsTrain[i]);
    if (rng() < mutationRate) ind.w[i] = clip(ind.w[i] + gauss(0, 0.1), 0, 1);
  }
  if (rng() < mutationRate) ind.k = clip(ind.k + gauss(0, 0.1), 0, 1);
}

// ---------- Função objetivo (porte de calcular_aptidao) ----------
function calcularAptidao(ind, X, y, cvFolds) {
  const { s, t, d, k, w } = ind;
  if (sum(s) === 0) return 0.0;
  const activeW = w.map((wi, i) => (s[i] === 1 ? wi : 0));
  const totalWeight = sum(activeW);
  if (totalWeight <= 0) return 0.0;

  const f1s = [];
  for (const [, valIdx] of cvFolds) {
    const yval = [],
      preds = [];
    for (const idx of valIdx) {
      const x = X[idx];
      let score = 0;
      for (let i = 0; i < N_FEATURES; i++) {
        if (s[i] === 1) {
          const met =
            (d[i] === 1 && x[i] > t[i]) || (d[i] === 0 && x[i] < t[i]);
          if (met) score += activeW[i];
        }
      }
      score /= totalWeight;
      preds.push(score >= k ? 1 : 0);
      yval.push(y[idx]);
    }
    f1s.push(f1Macro(yval, preds));
  }
  return mean(f1s);
}

function f1Macro(yTrue, yPred) {
  let total = 0;
  for (const c of [0, 1]) {
    let tp = 0,
      fp = 0,
      fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const p = yPred[i] === c,
        a = yTrue[i] === c;
      if (p && a) tp++;
      else if (p && !a) fp++;
      else if (!p && a) fn++;
    }
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    total += prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  }
  return total / 2;
}

// ---------- Inferência no conjunto de teste (porte de predict) ----------
function pontuarTeste(ind, Xte) {
  const { s, t, d, k, w } = ind;
  const activeW = w.map((wi, i) => (s[i] === 1 ? wi : 0));
  const totalWeight = sum(activeW);
  const scores = [],
    preds = [];
  for (const x of Xte) {
    let score = 0;
    for (let i = 0; i < N_FEATURES; i++) {
      if (s[i] === 1) {
        const met = (d[i] === 1 && x[i] > t[i]) || (d[i] === 0 && x[i] < t[i]);
        if (met) score += activeW[i];
      }
    }
    score = totalWeight > 0 ? score / totalWeight : 0;
    scores.push(score);
    preds.push(score >= k ? 1 : 0);
  }
  return { scores, preds, k };
}

// ============================================================
// 3. ESTADO DA EVOLUÇÃO E LAÇO DE ANIMAÇÃO
// ============================================================
const ui = {};
let DATA = null; // dados crus carregados uma vez
let PREP = null; // dados preparados da execução atual
let state = null; // estado da evolução em andamento
let rafId = null;

function setStatus(msg) {
  if (ui.status) ui.status.textContent = msg;
}

function iniciarEvolucao(cfg) {
  if (rafId) cancelAnimationFrame(rafId);

  rng = mulberry32(Date.now());
  PREP = prepararDados(DATA.X, DATA.y, rng);

  const pop = Array.from({ length: cfg.popSize }, () =>
    novoIndividuo(PREP.tBounds),
  );

  state = {
    cfg,
    pop,
    gen: 0,
    bestGlobal: null,
    bestFitnessGlobal: -1,
    semMelhora: 0,
    bestHist: [],
    avgHist: [],
    lastStep: 0,
    running: true,
  };

  ui.dashboard.classList.add("active");
  rafId = requestAnimationFrame(loop);
}

function passoGeracao() {
  const { pop, cfg } = state;

  // Avaliação
  for (const ind of pop) {
    if (ind.fitness === -1.0)
      ind.fitness = calcularAptidao(
        ind,
        PREP.XtrScaled,
        PREP.yTrain,
        PREP.cvFolds,
      );
  }
  pop.sort((a, b) => b.fitness - a.fitness);

  if (pop[0].fitness > state.bestFitnessGlobal) {
    state.bestFitnessGlobal = pop[0].fitness;
    state.bestGlobal = copiar(pop[0]);
    state.semMelhora = 0;
  } else {
    state.semMelhora++;
  }

  state.bestHist.push(state.bestFitnessGlobal);
  state.avgHist.push(mean(pop.map((p) => p.fitness)));

  // Critérios de parada
  const parar = state.semMelhora >= cfg.patience || state.gen >= cfg.maxGen - 1;
  if (parar) {
    state.running = false;
    return;
  }

  // Nova população: elitismo + torneio + crossover + mutação
  const elite = Math.min(10, Math.floor(cfg.popSize * 0.1)) || 1;
  const novo = pop.slice(0, elite).map(copiar);
  while (novo.length < cfg.popSize) {
    const t1 = torneio(pop);
    const t2 = torneio(pop);
    const [c1, c2] = crossover(t1, t2, cfg.crossoverRate);
    mutate(c1, PREP.stdsTrain, cfg.mutationRate);
    mutate(c2, PREP.stdsTrain, cfg.mutationRate);
    c1.fitness = -1.0;
    c2.fitness = -1.0;
    novo.push(c1, c2);
  }
  state.pop = novo.slice(0, cfg.popSize);
  state.gen++;
}

function torneio(pop) {
  let best = null;
  for (let i = 0; i < 5; i++) {
    const cand = pop[Math.floor(rng() * pop.length)];
    if (!best || cand.fitness > best.fitness) best = cand;
  }
  return best;
}

function loop(ts) {
  if (!state) return;
  const gap = 1000 / state.cfg.genPerSec;
  if (state.running && ts - state.lastStep >= gap) {
    state.lastStep = ts;
    passoGeracao();
    desenharTudo();
  }
  if (state.running) {
    rafId = requestAnimationFrame(loop);
  } else {
    desenharTudo();
    setStatus(
      `Evolução concluída — Geração ${state.gen} | Melhor F1-macro (CV): ` +
        state.bestFitnessGlobal.toFixed(4),
    );
  }
}

// ============================================================
// 4. RENDERIZAÇÃO
// ============================================================
function desenharTudo() {
  const best = state.bestGlobal;
  if (!best) return;
  setStatus(
    `Geração ${state.gen} / ${state.cfg.maxGen}  •  ` +
      `Melhor F1-macro (CV): ${state.bestFitnessGlobal.toFixed(4)}  •  ` +
      (state.running
        ? `estagnação ${state.semMelhora}/${state.cfg.patience}`
        : "parado"),
  );
  desenharFitness();
  atualizarRegra(best);
  const teste = pontuarTeste(best, PREP.XteScaled);
  desenharHistograma(teste);
  atualizarMetricas(teste);
}

function fitCanvas(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (
    canvas.width !== Math.round(r.width * dpr) ||
    canvas.height !== Math.round(r.height * dpr)
  ) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

function desenharFitness() {
  const { ctx, w, h } = fitCanvas(ui.fitnessCanvas);
  ctx.clearRect(0, 0, w, h);
  const padL = 38,
    padB = 22,
    padT = 12,
    padR = 10;
  const plotW = w - padL - padR,
    plotH = h - padT - padB;
  const n = state.bestHist.length;
  const maxX = Math.max(state.cfg.maxGen, 1);

  // grade + eixos
  ctx.strokeStyle = "rgba(120,150,180,0.15)";
  ctx.fillStyle = "#7790a5";
  ctx.font = "10px Segoe UI";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const val = i / 5;
    const yy = padT + plotH * (1 - val);
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(w - padR, yy);
    ctx.stroke();
    ctx.fillText(val.toFixed(1), 6, yy + 3);
  }

  const px = (i) => padL + (maxX > 1 ? (i / maxX) * plotW : 0);
  const py = (v) => padT + plotH * (1 - clip(v, 0, 1));

  const drawLine = (arr, color, width) => {
    if (arr.length === 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    arr.forEach((v, i) =>
      i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)),
    );
    ctx.stroke();
  };
  drawLine(state.avgHist, "#8a6dff", 1.5);
  drawLine(state.bestHist, "#00e5ff", 2);

  // marcador do melhor atual
  if (n > 0) {
    const v = state.bestHist[n - 1];
    ctx.fillStyle = "#00e5ff";
    ctx.beginPath();
    ctx.arc(px(n - 1), py(v), 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

let regraConstruida = false;
function construirRegra() {
  const cont = ui.ruleRows;
  cont.innerHTML = "";
  for (let i = 0; i < N_FEATURES; i++) {
    const row = document.createElement("div");
    row.className = "rule-row";
    row.innerHTML = `
      <span class="rule-feat">${FEATURE_PT[i]}<small>${FEATURE_NAMES[i]}</small></span>
      <span class="rule-op"></span>
      <span class="rule-thr"></span>
      <span class="rule-wbar"><span class="rule-wfill"></span></span>
      <span class="rule-wval"></span>`;
    cont.appendChild(row);
  }
  regraConstruida = true;
}

function atualizarRegra(best) {
  if (!regraConstruida) construirRegra();
  const rows = ui.ruleRows.children;
  for (let i = 0; i < N_FEATURES; i++) {
    const row = rows[i];
    const ativo = best.s[i] === 1;
    row.classList.toggle("inactive", !ativo);
    // limiar em unidades originais (inverso do StandardScaler)
    const tOrig = best.t[i] * PREP.stds[i] + PREP.means[i];
    row.querySelector(".rule-op").textContent = ativo
      ? best.d[i] === 1
        ? ">"
        : "<"
      : "—";
    row.querySelector(".rule-thr").textContent = ativo ? tOrig.toFixed(1) : "";
    row.querySelector(".rule-wfill").style.width =
      (best.w[i] * 100).toFixed(0) + "%";
    row.querySelector(".rule-wval").textContent = ativo
      ? best.w[i].toFixed(2)
      : "";
  }
  ui.kValue.textContent = best.k.toFixed(3);
  ui.kFill.style.width = (best.k * 100).toFixed(0) + "%";
}

// Strip plot: cada paciente é um ponto, duas faixas (real 0 / real 1).
// Corretos = preenchidos, errados (FP/FN) = contorno vermelho/laranja.
function desenharHistograma(teste) {
  const { ctx, w, h } = fitCanvas(ui.histCanvas);
  ctx.clearRect(0, 0, w, h);

  const padL = 38,
    padR = 12,
    padT = 12,
    padB = 22;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Duas faixas horizontais: topo = saudável (0), baixo = diabético (1)
  const laneH = plotH / 2;
  const lane = [padT + laneH * 0.5, padT + laneH * 1.5]; // centro de cada faixa

  // Rótulos das faixas
  ctx.font = "10px Segoe UI";
  ctx.textAlign = "right";
  ctx.fillStyle = "rgba(0,229,255,0.85)";
  ctx.fillText("Saudável", padL - 5, lane[0] + 4);
  ctx.fillStyle = "rgba(255,99,132,0.9)";
  ctx.fillText("Diabético", padL - 5, lane[1] + 4);
  ctx.textAlign = "left";

  // Divisória entre faixas
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT + laneH);
  ctx.lineTo(w - padR, padT + laneH);
  ctx.stroke();

  // Jitter vertical determinístico por paciente (sem aleatoriedade a cada frame)
  const jitterAmp = laneH * 0.36;
  const jitter = (i) => {
    // hash simples: distribui uniformemente dentro da faixa
    const h = Math.sin(i * 127.1 + 311.7) * 43758.5453;
    return (h - Math.floor(h) - 0.5) * 2 * jitterAmp;
  };

  const px = (sc) => padL + sc * plotW;
  const R = 3.5;

  teste.scores.forEach((sc, i) => {
    const realClass = PREP.yTest[i];
    const pred = teste.preds[i];
    const correto = pred === realClass;
    const cx = px(sc);
    const cy = lane[realClass] + jitter(i);

    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);

    if (correto) {
      // TN (azul) ou TP (rosa)
      ctx.fillStyle =
        realClass === 0 ? "rgba(0,229,255,0.65)" : "rgba(255,99,132,0.75)";
      ctx.fill();
    } else {
      // FP ou FN: contorno laranja destacado, interior semitransparente
      ctx.fillStyle = "rgba(255,180,0,0.18)";
      ctx.fill();
      ctx.strokeStyle = "#ffb300";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  });

  // Linha vertical do limiar k
  const kx = px(teste.k);
  ctx.strokeStyle = "#ffd166";
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(kx, padT);
  ctx.lineTo(kx, padT + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Rótulo do k
  const kLabel = "k=" + teste.k.toFixed(2);
  ctx.font = "10px Segoe UI";
  ctx.fillStyle = "#ffd166";
  const kLabelX = kx + 5 + 36 > w - padR ? kx - 40 : kx + 5;
  ctx.fillText(kLabel, kLabelX, padT + 10);

  // Eixo X: linha base + ticks em 0, 0.25, 0.5, 0.75, 1
  ctx.strokeStyle = "rgba(120,150,180,0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  ctx.strokeStyle = "rgba(120,150,180,0.5)";
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    const tx = padL + v * plotW;
    ctx.beginPath();
    ctx.moveTo(tx, padT + plotH);
    ctx.lineTo(tx, padT + plotH + 4);
    ctx.stroke();
  }
}

function atualizarMetricas(teste) {
  const y = PREP.yTest,
    p = teste.preds;
  let tp = 0,
    tn = 0,
    fp = 0,
    fn = 0;
  for (let i = 0; i < y.length; i++) {
    if (y[i] === 1 && p[i] === 1) tp++;
    else if (y[i] === 0 && p[i] === 0) tn++;
    else if (y[i] === 0 && p[i] === 1) fp++;
    else fn++;
  }
  const acc = (tp + tn) / y.length;
  const f1 = f1Macro(y, p);
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec = tp + fn > 0 ? tp / (tp + fn) : 0;

  ui.cmTN.textContent = tn;
  ui.cmFP.textContent = fp;
  ui.cmFN.textContent = fn;
  ui.cmTP.textContent = tp;
  ui.mAcc.textContent = (acc * 100).toFixed(1) + "%";
  ui.mF1.textContent = f1.toFixed(3);
  ui.mPrec.textContent = (prec * 100).toFixed(1) + "%";
  ui.mRec.textContent = (rec * 100).toFixed(1) + "%";
}

// ============================================================
// 5. WIRING DA INTERFACE
// ============================================================
function bindUI() {
  ui.status = document.getElementById("cls-status");
  ui.dashboard = document.getElementById("cls-dashboard");
  ui.fitnessCanvas = document.getElementById("fitnessCanvas");
  ui.histCanvas = document.getElementById("histCanvas");
  ui.ruleRows = document.getElementById("ruleRows");
  ui.kValue = document.getElementById("kValue");
  ui.kFill = document.getElementById("kFill");
  ui.cmTN = document.getElementById("cmTN");
  ui.cmFP = document.getElementById("cmFP");
  ui.cmFN = document.getElementById("cmFN");
  ui.cmTP = document.getElementById("cmTP");
  ui.mAcc = document.getElementById("mAcc");
  ui.mF1 = document.getElementById("mF1");
  ui.mPrec = document.getElementById("mPrec");
  ui.mRec = document.getElementById("mRec");

  const bind = (id, valId, suffix = "%") => {
    const el = document.getElementById(id);
    const out = document.getElementById(valId);
    el.addEventListener("input", () => (out.textContent = el.value + suffix));
  };
  bind("clsCrossover", "clsCrossoverVal");
  bind("clsMutation", "clsMutationVal");
  document.getElementById("clsSpeed").addEventListener("input", function () {
    document.getElementById("clsSpeedVal").textContent = this.value + " ger/s";
  });

  document.getElementById("clsStartBtn").addEventListener("click", async () => {
    if (!DATA) {
      setStatus("Baixando dataset Pima Indians Diabetes...");
      try {
        DATA = await carregarDados();
      } catch (e) {
        setStatus("Erro: " + e.message);
        return;
      }
    }
    const cfg = {
      popSize: clip(
        parseInt(document.getElementById("clsPop").value) || 80,
        10,
        500,
      ),
      maxGen: clip(
        parseInt(document.getElementById("clsGen").value) || 150,
        1,
        1000,
      ),
      patience: clip(
        parseInt(document.getElementById("clsPatience").value) || 25,
        1,
        200,
      ),
      crossoverRate:
        (parseFloat(document.getElementById("clsCrossover").value) || 80) / 100,
      mutationRate:
        (parseFloat(document.getElementById("clsMutation").value) || 2) / 100,
      genPerSec: clip(
        parseFloat(document.getElementById("clsSpeed").value) || 12,
        1,
        60,
      ),
    };
    iniciarEvolucao(cfg);
  });

  window.addEventListener("resize", () => {
    if (state && state.bestGlobal) desenharTudo();
  });
}

document.addEventListener("DOMContentLoaded", bindUI);
