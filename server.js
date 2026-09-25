const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;
const MAX_SESSIONS = 50;
const PREDS_FILE = path.join('/tmp', 'preds.json');

const API_CONFIGS = {
  1: { name: "SUNWIN", url: "https://kwinstore.com/sunwin/tx/history/c806cf04a7fdf1cace25db6c7a8bdd8e048242145ee726dc", algorithm: "predictPattern", type: "standard" },
  2: { name: "LC79-MD5", url: "https://wtxmd52.tele68.com/v1/txmd5/sessions", algorithm: "predictPattern", type: "lc79" },
  3: { name: "MAX789", url: "https://taixiu.maksh3979madfw.com/api/luckydice/GetSoiCau", algorithm: "predictPattern", type: "max789" },
  4: { name: "HITCLUB", url: "https://kwinstore.com/hitclub/tx/history/c806cf04a7fdf1cace25db6c7a8bdd8e048242145ee726dc", algorithm: "predictPattern", type: "standard" },
  5: { name: "68GB", url: "https://winds-fonts-seq-jaguar.trycloudflare.com/history", algorithm: "predictPattern", type: "68gb" },
  6: { name: "SUMCLUB", url: "https://kwinstore.com/sumclub/tx/history/c806cf04a7fdf1cace25db6c7a8bdd8e048242145ee726dc", algorithm: "predictPattern", type: "standard" },
  7: { name: "RIKVIP", url: "https://kwinstore.com/rikvip/tx/history/c806cf04a7fdf1cace25db6c7a8bdd8e048242145ee726dc", algorithm: "predictPattern", type: "standard" },
  8: { name: "SICSUN", url: "https://api.wsktnus8.net/v2/history/getLastResult?gameId=ktrng_3979&size=100&tableId=39791215743193&curPage=1", algorithm: "predictPattern", type: "sicsun" },
  9: { name: "LC79-HU", url: "https://wtx.tele68.com/v1/tx/sessions", algorithm: "predictPattern", type: "lc79" }
};

/* =====================================================================
   SIÊU ENSEMBLE ĐA CHUYÊN GIA - MULTI-EXPERT ENSEMBLE PREDICTOR
   ---------------------------------------------------------------------
   Triết lý: KHÔNG trọng số cố định. KHÔNG thiên vị Tài/Xỉu. KHÔNG bẻ
   cầu máy móc. Mỗi chuyên gia tự đánh giá độ tin cậy từ dữ liệu thực,
   hợp nhất bằng confidence-weighted voting (bỏ phiếu theo độ tin cậy).
   ===================================================================== */

function _seqFrom(hist, maxLen) {
  if (!hist || !hist.length) return [];
  const raw = hist.slice(0, maxLen).map(d => d["kết quả"]);
  raw.reverse();
  return raw.filter(x => x === 'Tài' || x === 'Xỉu');
}

/* 1. Markov bậc N: P(next | N ký hiệu gần nhất) */
function _expMarkov(seq, order) {
  if (seq.length < order + 5) return null;
  const ctx = seq.slice(-order).join('');
  const lastSym = seq[seq.length - 1];
  let cSame = 0, cDiff = 0, total = 0;
  for (let i = 0; i + order < seq.length; i++) {
    if (seq.slice(i, i + order).join('') === ctx) {
      if (seq[i + order] === lastSym) cSame++; else cDiff++;
      total++;
    }
  }
  if (total < 2) return null;
  const pSame = cSame / total;
  const edge = Math.abs(pSame - 0.5) * 2;
  const conf = edge * Math.min(1, total / 8) * (1 + order * 0.15);
  if (conf < 0.05) return null;
  const pred = pSame > 0.5 ? lastSym : (lastSym === 'Tài' ? 'Xỉu' : 'Tài');
  return { pred, conf: Math.min(conf, 1.5) };
}

/* 2. N-gram khớp mẫu tuyệt đối */
function _expNgram(seq, n) {
  if (seq.length < n + 4) return null;
  const ctx = seq.slice(-n).join('');
  let cTai = 0, cXiu = 0, total = 0;
  for (let i = 0; i + n < seq.length; i++) {
    if (seq.slice(i, i + n).join('') === ctx) {
      if (seq[i + n] === 'Tài') cTai++; else cXiu++;
      total++;
    }
  }
  if (total < 1) return null;
  const edge = Math.abs(cTai - cXiu) / total;
  const conf = edge * Math.min(1, total / 4) * (1 + n * 0.05);
  if (conf < 0.05) return null;
  return { pred: cTai >= cXiu ? 'Tài' : 'Xỉu', conf: Math.min(conf, 1.5) };
}

/* 3. Tương đồng mờ: đuôi hiện tại vs mọi vị trí lịch sử (soft-match) */
function _expSimilarity(seq, k) {
  if (seq.length < k + 12) return null;
  const tail = seq.slice(-k);
  let wTai = 0, wXiu = 0, totalW = 0;
  for (let i = 0; i + k < seq.length; i++) {
    const win = seq.slice(i, i + k);
    let match = 0;
    for (let j = 0; j < k; j++) if (win[j] === tail[j]) match++;
    const sim = match / k;
    if (sim < 0.6) continue;
    const w = sim * sim * sim;
    if (seq[i + k] === 'Tài') wTai += w; else wXiu += w;
    totalW += w;
  }
  if (totalW < 0.4) return null;
  const pTai = wTai / totalW;
  const edge = Math.abs(pTai - 0.5) * 2;
  const conf = edge * Math.min(1, totalW / 4) * 0.9;
  if (conf < 0.05) return null;
  return { pred: pTai > 0.5 ? 'Tài' : 'Xỉu', conf: Math.min(conf, 1.5) };
}

/* 4. Streak data-driven: KHÔNG bẻ cầu cứng, dùng phân bố lịch sử */
function _expStreak(seq) {
  if (seq.length < 12) return null;
  const lastSym = seq[seq.length - 1];
  let curLen = 1;
  for (let i = seq.length - 2; i >= 0 && seq[i] === lastSym; i--) curLen++;

  const runLens = [];
  let run = 0;
  const cutoff = seq.length - curLen;
  for (let i = 0; i < cutoff; i++) {
    if (seq[i] === lastSym) run++;
    else { if (run > 0) runLens.push(run); run = 0; }
  }
  if (run > 0) runLens.push(run);
  if (runLens.length < 4) return null;

  const reached = runLens.filter(l => l >= curLen).length;
  const exceeded = runLens.filter(l => l > curLen).length;
  if (reached < 2) return null;
  const pContinue = exceeded / reached;

  const edge = Math.abs(pContinue - 0.5) * 2;
  const conf = edge * Math.min(1, reached / 6) * 0.95;
  if (conf < 0.05) return null;

  if (pContinue > 0.5) return { pred: lastSym, conf };
  return { pred: lastSym === 'Tài' ? 'Xỉu' : 'Tài', conf };
}

/* 5. Alternation - phải được lịch sử xác nhận mới tin */
function _expAlternation(seq) {
  if (seq.length < 10) return null;
  const L = Math.min(7, seq.length);
  const tail = seq.slice(-L);
  let ac = 0;
  for (let i = 1; i < L; i++) if (tail[i] !== tail[i - 1]) ac++;
  if (ac / (L - 1) < 0.85) return null;

  let contCount = 0, breakCount = 0;
  for (let i = 0; i + L < seq.length; i++) {
    const w = seq.slice(i, i + L);
    let a = 0;
    for (let j = 1; j < L; j++) if (w[j] !== w[j - 1]) a++;
    if (a / (L - 1) >= 0.85) {
      if (seq[i + L] !== w[L - 1]) contCount++; else breakCount++;
    }
  }
  const total = contCount + breakCount;
  if (total < 2) return null;
  const pCont = contCount / total;
  const edge = Math.abs(pCont - 0.5) * 2;
  const conf = edge * Math.min(1, total / 5) * 0.85;
  if (conf < 0.05) return null;
  if (pCont > 0.5) return { pred: tail[L - 1] === 'Tài' ? 'Xỉu' : 'Tài', conf };
  return { pred: tail[L - 1], conf };
}

/* 6. Bayesian Dirichlet bậc 1 */
function _expBayes(seq) {
  if (seq.length < 8) return null;
  const lastSym = seq[seq.length - 1];
  let aSame = 1, aDiff = 1;
  for (let i = 0; i < seq.length - 1; i++) {
    if (seq[i] === lastSym) {
      if (seq[i + 1] === lastSym) aSame++; else aDiff++;
    }
  }
  const total = aSame + aDiff - 2;
  if (total < 4) return null;
  const pSame = aSame / (aSame + aDiff);
  const edge = Math.abs(pSame - 0.5) * 2;
  const conf = edge * Math.min(1, total / 12);
  if (conf < 0.05) return null;
  const pred = pSame > 0.5 ? lastSym : (lastSym === 'Tài' ? 'Xỉu' : 'Tài');
  return { pred, conf };
}

/* 7. Z-score tần suất (mean reversion data-driven) */
function _expFrequencyZ(seq) {
  if (seq.length < 20) return null;
  const n = seq.length;
  const cTai = seq.filter(x => x === 'Tài').length;
  const p = cTai / n;
  const z = (p - 0.5) / Math.sqrt(0.25 / n);
  if (Math.abs(z) < 1.6) return null;
  const conf = Math.min(0.6, (Math.abs(z) - 1.6) * 0.18);
  if (conf < 0.05) return null;
  return { pred: z > 0 ? 'Xỉu' : 'Tài', conf };
}

/* 8. Recency exponential (momentum có trọng số thời gian) */
function _expRecencyFreq(seq) {
  if (seq.length < 10) return null;
  let wTai = 0, wXiu = 0, w = 1;
  for (let i = seq.length - 1; i >= 0; i--) {
    if (seq[i] === 'Tài') wTai += w; else wXiu += w;
    w *= 0.9;
  }
  const total = wTai + wXiu;
  if (total < 1) return null;
  const pTai = wTai / total;
  const edge = Math.abs(pTai - 0.5);
  if (edge < 0.12) return null;
  return { pred: pTai > 0.5 ? 'Tài' : 'Xỉu', conf: Math.min(0.6, edge * 1.2) };
}

/* 9. Entropy bigram: chỉ tin khi chuỗi có cấu trúc thực sự */
function _expEntropy(seq) {
  if (seq.length < 15) return null;
  const counts = {};
  for (let i = 0; i < seq.length - 1; i++) {
    const bg = seq[i][0] + seq[i + 1][0];
    counts[bg] = (counts[bg] || 0) + 1;
  }
  const total = seq.length - 1;
  let H = 0;
  for (const k in counts) {
    const p = counts[k] / total;
    H -= p * Math.log2(p);
  }
  const normH = H / 2;
  if (normH > 0.92) return null;
  const lastBg = seq[seq.length - 2][0] + seq[seq.length - 1][0];
  let cTai = 0, cXiu = 0;
  for (let i = 0; i < seq.length - 2; i++) {
    if (seq[i][0] + seq[i + 1][0] === lastBg) {
      if (seq[i + 2] === 'Tài') cTai++; else cXiu++;
    }
  }
  const tot2 = cTai + cXiu;
  if (tot2 < 2) return null;
  const edge = Math.abs(cTai - cXiu) / tot2;
  const conf = edge * (1 - normH) * Math.min(1, tot2 / 6) * 1.1;
  if (conf < 0.05) return null;
  return { pred: cTai > cXiu ? 'Tài' : 'Xỉu', conf: Math.min(conf, 1.2) };
}

/* Hợp nhất ensemble - bỏ phiếu có trọng số ĐỘNG */
function _predictEnsemble(hist) {
  const seq = _seqFrom(hist, 90);
  if (seq.length < 6) return null;

  const experts = [];
  for (let o = 1; o <= 4; o++) { const e = _expMarkov(seq, o); if (e) experts.push(e); }
  for (let n = 3; n <= 5; n++) { const e = _expNgram(seq, n); if (e) experts.push(e); }
  for (let k = 4; k <= 7; k++) { const e = _expSimilarity(seq, k); if (e) experts.push(e); }

  const add = e => { if (e) experts.push(e); };
  add(_expStreak(seq));
  add(_expAlternation(seq));
  add(_expBayes(seq));
  add(_expFrequencyZ(seq));
  add(_expRecencyFreq(seq));
  add(_expEntropy(seq));

  if (!experts.length) return null;

  let tai = 0, xiu = 0;
  for (const e of experts) {
    if (e.pred === 'Tài') tai += e.conf;
    else if (e.pred === 'Xỉu') xiu += e.conf;
  }
  const total = tai + xiu;
  if (total <= 1e-9) return null;

  let pred;
  if (Math.abs(tai - xiu) < 1e-6) {
    pred = seq[seq.length - 1] === 'Tài' ? 'Xỉu' : 'Tài';
  } else {
    pred = tai > xiu ? 'Tài' : 'Xỉu';
  }

  const dom = Math.max(tai, xiu) / total;
  const strength = Math.min(1, total / 4);
  const confidence = 50 + 47 * (2 * dom - 1) * (0.6 + 0.4 * strength);

  return { pred, conf: confidence, tai, xiu, nExperts: experts.length };
}

function predictPattern(phien, hist) {
  const r = _predictEnsemble(hist);
  if (r) return r.pred;
  // Fallback cổ điển khi thiếu dữ liệu
  if (!hist || hist.length < 5) return (phien % 2 === 0) ? 'Tài' : 'Xỉu';
  const recent = hist.slice(0, 5).map(d => d["kết quả"]).reverse();
  let isAlt = true;
  for (let i = 0; i < 4; i++) if (recent[i] === recent[i + 1]) { isAlt = false; break; }
  if (isAlt) return recent[4] === 'Tài' ? 'Xỉu' : 'Tài';
  return recent[4];
}

const ALGORITHMS = { predictPattern };

function normalizeStandard(apiId, raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  raw.forEach(item => {
    const phien = item.phiên || item.phien;
    const result = item["kết quả"] || item.ket_qua;
    const time = item.updatedAt || item.thoi_gian || '';
    if (!phien || !result) return;
    if (item.status === 'đang chạy') return;
    const clean = (String(result).trim().toLowerCase() === 'tài' || String(result).trim().toLowerCase() === 'tai') ? 'Tài' : 'Xỉu';
    out.push({
      "phiên": Number(phien),
      "kết quả": clean,
      "updatedAt": time,
      "d1": item.d1 || item.xuc_xac_1,
      "d2": item.d2 || item.xuc_xac_2,
      "d3": item.d3 || item.xuc_xac_3
    });
  });
  return out.sort((a, b) => b["phiên"] - a["phiên"]);
}

function normalizeLC79(raw) {
  if (!raw || !Array.isArray(raw.list)) return [];
  const out = [];
  raw.list.forEach(item => {
    const phien = item.id;
    const result = item.resultTruyenThong;
    if (!phien || !result) return;
    const lower = String(result).trim().toLowerCase();
    let clean = '';
    if (lower === 'tai' || lower === 'tài') clean = 'Tài';
    else if (lower === 'xiu' || lower === 'xỉu') clean = 'Xỉu';
    else return;
    out.push({ "phiên": Number(phien), "kết quả": clean, "updatedAt": '', "dices": item.dices });
  });
  return out.sort((a, b) => b["phiên"] - a["phiên"]);
}

function normalizeMax789(raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  raw.forEach(item => {
    const phien = item.SessionId;
    const sum = item.DiceSum;
    if (!phien || sum === undefined) return;
    const clean = sum >= 11 ? 'Tài' : 'Xỉu';
    out.push({
      "phiên": Number(phien),
      "kết quả": clean,
      "updatedAt": item.CreatedDate || '',
      "d1": item.FirstDice,
      "d2": item.SecondDice,
      "d3": item.ThirdDice,
      "tong": sum
    });
  });
  return out.sort((a, b) => b["phiên"] - a["phiên"]);
}

function normalize68GB(raw) {
  if (!raw || !Array.isArray(raw.lich_su_50)) return [];
  const out = [];
  raw.lich_su_50.forEach(item => {
    const phien = item.phien;
    const result = item.ket_qua;
    if (!phien || !result) return;
    const lower = String(result).trim().toLowerCase();
    let clean = '';
    if (lower === 'tài' || lower === 'tai') clean = 'Tài';
    else if (lower === 'xỉu' || lower === 'xiu') clean = 'Xỉu';
    else if (lower === 'bão' || lower === 'bao') clean = 'Bão';
    else return;
    out.push({
      "phiên": Number(phien),
      "kết quả": clean,
      "updatedAt": item.thoi_gian || '',
      "d1": item.xuc_xac_1,
      "d2": item.xuc_xac_2,
      "d3": item.xuc_xac_3,
      "tong": item.tong
    });
  });
  return out.sort((a, b) => b["phiên"] - a["phiên"]);
}

function normalizeSicsun(raw) {
  if (!raw || !raw.data || !Array.isArray(raw.data.resultList)) return [];
  const out = [];
  raw.data.resultList.forEach(item => {
    const phienStr = String(item.gameNum).replace("#", "");
    const phien = Number(phienStr);
    const score = item.score;
    if (!phien || score === undefined) return;
    const clean = score >= 11 ? 'Tài' : 'Xỉu';
    out.push({
      "phiên": phien,
      "kết quả": clean,
      "updatedAt": '',
      "d1": item.facesList[0],
      "d2": item.facesList[1],
      "d3": item.facesList[2],
      "tong": score
    });
  });
  return out.sort((a, b) => b["phiên"] - a["phiên"]);
}

function formatTime(str) {
  if (!str) return '';
  const d = new Date(str);
  if (isNaN(d.getTime())) return str;
  const p = n => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())} ${p(d.getDate())}/${p(d.getMonth() + 1)}`;
}

let serverPreds = { 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {} };
let reverseMap = { 1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false };

function applyReverse(apiId, prediction) {
  if (!reverseMap[apiId]) return prediction;
  if (prediction === 'Tài') return 'Xỉu';
  if (prediction === 'Xỉu') return 'Tài';
  return prediction;
}

function loadPreds() {
  try {
    if (fs.existsSync(PREDS_FILE)) {
      const parsed = JSON.parse(fs.readFileSync(PREDS_FILE, 'utf8'));
      for (const k in parsed) {
        if (!serverPreds[k]) serverPreds[k] = {};
        Object.assign(serverPreds[k], parsed[k]);
      }
      console.log('✓ loaded preds.json');
    }
  } catch (e) {}
}

let saveTimer = null;
function savePreds() {
  if (saveTimer) return;
  saveTimer = setTimeout(() => {
    saveTimer = null;
    try { fs.writeFileSync(PREDS_FILE, JSON.stringify(serverPreds)); } catch (e) {}
  }, 1000);
}

loadPreds();

const rawCache = {};
const CACHE_MS = 6000;

async function fetchRaw(apiId) {
  const now = Date.now();
  if (rawCache[apiId] && (now - rawCache[apiId].ts) < CACHE_MS) {
    return rawCache[apiId].data;
  }
  const cfg = API_CONFIGS[apiId];
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);
  let response;
  try {
    response = await fetch(cfg.url, { signal: controller.signal });
  } catch (e) {
    try {
      response = await fetch("https://corsproxy.io/?" + encodeURIComponent(cfg.url), { signal: controller.signal });
    } catch (e2) {
      clearTimeout(timer);
      throw new Error('fetch failed');
    }
  }
  clearTimeout(timer);
  if (!response.ok) throw new Error('upstream ' + response.status);
  const raw = await response.json();
  rawCache[apiId] = { data: raw, ts: now };
  return raw;
}

function buildCleanData(apiId, raw) {
  const cfg = API_CONFIGS[apiId];
  if (cfg.type === "68gb") return normalize68GB(raw);
  if (cfg.type === "lc79") return normalizeLC79(raw);
  if (cfg.type === "max789") return normalizeMax789(raw);
  if (cfg.type === "sicsun") return normalizeSicsun(raw);
  if (raw.status === "OK" && Array.isArray(raw.data)) return normalizeStandard(apiId, raw.data);
  if (Array.isArray(raw)) return normalizeStandard(apiId, raw);
  if (Array.isArray(raw.data)) return normalizeStandard(apiId, raw.data);
  return [];
}

async function getRows(apiId) {
  let raw;
  try {
    raw = await fetchRaw(apiId);
  } catch (e) {
    if (rawCache[apiId]) raw = rawCache[apiId].data;
    else return { game: API_CONFIGS[apiId].name, rows: [] };
  }

  let cleanData = buildCleanData(apiId, raw);
  if (cleanData.length === 0) return { game: API_CONFIGS[apiId].name, rows: [] };

  cleanData = cleanData.slice(0, MAX_SESSIONS);

  const cfg = API_CONFIGS[apiId];
  const algo = ALGORITHMS[cfg.algorithm];
  const rows = [];

  const latestPhien = Number(cleanData[0]["phiên"]);
  const nextPhien = latestPhien + 1;

  if (!serverPreds[apiId][nextPhien]) {
    serverPreds[apiId][nextPhien] = algo(nextPhien, cleanData);
    savePreds();
  }

  rows.push({
    game: cfg.name,
    phien: nextPhien,
    time: 'running...',
    prediction: applyReverse(apiId, serverPreds[apiId][nextPhien]),
    result: 'Chờ...',
    verify: '⏳',
    isNext: true
  });

  const keepFrom = Number(cleanData[cleanData.length - 1]["phiên"]);
  let cleaned = false;
  for (const key in serverPreds[apiId]) {
    if (Number(key) < keepFrom) {
      delete serverPreds[apiId][key];
      cleaned = true;
    }
  }
  if (cleaned) savePreds();

  for (let i = 0; i < cleanData.length; i++) {
    const item = cleanData[i];
    const phienNum = Number(item["phiên"]);
    const actualResult = item["kết quả"];

    if (!serverPreds[apiId][phienNum]) {
      const pastData = cleanData.slice(i + 1);
      serverPreds[apiId][phienNum] = algo(phienNum, pastData);
      savePreds();
    }

    const prediction = applyReverse(apiId, serverPreds[apiId][phienNum]);
    const isCorrect = (prediction === actualResult);

    rows.push({
      game: cfg.name,
      phien: phienNum,
      time: formatTime(item["updatedAt"]),
      prediction: prediction,
      result: actualResult,
      verify: isCorrect ? '✅Đúng' : '❌Sai',
      isCorrect,
      isNext: false
    });
  }

  return { game: cfg.name, rows };
}

function calcConfidence(phien, hist) {
  const r = _predictEnsemble(hist);
  if (!r) return '52.0%';
  return r.conf.toFixed(1) + '%';
}

async function buildCommonJSON(apiId) {
  let raw;
  try {
    raw = await fetchRaw(apiId);
  } catch (e) {
    return null;
  }

  let cleanData = buildCleanData(apiId, raw);
  if (cleanData.length === 0) return null;

  cleanData = cleanData.slice(0, MAX_SESSIONS);

  const cfg = API_CONFIGS[apiId];
  const algo = ALGORITHMS[cfg.algorithm];

  const prev = cleanData[0];
  const phienTruoc = prev["phiên"];
  const ketQuaTruoc = prev["kết quả"];
  const timeTruoc = prev["updatedAt"] || '';

  let xucXac = [];
  if (prev.d1 && prev.d2 && prev.d3) {
    xucXac = [prev.d1, prev.d2, prev.d3];
  } else if (prev.dices && Array.isArray(prev.dices)) {
    xucXac = prev.dices;
  }

  const phienNay = phienTruoc + 1;

  if (!serverPreds[apiId][phienNay]) {
    serverPreds[apiId][phienNay] = algo(phienNay, cleanData);
    savePreds();
  }

  const duDoan = applyReverse(apiId, serverPreds[apiId][phienNay]);
  const doTinCay = calcConfidence(phienNay, cleanData);

  return {
    id: "@ZukaNoPro2",
    game: cfg.name,
    phien_truoc: phienTruoc,
    xuc_xac: xucXac,
    ket_qua: ketQuaTruoc,
    thoi_gian: formatTime(timeTruoc),
    phien_nay: phienNay,
    du_doan: duDoan,
    do_tin_cay: doTinCay
  };
}

app.use(express.json({ limit: '50kb' }));

app.post('/api/data', async (req, res) => {
  try {
    const { apiId } = req.body;
    if (!API_CONFIGS[apiId]) return res.status(404).json({ error: 'invalid' });
    const result = await getRows(apiId);
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: 'err' });
  }
});

app.get('/adm', (req, res) => {
  const items = Object.keys(API_CONFIGS).map(id => {
    const numId = Number(id);
    const g = API_CONFIGS[id];
    const rev = reverseMap[numId];
    return `<div class="item">
      <div class="ten">${g.name}</div>
      <button class="nut ${rev ? 'bat' : ''}" onclick="toggle(${numId}, this)" data-id="${numId}">${rev ? 'ĐẢO' : 'KHÔNG ĐẢO'}</button>
    </div>`;
  }).join('');

  res.send(`<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ADM</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; }
  body { background: #1a0820; color: #f0d5e8; padding: 16px; min-height: 100vh; }
  h1 { color: #ff6ec7; font-size: 16px; text-align: center; margin-bottom: 20px; letter-spacing: 1px; font-weight: 700; }
  .khung { max-width: 480px; margin: 0 auto; }
  .item {
    display: flex; justify-content: space-between; align-items: center;
    background: #2a0d26; border: 1px solid #7a2d5c; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
  }
  .ten { font-size: 14px; font-weight: 700; color: #ffd6ee; letter-spacing: 0.5px; }
  .nut {
    background: #4a1535; color: #ff9ecb; border: 1px solid #7a2d5c;
    padding: 8px 16px; border-radius: 16px; font-size: 11px; font-weight: 700;
    cursor: pointer; min-width: 110px; transition: all 0.2s;
  }
  .nut.bat { background: #d63384; color: #fff; border-color: #ff6ec7; }
  .nut:active { transform: scale(0.95); }
</style>
</head>
<body>
<h1>⚙️ QUẢN LÝ ĐẢO DỰ ĐOÁN</h1>
<div class="khung">${items}</div>
<script>
async function toggle(id, btn) {
  const res = await fetch('/adm/toggle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ apiId: id })
  });
  if (!res.ok) return;
  const json = await res.json();
  if (json.reverse) {
    btn.classList.add('bat');
    btn.innerText = 'ĐẢO';
  } else {
    btn.classList.remove('bat');
    btn.innerText = 'KHÔNG ĐẢO';
  }
}
</script>
</body>
</html>`);
});

app.post('/adm/toggle', (req, res) => {
  const { apiId } = req.body;
  if (!reverseMap.hasOwnProperty(apiId)) {
    return res.status(400).json({ error: 'invalid' });
  }
  reverseMap[apiId] = !reverseMap[apiId];
  res.json({ apiId, reverse: reverseMap[apiId] });
});

app.get('/index.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/health', (req, res) => res.json({ ok: true, games: 9 }));

app.get('/preds-status', (req, res) => {
  const counts = {};
  for (const k in serverPreds) counts[k] = Object.keys(serverPreds[k]).length;
  res.json({ file: PREDS_FILE, exists: fs.existsSync(PREDS_FILE), counts });
});

app.get('/sunwin', async (req, res) => {
  const json = await buildCommonJSON(1);
  if (!json) return res.status(503).json({ error: 'no data' });
  res.json(json);
});

app.get('/lc79-md5', async (req, res) => {
  const json = await buildCommonJSON(2);
  if (!json) return res.status(503).json({ error: 'no data' });
  res.json(json);
});

app.get('/max789', async (req, res) => {
  const json = await buildCommonJSON(3);
  if (!json) return res.status(503).json({ error: 'no data' });
  res.json(json);
});

app.get('/hitclub', async (req, res) => {
  const json = await buildCommonJSON(4);
  if (!json) return res.status(503).json({ error: 'no data' });
  res.json(json);
});

app.get('/68gb', async (req, res) => {
  const json = await buildCommonJSON(5);
  if (!json) return res.status(503).json({ error: 'no data' });
  res.json(json);
});

app.get('/sumclub', async (req, res) => {
  const json = await buildCommonJSON(6);
  if (!json) return res.status(503).json({ error: 'no data' });
  res.json(json);
});

app.get('/rikvip', async (req, res) => {
  const json = await buildCommonJSON(7);
  if (!json) return res.status(503).json({ error: 'no data' });
  res.json(json);
});

app.get('/sicsun', async (req, res) => {
  const json = await buildCommonJSON(8);
  if (!json) return res.status(503).json({ error: 'no data' });
  res.json(json);
});

app.get('/lc79-hu', async (req, res) => {
  const json = await buildCommonJSON(9);
  if (!json) return res.status(503).json({ error: 'no data' });
  res.json(json);
});

app.use((req, res) => {
  res.status(404).send('Not Found');
});

async function preload() {
  for (const id of Object.keys(API_CONFIGS)) {
    try {
      await fetchRaw(Number(id));
      console.log(`✓ preload game ${id}`);
    } catch (e) {
      console.log(`✗ preload game ${id} failed`);
    }
  }
}

setInterval(async () => {
  for (const id of Object.keys(API_CONFIGS)) {
    try { await fetchRaw(Number(id)); } catch (e) {}
  }
}, 5000);

app.listen(PORT, () => {
  console.log(`ZukaNoPro2 running on port ${PORT}`);
  preload();
});