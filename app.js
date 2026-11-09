/* 객석 소리지도 — 정적 웹 (외부 라이브러리·서버 요청 없음)
 * 데이터: data/seats.js(SEAT_DATA), data/env.js(ENV_B64), data/dry.js(DRY_B64) — 실험/scripts/build_web_data.py 가 만든다
 */
(() => {
  'use strict';
  const D = window.SEAT_DATA, META = D.meta, SEATS = D.seats;
  const MODES = META.modes, BANDS = META.bands, NBIN = META.nbin, FS = META.fs;
  const STAGE_X = META.stage_edge_x;
  const env = Uint8Array.from(atob(window.ENV_B64), c => c.charCodeAt(0));
  const envAt = (i, mi, bi) => env.subarray(((i * MODES.length + mi) * BANDS.length + bi) * NBIN, ((i * MODES.length + mi) * BANDS.length + bi + 1) * NBIN);
  const qToE = q => (q > 0 ? Math.pow(10, (META.env_db_ref + (q - 255) / 2) / 10) : 0);

  const METRIC = {
    TOT: { name: '종합', desc: '대사 또렷함·소리 크기·감싸는 느낌의 순위를 같은 비중으로 평균한 뒤 다시 순위를 매긴 값. 세 가지가 모두 고르게 높은 자리일수록 높습니다' },
    D50: { name: '대사 또렷함', desc: '소리가 도착하고 0.05초 안에 들어오는 비율(D50). 높을수록 말이 또렷하게 들립니다' },
    C80: { name: '음 또렷함', desc: '0.08초 전후 소리의 비(C80). 높으면 음이 또렷하고, 낮으면 울림이 풍부합니다' },
    G:   { name: '소리 크기', desc: '같은 소리를 냈을 때 이 자리에 도착하는 크기(G)' },
    LF:  { name: '감싸는 느낌', desc: '옆에서 오는 반사의 비율(LF). 높을수록 소리에 둘러싸인 느낌이 납니다' },
  };
  const UNIT = { C80: ' dB', G: ' dB', D50: '', LF: '' };
  const MODE_NOTE = {
    natural: '배우가 마이크 없이 무대 앞 가운데에서 말하거나 노래할 때의 소리입니다(낭독·어쿠스틱 공연 기준). 스피커를 쓰는 공연은 “스피커 확성”도 함께 보세요.',
    pa: '무대 입구 양옆과 위쪽 가운데 스피커로 소리를 키웠다고 가정한 결과입니다. 이 공연장의 음향장비 목록은 공개돼 있지만 설치 위치가 없어, 위치와 구성을 단순화해 가정했습니다. 참고용으로만 보세요.',
  };

  // 종합 순위: 모드별 D50·G·LF 백분위 평균을 다시 백분위로
  for (const m of MODES) {
    const avg = SEATS.map((s, i) => [i, (s[m].p.D50 + s[m].p.G + s[m].p.LF) / 3]);
    avg.sort((a, b) => a[1] - b[1]);
    avg.forEach(([i], r) => { SEATS[i][m].p.TOT = Math.round(r / (avg.length - 1) * 100); });
  }

  const state = { mode: 'natural', metric: 'TOT', floor: 1, sel: null, cmp: [] };
  const $ = id => document.getElementById(id);

  // viridis 근사
  const VIR = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]];
  function color(p) {
    const t = Math.max(0, Math.min(1, p / 100)) * (VIR.length - 1), k = Math.min(VIR.length - 2, Math.floor(t)), f = t - k;
    const c = VIR[k].map((v, j) => Math.round(v + (VIR[k + 1][j] - v) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  const rankText = p => (p >= 50 ? `상위 ${Math.max(1, 100 - p)}%` : `하위 ${p + 1}%`);
  function seatInfo(s) {
    const dist = STAGE_X - s.x, dz = s.z - (-3.86);
    const side = Math.abs(dz) < 5.2 ? '가운데' : (dz > 0 ? '오른쪽' : '왼쪽');
    return { title: `${s.f}층 ${side} 구역`, sub: `무대 앞 끝에서 약 ${dist.toFixed(0)}m · 좌석 번호 ${s.id + 1} (이 지도에서 매긴 번호)` };
  }

  // ---------- 지도 ----------
  const svg = $('map'), NS = 'http://www.w3.org/2000/svg';
  const xs = SEATS.map(s => s.x), zs = SEATS.map(s => s.z);
  const zmin = Math.min(...zs) - 0.8, zmax = Math.max(...zs) + 0.8, xmin = Math.min(...xs) - 0.8;
  // 층마다 좌석 영역만 보이게(2층은 무대에서 멀어 빈 공간이 크다). 무대까지 거리는 무대 띠에 글로 표시
  const VIEW = {};
  for (const f of [1, 2]) {
    const fx = SEATS.filter(s => s.f === f).map(s => s.x);
    const top = f === 1 ? 0 : STAGE_X - Math.max(...fx) - 0.8;
    VIEW[f] = { box: `${zmin} ${top} ${zmax - zmin} ${STAGE_X - xmin - top}`, gap: Math.round(STAGE_X - Math.max(...fx)) };
  }
  const circles = SEATS.map((s, i) => {
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', s.z); c.setAttribute('cy', STAGE_X - s.x); c.setAttribute('r', 0.24);
    c.dataset.i = i;
    const t = document.createElementNS(NS, 'title'); c.appendChild(t);
    svg.appendChild(c); return c;
  });
  const marks = document.createElementNS(NS, 'g'); svg.appendChild(marks);
  // 누른 위치에서 가장 가까운 좌석(같은 층, 1m 이내)을 고른다 — 휴대폰에서 작은 점을 정확히 누르기 어렵기 때문
  svg.addEventListener('click', e => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    let best = -1, bd = 1.0;
    SEATS.forEach((s, i) => { if (s.f !== state.floor) return; const d = Math.hypot(s.z - p.x, STAGE_X - s.x - p.y); if (d < bd) { bd = d; best = i; } });
    if (best >= 0) select(best);
  });
  $('n1').textContent = SEATS.filter(s => s.f === 1).length + '석';
  $('n2').textContent = SEATS.filter(s => s.f === 2).length + '석';

  function drawMap() {
    const m = state.mode, k = state.metric;
    svg.setAttribute('viewBox', VIEW[state.floor].box);
    document.querySelector('.stage-label').textContent = state.floor === 1 ? '무대' : `무대 (2층 첫 줄에서 약 ${VIEW[2].gap}m 앞)`;
    SEATS.forEach((s, i) => {
      const c = circles[i], on = s.f === state.floor;
      c.style.display = on ? '' : 'none';
      if (!on) return;
      const p = s[m].p[k];
      c.setAttribute('fill', color(p));
      c.firstChild.textContent = `${s.f}층 · ${METRIC[k].name} ${rankText(p)}`;
      c.classList.toggle('sel', i === state.sel);
    });
    marks.innerHTML = '';
    state.cmp.forEach((i, j) => {
      const s = SEATS[i]; if (s.f !== state.floor) return;
      const t = document.createElementNS(NS, 'text'); t.setAttribute('class', 'cmp-mark');
      t.setAttribute('x', s.z); t.setAttribute('y', STAGE_X - s.x - 0.55); t.setAttribute('fill', j ? 'var(--cmp-b)' : 'var(--cmp-a)');
      t.textContent = j ? 'B' : 'A'; marks.appendChild(t);
    });
    // 상위 10석 (공연장 전체)
    const top = SEATS.map((s, i) => [i, s[m].p[k], s[m][k === 'TOT' ? 'D50' : k]]).sort((a, b) => b[1] - a[1] || b[2] - a[2]).slice(0, 10);
    $('top-title').textContent = `${METRIC[k].name} 상위 10석 (${m === 'natural' ? '무대 육성' : '스피커 확성'})`;
    $('top-list').innerHTML = '';
    top.forEach(([i], r) => {
      const li = document.createElement('li'), b = document.createElement('button'), inf = seatInfo(SEATS[i]);
      b.type = 'button'; b.textContent = `${r + 1}. ${inf.title} ${i + 1}번 · 무대에서 ${(STAGE_X - SEATS[i].x).toFixed(0)}m`;
      b.addEventListener('click', () => select(i, true)); li.appendChild(b); $('top-list').appendChild(li);
    });
    $('mode-note').textContent = MODE_NOTE[m];
  }

  // ---------- 좌석 상세 ----------
  function select(i, switchFloor) {
    state.sel = i;
    if (switchFloor && SEATS[i].f !== state.floor) setFloor(SEATS[i].f);
    render(); writeHash();
  }
  function renderDetail() {
    if (state.sel === null) { $('empty').hidden = false; $('seat').hidden = true; return; }
    $('empty').hidden = true; $('seat').hidden = false;
    const s = SEATS[state.sel], m = state.mode, v = s[m], inf = seatInfo(s);
    $('seat-title').textContent = inf.title; $('seat-sub').textContent = inf.sub;
    const bars = $('bars'); bars.innerHTML = '';
    for (const k of ['TOT', 'D50', 'C80', 'G', 'LF']) {
      const row = document.createElement('div'); row.className = 'bar-row';
      row.innerHTML = `<span>${METRIC[k].name}</span><div class="bar"><i style="width:${Math.max(3, v.p[k])}%;background:${color(v.p[k])}"></i></div><span class="pct">${rankText(v.p[k])}</span><span class="bar-desc">${METRIC[k].desc}</span>`;
      bars.appendChild(row);
    }
    const rt = $('raw-table');
    rt.innerHTML = '<tr><th>대역</th><th>C80</th><th>G</th></tr>' +
      BANDS.map(b => `<tr><td>${b >= 1000 ? b / 1000 + 'kHz' : b + 'Hz'}</td><td>${v.bands[b].C80} dB</td><td>${v.bands[b].G} dB</td></tr>`).join('') +
      `<tr><td>1kHz D50 / LF</td><td colspan="2">${v.D50} / ${v.LF}</td></tr>` +
      `<tr><td>잔향시간(가정 기반)</td><td colspan="2">${META.T_room_s.map(t => t.toFixed(1)).join(' / ')}초 (저·중·고)</td></tr>`;
    drawIR();
  }

  function drawIR() {
    const cv = $('ir'), g = cv.getContext('2d'), W = cv.width, H = cv.height;
    const cs = getComputedStyle(document.documentElement), ink = cs.getPropertyValue('--ink').trim(), muted = cs.getPropertyValue('--muted').trim(), acc = cs.getPropertyValue('--accent').trim(), line = cs.getPropertyValue('--line').trim();
    const s = SEATS[state.sel], mi = MODES.indexOf(state.mode), v = s[state.mode], bi = BANDS.indexOf(1000);
    const e = Array.from(envAt(state.sel, mi, bi), qToE), Ed = v.Ed[bi], T = META.T_room_s[bi];
    const L = 40, R = 10, Tp = 10, B = 26, tmax = 300, dbmin = -50, dbmax = 5;
    const X = t => L + (W - L - R) * t / tmax, Y = db => Tp + (H - Tp - B) * (dbmax - db) / (dbmax - dbmin);
    g.clearRect(0, 0, W, H); g.font = '11px sans-serif'; g.fillStyle = muted; g.strokeStyle = line; g.lineWidth = 1;
    for (let db = 0; db >= dbmin; db -= 10) { g.beginPath(); g.moveTo(L, Y(db)); g.lineTo(W - R, Y(db)); g.stroke(); g.fillText(db + 'dB', 2, Y(db) + 4); }
    for (let t = 0; t <= tmax; t += 50) g.fillText(t + 'ms', X(t) - 12, H - 8);
    for (const [dt, lb] of [[50, '50ms'], [80, '80ms']]) { g.setLineDash([3, 3]); g.strokeStyle = muted; g.beginPath(); g.moveTo(X(v.t0 + dt), Tp); g.lineTo(X(v.t0 + dt), H - B); g.stroke(); g.setLineDash([]); g.fillText(lb, X(v.t0 + dt) + 2, Tp + 10); }
    g.fillStyle = acc;
    for (let k = 0; k < NBIN; k++) {
      if (e[k] <= 0) continue; const db = 10 * Math.log10(e[k] / Ed);
      if (db < dbmin) continue; g.fillRect(X(k * 2), Y(db), Math.max(1, X(2) - X(0) - 0.5), Y(dbmin) - Y(db));
    }
    const rho = e.slice(75).reduce((a, b) => a + b, 0) / 25;
    g.strokeStyle = acc; g.setLineDash([5, 4]); g.beginPath();
    for (let t = 200; t <= tmax; t += 5) { const db = 10 * Math.log10(rho * Math.exp(-13.82 * (t - 200) / 1000 / T) / Ed); t === 200 ? g.moveTo(X(t), Y(db)) : g.lineTo(X(t), Y(db)); }
    g.stroke(); g.setLineDash([]);
    g.fillStyle = ink; g.fillRect(X(v.t0) - 1, Y(0), 3, Y(dbmin) - Y(0)); g.fillText('직접음', X(v.t0) + 4, Y(0) + 10);
  }

  // ---------- 비교 ----------
  $('add-compare').addEventListener('click', () => {
    if (state.sel === null || state.cmp.includes(state.sel)) return;
    state.cmp = [...state.cmp, state.sel].slice(-2); render();
  });
  $('cmp-clear').addEventListener('click', () => { state.cmp = []; render(); });
  function renderCompare() {
    const box = $('compare'); box.hidden = state.cmp.length === 0; if (box.hidden) return;
    const m = state.mode, S2 = state.cmp.map(i => SEATS[i]);
    const head = '<tr><th></th>' + S2.map((s, j) => `<th><span class="tag" style="background:var(${j ? '--cmp-b' : '--cmp-a'})">${j ? 'B' : 'A'}</span>${seatInfo(s).title}</th>`).join('') + '</tr>';
    const rows = ['TOT', 'D50', 'C80', 'G', 'LF'].map(k => `<tr><td>${METRIC[k].name}</td>` + S2.map(s => `<td>${rankText(s[m].p[k])}${k === 'TOT' ? '' : ` <span class="small">(${s[m][k]}${UNIT[k]})</span>`}</td>`).join('') + '</tr>').join('');
    $('cmp-table').innerHTML = head + rows;
    const bp = $('cmp-play'); bp.innerHTML = '';
    state.cmp.forEach((i, j) => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'btn'; b.textContent = `▶ ${j ? 'B' : 'A'} 듣기`;
      b.addEventListener('click', () => playSeat(i, b)); bp.appendChild(b);
    });
  }

  // ---------- 미리듣기: 브라우저에서 좌석 임펄스응답 합성 ----------
  let ctx = null, dryBuf = null, tmpl = null, cur = null;
  const LEN = Math.round(2.5 * FS);
  const QDB = [0.5098, 0.6013, 0.9, 2.5629].map(q => 20 * Math.log10(q)); // 8차 버터워스, 웹오디오 LP/HP Q는 dB 단위
  function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  async function filtered(input, band) {
    const oc = new OfflineAudioContext(1, input.length, FS), b = oc.createBuffer(1, input.length, FS); b.copyToChannel(input, 0);
    const src = oc.createBufferSource(); src.buffer = b; let node = src;
    const stage = (type, f) => { for (const q of QDB) { const bq = oc.createBiquadFilter(); bq.type = type; bq.frequency.value = f; bq.Q.value = q; node.connect(bq); node = bq; } };
    if (band === 250) stage('lowpass', 500);
    if (band === 1000) { stage('highpass', 500); stage('lowpass', 2000); }
    if (band === 4000) stage('highpass', 2000);
    node.connect(oc.destination); src.start(); return (await oc.startRendering()).getChannelData(0);
  }
  async function ensureAudio() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: FS });
      const bin = Uint8Array.from(atob(window.DRY_B64), c => c.charCodeAt(0));
      dryBuf = await ctx.decodeAudioData(bin.buffer);
      const r = rng(20261109); tmpl = { noise: [], direct: [] };
      for (const b of BANDS) {
        const wn = new Float32Array(LEN); for (let n = 0; n < LEN; n += 2) { const u = Math.max(1e-12, r()), w = 2 * Math.PI * r(), rad = Math.sqrt(-2 * Math.log(u)); wn[n] = rad * Math.cos(w); if (n + 1 < LEN) wn[n + 1] = rad * Math.sin(w); }
        const nz = await filtered(wn, b); let ss = 0; for (const x of nz) ss += x * x; const sd = Math.sqrt(ss / nz.length);
        tmpl.noise.push(nz.map(x => x / sd));
        const imp = new Float32Array(4096); imp[0] = 1; const di = await filtered(imp, b); let es = 0; for (const x of di) es += x * x;
        tmpl.direct.push(di.map(x => x / Math.sqrt(es)));
      }
    }
    if (ctx.state === 'suspended') await ctx.resume();
  }
  function seatIR(i, mode) {
    const mi = MODES.indexOf(mode), v = SEATS[i][mode], ir = new Float32Array(LEN), k0 = Math.round(v.t0 / 1000 * FS), spb = 2 * FS / 1000;
    BANDS.forEach((b, bi) => {
      const e = Array.from(envAt(i, mi, bi), qToE), T = META.T_room_s[bi], nz = tmpl.noise[bi];
      const rho200 = e.slice(75).reduce((a, c) => a + c, 0) / 50 / (FS / 1000);   // 200ms 시점 표본당 에너지
      for (let n = 0; n < LEN; n++) {
        const t = n / FS; let en;
        if (t < 0.2) en = e[Math.floor(n / spb)] / spb; else en = rho200 * Math.exp(-13.82 * (t - 0.2) / T);
        ir[n] += nz[n] * Math.sqrt(en);
      }
      const di = tmpl.direct[bi], a = Math.sqrt(v.Ed[bi]);
      for (let m = 0; m < di.length && k0 + m < LEN; m++) ir[k0 + m] += di[m] * a;
    });
    const gscale = META.play_gain / Math.sqrt(META.E_ref);
    for (let n = 0; n < LEN; n++) ir[n] *= gscale;
    const buf = ctx.createBuffer(1, LEN, FS); buf.copyToChannel(ir, 0); return buf;
  }
  function stop() { if (cur) { try { cur.stop(); } catch (_) {} cur = null; } $('stop').hidden = true; }
  async function play(buildGraph, btn) {
    stop(); const old = btn.textContent; btn.disabled = true; btn.textContent = '준비 중…';
    try {
      await ensureAudio();
      const src = ctx.createBufferSource(); src.buffer = dryBuf;
      const lim = ctx.createDynamicsCompressor(); lim.threshold.value = -2; lim.ratio.value = 20; lim.attack.value = 0.002; lim.release.value = 0.1;
      buildGraph(src).connect(lim); lim.connect(ctx.destination);
      src.onended = () => { if (cur === src) stop(); }; cur = src; src.start(); $('stop').hidden = false;
    } catch (err) { alert('이 브라우저에서는 미리듣기를 재생할 수 없습니다: ' + err.message); }
    finally { btn.disabled = false; btn.textContent = old; }
  }
  function playSeat(i, btn) { play(src => { const cv = ctx.createConvolver(); cv.normalize = false; cv.buffer = seatIR(i, state.mode); src.connect(cv); return cv; }, btn); }
  $('play-seat').addEventListener('click', e => { if (state.sel !== null) playSeat(state.sel, e.currentTarget); });
  $('play-dry').addEventListener('click', e => play(src => { const g = ctx.createGain(); g.gain.value = 1; src.connect(g); return g; }, e.currentTarget));
  $('stop').addEventListener('click', stop);

  // ---------- 설정·주소 ----------
  function setRadio(sel, attr, val) { document.querySelectorAll(sel).forEach(b => b.setAttribute('aria-checked', String(b.dataset[attr] === String(val)))); }
  function setFloor(f) { state.floor = f; setRadio('[data-floor]', 'floor', f); }
  document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => { state.mode = b.dataset.mode; setRadio('[data-mode]', 'mode', state.mode); stop(); render(); writeHash(); }));
  document.querySelectorAll('[data-floor]').forEach(b => b.addEventListener('click', () => { setFloor(+b.dataset.floor); render(); writeHash(); }));
  $('metric').addEventListener('change', e => { state.metric = e.target.value; render(); writeHash(); });
  function writeHash() { const h = `m=${state.mode}&k=${state.metric}&f=${state.floor}` + (state.sel !== null ? `&s=${state.sel + 1}` : ''); history.replaceState(null, '', '#' + h); }
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (MODES.includes(p.get('m'))) state.mode = p.get('m');
    if (METRIC[p.get('k')]) state.metric = p.get('k');
    if (p.get('f') === '2') state.floor = 2;
    const s = parseInt(p.get('s'), 10); if (s >= 1 && s <= SEATS.length) { state.sel = s - 1; state.floor = SEATS[s - 1].f; }
    setRadio('[data-mode]', 'mode', state.mode); setFloor(state.floor); $('metric').value = state.metric;
  }
  // 검증용(브라우저 합성 IR이 계산값과 맞는지 확인할 때만 사용)
  window.__seatSoundCheck = async (i, mode) => {
    await ensureAudio(); const ir = seatIR(i, mode).getChannelData(0), y = await filtered(ir, 1000);
    const k = Math.round(SEATS[i][mode].t0 / 1000 * FS) + Math.round(0.08 * FS); let a = 0, b = 0;
    for (let n = 0; n < y.length; n++) (n < k ? (a += y[n] * y[n]) : (b += y[n] * y[n]));
    return { C80_browser: +(10 * Math.log10(a / b)).toFixed(2), C80_metric: SEATS[i][mode].C80 };
  };
  function render() { drawMap(); renderDetail(); renderCompare(); }
  readHash(); render();
})();
