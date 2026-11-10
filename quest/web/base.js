/* 量化教室 — 共用 JavaScript（每個互動元件都會載入） */
'use strict';

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

/* ---------- 讓 iframe 的高度自動貼合內容 ---------- */
function fit() {
  try {
    const h = Math.ceil(document.body.getBoundingClientRect().height);
    if (window.frameElement && h > 40) window.frameElement.style.height = h + 'px';
  } catch (e) { /* 跨網域就算了，用預設高度 */ }
}
window.addEventListener('load', () => {
  fit();
  if (window.ResizeObserver) new ResizeObserver(fit).observe(document.body);
  setTimeout(fit, 300);
});

/* ---------- 小工具 ---------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const hex2 = (n) => n.toString(16).toUpperCase().padStart(2, '0');
function fmt(x, digits) {
  if (!isFinite(x)) return '—';
  return x.toFixed(digits === undefined ? 3 : digits);
}
function fmtBytes(bytes) {
  if (bytes >= 1e12) return (bytes / 1e12).toFixed(2) + ' TB';
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB';
  return bytes.toFixed(0) + ' B';
}

/* float32 → float16 的 16 個 bit（跟 numpy 的 np.float16 一樣，四捨六入五成雙） */
function f16bits(value) {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let man = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (man ? 0x200 : 0);
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    man = (man | 0x800000) >>> (1 - e);
    let half = man >>> 13;
    const rem = man & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++;
    return sign | half;
  }
  let half = (e << 10) | (man >>> 13);
  const rem = man & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++;
  return sign | half;
}
function f16value(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const man = bits & 0x3ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (man / 1024);
  if (exp === 0x1f) return man ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + man / 1024);
}

/* Q4_0：把 32 個權重編成 {d, codes, bytes}（跟 llama.cpp 的規格一致） */
function q4encode(block) {
  let m = 0;
  for (const v of block) if (Math.abs(v) > Math.abs(m)) m = v;
  const d = Math.fround(m / -8);
  const inv = d === 0 ? 0 : Math.fround(1 / d);
  const codes = block.map((v) => clamp(Math.trunc(Math.fround(Math.fround(v * inv) + 8.5)), 0, 15));
  const dbits = f16bits(d);
  const bytes = [dbits & 0xff, dbits >>> 8];
  for (let j = 0; j < 16; j++) bytes.push(codes[j] | (codes[j + 16] << 4));
  return { m, d, d16: f16value(dbits), codes, bytes };
}

/* ---------- 像素角色（每個像素只用調色盤裡的一種顏色 —— 這也是量化！） ---------- */
const PALETTE = {
  k: '#1a1530', w: '#fff8ea', c: '#e8dcc6', t: '#e0b98a', y: '#ffd95a', o: '#ff9f43',
  p: '#ff8fb1', v: '#8e5bd6', u: '#6a3fb0', r: '#ff5964', b: '#4f7df0', n: '#2c3f9e',
  g: '#3b2a55', s: '#ffd3ae', m: '#5ee6c0', l: '#b9f3ff',
};
const SPRITES = {
  smol: [
    '................',
    '.....kkkkkk.....',
    '...kkyyyyyykk...',
    '..kyyyyyyyyyyk..',
    '.kyyyyyyyyyyyyk.',
    '.kyykkyyyykkyyk.',
    'kyyykwyyyykwyyyk',
    'kyyyyyyooyyyyyyk',
    'kyppyyyooyyyppyk',
    'kyyyyyyyyyyyyyyk',
    'kyyyyyyyyyyyyyyk',
    '.kyyyyyyyyyyyyk.',
    '..kyyyyyyyyyyk..',
    '...kkyyyyyykk...',
    '....kok..kok....',
    '....kkk..kkk....',
  ],
  gguf: [
    '..kk........kk..',
    '.kwwk......kwwk.',
    '.kwpk......kpwk.',
    '.kwwkkkkkkkkwwk.',
    '.kwwwwwwwwwwwwk.',
    'kggggggwwggggggk',
    'kglllkgwwglllkgk',
    'kglkllgggglkllgk',
    'kggggggwwggggggk',
    '.kwwwttttttwwwk.',
    '.kwwttkkkkttwwk.',
    '.kwwtttkktttwwk.',
    '..kwwttttttwwk..',
    '...kwwwwwwwwk...',
    '...kwwwwwwwwk...',
    '...kkkkkkkkkk...',
  ],
  oom: [
    'kk............kk',
    'kuk..........kuk',
    'kuuk..kkkk..kuuk',
    '.kuukkvvvvkkuuk.',
    '..kvvvvvvvvvvk..',
    '.kvkkkvvvvkkkvk.',
    '.kvvwkkvvkkwvvk.',
    'kvvvwrkvvkrwvvvk',
    'kvvvvvvvvvvvvvvk',
    'kvkkkkkkkkkkkkvk',
    'kvkwkwkwkwkwkkvk',
    'kvkkwkwkwkwkwkvk',
    'kvkkkkkkkkkkkkvk',
    '.kvvvvvvvvvvvvk.',
    '..kuvvvvvvvvuk..',
    '...kkkkkkkkkk...',
  ],
  you: [
    '.......kk.......',
    '......kbbk......',
    '.....kbbbbk.....',
    '....kbbybbbk....',
    '...kbbbbbbbbk...',
    '.kkkkkkkkkkkkkk.',
    '.knnnnnnnnnnnnk.',
    '..kkkkkkkkkkkk..',
    '..kssssssssssk..',
    '..kssksssskssk..',
    '..kssksssskssk..',
    '..kspsssssspsk..',
    '..ksssskkssssk..',
    '...kssssssssk...',
    '..kkbbbbbbbbkk..',
    '.kbbbbbbbbbbbbk.',
  ],
};
const WHO = {
  gguf: { name: '咕咕夫教授', color: '#ffd166' },
  smol: { name: '小史莫', color: '#ffd95a' },
  oom: { name: 'OOM 魔王', color: '#b794ff' },
  you: { name: '你', color: '#74b9ff' },
  narrator: { name: '旁白', color: '#a3abd6' },
};
function sprite(name, px) {
  const rows = SPRITES[name];
  if (!rows) return '';
  px = px || 6;
  const n = rows.length;
  let rects = '';
  for (let y = 0; y < n; y++) {
    let x = 0;
    while (x < rows[y].length) {
      const ch = rows[y][x];
      let run = 1;
      while (x + run < rows[y].length && rows[y][x + run] === ch) run++;
      if (ch !== '.') rects += `<rect x="${x}" y="${y}" width="${run}" height="1" fill="${PALETTE[ch] || '#f0f'}"/>`;
      x += run;
    }
  }
  return `<svg class="spr" width="${px * 16}" height="${px * n}" viewBox="0 0 16 ${n}" shape-rendering="crispEdges" role="img" aria-label="${(WHO[name] || {}).name || name}">${rects}</svg>`;
}

/* ---------- 過關彩帶 ---------- */
function confetti(ms) {
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const cv = document.createElement('canvas');
  cv.className = 'confetti';
  document.body.appendChild(cv);
  const ctx = cv.getContext('2d');
  const W = (cv.width = window.innerWidth);
  const H = (cv.height = window.innerHeight);
  const colors = ['#ffd166', '#ff7aa8', '#5ee6c0', '#74b9ff', '#ff9f5a', '#b794ff'];
  const bits = Array.from({ length: 110 }, () => ({
    x: Math.random() * W, y: -20 - Math.random() * H * 0.5, s: 5 + Math.random() * 6,
    vx: -1.5 + Math.random() * 3, vy: 2 + Math.random() * 4, c: colors[(Math.random() * colors.length) | 0],
  }));
  const end = performance.now() + (ms || 2200);
  (function tick(now) {
    ctx.clearRect(0, 0, W, H);
    for (const b of bits) {
      b.x += b.vx; b.y += b.vy; b.vy += 0.05;
      ctx.fillStyle = b.c;
      ctx.fillRect(Math.round(b.x), Math.round(b.y), b.s, b.s);
    }
    if (now < end) requestAnimationFrame(tick);
    else cv.remove();
  })(performance.now());
}

/* 顯示「通關密語」的小卡 */
function passBox(text, note) {
  return `<div class="quest pop">🔑 <b>通關密語：</b><span class="code" style="font-size:18px;letter-spacing:.12em;user-select:all">${text}</span>
    <span class="muted">　${note || '把它填進下一格程式碼裡！'}</span></div>`;
}
