/* EV Predict — Frontend JS
   All API calls → Python server.py at /api/*
*/

// ── State ────────────────────────────────────────────────────
const S = {
  battery_capacity: 75,
  charger_power:    7.2,
  initial_soc:      35,
  target_soc:       80,
  temperature:      15,
  battery_age:      400,
  vehicle_weight:   1844,
  hour_of_day:      20,
};

// ── Nav ──────────────────────────────────────────────────────
document.querySelectorAll('.tnav').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tnav').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.page).classList.add('active');
  });
});

// ── EV Model Cards ───────────────────────────────────────────
document.querySelectorAll('.model-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    S.battery_capacity = parseFloat(card.dataset.bat);
    S.vehicle_weight   = parseFloat(card.dataset.wt);
  });
});

// ── Charger Buttons ──────────────────────────────────────────
document.querySelectorAll('.charger-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.charger-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    S.charger_power = parseFloat(btn.dataset.val);
  });
});

// ── Sliders ──────────────────────────────────────────────────
function bindSlider(sliderId, barId, displayId, stateKey, fmt, min, max) {
  const slider  = document.getElementById(sliderId);
  const bar     = document.getElementById(barId);
  const display = document.getElementById(displayId);

  function update() {
    const val = parseFloat(slider.value);
    S[stateKey]  = val;
    const pct    = ((val - min) / (max - min)) * 100;
    bar.style.width = pct + '%';
    display.textContent = fmt(val);
  }

  slider.addEventListener('input', update);
  update();
}

bindSlider('sl-isoc', 'sb-isoc', 'fv-isoc', 'initial_soc',  v => Math.round(v) + '%',        5,   85);
bindSlider('sl-tsoc', 'sb-tsoc', 'fv-tsoc', 'target_soc',   v => Math.round(v) + '%',        30,  100);
bindSlider('sl-temp', 'sb-temp', 'fv-temp', 'temperature',  v => Math.round(v) + '°C',       -12, 42);
bindSlider('sl-age',  'sb-age',  'fv-age',  'battery_age',  v => Math.round(v) + ' cycles',  0,   1500);
bindSlider('sl-hr',   'sb-hr',   'fv-hr',   'hour_of_day',  v => String(Math.round(v)).padStart(2,'0') + ':00', 0, 23);

// ── Server Status ────────────────────────────────────────────
async function checkStatus() {
  const pill = document.getElementById('model-pill');
  const txt  = document.getElementById('pill-txt');
  try {
    const d = await fetch('/api/status').then(r => r.json());
    if (d.ok) {
      pill.className = 'model-pill';
      txt.textContent = (d.model || 'Model').replace('Regressor','').replace('Classifier','') + ' · Active';
    } else {
      pill.className = 'model-pill off';
      txt.textContent = 'Models Not Loaded';
    }
  } catch {
    pill.className = 'model-pill off';
    txt.textContent = 'Server Offline';
  }
}

// ── Predict ──────────────────────────────────────────────────
async function runPredict() {
  if (S.initial_soc >= S.target_soc) {
    showToast('Target battery must be higher than current battery %', 'warn');
    return;
  }

  const btn    = document.getElementById('predict-btn');
  btn.classList.add('loading');

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(S),
    });
    const data = await res.json();

    if (data.error) { showToast(data.error, 'warn'); return; }

    renderResults(data);
    showToast('Prediction complete', 'ok');

  } catch {
    showToast('Cannot reach Python server — is server.py running?', 'warn');
  } finally {
    btn.classList.remove('loading');
  }
}

// ── Render ───────────────────────────────────────────────────
function renderResults(d) {
  const h = Math.floor(d.time_hours);
  const m = Math.round((d.time_hours - h) * 60);

  // Cards
  setCard('rv-time',   d.time_display || `${h}h ${m}m`);
  setCard('rv-energy', d.energy_kwh + ' kWh');
  setCard('rv-cost',   '₹' + Math.round(d.cost_inr).toLocaleString('en-IN'));

  document.getElementById('rs-time').textContent   = d.time_hours + ' hours total';
  document.getElementById('rs-energy').textContent = '+' + d.soc_delta + '% SOC gained';

  ['rc-time','rc-energy','rc-cost'].forEach(id =>
    document.getElementById(id).classList.add('lit'));

  // Battery visual
  document.getElementById('bfill-e').style.width     = S.initial_soc + '%';
  document.getElementById('bfill-c').style.left      = S.initial_soc + '%';
  document.getElementById('bfill-c').style.width     = d.soc_delta + '%';
  document.getElementById('bfill-glow').style.left   = S.initial_soc + '%';
  document.getElementById('bfill-glow').style.width  = d.soc_delta + '%';
  document.getElementById('soc-tag').textContent     =
    S.initial_soc + '% → ' + Math.min(100, S.initial_soc + d.soc_delta).toFixed(0) + '%';

  // Chart
  renderChart(d.comparison);

  // Insights
  renderInsights(d);

  // Model strip
  document.getElementById('ms-algo').textContent    =
    (d.model_used || 'GradientBoosting').replace('Regressor','');
  document.getElementById('model-strip').style.display = 'flex';
}

function setCard(id, val) {
  const el = document.getElementById(id);
  el.style.opacity = '0';
  el.style.transform = 'translateY(6px)';
  setTimeout(() => {
    el.textContent = val;
    el.style.transition = 'opacity 0.25s, transform 0.25s';
    el.style.opacity = '1';
    el.style.transform = 'translateY(0)';
  }, 120);
}

// ── Chart ────────────────────────────────────────────────────
function renderChart(comparison) {
  const area = document.getElementById('chart-area');
  if (!comparison || !comparison.length) return;
  area.innerHTML = '';

  const maxH = Math.max(...comparison.map(c => c.hours), 0.1);

  comparison.forEach(item => {
    const pct    = Math.max((item.hours / maxH) * 90, 3);
    const isSel  = Math.abs(item.power - S.charger_power) < 0.1;

    const col    = document.createElement('div');
    col.className = 'bar-col';
    col.innerHTML = `
      <div class="bar-val">${item.hours}h</div>
      <div class="bar-rect ${isSel ? 'active-bar' : ''}" style="height:${pct}%"></div>
      <div class="bar-label">${item.label.split(' ').slice(-1)[0]}</div>
    `;
    area.appendChild(col);
  });
}

// ── Insights ─────────────────────────────────────────────────
function renderInsights(d) {
  const box    = document.getElementById('insights-row');
  const items  = [];

  if (d.is_cold)
    items.push({ type:'warn', icon:'🥶', text:'Cold weather detected. Battery efficiency drops up to 30% below 5°C. Pre-heat your battery before charging.' });
  else if (d.is_hot)
    items.push({ type:'warn', icon:'🌡️', text:'High temperature. Avoid charging in direct sunlight above 38°C — heat degrades cell chemistry.' });
  else
    items.push({ type:'good', icon:'✅', text:`Temperature is optimal at ${d.temp_eff}% efficiency. Room-temperature charging is best for battery longevity.` });

  if (d.old_battery)
    items.push({ type:'alert', icon:'🔋', text:`Battery health estimated at ~${d.health_score}% after ${S.battery_age} cycles. Real capacity may be lower than rated.` });
  else
    items.push({ type:'good', icon:'🔋', text:`Battery health ~${d.health_score}%. Avoid charging to 100% daily to extend battery life beyond 1000 cycles.` });

  if (d.is_night)
    items.push({ type:'info', icon:'🌙', text:'Night charging active. Off-peak electricity rates (10pm–6am) are typically 20–30% cheaper across India.' });
  else
    items.push({ type:'info', icon:'💡', text:'Schedule overnight charging for lower rates. Most Indian DISCOMs offer ToD tariffs with discounted night slots.' });

  box.innerHTML = items.map(item => `
    <div class="insight ${item.type}">
      <span class="insight-icon">${item.icon}</span>
      ${item.text}
    </div>
  `).join('');
}

// ── History ──────────────────────────────────────────────────
async function loadHistory() {
  const wrap = document.getElementById('hist-wrap');
  wrap.innerHTML = '<div class="table-empty">Loading from Python backend...</div>';

  try {
    const data = await fetch('/api/history').then(r => r.json());
    if (data.error) { wrap.innerHTML = `<div class="table-empty">${data.error}</div>`; return; }

    const cols = {
      'battery_capacity_kwh': 'Battery (kWh)',
      'charger_power_kw':     'Charger (kW)',
      'initial_soc_pct':      'Start %',
      'target_soc_pct':       'End %',
      'temperature_c':        'Temp °C',
      'battery_age_cycles':   'Cycles',
      'charging_time_hours':  'Time (h)',
      'energy_required_kwh':  'Energy (kWh)',
      'estimated_cost_inr':   'Cost ₹',
    };
    const keys = Object.keys(cols).filter(k => data[0] && k in data[0]);

    wrap.innerHTML = `
      <table>
        <thead>
          <tr>${keys.map(k => `<th>${cols[k]}</th>`).join('')}</tr>
        </thead>
        <tbody>
          ${data.map(row => `
            <tr>
              ${keys.map((k, i) => {
                const cls = k === 'charging_time_hours' ? 'td-amber'
                          : k === 'energy_required_kwh' ? 'td-lime'
                          : k === 'estimated_cost_inr'  ? 'td-green'
                          : '';
                return `<td class="${cls}">${row[k] ?? '—'}</td>`;
              }).join('')}
            </tr>
          `).join('')}
        </tbody>
      </table>`;
  } catch {
    wrap.innerHTML = '<div class="table-empty">Cannot reach Python server — is server.py running?</div>';
  }
}

// ── Analytics ─────────────────────────────────────────────────
async function loadStats() {
  const wrap = document.getElementById('ana-wrap');
  wrap.innerHTML = '<div class="table-empty">Loading dataset analytics...</div>';

  try {
    const d    = await fetch('/api/stats').then(r => r.json());
    if (d.error) { wrap.innerHTML = `<div class="table-empty">${d.error}</div>`; return; }

    const cdTotal = d.charger_dist
      ? Object.values(d.charger_dist).reduce((a,b) => a+b, 0) : 1;

    const vehicleBlock = d.vehicle_dist ? `
      <div class="dist-card">
        <div class="dist-title">EV Model Distribution</div>
        <div class="vehicle-grid">
          ${Object.entries(d.vehicle_dist).map(([name, count]) => `
            <div class="vehicle-card">
              <div class="vc-name">${name}</div>
              <div class="vc-count">${count} sessions</div>
            </div>
          `).join('')}
        </div>
      </div>` : '';

    const chargerBlock = d.charger_dist ? `
      <div class="dist-card">
        <div class="dist-title">Charger Type Distribution</div>
        ${Object.entries(d.charger_dist)
          .sort((a,b) => parseFloat(a[0]) - parseFloat(b[0]))
          .map(([k, v]) => `
            <div class="dist-row">
              <div class="dist-lbl">${k}</div>
              <div class="dist-bar">
                <div class="dist-fill" style="width:${(v/cdTotal*100).toFixed(1)}%"></div>
              </div>
              <div class="dist-pct">${(v/cdTotal*100).toFixed(0)}%</div>
            </div>
          `).join('')}
      </div>` : '';

    wrap.innerHTML = `
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-num">${d.total.toLocaleString()}</div>
          <div class="stat-lbl">Real Sessions</div>
        </div>
        <div class="stat-card">
          <div class="stat-num">${d.avg_time}h</div>
          <div class="stat-lbl">Avg Charge Time</div>
        </div>
        <div class="stat-card">
          <div class="stat-num">${d.avg_energy} kWh</div>
          <div class="stat-lbl">Avg Energy Used</div>
        </div>
        <div class="stat-card">
          <div class="stat-num">₹${Math.round(d.avg_cost).toLocaleString('en-IN')}</div>
          <div class="stat-lbl">Avg Charging Cost</div>
        </div>
        <div class="stat-card">
          <div class="stat-num">${d.max_time}h</div>
          <div class="stat-lbl">Max Charge Time</div>
        </div>
        <div class="stat-card">
          <div class="stat-num">${d.fast_pct}%</div>
          <div class="stat-lbl">DC Fast Charge %</div>
        </div>
      </div>
      ${chargerBlock}
      ${vehicleBlock}
    `;

    // Animate dist bars
    setTimeout(() => {
      document.querySelectorAll('.dist-fill').forEach(el => {
        const w = el.style.width; el.style.width = '0%';
        requestAnimationFrame(() => { el.style.width = w; });
      });
    }, 50);

  } catch {
    wrap.innerHTML = '<div class="table-empty">Cannot reach Python server — is server.py running?</div>';
  }
}

// ── Toast ─────────────────────────────────────────────────────
function showToast(msg, type = 'ok') {
  const el = document.getElementById('toast');
  el.textContent  = msg;
  el.className    = `toast ${type}`;
  el.style.display = 'block';
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.style.display = 'none'; }, 3000);
}

// ── Init ──────────────────────────────────────────────────────
checkStatus();
