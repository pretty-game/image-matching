/* 图片查重控制台 - 前端逻辑 */
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));

/* ---------------- 全局状态 ---------------- */
const S = {
  tab: 'overview',
  status: null,
  followId: null,        // 当前正在运行的任务 id
  taskMeta: {},          // tid -> brief（用于计时）
  logCache: {},          // tid -> { text: '', next: 0 }
  expandedTask: null,
};

/* ---------------- 工具 ---------------- */
function toast(msg, type = 'info', ms = 6000) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = msg;
  $('#toasts').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

function fmtDur(sec) {
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return m > 0 ? `${m}分${s.toString().padStart(2, '0')}秒` : `${s}秒`;
}

function basename(p) { return p.replace(/\\/g, '/').split('/').pop(); }

async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

function copyText(t) {
  navigator.clipboard?.writeText(t).then(
    () => toast('路径已复制', 'success', 2500),
    () => toast('复制失败，请手动选择', 'error', 2500),
  );
}

/* ---------------- 标签页切换 ---------------- */
$$('.tab').forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));
$$('[data-goto]').forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.goto)));

function switchTab(tab) {
  S.tab = tab;
  $$('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  $$('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${tab}`));
  if (tab === 'reports') fetchReports();
  if (tab === 'tasks') renderTaskList();
}

/* ---------------- 状态轮询 ---------------- */
async function fetchStatus() {
  let st;
  try { st = await api('/api/status'); }
  catch (e) {
    $('#dot-cache').className = 'dot off';
    $('#dot-task').className = 'dot off';
    return;
  }
  S.status = st;
  renderStatus(st);
  handleFollow(st);
}

function renderStatus(st) {
  const errBanner = $('#cfg-error');
  if (!st.ok) {
    errBanner.textContent = st.error || '配置加载失败';
    errBanner.classList.remove('hidden');
    return;
  }
  errBanner.classList.add('hidden');

  const cfg = st.config, cache = st.cache, lib = st.library, rep = st.reports;

  /* 顶栏指示灯 */
  $('#dot-cache').className = `dot ${cache.index_exists ? 'on' : 'off'}`;
  $('#dot-task').className = `dot ${st.tasks.running ? 'run' : 'on'}`;

  /* 配置卡片 */
  const dirsHtml = cfg.asset_directories.map(d => {
    const info = (lib.dirs || []).find(x => x.dir === d);
    const ok = info ? info.exists : false;
    return `${escapeHtml(d)}<span class="badge-dir ${ok ? 'ok' : 'bad'}">${ok ? '✓ 存在' : '✗ 不存在'}</span>`;
  }).join('<br>');
  kvSet('#kv-config', [
    ['资产目录', dirsHtml],
    ['预处理尺寸 / DCT', `${cfg.image_size} / ${cfg.dct_size}`],
    ['特征维度', cfg.feature_dim],
    ['聚类数 / 批大小', `${cfg.n_clusters} / ${cfg.batch_size}`],
    ['TopK / 相似阈值', `${cfg.default_top_k} / ${cfg.similarity_threshold}`],
    ['缓存目录', cfg.cache_dir],
  ]);

  /* 缓存卡片 */
  const pillIndex = $('#pill-index');
  pillIndex.textContent = cache.index_exists ? '已构建' : '未构建';
  pillIndex.className = `pill ${cache.index_exists ? 'ok' : 'bad'}`;
  kvSet('#kv-cache', [
    ['索引文件', cache.index_exists
      ? `${escapeHtml(cache.index_path)}<br><span class="muted">${cache.index_size_mb} MB · ${cache.index_mtime}</span>`
      : `<span class="muted">尚未生成（${escapeHtml(cache.index_path)}）</span>`],
    ['DCT 特征缓存', `${cache.dct_cache_files} 个文件`],
  ]);

  /* 图片库卡片 */
  kvSet('#kv-library', [
    ['图片数量', `<b>${lib.image_count.toLocaleString()}</b> 张 <span class="muted">（扫描耗时 ${lib.scan_seconds}s）</span>`],
    ['目录状态', lib.dirs.map(d =>
      `${escapeHtml(d.dir)} <span class="badge-dir ${d.exists ? 'ok' : 'bad'}">${d.exists ? '✓' : '✗'}</span>`
    ).join('<br>')],
  ]);

  /* 报告卡片 */
  kvSet('#kv-reports', [
    ['报告数量', `${rep.count} 份`],
    ['最新报告', rep.latest ? `${escapeHtml(rep.latest.name)}<br><span class="muted">${rep.latest.mtime} · ${rep.latest.size_kb} KB</span>` : '<span class="muted">暂无</span>'],
  ]);

  /* 查询就绪状态 */
  const ready = st.query_ready;
  [['#pill-ready-update'], ['#pill-ready-query']].forEach(([sel]) => {
    const el = $(sel);
    el.textContent = ready ? '查询就绪' : '未构建缓存';
    el.className = `pill ${ready ? 'ok' : 'warn'}`;
  });

  /* 表单默认值（仅首次填充） */
  if (!renderStatus._filled) {
    $('#q-topk').value = cfg.default_top_k || 10;
    $('#q-threshold').value = cfg.similarity_threshold ?? 0.9;
    $('#r-threshold').value = cfg.similarity_threshold ?? 0.9;
    renderStatus._filled = true;
  }

  /* 按钮状态 */
  $('#btn-update').disabled = st.tasks.running;
  $('#btn-report').disabled = st.tasks.running;
  $('#btn-terminate').classList.toggle('hidden', !st.tasks.running);

  if (!st.tasks.running) {
    $('#update-state').textContent = st.tasks.recent.length
      ? `上次任务：${st.tasks.recent[0].label} · ${statusText(st.tasks.recent[0])}` : '';
    $('#report-run-state').textContent = '空闲';
    $('#report-run-state').className = 'pill';
  }

  if (S.tab === 'tasks') renderTaskList();
}

function statusText(t) {
  if (t.status === 'running') return '运行中';
  if (t.status === 'success') return `成功 · ${fmtDur((t.ended_at_ts || 0) - t.started_at_ts)}`;
  return `失败（${t.error || t.exit_code}）`;
}

function kvSet(sel, rows) {
  const tbody = $(sel + ' tbody') || $(sel);
  tbody.innerHTML = rows.map(([k, v]) => `<tr><td class="k">${k}</td><td class="v">${v}</td></tr>`).join('');
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ---------------- 任务跟随 ---------------- */
function handleFollow(st) {
  const cur = st.tasks && st.tasks.current;
  if (cur) {
    S.taskMeta[cur.id] = cur;
    if (S.followId !== cur.id) {
      S.followId = cur.id;
      S.logCache[cur.id] = { text: '', next: 0 };
      $('#run-console-card').classList.remove('hidden');
      $('#run-task-label').textContent = `· ${cur.label}`;
      $('#update-state').textContent = `${cur.label} 运行中...`;
      $('#report-run-state').textContent = '生成中';
      $('#report-run-state').className = 'pill run';
    }
  } else if (S.followId) {
    const finishedId = S.followId;
    S.followId = null;
    finalizeTask(finishedId);
  }
}

async function pollTask(tid) {
  let d;
  try {
    d = await api(`/api/tasks/${tid}?since=${S.logCache[tid].next}`);
  } catch (e) { return; }
  const cache = S.logCache[tid];
  if (d.lines.length) {
    cache.text += (cache.text ? '\n' : '') + d.lines.join('\n');
    cache.next = d.next;
  }
  S.taskMeta[tid] = d;

  const pill = $('#run-task-pill');
  pill.textContent = d.status === 'running' ? '运行中' : (d.status === 'success' ? '成功' : '失败');
  pill.className = `pill ${d.status === 'running' ? 'run' : (d.status === 'success' ? 'ok' : 'bad')}`;
  $('#update-state').textContent = d.status === 'running'
    ? `${d.label} 运行中 · 已用 ${fmtDur(d.duration)} · ${d.total_lines} 行日志`
    : `${d.label} ${statusText(d)}`;

  appendConsole('#run-console', cache.text);
  appendConsole(`#console-${tid}`, cache.text);
  updateTaskRow(tid, d);
}

function appendConsole(sel, text) {
  const pre = $(sel);
  if (!pre) return;
  const nearBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 40;
  if (pre.textContent !== text) pre.textContent = text;
  if (nearBottom) pre.scrollTop = pre.scrollHeight;
}

async function finalizeTask(tid) {
  let d;
  try { d = await api(`/api/tasks/${tid}?since=0`); }
  catch (e) { return; }
  S.taskMeta[tid] = d;
  const cache = S.logCache[tid];
  cache.text = d.lines.join('\n');
  cache.next = d.next;

  const pill = $('#run-task-pill');
  pill.textContent = d.status === 'success' ? '成功' : '失败';
  pill.className = `pill ${d.status === 'success' ? 'ok' : 'bad'}`;
  $('#update-state').textContent = `${d.label} ${statusText(d)}`;
  appendConsole('#run-console', cache.text);

  if (d.status === 'success') {
    if (d.name === 'report') {
      const s = d.summary || {};
      toast(`📊 报告生成完成：相似组 <b>${s.total_groups ?? '-'}</b> · 相似图片 <b>${s.total_similar_images ?? '-'}</b> · 耗时 ${s.execution_time ?? '-'}s`
        + (d.new_report ? ` · <a href="/api/reports/view?name=${encodeURIComponent(d.new_report)}" target="_blank">查看报告</a>` : ''),
        'success', 10000);
      fetchReports();
    } else if (d.name === 'update') {
      toast(`✅ 图片库更新完成，耗时 ${fmtDur(d.duration)}，现在可以进行相似查询了`, 'success');
    }
  } else {
    toast(`❌ ${d.label} 失败：${d.error || '请查看任务日志'}`, 'error', 10000);
  }
  fetchStatus();
}

/* ---------------- 任务列表 ---------------- */
function renderTaskList() {
  const items = (S.status?.tasks?.recent) || [];
  const box = $('#task-list');
  if (!items.length) { box.innerHTML = '<div class="empty">暂无任务</div>'; return; }
  box.innerHTML = '';
  for (const t of items) {
    const item = document.createElement('div');
    item.className = 'task-item';
    item.dataset.tid = t.id;
    const pillCls = t.status === 'running' ? 'run' : (t.status === 'success' ? 'ok' : 'bad');
    const dur = t.status === 'running'
      ? fmtDur((Date.now() / 1000) - t.started_at_ts)
      : fmtDur((t.ended_at_ts || t.started_at_ts) - t.started_at_ts);
    item.innerHTML = `
      <div class="task-item-head">
        <span class="task-label">${escapeHtml(t.label)}</span>
        <span class="pill ${pillCls}">${t.status === 'running' ? '运行中' : (t.status === 'success' ? '成功' : '失败')}</span>
        <span class="task-info">${t.started_at} · ${dur}${t.exit_code != null ? ` · 退出码 ${t.exit_code}` : ''}</span>
        ${t.new_report ? `<a class="task-info" href="/api/reports/view?name=${encodeURIComponent(t.new_report)}" target="_blank">📄 ${escapeHtml(t.new_report)}</a>` : ''}
        <span class="task-toggle">${S.expandedTask === t.id ? '收起 ▲' : '展开日志 ▼'}</span>
      </div>
      <div class="task-console-wrap ${S.expandedTask === t.id ? '' : 'hidden'}">
        <pre class="console" id="console-${t.id}"></pre>
      </div>`;
    item.querySelector('.task-item-head').addEventListener('click', (e) => {
      if (e.target.tagName === 'A') return;
      S.expandedTask = S.expandedTask === t.id ? null : t.id;
      renderTaskList();
    });
    box.appendChild(item);

    /* 展开时填充已有日志 */
    const pre = $(`#console-${t.id}`);
    if (pre) {
      const cache = S.logCache[t.id];
      if (cache && cache.text) {
        pre.textContent = cache.text;
        pre.scrollTop = pre.scrollHeight;
      } else if (t.status !== 'running') {
        api(`/api/tasks/${t.id}?since=0`).then(d => {
          S.logCache[t.id] = { text: d.lines.join('\n'), next: d.next };
          const p = $(`#console-${t.id}`);
          if (p) { p.textContent = S.logCache[t.id].text; p.scrollTop = p.scrollHeight; }
        }).catch(() => {});
      }
    }
  }
}

function updateTaskRow(tid, d) {
  const item = $(`.task-item[data-tid="${tid}"]`);
  if (!item) return;
  const info = item.querySelector('.task-info');
  if (info) info.textContent = `${d.started_at} · ${fmtDur(d.duration)}${d.exit_code != null ? ` · 退出码 ${d.exit_code}` : ''}`;
}

/* ---------------- 更新图片库 ---------------- */
$('#btn-update').addEventListener('click', async () => {
  try {
    const r = await api('/api/update', { method: 'POST' });
    toast(`🚀 已开始更新图片库（任务 ${r.task_id}），请在下方查看实时日志`, 'info');
    fetchStatus();
  } catch (e) { toast(`启动失败：${e.message}`, 'error'); }
});

$('#btn-terminate').addEventListener('click', async () => {
  if (!S.followId) return;
  if (!confirm('确定要终止当前任务吗？可能导致缓存不完整。')) return;
  try {
    await api(`/api/tasks/${S.followId}/terminate`, { method: 'POST' });
    toast('已发送终止信号', 'warn');
  } catch (e) { toast(`终止失败：${e.message}`, 'error'); }
});

/* ---------------- 相似查询 ---------------- */
$('#btn-query').addEventListener('click', runQuery);
$('#q-path').addEventListener('keydown', e => { if (e.key === 'Enter') runQuery(); });

async function runQuery() {
  const path = $('#q-path').value.trim();
  if (!path) { toast('请输入查询图片路径，或从图片库搜索选择', 'error'); return; }
  const topk = parseInt($('#q-topk').value) || 10;
  let threshold = parseFloat($('#q-threshold').value);
  if (isNaN(threshold)) threshold = null;

  $('#btn-query').disabled = true;
  $('#q-status').textContent = '查询中...（首次查询需加载模型，可能稍慢）';
  try {
    const res = await api('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: path, top_k: topk, threshold }),
    });
    renderQueryResult(res);
  } catch (e) {
    toast(`查询失败：${e.message}`, 'error');
  } finally {
    $('#btn-query').disabled = false;
    $('#q-status').textContent = '';
  }
}

function renderQueryResult(res) {
  if (!res.success) {
    toast(`查询失败：${res.error || '未知错误'}`, 'error', 8000);
    return;
  }
  $('#query-result').classList.remove('hidden');
  $('#q-origin-img').src = `/api/thumb?path=${encodeURIComponent(res.query_image)}&size=320`;
  $('#q-origin-name').textContent = basename(res.query_image);
  $('#q-origin-name').title = res.query_image;
  $('#q-origin-name').style.cursor = 'pointer';
  $('#q-origin-name').onclick = () => copyText(res.query_image);
  $('#q-summary').textContent = `· 找到 ${res.similar_images.length} 张相似 · 耗时 ${res.execution_time.toFixed(2)}s`;

  const grid = $('#q-matches');
  grid.innerHTML = '';
  if (!res.similar_images.length) {
    grid.innerHTML = '<div class="empty">没有满足阈值的相似图片，可尝试降低相似度阈值</div>';
    return;
  }
  for (const img of res.similar_images) {
    const card = document.createElement('div');
    card.className = 'match-card';
    const sim = img.similarity;
    const cls = sim >= 0.98 ? 'hi' : (sim >= 0.9 ? 'mid' : 'lo');
    card.innerHTML = `
      <img loading="lazy" src="/api/thumb?path=${encodeURIComponent(img.image_path)}&size=200" alt="">
      <div class="filename" title="${escapeHtml(img.image_path)}">${escapeHtml(img.filename)}</div>
      <div class="match-meta">
        <span class="sim-badge ${cls}">${(sim * 100).toFixed(1)}%</span>
        <span class="size-badge">${img.file_size_mb.toFixed(2)} MB</span>
      </div>
      <div class="match-path" title="点击复制完整路径">${escapeHtml(img.image_path)}</div>`;
    card.querySelector('.match-path').addEventListener('click', () => copyText(img.image_path));
    card.querySelector('.filename').addEventListener('click', () => copyText(img.image_path));
    grid.appendChild(card);
  }
}

/* ---------------- 图片库搜索 ---------------- */
let libTimer = null;
$('#lib-search').addEventListener('input', () => {
  clearTimeout(libTimer);
  libTimer = setTimeout(searchLib, 300);
});

async function searchLib() {
  const q = $('#lib-search').value.trim();
  const box = $('#lib-results');
  if (!q) { box.innerHTML = ''; return; }
  try {
    const res = await api(`/api/library?search=${encodeURIComponent(q)}&limit=30`);
    if (!res.matches.length) {
      box.innerHTML = '<div class="lib-item"><span class="dir">没有匹配的文件</span></div>';
      return;
    }
    box.innerHTML = '';
    for (const m of res.matches) {
      const item = document.createElement('button');
      item.className = 'lib-item';
      item.innerHTML = `<div class="fn">${escapeHtml(m.filename)}</div><div class="dir">${escapeHtml(m.directory)}</div>`;
      item.addEventListener('click', () => {
        $('#q-path').value = m.path;
        box.innerHTML = '';
        $('#lib-search').value = '';
        toast(`已填入：${m.filename}`, 'success', 2500);
      });
      box.appendChild(item);
    }
    if (res.total > res.matches.length) {
      const more = document.createElement('div');
      more.className = 'lib-item';
      more.innerHTML = `<span class="dir">共 ${res.total} 个匹配，仅显示前 ${res.matches.length} 个，请细化关键字</span>`;
      box.appendChild(more);
    }
  } catch (e) { box.innerHTML = ''; }
}

/* ---------------- 报告 ---------------- */
$('#btn-report').addEventListener('click', async () => {
  const maxGroups = parseInt($('#r-maxgroups').value);
  let threshold = parseFloat($('#r-threshold').value);
  if (isNaN(threshold)) threshold = null;
  const body = {
    threshold,
    min_size: parseInt($('#r-minsize').value) || 2,
    max_groups: isNaN(maxGroups) ? null : maxGroups,
    lightweight: $('#r-lightweight').checked,
    use_optimized: !$('#r-noopt').checked,
    format_type: $('#r-format').value,
  };
  try {
    const r = await api('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    toast('📊 报告生成任务已启动，可在「图片库更新」页或「任务日志」页查看实时进度', 'info', 8000);
    fetchStatus();
  } catch (e) { toast(`启动失败：${e.message}`, 'error'); }
});

$('#btn-refresh-reports').addEventListener('click', fetchReports);

let reportsTimer = null;
async function fetchReports() {
  let res;
  try { res = await api('/api/reports'); }
  catch (e) { return; }
  const body = $('#reports-body');
  body.innerHTML = '';
  $('#reports-empty').classList.toggle('hidden', res.reports.length > 0);
  for (const r of res.reports) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="fn">${r.mtime}</td>
      <td class="fn" title="点击复制文件名">${escapeHtml(r.name)}</td>
      <td>${r.type.toUpperCase()}</td>
      <td>${r.size_kb} KB</td>
      <td>
        <a class="btn primary sm" href="/api/reports/view?name=${encodeURIComponent(r.name)}" target="_blank">查看</a>
        <a class="btn ghost sm" href="/api/reports/download?name=${encodeURIComponent(r.name)}">下载</a>
      </td>`;
    tr.querySelector('td:nth-child(2)').addEventListener('click', () => copyText(r.name));
    body.appendChild(tr);
  }
}

/* ---------------- 刷新与轮询 ---------------- */
$('#btn-reload').addEventListener('click', () => { fetchStatus(); toast('已刷新', 'success', 2000); });

fetchStatus();
setInterval(fetchStatus, 5000);
setInterval(() => { if (S.followId) pollTask(S.followId); }, 1500);
setInterval(() => { if (S.tab === 'reports' && !S.followId) fetchReports(); }, 20000);
