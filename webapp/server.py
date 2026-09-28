#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片匹配系统 Web 控制台（独立于 src/）

功能：
  - 触发 update / report 命令（子进程执行，实时日志流）
  - 在线相似查询（进程内调用 ImageMatchingAPI）
  - 报告列表 / 在线查看 / 下载
  - 图片与缩略图服务（PIL 生成，本地缓存）
  - 系统状态总览（配置 / 索引缓存 / 图片库 / 任务）

启动：
  cd webapp
  python server.py [--config ../config/settings.yaml] [--host 127.0.0.1] [--port 8899]

  - 推荐（自动使用项目虚拟环境）： .venv/Scripts/python server.py
  - 若用其他解释器启动且其缺少依赖，服务会自动切换到 .venv 重新运行
  - 子进程任务（update / report）始终优先使用 .venv 的解释器
"""

import argparse
import hashlib
import importlib.util
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from flask import Flask, abort, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 关键：进程内查询需要导入 src 包；python webapp/server.py 时 sys.path[0] 是 webapp/，
# 项目根不在路径中会导致 "No module named 'src'"，这里显式加入
sys.path.insert(0, str(PROJECT_ROOT))
WEBAPP_DIR = Path(__file__).resolve().parent
THUMB_CACHE_DIR = WEBAPP_DIR / "thumb_cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tga', '.webp', '.gif')

# 项目虚拟环境解释器（任务子进程优先使用）
VENV_PYTHON = PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if os.name == 'nt' else "bin/python")

# 核心依赖（src 模块运行所需）
REQUIRED_MODULES = ('cv2', 'sklearn', 'scipy', 'PIL', 'yaml')


def has_required_modules() -> bool:
    """检查当前解释器是否具备 src 模块所需的全部依赖"""
    try:
        return all(importlib.util.find_spec(m) for m in REQUIRED_MODULES)
    except Exception:
        return False


def get_worker_python() -> str:
    """任务子进程使用的解释器：优先 .venv，其次当前解释器"""
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable

app = Flask(__name__)
CONFIG_PATH = str((PROJECT_ROOT / "config" / "settings.yaml").resolve())

THUMB_CACHE_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config() -> dict:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def resolve_cache_info(cfg: dict):
    """与 src/main.py 保持一致的索引缓存定位逻辑"""
    dirs = sorted(cfg.get('data_sources', {}).get('asset_directories', []))
    dirs_hash = hashlib.md5("".join(dirs).encode()).hexdigest()
    cache_dir = Path(cfg.get('performance', {}).get('cache_dir', './cache'))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    cache_dir = cache_dir.resolve()
    index_path = cache_dir / f"index_{dirs_hash}.pkl"
    return cache_dir, index_path


class LibraryCache:
    """资产目录扫描结果缓存（带 TTL），避免状态轮询反复遍历磁盘"""

    def __init__(self, ttl: float = 15.0):
        self.ttl = ttl
        self._lock = threading.Lock()
        self._key = None
        self._paths = []
        self._scan_seconds = 0.0
        self._scanned_at = 0.0

    def get(self, dirs):
        key = tuple(sorted(dirs))
        with self._lock:
            if key == self._key and time.time() - self._scanned_at < self.ttl:
                return self._paths, self._scan_seconds
            t0 = time.time()
            paths = []
            for d in dirs:
                if not os.path.isdir(d):
                    continue
                for root, _, files in os.walk(d):
                    for fn in files:
                        if fn.lower().endswith(IMAGE_EXTS):
                            paths.append(os.path.join(root, fn))
            self._paths = paths
            self._scan_seconds = time.time() - t0
            self._scanned_at = time.time()
            self._key = key
            return paths, self._scan_seconds

    def invalidate(self):
        with self._lock:
            self._scanned_at = 0.0


LIB_CACHE = LibraryCache()


def list_report_files():
    if not REPORTS_DIR.is_dir():
        return []
    items = []
    for p in REPORTS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in ('.html', '.json'):
            st = p.stat()
            items.append({
                'name': p.name,
                'path': str(p),
                'type': p.suffix.lower().lstrip('.'),
                'size_kb': round(st.st_size / 1024, 1),
                'mtime': datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                'mtime_ts': st.st_mtime,
            })
    items.sort(key=lambda x: x['mtime_ts'], reverse=True)
    return items


# --------------------------------------------------------------------------
# 任务管理（子进程执行 app.py 命令）
# --------------------------------------------------------------------------

class TaskManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.tasks = {}   # id -> task dict
        self.procs = {}   # id -> Popen

    def _brief(self, t):
        return {
            'id': t['id'], 'name': t['name'], 'label': t['label'],
            'status': t['status'], 'cmd': t['cmd'],
            'started_at': t['started_at'], 'started_at_ts': t['started_at_ts'],
            'ended_at': t['ended_at'], 'ended_at_ts': t['ended_at_ts'],
            'exit_code': t['exit_code'], 'log_count': len(t['logs']),
            'new_report': t.get('new_report'), 'summary': t.get('summary'),
            'error': t.get('error'),
        }

    def list(self):
        with self._lock:
            items = sorted(self.tasks.values(),
                           key=lambda t: t['started_at_ts'], reverse=True)
            return [self._brief(t) for t in items[:30]]

    def running(self):
        with self._lock:
            for t in self.tasks.values():
                if t['status'] == 'running':
                    return self._brief(t)
        return None

    def get(self, tid):
        with self._lock:
            return self.tasks.get(tid)

    def start(self, name, label, cmd, before_hook=None, success_hook=None):
        current = self.running()
        if current:
            return None, f"已有任务在运行：{current['label']}，请等待完成后再操作"
        tid = 't' + uuid.uuid4().hex[:8]
        task = {
            'id': tid, 'name': name, 'label': label, 'cmd': cmd,
            'status': 'running',
            'started_at': now_str(), 'started_at_ts': time.time(),
            'ended_at': None, 'ended_at_ts': None, 'exit_code': None,
            'logs': [], 'new_report': None, 'summary': None, 'error': None,
        }
        with self._lock:
            self.tasks[tid] = task
        threading.Thread(target=self._run,
                         args=(task, before_hook, success_hook),
                         daemon=True).start()
        return task, None

    def _run(self, task, before_hook, success_hook):
        ctx = {}
        if before_hook:
            try:
                ctx = before_hook() or {}
            except Exception as e:
                task['logs'].append(f"[webapp] 前置检查失败: {e}")

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        try:
            proc = subprocess.Popen(
                task['cmd'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
                text=True, encoding='utf-8', errors='replace',
                bufsize=1,
            )
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = f'子进程启动失败: {e}'
            task['ended_at'] = now_str()
            task['ended_at_ts'] = time.time()
            return

        with self._lock:
            self.procs[task['id']] = proc

        for line in proc.stdout:
            line = line.rstrip('\r\n')
            with self._lock:
                task['logs'].append(line)

        code = proc.wait()
        task['exit_code'] = code
        task['status'] = 'success' if code == 0 else 'failed'
        task['ended_at'] = now_str()
        task['ended_at_ts'] = time.time()
        if task['status'] == 'failed' and not task['error']:
            task['error'] = f'进程退出码 {code}'

        if success_hook and task['status'] == 'success':
            try:
                success_hook(task, ctx)
            except Exception as e:
                with self._lock:
                    task['logs'].append(f"[webapp] 收尾处理出错: {e}")

        with self._lock:
            self.procs.pop(task['id'], None)

    def terminate(self, tid):
        with self._lock:
            proc = self.procs.get(tid)
        if proc is None:
            return False, "任务不在运行中"
        try:
            proc.terminate()
            return True, "已发送终止信号"
        except Exception as e:
            return False, str(e)


TASKS = TaskManager()


def summarize_report_logs(logs):
    """从 CLI 日志中提取报告结果摘要"""
    pats = {
        'total_groups': r'相似组数量:\s*(\d+)',
        'total_similar_images': r'相似图片数:\s*(\d+)',
        'similarity_ratio': r'相似度比例:\s*([\d.]+%)',
        'output_file': r'输出文件:\s*(\S+)',
        'execution_time': r'执行时间:\s*([\d.]+)秒',
    }
    summary = {}
    for key, pat in pats.items():
        for line in logs:
            m = re.search(pat, line)
            if m:
                summary[key] = m.group(1)
    return summary or None


# --------------------------------------------------------------------------
# 进程内查询 API（懒加载，仅用于相似查询）
# --------------------------------------------------------------------------

_query_state = {'api': None, 'lock': threading.Lock()}


def reset_query_api():
    with _query_state['lock']:
        _query_state['api'] = None


def get_query_api():
    with _query_state['lock']:
        if _query_state['api'] is not None:
            return _query_state['api']
        if not has_required_modules():
            missing = [m for m in REQUIRED_MODULES if not importlib.util.find_spec(m)]
            raise RuntimeError(
                f'当前服务进程缺少依赖: {", ".join(missing)}，'
                f'请使用虚拟环境启动: .venv\\Scripts\\python webapp\\server.py')
        # 延迟导入：src 依赖 numpy/cv2/sklearn，较重
        from src.apis import ImageMatchingAPI
        api = ImageMatchingAPI(CONFIG_PATH)
        if not api.initialize():
            raise RuntimeError('图片匹配系统初始化失败，详见日志')
        _query_state['api'] = api
        return api


# --------------------------------------------------------------------------
# 路由：页面
# --------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# --------------------------------------------------------------------------
# 路由：状态
# --------------------------------------------------------------------------

@app.route('/api/status')
def api_status():
    try:
        cfg = load_config()
    except Exception as e:
        return jsonify({'ok': False, 'error': f'配置加载失败: {e}'})

    cache_dir, index_path = resolve_cache_info(cfg)
    asset_dirs = cfg.get('data_sources', {}).get('asset_directories', [])
    paths, scan_seconds = LIB_CACHE.get(asset_dirs)

    index_exists = index_path.is_file()
    index_info = {
        'dir': str(cache_dir),
        'dir_exists': cache_dir.is_dir(),
        'index_path': str(index_path),
        'index_exists': index_exists,
        'index_size_mb': round(index_path.stat().st_size / (1024 * 1024), 2) if index_exists else 0,
        'index_mtime': datetime.fromtimestamp(index_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if index_exists else None,
        'dct_cache_files': len(list((cache_dir / 'dct_cache').glob('*.pkl'))) if (cache_dir / 'dct_cache').is_dir() else 0,
    }

    fe = cfg.get('feature_extraction', {})
    cl = cfg.get('clustering', {})
    mt = cfg.get('matching', {})

    reports = list_report_files()
    running = TASKS.running()

    return jsonify({
        'ok': True,
        'config_path': CONFIG_PATH,
        'project_root': str(PROJECT_ROOT),
        'reports_dir': str(REPORTS_DIR),
        'config': {
            'asset_directories': asset_dirs,
            'image_size': fe.get('image_size'),
            'dct_size': fe.get('dct_size'),
            'feature_dim': fe.get('feature_dim'),
            'n_clusters': cl.get('n_clusters'),
            'batch_size': cl.get('batch_size'),
            'default_top_k': mt.get('default_top_k', 10),
            'similarity_threshold': mt.get('similarity_threshold', 0.9),
            'cache_dir': str(cache_dir),
        },
        'cache': index_info,
        'library': {
            'image_count': len(paths),
            'scan_seconds': round(scan_seconds, 3),
            'dirs': [{'dir': d, 'exists': os.path.isdir(d)} for d in asset_dirs],
        },
        'reports': {
            'count': len(reports),
            'latest': reports[0] if reports else None,
        },
        'query_ready': index_exists,
        'tasks': {
            'running': bool(running),
            'current': running,
            'recent': TASKS.list(),
        },
    })


# --------------------------------------------------------------------------
# 路由：任务触发与查询
# --------------------------------------------------------------------------

@app.route('/api/update', methods=['POST'])
def api_update():
    cmd = [get_worker_python(), 'app.py', 'update', '--config', CONFIG_PATH]
    task, err = TASKS.start(
        'update', '更新图片库', cmd,
        success_hook=lambda task, ctx: (
            reset_query_api(),
            LIB_CACHE.invalidate(),
        ),
    )
    if err:
        return jsonify({'ok': False, 'error': err}), 409
    return jsonify({'ok': True, 'task_id': task['id']})


@app.route('/api/report', methods=['POST'])
def api_report():
    data = request.get_json(force=True, silent=True) or {}

    cmd = [get_worker_python(), 'app.py', 'report',
           '--config', CONFIG_PATH,
           '--output', str(REPORTS_DIR) + os.sep]

    threshold = data.get('threshold')
    if threshold is not None:
        cmd += ['--threshold', str(float(threshold))]
    min_size = data.get('min_size')
    if min_size:
        cmd += ['--min-size', str(int(min_size))]
    max_groups = data.get('max_groups')
    if max_groups:
        cmd += ['--max-groups', str(int(max_groups))]
    fmt = data.get('format_type') or 'html'
    cmd += ['--format-type', fmt]
    if data.get('lightweight'):
        cmd.append('--lightweight')
    if data.get('use_optimized') is False:
        cmd.append('--no-optimize')

    def before_hook():
        return {'existing': {f['path'] for f in list_report_files()}}

    def success_hook(task, ctx):
        existing = ctx.get('existing', set())
        files = list_report_files()
        new_files = [f for f in files if f['path'] not in existing]
        if new_files:
            latest = max(new_files, key=lambda x: x['mtime_ts'])
            task['new_report'] = latest['name']
        task['summary'] = summarize_report_logs(task['logs'])

    task, err = TASKS.start('report', '生成相似报告', cmd,
                            before_hook=before_hook, success_hook=success_hook)
    if err:
        return jsonify({'ok': False, 'error': err}), 409
    return jsonify({'ok': True, 'task_id': task['id']})


@app.route('/api/tasks')
def api_tasks():
    return jsonify({'tasks': TASKS.list()})


@app.route('/api/tasks/<tid>')
def api_task(tid):
    t = TASKS.get(tid)
    if not t:
        abort(404)
    since = request.args.get('since', 0, type=int)
    with TASKS._lock:
        lines = t['logs'][since:]
        total = len(t['logs'])
        brief = TASKS._brief(t)
    brief.update({
        'lines': lines,
        'next': since + len(lines),
        'total_lines': total,
        'duration': round(((t['ended_at_ts'] or time.time()) - t['started_at_ts']), 1),
    })
    return jsonify(brief)


@app.route('/api/tasks/<tid>/terminate', methods=['POST'])
def api_task_terminate(tid):
    ok, msg = TASKS.terminate(tid)
    return jsonify({'ok': ok, 'message': msg}), (200 if ok else 409)


# --------------------------------------------------------------------------
# 路由：相似查询（进程内）
# --------------------------------------------------------------------------

@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.get_json(force=True, silent=True) or {}
    image_path = (data.get('image_path') or '').strip()
    top_k = int(data.get('top_k') or 10)
    threshold = data.get('threshold')
    if threshold is not None:
        threshold = float(threshold)

    if not image_path:
        return jsonify({'success': False, 'error': '请提供查询图片路径'})
    if not os.path.isfile(image_path):
        return jsonify({'success': False, 'error': f'查询图片不存在: {image_path}'})

    try:
        _, index_path = resolve_cache_info(load_config())
    except Exception as e:
        return jsonify({'success': False, 'error': f'读取配置失败: {e}'})
    if not index_path.is_file():
        return jsonify({'success': False,
                        'error': '索引缓存不存在，请先在「图片库更新」页执行更新'})

    try:
        api = get_query_api()
        result = api.find_similar_images(
            query_image_path=image_path,
            top_k=top_k,
            similarity_threshold=threshold,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': f'查询出错: {e}'})


# --------------------------------------------------------------------------
# 路由：图片库搜索 / 图片与缩略图 / 报告
# --------------------------------------------------------------------------

@app.route('/api/library')
def api_library():
    search = (request.args.get('search') or '').strip().lower()
    limit = request.args.get('limit', 30, type=int)
    limit = max(1, min(limit, 100))
    try:
        cfg = load_config()
    except Exception:
        return jsonify({'matches': [], 'total': 0})
    dirs = cfg.get('data_sources', {}).get('asset_directories', [])
    paths, _ = LIB_CACHE.get(dirs)

    if not search:
        matches = paths[:limit]
        return jsonify({'matches': [
            {'path': p, 'filename': os.path.basename(p),
             'directory': os.path.dirname(p)} for p in matches
        ], 'total': len(paths)})

    matched = [p for p in paths if search in os.path.basename(p).lower()]
    return jsonify({'matches': [
        {'path': p, 'filename': os.path.basename(p),
         'directory': os.path.dirname(p)} for p in matched[:limit]
    ], 'total': len(matched)})


@app.route('/api/image')
def api_image():
    path = request.args.get('path', '')
    if not os.path.isfile(path):
        abort(404)
    ext = os.path.splitext(path)[1].lower()
    if ext not in IMAGE_EXTS:
        abort(400)
    return send_file(path)


@app.route('/api/thumb')
def api_thumb():
    path = request.args.get('path', '')
    size = request.args.get('size', 160, type=int)
    size = max(32, min(size, 512))
    if not os.path.isfile(path):
        abort(404)
    ext = os.path.splitext(path)[1].lower()
    if ext not in IMAGE_EXTS:
        abort(400)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        abort(404)

    key = hashlib.md5(f"{path}|{mtime}|{size}".encode('utf-8')).hexdigest()
    cache_file = THUMB_CACHE_DIR / f"{key}.png"
    if not cache_file.exists():
        try:
            from PIL import Image
            img = Image.open(path)
            img.load()
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGBA')
            img.thumbnail((size, size))
            img.save(cache_file, 'PNG')
        except Exception:
            return send_file(path)  # 缩略图失败时回退原图
    return send_file(cache_file, mimetype='image/png')


@app.route('/api/reports')
def api_reports():
    return jsonify({'reports': list_report_files()})


@app.route('/api/reports/view')
def api_report_view():
    name = request.args.get('name', '')
    target = (REPORTS_DIR / name).resolve()
    if not str(target).startswith(str(REPORTS_DIR.resolve())) or not target.is_file():
        abort(404)
    return send_file(target)


@app.route('/api/reports/download')
def api_report_download():
    name = request.args.get('name', '')
    target = (REPORTS_DIR / name).resolve()
    if not str(target).startswith(str(REPORTS_DIR.resolve())) or not target.is_file():
        abort(404)
    return send_file(target, as_attachment=True)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def main():
    global CONFIG_PATH
    parser = argparse.ArgumentParser(description='图片匹配系统 Web 控制台')
    parser.add_argument('--config', default=CONFIG_PATH, help='配置文件路径')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址（默认 127.0.0.1）')
    parser.add_argument('--port', type=int, default=8899, help='监听端口（默认 8899）')
    args = parser.parse_args()
    CONFIG_PATH = str(Path(args.config).resolve())

    # 当前解释器缺少依赖且 .venv 可用 -> 自动切换到 .venv 重新运行本服务
    # 固定工作目录为项目根：子进程 cwd 与进程内 src 模块的相对路径行为保持一致
    os.chdir(PROJECT_ROOT)

    # 当前解释器缺少依赖且 .venv 可用 -> 用子进程切到 .venv 重新运行本服务后退出
    # （Windows 下 os.execv 会残留旧进程，改用 subprocess + 退出避免出现双服务进程）
    if os.environ.get('WEBAPP_RELAUNCHED') != '1' and not has_required_modules() and VENV_PYTHON.is_file():
        check = subprocess.run(
            [str(VENV_PYTHON), '-c', 'import ' + ', '.join(REQUIRED_MODULES)],
            capture_output=True, check=False,
            env={**os.environ, 'WEBAPP_RELAUNCHED': '1'})
        if check.returncode == 0:
            print(f"[webapp] 当前解释器缺少依赖，自动切换到虚拟环境: {VENV_PYTHON}")
            env = {**os.environ, 'WEBAPP_RELAUNCHED': '1'}
            proc = subprocess.Popen(
                [str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:],
                env=env, cwd=str(PROJECT_ROOT))
            print(f"[webapp] 新服务进程 PID={proc.pid}，旧进程即将退出")
            sys.exit(0)
        else:
            print(f"[webapp] 警告: .venv 也缺少依赖（{check.stderr.decode(errors='replace').strip()[:200]}）")

    # 让 src 模块的日志输出到控制台（否则进程内查询/初始化时 INFO 日志会被静默丢弃，
    # 出现"卡住但看不到任何日志"的情况）
    _fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler = logging.StreamHandler()
    _handler.setFormatter(_fmt)
    _handler.setLevel(logging.INFO)
    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    _root.addHandler(_handler)

    print("=" * 60)
    print("  图片匹配系统 Web 控制台")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  配置文件  : {CONFIG_PATH}")
    print(f"  服务进程  : {sys.executable}"
          + ("  (依赖完整)" if has_required_modules() else "  (缺少依赖，相似查询不可用!)"))
    print(f"  任务解释器: {get_worker_python()}")
    print(f"  地址      : http://{args.host}:{args.port}")
    print("=" * 60)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
