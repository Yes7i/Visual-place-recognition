#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉位置识别 Web 界面
- 打开网页(output/results.html),顶部显示当前聚类中心个数
- 在网页输入框填写新的聚类中心个数 → 自动修改 config.yaml 并重新训练(--build)
- 训练完成后自动用原查询图重新检索并刷新页面

用法(在项目目录):
    ./open_web.sh        # 推荐,会自动打开浏览器
    python3 webui.py     # 或直接运行
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(ROOT, "config", "config.yaml")
RESULTS_HTML = os.path.join(ROOT, "output", "results.html")
VPR = os.path.join(ROOT, "build", "vpr")
PORT = int(os.environ.get("VPR_PORT", "8000"))
DATA_DIR = os.path.join(ROOT, "data")
UPLOAD_DIR = os.path.join(ROOT, "uploads")
MAX_UPLOAD = 20 * 1024 * 1024  # 20MB
OUTPUT_DIR = os.path.join(ROOT, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")  # 按 K 缓存训练好的模型
MODEL_FILES = ("vocabulary.yml", "idf.yml", "tfidf_vectors.bin")

state = {
    "training": False,
    "target_k": None,
    "started_at": None,
    "action": None,  # "train"(真正训练) 或 "reuse"(复用缓存,不训练)
    "error": None,
}


def read_current_k():
    """读取 config.yaml 中的 dictionary_size"""
    try:
        with open(CONFIG, "r", encoding="utf-8") as f:
            m = re.search(r"^\s*dictionary_size\s*:\s*(\d+)", f.read(), re.M)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def write_new_k(k):
    """只替换 dictionary_size 的值,保留其他配置与注释"""
    with open(CONFIG, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"^(\s*dictionary_size\s*:\s*)\d+", r"\g<1>%d" % k, text,
                  count=1, flags=re.M)
    with open(CONFIG, "w", encoding="utf-8") as f:
        f.write(text)


def query_image_from_html():
    """从现有 results.html 中取出查询图相对路径,训练后用于重新检索"""
    try:
        with open(RESULTS_HTML, "r", encoding="utf-8") as f:
            text = f.read()
        m = re.search(r'<img class="query-img" src="([^"]+)"', text)
        if not m:
            return None
        src = m.group(1)
        if src.startswith("file://") or src.startswith("/"):
            return None  # 旧版绝对路径无法复用,跳过自动检索
        return src.replace("../", "", 1)  # "../data/..." -> "data/..."
    except Exception:
        return None


def trained_k_list():
    """已训练过并缓存的 K 列表(output/models/k<k>),升序"""
    ks = []
    if os.path.isdir(MODEL_DIR):
        for name in sorted(os.listdir(MODEL_DIR)):
            m = re.fullmatch(r"k(\d+)", name)
            if m and os.path.isdir(os.path.join(MODEL_DIR, name)):
                ks.append(int(m.group(1)))
    return sorted(ks)


def model_is_cached(k):
    """该 K 是否已有完整缓存(三个模型文件齐全)"""
    d = os.path.join(MODEL_DIR, "k%d" % k)
    return all(os.path.exists(os.path.join(d, fn)) for fn in MODEL_FILES)


def cache_model(k):
    """把当前 output/ 的训练产物存入模型缓存 output/models/k<k>/"""
    d = os.path.join(MODEL_DIR, "k%d" % k)
    os.makedirs(d, exist_ok=True)
    for fn in MODEL_FILES:
        src = os.path.join(OUTPUT_DIR, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(d, fn))


def restore_model(k):
    """把已缓存的模型拷回 output/,作为当前生效模型(不重新训练)"""
    d = os.path.join(MODEL_DIR, "k%d" % k)
    for fn in MODEL_FILES:
        src = os.path.join(d, fn)
        dst = os.path.join(OUTPUT_DIR, fn)
        if os.path.exists(src):
            shutil.copy2(src, dst)


def start_set_k(target_k):
    """统一入口:已训练过的 K 直接复用缓存(秒级),未训练过的才跑 --build 训练"""
    reuse = model_is_cached(target_k)
    state.update(training=True, target_k=target_k,
                 started_at=time.strftime("%H:%M:%S"),
                 action="reuse" if reuse else "train", error=None)

    def _work():
        try:
            if reuse:
                restore_model(target_k)  # 复用:把缓存模型设为当前生效模型
            else:
                proc = subprocess.run([VPR, "--build"], cwd=ROOT,
                                      capture_output=True, text=True, timeout=7200)
                if proc.returncode != 0:
                    state["error"] = (proc.stderr or proc.stdout)[-500:]
                    return
                cache_model(target_k)  # 训练成功 → 存入缓存,下次直接复用
            q = query_image_from_html()
            if q:
                subprocess.run([VPR, "--query", q], cwd=ROOT,
                               capture_output=True, text=True, timeout=600)
        except Exception as e:  # noqa: BLE001
            state["error"] = str(e)
        finally:
            state["training"] = False

    threading.Thread(target=_work, daemon=True).start()


def list_library_images():
    """扫描 data/ 下的图库图片,返回相对项目根的路径列表"""
    exts = (".png", ".jpg", ".jpeg")
    out = []
    if not os.path.isdir(DATA_DIR):
        return out
    for dirpath, _dirs, files in os.walk(DATA_DIR):
        for fn in sorted(files):
            if fn.lower().endswith(exts):
                full = os.path.join(dirpath, fn)
                out.append(os.path.relpath(full, ROOT).replace(os.sep, "/"))
    return out


def resolve_image(raw):
    """把用户输入的路径/文件名解析为真实文件路径"""
    p = (raw or "").strip().strip('"').strip("'")
    if not p:
        return None, "图片路径为空"
    candidates = []
    if os.path.isabs(p):
        candidates.append(p)
    else:
        candidates.append(os.path.join(ROOT, p))
    # 无论输入是路径还是纯文件名,都再按文件名在 data 图库中搜索一次
    base = os.path.basename(p)
    if base:
        for dirpath, _dirs, files in os.walk(DATA_DIR):
            if base in files:
                candidates.append(os.path.join(dirpath, base))
                break
    for c in candidates:
        if os.path.isfile(c):
            return c, None
    return None, "找不到图片: %s(支持图库文件名、相对路径或绝对路径)" % p


def run_query(path):
    """执行一次检索(--query),成功后 results.html 已被重写"""
    try:
        proc = subprocess.run([VPR, "--query", path], cwd=ROOT,
                              capture_output=True, text=True, timeout=300)
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout)[-300:]
    return True, None


TRAINING_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>处理中…</title>
<style>body{font-family:Arial,Helvetica,sans-serif;background:#f6f8fa;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.card{background:#fff;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.08);padding:32px 44px;text-align:center;max-width:560px}
.spinner{width:34px;height:34px;border:4px solid #e3edf7;border-top-color:#1e88e5;border-radius:50%;margin:0 auto 18px;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
h2{margin:0 0 8px;color:#222} p{color:#666;margin:6px 0} #err{color:#c62828;display:none;margin-top:10px;word-break:break-all}
</style></head><body><div class="card">
<div class="spinner"></div>
<h2 id="title">正在重新训练(K-Means 聚类中…)</h2>
<p id="sub">目标聚类中心个数:<b id="tk"></b> &nbsp;开始时间:<b id="st"></b></p>
<p id="timehint">K 越大耗时越久(K=2000 约 15 分钟),请勿关闭此页面。</p>
<div id="err"></div>
</div>
<script>
var tk=document.getElementById('tk'),st=document.getElementById('st'),err=document.getElementById('err'),
    title=document.getElementById('title'),sub=document.getElementById('sub'),timehint=document.getElementById('timehint');
function poll(){
  fetch('/status').then(function(r){return r.json()}).then(function(s){
    tk.textContent=s.target_k; st.textContent=s.started_at;
    if(s.action==='reuse'){
      title.textContent='正在复用已训练的模型…';
      sub.textContent='K='+s.target_k+' 之前已训练过,无需重新训练,正在恢复模型并刷新结果';
      timehint.textContent='';
    }
    if(s.error){ err.style.display='block'; err.textContent='出错: '+s.error; }
    if(!s.training){ location.href='/'; return; }
    setTimeout(poll, 1500);
  }).catch(function(){ setTimeout(poll, 3000); });
}
poll();
</script></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text, code=200):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            self._send_json({
                "training": state["training"],
                "target_k": state["target_k"],
                "current_k": read_current_k(),
                "started_at": state["started_at"],
                "action": state["action"],
                "error": state["error"],
            })
        elif parsed.path == "/trained":
            self._send_json({"ks": trained_k_list(),
                             "current_k": read_current_k()})
        elif parsed.path == "/images":
            imgs = list_library_images()
            self._send_json({"images": imgs, "count": len(imgs)})
        elif parsed.path in ("/", "/results.html"):
            if os.path.exists(RESULTS_HTML):
                with open(RESULTS_HTML, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_html(
                    "<h1>未找到 output/results.html</h1>"
                    "<p>请先完成首次构建与检索:</p>"
                    "<pre>./build/vpr --build\n./build/vpr --query data/Freiburg/imageCompressedCam0_0005650.png</pre>",
                    404)
        elif parsed.path == "/training":
            self._send_html(TRAINING_PAGE)
        else:
            super().do_GET()  # 静态文件(../data/... 等相对路径资源)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/set_k":
            length = int(self.headers.get("Content-Length", 0))
            data = parse_qs(self.rfile.read(length).decode("utf-8"))
            raw = (data.get("dictionary_size") or [""])[0].strip()
            try:
                k = int(raw)
                if k <= 0:
                    raise ValueError
            except ValueError:
                self._send_html("<h1>输入无效</h1>"
                                "<p>聚类中心个数必须是正整数。</p><a href='/'>返回</a>", 400)
                return
            if state["training"]:
                self._send_html("<h1>正在训练中</h1>"
                                "<p>请等待当前训练完成后再修改。</p><a href='/training'>查看进度</a>", 409)
                return
            write_new_k(k)
            start_set_k(k)
            self.send_response(303)
            self.send_header("Location", "/training")
            self.end_headers()
        elif parsed.path == "/query":
            if state["training"]:
                self._send_html("正在训练中,请稍候再查询", 409)
                return
            length = int(self.headers.get("Content-Length", 0))
            data = parse_qs(self.rfile.read(length).decode("utf-8"))
            raw = (data.get("image_path") or [""])[0]
            resolved, err = resolve_image(raw)
            if not resolved:
                self._send_html(err, 404)
                return
            ok, msg = run_query(resolved)
            if not ok:
                self._send_html("查询失败: " + msg, 500)
                return
            self._send_html("OK")
        elif parsed.path == "/upload":
            if state["training"]:
                self._send_html("正在训练中,请稍候再查询", 409)
                return
            qs = parse_qs(parsed.query)
            name = os.path.basename((qs.get("name") or ["upload.png"])[0]) or "upload.png"
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_UPLOAD:
                self._send_html("文件大小无效(上限 20MB)", 400)
                return
            body = self.rfile.read(length)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            dest = os.path.join(UPLOAD_DIR, name)
            with open(dest, "wb") as f:
                f.write(body)
            ok, msg = run_query(dest)
            if not ok:
                self._send_html("查询失败: " + msg, 500)
                return
            self._send_html("OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        print("[%s] %s" % (time.strftime("%H:%M:%S"), fmt % args))


def _open_browser(url):
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url = "http://127.0.0.1:%d/" % PORT
    print("=" * 56)
    print("视觉位置识别 Web UI 已启动: %s" % url)
    print("当前聚类中心个数: %s (config/config.yaml)" % read_current_k())
    print("按 Ctrl+C 停止服务")
    print("=" * 56)
    threading.Timer(0.6, _open_browser, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
