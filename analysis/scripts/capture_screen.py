"""기획서 그림 1(서비스 화면) 캡처 — 화면 없는 크롬으로 주어진 주소를 열어 PNG로 저장한다.
사용법: python capture_screen.py <작업 폴더> <출력.png> [--url URL] [--size 1100x775] [--scale 1.5]
"""
import argparse, base64, json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_demo import CDP, CHROME

ap = argparse.ArgumentParser(); ap.add_argument('work'); ap.add_argument('out')
ap.add_argument('--url', default='https://juhwanseok.github.io/seat-sound-map/#m=pa&k=D50&f=1&s=307')
ap.add_argument('--size', default='1100x775'); ap.add_argument('--scale', type=float, default=1.5)
a = ap.parse_args(); W, H = map(int, a.size.split('x')); PORT = 9378
os.makedirs(os.path.join(a.work, 'frames'), exist_ok=True)
proc = subprocess.Popen([CHROME, '--headless=new', f'--remote-debugging-port={PORT}', f"--user-data-dir={os.path.join(a.work, 'chrome-profile-shot')}",
                         f'--window-size={W},{H}', '--hide-scrollbars', '--no-first-run', '--disable-extensions', '--remote-allow-origins=*', 'about:blank'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/json')); break
        except Exception:
            time.sleep(0.25)
    c = CDP(next(t for t in tabs if t['type'] == 'page')['webSocketDebuggerUrl'], os.path.join(a.work, 'frames'))
    c.call('Page.enable'); c.call('Runtime.enable')
    c.call('Emulation.setDeviceMetricsOverride', {'width': W, 'height': H, 'deviceScaleFactor': a.scale, 'mobile': False})
    c.call('Page.navigate', {'url': a.url})
    for _ in range(80):
        if c.js("document.readyState === 'complete' && !!window.SEAT_DATA && !document.getElementById('seat').hidden"): break
        time.sleep(0.25)
    time.sleep(0.8)
    r = c.call('Page.captureScreenshot', {'format': 'png', 'clip': {'x': 0, 'y': 0, 'width': W, 'height': H, 'scale': 1}})
    open(a.out, 'wb').write(base64.b64decode(r['data']))
    print('saved', a.out, c.js("document.getElementById('seat-title').textContent"))
finally:
    proc.terminate()
