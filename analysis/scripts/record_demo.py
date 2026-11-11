"""시연 영상 녹화 — 공개된 서비스 화면을 화면 없는 크롬으로 조작하며 프레임과 소리를 받아 mp4로 만든다.

사용법: python record_demo.py <작업 폴더> <출력.mp4> [--url URL] [--size 1280x720]
- 크롬 원격 조작(CDP)으로 실제 클릭을 보내고, 화면은 Page.startScreencast 프레임으로, 소리는 페이지의
  오디오 출력에 녹음기를 붙여(MediaRecorder) 받는다. 설치가 필요한 것은 없다(크롬, websocket-client, imageio-ffmpeg).
- 화면 아래 자막과 가짜 마우스 커서는 녹화 세션에만 넣는 것이고 서비스 파일은 바꾸지 않는다.
"""
import argparse, base64, json, os, subprocess, sys, threading, time, urllib.request, queue

import imageio_ffmpeg
import websocket

CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
PORT = 9377

TAP_JS = r"""
(() => {
  window.__rec = { chunks: [], t0: null, err: null };
  const orig = AudioNode.prototype.connect;
  AudioNode.prototype.connect = function (dest, ...a) {
    const r = orig.call(this, dest, ...a);
    try {
      if (dest instanceof AudioDestinationNode && !(dest.context instanceof OfflineAudioContext)) {
        const c = dest.context;
        if (!c.__tap) {
          c.__tap = c.createMediaStreamDestination();
          const mr = new MediaRecorder(c.__tap.stream, { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 160000 });
          mr.ondataavailable = e => window.__rec.chunks.push(e.data);
          mr.start(); window.__rec.mr = mr; window.__rec.t0 = Date.now();
        }
        orig.call(this, c.__tap);
      }
    } catch (e) { window.__rec.err = String(e); }
    return r;
  };
})();
"""

OVERLAY_JS = r"""
(() => {
  const st = document.createElement('style');
  st.textContent = `
    #__cap { position: fixed; left: 0; right: 0; bottom: 0; z-index: 99998; background: rgba(20,20,20,.86); color: #fff;
      font: 600 21px/1.45 'Malgun Gothic', sans-serif; padding: 12px 28px 14px; text-align: center; letter-spacing: -.2px; }
    #__cap small { display: block; font: 400 15px/1.4 'Malgun Gothic', sans-serif; color: #d8d2c8; margin-top: 2px; }
    #__cur { position: fixed; left: 0; top: 0; width: 22px; height: 22px; margin: -11px 0 0 -11px; border-radius: 50%; z-index: 99999;
      background: rgba(180,70,47,.55); border: 2px solid #b4462f; pointer-events: none; transition: transform .55s ease; transform: translate(640px, 300px); }
    #__cur.hit { background: rgba(180,70,47,.95); box-shadow: 0 0 0 10px rgba(180,70,47,.25); }`;
  document.head.appendChild(st);
  const cap = document.createElement('div'); cap.id = '__cap'; document.body.appendChild(cap);
  const cur = document.createElement('div'); cur.id = '__cur'; document.body.appendChild(cur);
  window.__caption = (a, b) => { cap.innerHTML = ''; cap.append(a); if (b) { const s = document.createElement('small'); s.textContent = b; cap.appendChild(s); } };
  window.__moveTo = (x, y) => { cur.style.transform = `translate(${x}px, ${y}px)`; };
  window.__hit = () => { cur.classList.add('hit'); setTimeout(() => cur.classList.remove('hit'), 260); };
})();
"""


class CDP:
    def __init__(self, ws_url, frame_dir):
        self.ws = websocket.create_connection(ws_url, max_size=None, suppress_origin=True)
        self.lock = threading.Lock(); self.n = 0; self.res = {}; self.frames = []; self.frame_dir = frame_dir
        threading.Thread(target=self._reader, daemon=True).start()

    def _send(self, method, params=None):
        with self.lock:
            self.n += 1; i = self.n; self.res[i] = queue.Queue()
            self.ws.send(json.dumps({'id': i, 'method': method, 'params': params or {}}))
        return i

    def _reader(self):
        while True:
            try:
                m = json.loads(self.ws.recv())
            except Exception:
                return
            if 'id' in m:
                q = self.res.get(m['id'])
                if q: q.put(m)
            elif m.get('method') == 'Page.screencastFrame':
                p = m['params']; k = len(self.frames)
                path = os.path.join(self.frame_dir, f'f{k:05d}.jpg')
                open(path, 'wb').write(base64.b64decode(p['data']))
                self.frames.append((p['metadata']['timestamp'], path))
                self._send('Page.screencastFrameAck', {'sessionId': p['sessionId']})

    def call(self, method, params=None, timeout=60):
        m = self.res[self._send(method, params)].get(timeout=timeout)
        if 'error' in m: raise RuntimeError(f'{method}: {m["error"]}')
        return m['result']

    def js(self, expr, await_promise=False, timeout=60):
        r = self.call('Runtime.evaluate', {'expression': expr, 'returnByValue': True, 'awaitPromise': await_promise}, timeout)
        if 'exceptionDetails' in r: raise RuntimeError(json.dumps(r['exceptionDetails'], ensure_ascii=False)[:500])
        return r['result'].get('value')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('work'); ap.add_argument('out')
    ap.add_argument('--url', default='https://juhwanseok.github.io/seat-sound-map/')
    ap.add_argument('--size', default='1280x720')
    a = ap.parse_args()
    W, H = map(int, a.size.split('x'))
    fdir = os.path.join(a.work, 'frames'); os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir): os.remove(os.path.join(fdir, f))
    prof = os.path.join(a.work, 'chrome-profile')
    proc = subprocess.Popen([CHROME, '--headless=new', f'--remote-debugging-port={PORT}', f'--user-data-dir={prof}', f'--window-size={W},{H}',
                             '--autoplay-policy=no-user-gesture-required', '--hide-scrollbars', '--no-first-run', '--disable-extensions',
                             '--remote-allow-origins=*', 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/json')); break
            except Exception:
                time.sleep(0.25)
        page = next(t for t in tabs if t['type'] == 'page')
        c = CDP(page['webSocketDebuggerUrl'], fdir)
        c.call('Page.enable'); c.call('Runtime.enable')
        c.call('Emulation.setDeviceMetricsOverride', {'width': W, 'height': H, 'deviceScaleFactor': 1, 'mobile': False})
        c.call('Page.addScriptToEvaluateOnNewDocument', {'source': TAP_JS})
        c.call('Page.navigate', {'url': a.url})
        for _ in range(80):
            if c.js("document.readyState === 'complete' && !!window.SEAT_DATA && document.querySelectorAll('#map circle').length > 1000"): break
            time.sleep(0.25)
        else:
            sys.exit('페이지가 준비되지 않음')
        c.js(OVERLAY_JS)

        def cap(a_, b_=''): c.js(f'__caption({json.dumps(a_, ensure_ascii=False)}, {json.dumps(b_, ensure_ascii=False)})')
        def wait(s): time.sleep(s)
        def rect(sel):
            r = c.js(f"(() => {{ const e = document.querySelector({json.dumps(sel)}); e.scrollIntoView({{block: 'center', behavior: 'instant'}}); const b = e.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; }})()")
            return r
        def move(sel):
            x, y = rect(sel); c.js(f'__moveTo({x}, {y})'); wait(0.75); return x, y
        def click(sel):
            x, y = move(sel); c.js('__hit()')
            for t in ('mousePressed', 'mouseReleased'):
                c.call('Input.dispatchMouseEvent', {'type': t, 'x': x, 'y': y, 'button': 'left', 'clickCount': 1})
            wait(0.35)
        def scroll_top(): c.js("window.scrollTo({top: 0, behavior: 'smooth'})"); wait(0.8)
        def playing(): return c.js("!document.getElementById('stop').hidden")
        def wait_play(maxs):
            t = time.time(); wait(0.6)
            while playing() and time.time() - t < maxs: wait(0.2)

        seat2 = c.js("""(() => { const S = window.SEAT_DATA.seats.map((s, i) => [i, s]).filter(([i, s]) => s.f === 2);
            const xs = S.map(([i, s]) => s.x).sort((a, b) => a - b), xm = xs[Math.floor(xs.length * 0.35)];
            S.sort((a, b) => (Math.abs(a[1].z + 3.86) + Math.abs(a[1].x - xm)) - (Math.abs(b[1].z + 3.86) + Math.abs(b[1].x - xm))); return S[0][0]; })()""")
        dry_s = None

        cap('객석 소리지도 — 좌석마다 소리가 어떻게 다른지 미리 보고 듣습니다', '한국체육산업개발 공개 3D 데이터로 계산한 우리금융아트홀 1,172석 · 회원가입·로그인 없음')
        wait(0.5)
        c.call('Page.startScreencast', {'format': 'jpeg', 'quality': 88, 'maxWidth': W, 'maxHeight': H, 'everyNthFrame': 1})
        wait(4.5)

        cap('① 지표를 고르면 지도 색이 바뀝니다 — 소리 크기', '색은 값이 아니라 이 공연장 안에서의 순위입니다 (밝을수록 높음)')
        move('#metric'); c.js('__hit()')
        setm = lambda k: c.js("(() => { const m = document.getElementById('metric'); m.value = '%s'; m.dispatchEvent(new Event('change', {bubbles: true})); })()" % k)
        setm('G'); wait(3.0)
        cap('① 기본 지표는 "대사가 또렷한 정도"입니다', '색은 값이 아니라 이 공연장 안에서의 순위입니다 (밝을수록 높음)')
        c.js('__hit()'); setm('D50'); wait(2.5)

        cap('② 좌석을 누르면 그 자리의 소리 특성이 나옵니다', '지표별 순위(다섯 구간)와, 소리가 시간 순서로 도착하는 모습(소리 도착 그래프)')
        click('#map circle[data-i="306"]'); wait(4.5)

        cap('③ 먼저 원음을 듣습니다', '직접 합성한 박수·피아노 소리 (헤드폰 권장)')
        click('#play-dry'); t0 = time.time(); wait_play(9); dry_s = time.time() - t0; wait(0.4)

        cap('③ 같은 소리를 이 자리에서 들으면 — 1층 가운데', '이 좌석에 소리가 도착하는 모습을 원음에 입혀 들려줍니다')
        click('#play-seat'); wait_play(11); wait(0.4)

        cap('④ 다른 자리와 비교합니다 — 먼저 이 자리를 담고', '[비교에 담기]')
        click('#add-compare'); wait(1.6)
        scroll_top()
        cap('④ 2층에서 좌석을 고릅니다', '2층에서는 스피커 가정 안내가 지도 위에 함께 나옵니다')
        click('button[data-floor="2"]'); wait(1.5)
        click(f'#map circle[data-i="{seat2}"]'); wait(2.5)

        cap('④ 같은 소리를 2층 자리에서 들으면', '1층 가운데와 비교해 보세요')
        click('#play-seat'); wait_play(11); wait(0.4)

        cap('④ 두 자리를 표로 비교하고 각각 다시 들어 볼 수 있습니다', '[비교에 담기] → 두 자리 비교')
        click('#add-compare'); wait(1.0)
        c.js("document.getElementById('compare').scrollIntoView({block: 'center', behavior: 'smooth'})"); wait(5.0)

        scroll_top()
        cap('⑤ 마이크 없는 공연은 "무대 육성" 지도로 봅니다', '기본 지도는 뮤지컬처럼 스피커로 소리를 키운 경우입니다')
        click('button[data-mode="natural"]'); wait(4.5)

        cap('실제 측정값이 아닌 계산값이며, 가정은 화면에 밝혀 두었습니다', '사용한 공공데이터: 올림픽공원 공연장 실감형 콘텐츠 (공공데이터포털 15127487)')
        c.js("document.querySelector('.caveats').scrollIntoView({block: 'start', behavior: 'smooth'})"); wait(6.0)

        cap('객석 소리지도 · juhwanseok.github.io/seat-sound-map', '2026년도 한국체육산업개발(주) 공공데이터 활용 경진대회 출품 시연본')
        wait(3.5)
        c.call('Page.stopScreencast')
        rec = c.js("""new Promise(res => { const R = window.__rec; if (!R.mr) return res({err: R.err || 'no recorder'});
            R.mr.onstop = () => { const fr = new FileReader(); fr.onload = () => res({t0: R.t0, err: R.err, b64: fr.result.split(',')[1]}); fr.readAsDataURL(new Blob(R.chunks, {type: 'audio/webm'})); };
            R.mr.stop(); })""", await_promise=True, timeout=120)
        frames = list(c.frames)
    finally:
        proc.terminate()

    if not frames: sys.exit('프레임 없음')
    lst = os.path.join(a.work, 'frames.txt'); tv = frames[0][0]; end = frames[-1][0] + 0.5
    with open(lst, 'w', encoding='utf-8') as f:
        for (t, p), nxt in zip(frames, [x[0] for x in frames[1:]] + [end]):
            f.write(f"file '{p.replace(os.sep, '/')}'\nduration {max(0.02, nxt - t):.4f}\n")
        f.write(f"file '{frames[-1][1].replace(os.sep, '/')}'\n")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst]
    has_audio = bool(rec and rec.get('b64'))
    if has_audio:
        ap_ = os.path.join(a.work, 'audio.webm'); open(ap_, 'wb').write(base64.b64decode(rec['b64']))
        off = rec['t0'] / 1000 - tv
        cmd += ['-itsoffset', f'{off:.3f}', '-i', ap_]
    cmd += ['-vf', 'fps=25,format=yuv420p', '-c:v', 'libx264', '-preset', 'slow', '-crf', '24', '-movflags', '+faststart']
    if has_audio: cmd += ['-c:a', 'aac', '-b:a', '128k', '-af', 'aresample=async=1:first_pts=0']
    cmd += [a.out]
    subprocess.run(cmd, check=True)
    print(json.dumps({'frames': len(frames), 'video_s': round(end - tv, 1), 'audio': has_audio, 'audio_err': rec.get('err') if rec else None,
                      'audio_offset_s': round(rec['t0'] / 1000 - tv, 2) if has_audio else None, 'dry_play_s': round(dry_s, 1) if dry_s else None,
                      'seat2': seat2, 'size_MB': round(os.path.getsize(a.out) / 1e6, 2)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
