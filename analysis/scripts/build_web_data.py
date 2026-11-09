"""웹 데이터 빌드 → 02_제출물/web/data/*.js
서버 없이(file://)도, GitHub Pages에서도 열리도록 데이터를 JS 전역 변수로 싼다.
- seats.js : window.SEAT_DATA = {meta, seats}  (out/web/seats.json + 재생 배율)
- env.js   : window.ENV_B64 = "..."            (out/web/env.bin, base64)
- dry.js   : window.DRY_B64 = "..."            (out/audio/00_dry.wav, base64)
재생 배율(play_gain): 가장 에너지가 큰 좌석의 미리듣기 최고점이 0.9가 되도록 Python 합성으로 계산.
"""
import sys, io, os, json, base64
import numpy as np
from scipy.signal import fftconvolve
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
OUT = '../data'   # 저장소 배치: analysis/ 에서 실행, 저장소 루트의 data/ 에 쓴다; os.makedirs(OUT, exist_ok=True)
d = json.load(open('out/web/seats.json', encoding='utf-8')); meta, S = d['meta'], d['seats']
env = np.fromfile('out/web/env.bin', np.uint8).reshape(len(S), len(meta['modes']), len(meta['bands']), meta['nbin'])
E = np.where(env > 0, 10 ** ((meta['env_db_ref'] + (env.astype(float) - 255) / 2) / 10), 0)
T = np.array(meta['T_room_s'])
tail = E[..., 75:].sum(-1) / 50 * 1000 * T[None, None, :] / 13.82
Ed = np.array([[s[m]['Ed'] for m in meta['modes']] for s in S])
Etot = (E.sum(-1) + tail + Ed).sum(-1)       # 좌석 × 모드
meta['E_ref'] = float(Etot.max())
# 재생 배율: Python 합성기로 가장 큰 좌석을 렌더해 최고점 측정
src = open('scripts/ir_synth.py', encoding='utf-8').read().split("os.makedirs('out/audio'")[0]
src = src.replace("sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')", '')
ns = {}; exec(compile(src, 'ir_synth_head', 'exec'), ns)
idx = np.nonzero(~ns['M']['excluded'])[0]
j, mi = np.unravel_index(np.argmax(Etot), Etot.shape)
ir, _ = ns['synth_ir'](meta['modes'][mi], int(idx[j])); ir /= np.sqrt(meta['E_ref'])
x = ns['dry_signal'](); peak = np.abs(fftconvolve(x, ir)).max()
meta['play_gain'] = round(float(0.9 / peak), 4)
meta['fs'] = ns['FS']
open(f'{OUT}/seats.js', 'w', encoding='utf-8').write('window.SEAT_DATA=' + json.dumps({'meta': meta, 'seats': S}, ensure_ascii=False, separators=(',', ':')) + ';\n')
open(f'{OUT}/env.js', 'w', encoding='ascii').write('window.ENV_B64="' + base64.b64encode(env.tobytes()).decode() + '";\n')
open(f'{OUT}/dry.js', 'w', encoding='ascii').write('window.DRY_B64="' + base64.b64encode(open('out/audio/00_dry.wav', 'rb').read()).decode() + '";\n')
for f in ('seats.js', 'env.js', 'dry.js'):
    print(f, os.path.getsize(f'{OUT}/{f}') // 1024, 'KB')
print('E_ref', meta['E_ref'], 'play_gain', meta['play_gain'], 'loudest seat', int(j), meta['modes'][mi])
