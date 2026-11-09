"""정적 웹용 데이터 내보내기 → out/web/
- seats.json : 좌석 위치(평면 x,z·층), 모드별 1kHz 지표와 공연장 내 백분위, 대역별 C80·G
- env.bin    : 좌석×모드×대역×100칸(2ms) 반사 에너지, uint8 = 0.5dB 단위(0 = 무음). 브라우저에서 IR 합성용
- 직접음(Ed, 도착 ms)은 seats.json에 넣는다
흡음률·스피커 배치가 가정값이므로 화면에는 절대 수치보다 '공연장 내 백분위'를 먼저 보여 준다.
"""
import sys, io, os, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
M = np.load('out/v2/metrics.npz'); seats = M['seats']; ok = ~M['excluded']
MODES = ('natural', 'pa'); BANDS = (250, 1000, 4000); NBIN = 100
idx = np.nonzero(ok)[0]; bal = seats[:, 1] > 6.6
os.makedirs('out/web', exist_ok=True)


def pct(v):  # 공연장 내 백분위(0=최하, 100=최상), 유효 좌석 기준
    r = np.argsort(np.argsort(v[idx])); return np.round(r / (len(idx) - 1) * 100).astype(int)


rows = []
P = {m: {k: pct(M[f'{m}_1000_{k}']) for k in ('C80', 'D50', 'G', 'LF')} for m in MODES}
allE = np.concatenate([M[f'{m}_{b}_hist'][idx].reshape(len(idx), -1, 2).sum(2).ravel() for m in MODES for b in BANDS])
Lref = 10 * np.log10(allE[allE > 0].max())
env = np.zeros((len(idx), len(MODES), len(BANDS), NBIN), np.uint8)
for j, i in enumerate(idx):
    row = {'id': int(j), 'x': round(float(seats[i, 0]), 2), 'z': round(float(seats[i, 2]), 2), 'f': 2 if bal[i] else 1}
    for mi, m in enumerate(MODES):
        row[m] = {k: round(float(M[f'{m}_1000_{k}'][i]), 2 if k in ('D50', 'LF') else 1) for k in ('C80', 'D50', 'G', 'LF')}
        row[m]['p'] = {k: int(P[m][k][j]) for k in ('C80', 'D50', 'G', 'LF')}
        row[m]['bands'] = {str(b): {'C80': round(float(M[f'{m}_{b}_C80'][i]), 1), 'G': round(float(M[f'{m}_{b}_G'][i]), 1)} for b in BANDS}
        row[m]['Ed'] = [float(f'{M[f"{m}_{b}_Ed"][i]:.4g}') for b in BANDS]
        row[m]['t0'] = round(float(M[f'{m}_1000_tdir'][i]), 1)
        for bi, b in enumerate(BANDS):
            h = M[f'{m}_{b}_hist'][i].reshape(NBIN, 2).sum(1)
            q = np.where(h > 0, np.round((10 * np.log10(np.maximum(h, 1e-30)) - Lref) * 2) + 255, 0)
            env[j, mi, bi] = np.clip(q, 0, 255).astype(np.uint8)
    rows.append(row)
meta = {'modes': list(MODES), 'bands': list(BANDS), 'bin_ms': 2, 'nbin': NBIN, 'env_db_ref': round(float(Lref), 3),
        'env_encoding': 'uint8 q: 0=무음, 그 외 에너지[dB] = env_db_ref + (q-255)/2  (2ms 칸 에너지)',
        'T_room_s': [round(float(t), 2) for t in M['T_room']],
        'stage_edge_x': -20.5, 'n_seats': len(rows),
        'note': '흡음률·산란계수·스피커 배치는 가정값. 좌석 간 상대 비교용이며 실측 임펄스응답으로 검증되지 않음.'}
json.dump({'meta': meta, 'seats': rows}, open('out/web/seats.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
env.tofile('out/web/env.bin')
print('seats', len(rows), ' seats.json', os.path.getsize('out/web/seats.json') // 1024, 'KB  env.bin', os.path.getsize('out/web/env.bin') // 1024, 'KB')
