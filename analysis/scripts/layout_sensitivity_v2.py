"""v2 스피커 배치 민감도: 장비 목록 구성(pa) 안에서 가정을 바꿨을 때 좌석 순위와 구역 비교가 유지되는지 (r03 할 일 #6)
- 사용: python -I scripts/layout_sensitivity_v2.py
- 기준: 장비 목록 구성 pa (out/v2/pa_1000_s1·s2)
- 변형 (out/v2layout/<이름>_1000_s1·s2, raytrace_v2.py의 _variant):
    pa_noub  발코니 아래 4대 끔 (나머지는 그대로)
    pa_ub1   발코니 아래 1대당 출력 비율 3% → 1%
    pa_ub5   발코니 아래 1대당 출력 비율 3% → 5%
    pa_ceil  2층 객석 위 천장 보조 스피커 4대 추가 (1대당 3%, 위치는 가정)
  + pa_simple 단순 배치(메인 2 + 센터 1, out/v2_pa_simple, 시드 1·2)
- 1kHz, 시드 1·2 평균(기준과 같은 난수). 200ms 이후 꼬리의 감쇠율은 무대 육성 추적값(out/v2 natural) 사용
산출: out/v2layout/layout_sensitivity.json
"""
import sys, io, json
import numpy as np
from scipy.stats import spearmanr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
NB = 200; G_REF = 1 / (4 * np.pi * 100); K = ('C80', 'D50', 'G', 'LF')
meta = np.load('out/v2/meta.npz'); seats = meta['seats']; ok = ~meta['excluded']
r = meta['r_natural']   # 무대 앞 가운데(무대 육성 음원)에서 좌석까지 거리 — make_figures.py와 같은 기준
bal = seats[:, 1] > 6.6
# 발코니 아래 판정: 원 점군 격자에서 귀 위 4m 이내에 장애물 (make_figures.py와 같은 기준)
ear = meta['ear']; g = np.load('out/grid.npz'); R_ = g['occ_raw']; lo = g['lo']; H = float(g['H'])
under = np.zeros(len(seats), bool)
for i in np.nonzero(ok & ~bal)[0]:
    q = np.floor((ear[i] - lo) / H).astype(int); under[i] = R_[q[0], q[1] + 1:q[1] + int(4 / H), q[2]].any()


def load(paths):
    Hs = [np.load(p) for p in paths]
    return np.mean([h['hist'] for h in Hs], 0), np.mean([h['lat'] for h in Hs], 0), Hs[0]['Ed'], Hs[0]['tdir']


T = float(np.mean([np.load(f'out/v2/natural_1000_s{s}.npz')['T_fit'] for s in (1, 2)]))


def metrics(hist, lat, Ed, tdir):
    td = np.round(tdir).astype(int); rows = np.arange(len(td))
    tail = (hist[:, 150:200].sum(1) / 0.05) * (T / 13.82); cum = np.cumsum(hist, 1)
    def upto(ms):
        k = np.clip(td + ms, 0, NB) - 1
        return np.where(k >= 0, cum[rows, np.maximum(k, 0)], 0.0)
    E = Ed + hist.sum(1) + tail; e80 = Ed + upto(80); e50 = Ed + upto(50)
    lc = np.cumsum(lat, 1); k5 = np.clip(td + 5, 0, NB) - 1; k80 = np.clip(td + 80, 0, NB) - 1
    l = lc[rows, k80] - np.where(k5 >= 0, lc[rows, np.maximum(k5, 0)], 0)
    return {'C80': 10 * np.log10(e80 / np.maximum(E - e80, 1e-3 * E)), 'D50': e50 / E, 'G': 10 * np.log10(E / G_REF), 'LF': l / e80}


def zone(M):
    """구역 비교: 발코니 아래 vs 1층 나머지의 G 차이, 같은 거리(22~28m)에서 2층 vs 발코니 아래."""
    same = ok & (r > 22) & (r < 28)
    med = lambda k, m: float(np.median(M[k][m]))
    return {
        'under_minus_rest_floor1_G_dB': round(med('G', ok & under) - med('G', ok & ~bal & ~under), 2),
        'same_dist_2F_minus_under_G_dB': round(med('G', same & bal) - med('G', same & under), 2),
        'same_dist_D50_2F': round(med('D50', same & bal), 2), 'same_dist_D50_under': round(med('D50', same & under), 2),
        'same_dist_G_2F': round(med('G', same & bal), 1), 'same_dist_G_under': round(med('G', same & under), 1),
        'floor2_minus_floor1_G_dB': round(med('G', ok & bal) - med('G', ok & ~bal), 2),
        'floor2_minus_floor1_D50': round(med('D50', ok & bal) - med('D50', ok & ~bal), 3),
    }


base = metrics(*load([f'out/v2/pa_1000_s{s}.npz' for s in (1, 2)]))
res = {'condition': '스피커 확성, 1kHz, 시드 1·2 평균. 기준 = 공개 장비 목록 구성(pa)',
       'n_under_balcony': int((ok & under).sum()), 'base': zone(base), 'variants': {}}
V = (('pa_noub', '발코니 아래 4대 끔', 'out/v2layout'), ('pa_ub1', '발코니 아래 출력 1%', 'out/v2layout'),
     ('pa_ub5', '발코니 아래 출력 5%', 'out/v2layout'), ('pa_ceil', '2층 천장 보조 4대 추가(가정)', 'out/v2layout'),
     ('pa_simple', '단순 배치(메인 2 + 센터 1)', 'out/v2_pa_simple'))
for name, label, d in V:
    f = name if d != 'out/v2_pa_simple' else 'pa'
    M = metrics(*load([f'{d}/{f}_1000_s{s}.npz' for s in (1, 2)]))
    res['variants'][name] = {'label': label,
                             'rank_corr_vs_base': {k: round(float(spearmanr(base[k][ok], M[k][ok]).correlation), 3) for k in K},
                             **zone(M)}
inside = [v for n, x in res['variants'].items() if n != 'pa_simple' for v in x['rank_corr_vs_base'].values()]
res['rank_corr_range_within_list_config'] = [min(inside), max(inside)]
json.dump(res, open('out/v2layout/layout_sensitivity.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
