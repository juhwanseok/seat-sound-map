"""v2 흡음 민감도 (r01 할 일 #7): 무대 육성·1kHz, 시나리오별 시드 1·2 평균 vs 기준(같은 시드 1·2 평균)
- 기준: 객석 0.90, 벽·천장 0.15 (out/v2/natural_1000_s1·s2)
- 시나리오: 객석 0.80 / 0.95, 벽·천장 0.08 / 0.25 (out/v2sens/*)
- 잡음 바닥: 기준 조건에서 시드 1·2 평균 vs 시드 3·4 평균의 순위 상관 (시드만 바뀌어도 이만큼은 흔들린다)
같은 시드를 써서 시나리오 간 비교의 난수 잡음을 줄였다(공통 난수).
산출: out/v2sens/sensitivity.json
"""
import sys, io, json
import numpy as np
from scipy.stats import spearmanr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
NB = 200; G_REF = 1 / (4 * np.pi * 100); K = ('C80', 'D50', 'G', 'LF')
meta = np.load('out/v2/meta.npz'); ok = ~meta['excluded']


def load(paths):
    H = [np.load(p) for p in paths]
    return (np.mean([h['hist'] for h in H], 0), np.mean([h['lat'] for h in H], 0),
            H[0]['Ed'], H[0]['tdir'], float(np.mean([h['T_fit'] for h in H])))


def metrics(hist, lat, Ed, tdir, T):
    td = np.round(tdir).astype(int); n = len(td); rows = np.arange(n)
    tail = (hist[:, 150:200].sum(1) / 0.05) * (T / 13.82)
    cum = np.cumsum(hist, 1)
    def upto(ms):
        k = np.clip(td + ms, 0, NB) - 1
        return np.where(k >= 0, cum[rows, np.maximum(k, 0)], 0.0)
    E = Ed + hist.sum(1) + tail; e80 = Ed + upto(80); e50 = Ed + upto(50)
    late = np.maximum(E - e80, 1e-3 * E)
    lc = np.cumsum(lat, 1); k5 = np.clip(td + 5, 0, NB) - 1; k80 = np.clip(td + 80, 0, NB) - 1
    l = lc[rows, k80] - np.where(k5 >= 0, lc[rows, np.maximum(k5, 0)], 0)
    return {'C80': 10 * np.log10(e80 / late), 'D50': e50 / E, 'G': 10 * np.log10(E / G_REF), 'LF': l / e80}


base = load([f'out/v2/natural_1000_s{s}.npz' for s in (1, 2)]); Mb = metrics(*base)
alt = load([f'out/v2/natural_1000_s{s}.npz' for s in (3, 4)]); Ma = metrics(*alt)
res = {'condition': '무대 육성, 1kHz, 시드 1·2 평균. 기준 객석 0.90 / 벽·천장 0.15',
       'noise_floor_seed12_vs_seed34': {k: round(float(spearmanr(Mb[k][ok], Ma[k][ok]).correlation), 3) for k in K},
       'T_base_s': round(base[4], 2), 'scenarios': {}}
for tag, label in (('aud080', '객석 0.80'), ('aud095', '객석 0.95'), ('wall008', '벽·천장 0.08'), ('wall025', '벽·천장 0.25')):
    sc = load([f'out/v2sens/{tag}/natural_1000_s{s}.npz' for s in (1, 2)]); Ms = metrics(*sc)
    res['scenarios'][label] = {'T_fit_s': round(sc[4], 2),
                               'rank_corr_vs_base': {k: round(float(spearmanr(Mb[k][ok], Ms[k][ok]).correlation), 3) for k in K},
                               'median_shift': {k: round(float(np.median(Ms[k][ok] - Mb[k][ok])), 3) for k in K}}
allc = [v for s in res['scenarios'].values() for v in s['rank_corr_vs_base'].values()]
res['rank_corr_range_all'] = [min(allc), max(allc)]
json.dump(res, open('out/v2sens/sensitivity.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
