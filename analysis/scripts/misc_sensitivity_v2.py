"""v2 그 밖의 민감도 (r04 할 일 #4·#5): 무대 육성 음원 위치, 소리를 받는 공 반지름, 종합 지표, 층 안 대사 또렷함 차이
- 사용: python -I scripts/misc_sensitivity_v2.py
- 기준: out/v2/<mode>_1000_s1·s2 (R 1.0m, 무대 육성 음원 x −17.0·z −3.86)
- 변형 (1kHz, 시드 1·2 — 기준과 같은 난수):
    out/v2misc/src/natural_back  무대 육성 음원을 무대 안쪽으로 3m (x −14.0)
    out/v2misc/src/natural_side  무대 육성 음원을 옆으로 3m (z −0.86)
    out/v2misc/r05/natural·pa    소리를 받는 공 반지름 1.0 → 0.5m
- 200ms 이후 꼬리의 감쇠율: 각 조건의 무대 육성 추적값(T_fit). 확성(pa)은 같은 폴더의 무대 육성 값을 쓴다(sensitivity_v2.py와 같은 규칙)
- 종합(TOT) = 대사 또렷함·소리 크기·감싸는 느낌 백분위의 평균 (화면 app.js와 같은 정의)
- 층 안 대사 또렷함 차이: out/v2/metrics.npz (시드 4개 평균)의 pa_1000_D50
산출: out/v2misc/misc_sensitivity.json
"""
import sys, io, json
import numpy as np
from scipy.stats import spearmanr, rankdata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
NB = 200; G_REF = 1 / (4 * np.pi * 100); K = ('C80', 'D50', 'G', 'LF', 'TOT')
meta = np.load('out/v2/meta.npz'); seats = meta['seats']; ok = ~meta['excluded']; bal = seats[:, 1] > 6.6


def load(d, mode, seeds=(1, 2), tdir_=None, tmode='natural'):
    Hs = [np.load(f'{d}/{mode}_1000_s{s}.npz') for s in seeds]
    T = float(np.mean([np.load(f'{tdir_ or d}/{tmode}_1000_s{s}.npz')['T_fit'] for s in seeds]))
    return np.mean([h['hist'] for h in Hs], 0), np.mean([h['lat'] for h in Hs], 0), Hs[0]['Ed'], Hs[0]['tdir'], T


def metrics(hist, lat, Ed, tdir, T):
    td = np.round(tdir).astype(int); rows = np.arange(len(td))
    tail = (hist[:, 150:200].sum(1) / 0.05) * (T / 13.82); cum = np.cumsum(hist, 1)
    def upto(ms):
        k = np.clip(td + ms, 0, NB) - 1
        return np.where(k >= 0, cum[rows, np.maximum(k, 0)], 0.0)
    E = Ed + hist.sum(1) + tail; e80 = Ed + upto(80); e50 = Ed + upto(50)
    lc = np.cumsum(lat, 1); k5 = np.clip(td + 5, 0, NB) - 1; k80 = np.clip(td + 80, 0, NB) - 1
    l = lc[rows, k80] - np.where(k5 >= 0, lc[rows, np.maximum(k5, 0)], 0)
    M = {'C80': 10 * np.log10(e80 / np.maximum(E - e80, 1e-3 * E)), 'D50': e50 / E, 'G': 10 * np.log10(E / G_REF), 'LF': l / e80}
    pct = lambda v: rankdata(v[ok]) / ok.sum()
    tot = np.full(len(td), np.nan); tot[ok] = (pct(M['D50']) + pct(M['G']) + pct(M['LF'])) / 3
    M['TOT'] = tot; return M


def corr(A, B): return {k: round(float(spearmanr(A[k][ok], B[k][ok]).correlation), 3) for k in K}


res = {'condition': '1kHz, 시드 1·2 평균. 순위 일치도 = 스피어만 순위 상관(유효 1,172석)', 'variants': {}}
base = {m: metrics(*load('out/v2', m)) for m in ('natural', 'pa')}
# 다시 계산했을 때의 흔들림(시드 1·2 대 3·4) — 종합 지표 포함
res['repeat_s12_vs_s34'] = {m: corr(base[m], metrics(*load('out/v2', m, seeds=(3, 4)))) for m in ('natural', 'pa')}
V = (('natural_back', '무대 육성 음원을 무대 안쪽으로 3m', 'natural', 'out/v2misc/src', 'natural_back'),
     ('natural_side', '무대 육성 음원을 옆으로 3m', 'natural', 'out/v2misc/src', 'natural_side'),
     ('natural_R05', '무대 육성, 받는 공 반지름 0.5m', 'natural', 'out/v2misc/r05', 'natural'),
     ('pa_R05', '스피커 확성, 받는 공 반지름 0.5m', 'pa', 'out/v2misc/r05', 'pa'))
for name, label, bm, d, f in V:
    tmode = f if f.startswith('natural') else 'natural'
    M = metrics(*load(d, f, tmode=tmode))
    res['variants'][name] = {'label': label, 'base': bm, 'T_fit_s': round(load(d, f, tmode=tmode)[4], 2), 'rank_corr_vs_base': corr(base[bm], M)}
four = lambda x: {k: v for k, v in x.items() if k != 'TOT'}
res['summary'] = {
    'source_position_min_max_4metrics': [min(min(four(res['variants'][n]['rank_corr_vs_base']).values()) for n in ('natural_back', 'natural_side')),
                                         max(max(four(res['variants'][n]['rank_corr_vs_base']).values()) for n in ('natural_back', 'natural_side'))],
    'receiver_radius_min_max_4metrics': [min(min(four(res['variants'][n]['rank_corr_vs_base']).values()) for n in ('natural_R05', 'pa_R05')),
                                         max(max(four(res['variants'][n]['rank_corr_vs_base']).values()) for n in ('natural_R05', 'pa_R05'))],
}
# 층 안 대사 또렷함: 상위 20% 대 하위 20% (스피커 확성, 1kHz, 시드 4개 평균)
Mx = np.load('out/v2/metrics.npz'); d50 = Mx['pa_1000_D50']
res['d50_within_floor_pa_1k'] = {}
for nm, msk in (('1층', ok & ~bal), ('2층', ok & bal)):
    v = np.sort(d50[msk]); n = len(v); k = int(round(n * 0.2))
    res['d50_within_floor_pa_1k'][nm] = {'n': n, 'n_top20': k, 'median_top20': round(float(np.median(v[-k:])), 2), 'median_bottom20': round(float(np.median(v[:k])), 2),
                                         'median_all': round(float(np.median(v)), 2), 'p10': round(float(np.percentile(v, 10)), 2), 'p90': round(float(np.percentile(v, 90)), 2)}
json.dump(res, open('out/v2misc/misc_sensitivity.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
