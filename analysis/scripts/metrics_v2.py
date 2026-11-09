"""v2 지표 집계: 시드 평균 → 모드·대역별 좌석 지표(C80, D50, G, LF) + 수렴·모드 비교
늦은 꼬리(200ms 이후)의 감쇠율은 공간의 성질이므로 두 모드 모두 생음(무지향) 추적의 T_fit을 쓴다.
산출: out/v2/metrics.npz, out/v2/metrics_summary.json, out/seat_map_v2.png
"""
import sys, io, json
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
D = 'out/v2'; MODES = ('natural', 'pa'); BANDS = (250, 1000, 4000); SEEDS = (1, 2, 3, 4)
meta = np.load(f'{D}/meta.npz'); seats, excl = meta['seats'], meta['excluded']; ok = ~excl
NS = len(seats); NB = 200; tt = np.arange(NB)
load = {(m, b, s): np.load(f'{D}/{m}_{b}_s{s}.npz') for m in MODES for b in BANDS for s in SEEDS}
T_room = {b: float(np.mean([load[('natural', b, s)]['T_fit'] for s in SEEDS])) for b in BANDS}
G_REF = 1 / (4 * np.pi * 100)


def metrics(hist, lat, Ed, tdir, T):
    td = np.round(tdir).astype(int)
    e_w = hist[:, 150:200].sum(1); tail = (e_w / 0.05) * (T / 13.82)
    cum = np.cumsum(hist, 1)
    def upto(ms):  # 직접음 도착 + ms 까지의 반사 에너지
        k = np.clip(td + ms, 0, NB) - 1
        return np.where(k >= 0, cum[np.arange(len(td)), np.maximum(k, 0)], 0.0)
    E_tot = Ed + hist.sum(1) + tail
    e80 = Ed + upto(80); e50 = Ed + upto(50)
    late = np.maximum(E_tot - e80, 1e-3 * E_tot)
    latc = np.cumsum(lat, 1); k5 = np.clip(td + 5, 0, NB) - 1; k80 = np.clip(td + 80, 0, NB) - 1
    l5_80 = latc[np.arange(len(td)), k80] - np.where(k5 >= 0, latc[np.arange(len(td)), np.maximum(k5, 0)], 0)
    return dict(C80=10 * np.log10(e80 / late), D50=e50 / E_tot, G=10 * np.log10(E_tot / G_REF), LF=l5_80 / e80)


R = {}; half = {}
for m in MODES:
    for b in BANDS:
        H_ = [load[(m, b, s)] for s in SEEDS]
        hist = np.mean([h['hist'] for h in H_], 0); lat = np.mean([h['lat'] for h in H_], 0)
        R[(m, b)] = metrics(hist, lat, H_[0]['Ed'], H_[0]['tdir'], T_room[b])
        R[(m, b)]['hist'] = hist; R[(m, b)]['Ed'] = H_[0]['Ed']; R[(m, b)]['tdir'] = H_[0]['tdir']
        A = metrics(np.mean([h['hist'] for h in H_[:2]], 0), np.mean([h['lat'] for h in H_[:2]], 0), H_[0]['Ed'], H_[0]['tdir'], T_room[b])
        B = metrics(np.mean([h['hist'] for h in H_[2:]], 0), np.mean([h['lat'] for h in H_[2:]], 0), H_[0]['Ed'], H_[0]['tdir'], T_room[b])
        half[(m, b)] = {k: round(float(spearmanr(A[k][ok], B[k][ok]).correlation), 3) for k in ('C80', 'D50', 'G', 'LF')}
summ = {'T_room_natural_fit_s': {b: round(T_room[b], 2) for b in BANDS},
        'T_fit_pa_s': {b: round(float(np.mean([load[('pa', b, s)]['T_fit'] for s in SEEDS])), 2) for b in BANDS},
        'split_half_rank_corr(2seeds_vs_2seeds)': {f'{m}_{b}': half[(m, b)] for m in MODES for b in BANDS}}
for m in MODES:
    for b in BANDS:
        summ[f'{m}_{b}'] = {k: [round(float(x), 2) for x in np.percentile(R[(m, b)][k][ok], [0, 10, 50, 90, 100])] for k in ('C80', 'D50', 'G', 'LF')}
# 모드 비교 (1kHz): 음량 고르기(G 범위), 명료도
g_nat, g_pa = R[('natural', 1000)]['G'][ok], R[('pa', 1000)]['G'][ok]
summ['mode_compare_1k'] = {'G_p90_minus_p10_natural_dB': round(float(np.subtract(*np.percentile(g_nat, [90, 10]))), 2),
                           'G_p90_minus_p10_pa_dB': round(float(np.subtract(*np.percentile(g_pa, [90, 10]))), 2),
                           'rank_corr_natural_vs_pa': {k: round(float(spearmanr(R[('natural', 1000)][k][ok], R[('pa', 1000)][k][ok]).correlation), 3) for k in ('C80', 'D50', 'G', 'LF')}}
np.savez_compressed(f'{D}/metrics.npz', seats=seats, excluded=excl, T_room=np.array([T_room[b] for b in BANDS]),
                    **{f'{m}_{b}_{k}': R[(m, b)][k] for m in MODES for b in BANDS for k in ('C80', 'D50', 'G', 'LF', 'hist', 'Ed', 'tdir')})
json.dump(summ, open(f'{D}/metrics_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1))

bal = seats[:, 1] > 6.6
fig, ax = plt.subplots(2, 3, figsize=(22, 13))
for row, m in enumerate(MODES):
    for col, k in enumerate(('C80', 'D50', 'G')):
        a = ax[row, col]; v = R[(m, 1000)][k]; vmin, vmax = np.percentile(v[ok], [2, 98])
        f1, f2 = ok & ~bal, ok & bal
        sc = a.scatter(seats[f1, 0], seats[f1, 2], c=v[f1], s=9, cmap='viridis', vmin=vmin, vmax=vmax)
        a.scatter(seats[f2, 0], seats[f2, 2] + 30, c=v[f2], s=9, marker='^', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=a); a.set_aspect('equal'); a.axvline(-20.5, c='k', lw=1)
        a.set_title(f"{'생음(무확성)' if m == 'natural' else '확성(PA, 가정 배치)'} — {k} 1kHz  (1층 ●, 2층 ▲ 위로 30m)", fontname='Malgun Gothic', fontsize=10)
plt.tight_layout(); plt.savefig('out/seat_map_v2.png', dpi=60); print('saved')
