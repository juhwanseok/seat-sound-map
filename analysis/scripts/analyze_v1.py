"""v1 분석: 수렴(시드 간), 흡음 가정 민감도(순위), 거리 대비 추가 정보, 구역별 비교, 좌석 지도"""
import sys, io, json, numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
L = {t: np.load(f'out/rt_v1_{t}.npz') for t in ('a15s1', 'a15s2', 'a08s1', 'a25s1')}
b = L['a15s1']; ok = ~b['excluded']; seats = b['seats'][ok]; r = b['r'][ok]
bal = seats[:,1] > 6.6
M = ('C80', 'D50', 'G', 'LF')
res = {'n_valid': int(ok.sum()), 'n_floor1': int((~bal).sum()), 'n_balcony': int(bal.sum()), 'official': {'floor1': 832, 'balcony': 352}}
res['convergence_seed1_vs_seed2'] = {m: {'spearman': round(float(spearmanr(b[m][ok], L['a15s2'][m][ok]).correlation), 3),
    'median_abs_diff': round(float(np.median(np.abs(b[m][ok] - L['a15s2'][m][ok]))), 3)} for m in M}
res['rank_vs_alpha_0.15'] = {t: {m: round(float(spearmanr(b[m][ok], L[t][m][ok]).correlation), 3) for m in M} for t in ('a08s1', 'a25s1')}
res['rank_corr_with_distance'] = {m: round(float(spearmanr(-r, b[m][ok]).correlation), 3) for m in M}
# 거리로 설명 안 되는 부분: 1층/2층 각각 거리 회귀 잔차의 표준편차 vs 시드 잡음
for m in ('C80', 'G'):
    v = 0.5*(b[m][ok] + L['a15s2'][m][ok]); noise = np.std(b[m][ok] - L['a15s2'][m][ok])/np.sqrt(2)
    resid = np.empty_like(v)
    for msk in (bal, ~bal):
        cf = np.polyfit(np.log(r[msk]), v[msk], 1); resid[msk] = v[msk] - np.polyval(cf, np.log(r[msk]))
    res[f'{m}_beyond_distance'] = {'resid_std': round(float(resid.std()), 2), 'seed_noise_std_of_mean': round(float(noise/np.sqrt(2)), 2)}
# 구역 비교: 발코니 아래(1층, 점군상 위에 발코니 슬래브) vs 1층 나머지 vs 2층, 같은 거리대(22~28m)
G_ = np.load('out/grid.npz'); R_ = G_['occ_raw']; lo = G_['lo']; H = float(G_['H'])
ear = b['ear'][ok]
under = np.zeros(len(seats), bool)
for i in np.nonzero(~bal)[0]:
    q = np.floor((ear[i] - lo)/H).astype(int); col = R_[q[0], q[1]+1:q[1]+int(4/H), q[2]]
    under[i] = col.any()
res['n_under_balcony'] = int(under.sum())
band = (r > 22) & (r < 28)
grp = {'1층 발코니 아래': under & band, '1층 개방부': (~bal) & (~under) & band, '2층': bal & band}
res['same_distance_22_28m'] = {k: {'n': int(v.sum()), **{m: round(float(np.median(0.5*(b[m][ok][v] + L['a15s2'][m][ok][v]))), 3) for m in M}} for k, v in grp.items()}
side = np.abs(seats[:,2] - np.median(seats[:,2])) > 8
res['LF_side_vs_center_floor1'] = {'side': round(float(np.median(b['LF'][ok][side & ~bal])), 3), 'center': round(float(np.median(b['LF'][ok][~side & ~bal])), 3)}
json.dump(res, open('out/analysis_v1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
avg = {m: 0.5*(b[m][ok] + L['a15s2'][m][ok]) for m in M}
fig, ax = plt.subplots(1, 4, figsize=(26, 7.5))
ttl = {'C80': 'C80 dB (명료도, 음악)', 'D50': 'D50 (명료도, 대사)', 'G': 'G dB (음량감)', 'LF': 'LF (공간감, 측면 반사)'}
for a, m in zip(ax, M):
    v = avg[m]; vmin, vmax = np.percentile(v, [2, 98])
    sc = a.scatter(seats[~bal,0], seats[~bal,2], c=v[~bal], s=10, cmap='viridis', vmin=vmin, vmax=vmax)
    a.scatter(seats[bal,0], seats[bal,2] + 30, c=v[bal], s=10, marker='^', cmap='viridis', vmin=vmin, vmax=vmax)
    a.scatter(seats[under,0], seats[under,2], facecolors='none', edgecolors='r', s=14, lw=0.4)
    plt.colorbar(sc, ax=a); a.set_title(ttl[m] + ' — 1층(●, 빨간 테두리=발코니 아래) / 2층(▲, 위로 30m)', fontname='Malgun Gothic', fontsize=9)
    a.set_aspect('equal'); a.axvline(-20.5, c='k', lw=1); a.text(-20, -16, '무대 →', fontname='Malgun Gothic')
plt.tight_layout(); plt.savefig('out/seat_map_v1.png', dpi=65); print('saved')
