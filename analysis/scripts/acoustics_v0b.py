"""v0b: 공식 수치 검증(층별 좌석 수·무대 끝~최후열 거리), 잔향 범위 민감도, 발코니 오버행(D/H), 좌석 지도"""
import sys, io, json, numpy as np, pye57
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
m = np.load('out/seat_metrics_v0.npz'); seats, rcv, src = m['seats'], m['rcv'], m['src']; V = json.load(open('out/acoustics_v0.json', encoding='utf-8'))['V_house_m3']
bal = seats[:,1] > 5.0
X_STAGE_EDGE = -20.5
d_edge = np.hypot(seats[:,0]-X_STAGE_EDGE, 0)  # 무대 끝선(x)까지 수평거리
res = {'n_total': int(len(seats)), 'n_floor1': int((~bal).sum()), 'n_balcony': int(bal.sum()),
       'official': {'total': 1184, 'floor1': 832, 'balcony': 352},
       'max_dist_stage_edge_to_balcony_m': round(float(d_edge[bal].max()),1), 'official_balcony_end_dist_m': 25}
# 잔향 민감도: T 1.0~2.0s에서 C80 순위 안정성
r = np.linalg.norm(rcv - src, axis=1)
def barron(T):
    k = 31200*T/V; dcy = np.exp(-0.04*r/T); d = 100/r**2
    e = k*dcy*(1-np.exp(-1.11/T)); l = k*dcy*np.exp(-1.11/T)
    return 10*np.log10((d+e)/l), 10*np.log10(d+e+l)
base = barron(1.4)[0]; res['rank_stability_C80_vs_T1.4'] = {}
for T in (1.0, 1.2, 1.6, 1.8, 2.0):
    res['rank_stability_C80_vs_T1.4'][T] = round(float(spearmanr(base, barron(T)[0]).correlation), 4)
C80_14, G_14 = barron(1.4)
res['C80_T1.4'] = [round(float(x),2) for x in (C80_14.min(), np.median(C80_14), C80_14.max())]
# 발코니 오버행: 1층 좌석 위에 발코니 슬래브(점군)가 있는지 → D(좌석~발코니 앞끝 수평거리), H(좌석 귀높이~슬래브 밑면)
e57 = pye57.E57('raw/woori_e57/x/woori_e57/WooriArtHall_1mm.e57'); dd = e57.read_scan_raw(0)
P = np.c_[dd['cartesianX'], dd['cartesianZ'], -np.asarray(dd['cartesianY'])][::3]
slab = P[(P[:,1] > 4.5) & (P[:,1] < 7.0) & (P[:,0] < -34)]       # 단면도상 발코니 밑면 y≈5~6.5, x<-35
h = 0.5; lo = slab[:,[0,2]].min(0); ix = ((slab[:,[0,2]]-lo)/h).astype(int); Wg = ix.max(0)+1
under = np.full(Wg, np.inf); np.fmin.at(under, (ix[:,0], ix[:,1]), slab[:,1])
front_x = np.full(Wg[1], np.nan)
for j in range(Wg[1]):
    col = np.where(np.isfinite(under[:,j]))[0]
    if len(col): front_x[j] = lo[0] + col.max()*h
D = np.zeros(len(seats)); Hh = np.full(len(seats), np.nan)
for i, (x, y, z) in enumerate(rcv):
    if bal[i]: continue
    a, b = int((x-lo[0])/h), int((z-lo[1])/h)
    if 0 <= a < Wg[0] and 0 <= b < Wg[1] and np.isfinite(under[a, b]):
        D[i] = front_x[b] - x; Hh[i] = under[a, b] - y
covered = D > 0
DH = np.where(covered, D/np.maximum(Hh, 0.5), 0)
res['under_balcony_seats'] = int(covered.sum())
res['D_over_H'] = {'max': round(float(DH.max()),2), 'n_DH_gt_1': int((DH > 1).sum()), 'n_DH_gt_2': int((DH > 2).sum())}
np.savez('out/seat_metrics_v0b.npz', seats=seats, bal=bal, C80=C80_14, G=G_14, DH=DH, D=D, H=Hh, r=r)
json.dump(res, open('out/acoustics_v0b.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
for a, val, ttl, cm in ((ax[0], C80_14, 'C80 dB (Barron, T=1.4s) — 높을수록 또렷', 'viridis'),
                        (ax[1], G_14, 'G dB (Barron, T=1.4s) — 높을수록 큼', 'magma'),
                        (ax[2], DH, '발코니 오버행 D/H (1층) — 1 초과 시 잔향 차단', 'Reds')):
    for msk, mk in ((~bal, 'o'), (bal, '^')):
        sc = a.scatter(seats[msk,0], seats[msk,2] + (0 if mk == 'o' else 30), c=val[msk], s=9, marker=mk, cmap=cm, vmin=val.min(), vmax=val.max())
    plt.colorbar(sc, ax=a); a.set_title(ttl, fontname='Malgun Gothic'); a.set_aspect('equal'); a.axvline(X_STAGE_EDGE, c='k', lw=1)
    a.text(-20, 30, '2층(위로 30m 이동 표시)', fontname='Malgun Gothic', fontsize=8); a.text(-20, -18, '무대 →', fontname='Malgun Gothic')
plt.tight_layout(); plt.savefig('out/seat_map_v0.png', dpi=70); print('saved')
