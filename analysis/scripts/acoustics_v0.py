"""v0 좌석별 음향 지표 — 체적·흡음 기반 잔향(Sabine/Eyring) + Barron 수정이론(1988)
가정(모두 out/acoustics_v0.json에 기록):
 - 객석(house)은 x < X_PROS. 무대 쪽 프로시니엄 개구부는 흡음률 ALPHA_OPEN으로 처리(무대 공간과 결합)
 - 체적: 점군 0.25m 격자 열(column)마다 (천장 max y - 바닥 min y). 발코니 슬래브 두께만큼 과대
 - 경계면적: 바닥(평면) + 천장(평면) + 벽(둘레 × 높이). 객석 면적 = 좌석 점유 격자 합(통로 0.5m 가산)
 - 흡음률: 문헌 대푯값(점유 객석 / 일반 마감) — 실측 아님
 - 음원: 무대 중앙, 지상 1.5m. 수음점: 좌석 쿠션 중심 + 0.75m
 결과는 '상대 비교'용. 절대값은 실측 임펄스응답이 없어 검증되지 않음.
"""
import sys, io, json, numpy as np, pye57
from scipy import ndimage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
BANDS = [125, 250, 500, 1000, 2000, 4000]
A_AUD  = [0.60, 0.74, 0.88, 0.96, 0.93, 0.85]   # 점유 객석(천 의자), Beranek 대푯값
A_WALL = [0.14, 0.10, 0.08, 0.07, 0.06, 0.06]   # 목재 패널·석고 등 일반 마감 대푯값
A_OPEN = [0.30, 0.35, 0.40, 0.40, 0.40, 0.40]   # 프로시니엄 개구부(무대 공간 결합) 등가 흡음
AIR_m  = [0.0, 0.0, 0.0006, 0.0011, 0.0024, 0.0072]  # 공기 흡수 m (Np/m 아닌 4m 계수용, 20°C 50%RH 근사)
X_PROS = -19.5; C = 343.0
e = pye57.E57('raw/woori_e57/x/woori_e57/WooriArtHall_1mm.e57'); d = e.read_scan_raw(0)
P = np.c_[d['cartesianX'], d['cartesianZ'], -np.asarray(d['cartesianY'])]
H = P[P[:,0] < X_PROS]
h = 0.25; lo = H.min(0); ix = ((H[:,[0,2]] - lo[[0,2]])/h).astype(int); W = ix.max(0)+1
top = np.full(W, np.nan); low = np.full(W, np.nan); cnt = np.zeros(W)
np.fmax.at(top, (ix[:,0], ix[:,1]), H[:,1]); np.fmin.at(low, (ix[:,0], ix[:,1]), H[:,1]); np.add.at(cnt, (ix[:,0], ix[:,1]), 1)
ht = top - low
inside = (cnt > 20) & (ht > 8)           # 천장까지 닿는 열만 실내로
inside = ndimage.binary_closing(inside, iterations=2) & ~np.isnan(ht)
inside = ndimage.binary_fill_holes(inside)
lab, n = ndimage.label(inside); k = np.argmax(ndimage.sum(inside, lab, range(1, n+1))) + 1; inside = lab == k
htf = np.where(inside, np.nan_to_num(ht, nan=np.nanmedian(ht[inside])), 0)
V = float((htf * h*h).sum()); A_plan = float(inside.sum()*h*h)
perim = float((inside ^ ndimage.binary_erosion(inside)).sum()*h)
mean_h = float(np.nanmean(ht[inside]))
seats = np.load('out/seat_centroids_raw.npy')
seat_area = np.array([0])  # placeholder
# 좌석 필터: 쿠션 면적 정상치만
import trimesh
s = trimesh.load('raw/woori_glb/x/woori_glb/WooriArtHall.glb', force='scene')
for node in s.graph.nodes_geometry:
    T, g = s.graph[node]; G = s.geometry[g]
    if getattr(getattr(G.visual,'material',None),'name','') == 'PDM_TheaterSeat_01_blue_carpet':
        G = G.copy(); G.apply_transform(T); parts = G.split(only_watertight=False); break
areas = np.array([p.area for p in parts]); cents = np.array([p.centroid for p in parts])
ok = areas > 0.5*np.median(areas); seats = cents[ok]
si = ((seats[:,[0,2]] - lo[[0,2]])/h).astype(int)
aud = np.zeros(W, bool); aud[si[:,0].clip(0,W[0]-1), si[:,1].clip(0,W[1]-1)] = True
aud = ndimage.binary_dilation(aud, iterations=2)       # 좌석 1개 격자 → 열 간격·통로 0.5m 포함
S_aud = float((aud & inside).sum()*h*h)
# 프로시니엄 개구부: 객석 경계 x=X_PROS 근처 열의 폭 × 평균 높이(13m까지)
edge_cols = inside[-3:, :].any(0); W_open = float(edge_cols.sum()*h); H_open = 14.0
S_open = W_open*H_open
S_floor = A_plan; S_ceil = A_plan; S_wall = perim*mean_h - S_open
S_tot = S_floor + S_ceil + S_wall + S_open
S_other = S_tot - S_aud - S_open
out = {'assumptions': __doc__, 'V_house_m3': round(V,1), 'plan_area_m2': round(A_plan,1), 'perimeter_m': round(perim,1),
       'mean_height_m': round(mean_h,2), 'S_total_m2': round(S_tot,1), 'S_audience_m2': round(S_aud,1),
       'S_opening_m2': round(S_open,1), 'n_seats': int(len(seats)), 'bands': {}}
for b, aa, aw, ao, m in zip(BANDS, A_AUD, A_WALL, A_OPEN, AIR_m):
    A = S_aud*aa + S_other*aw + S_open*ao
    am = A/S_tot
    T_sab = 0.161*V/(A + 4*m*V)
    T_ey = 0.161*V/(-S_tot*np.log(1-am) + 4*m*V)
    out['bands'][b] = {'A_m2': round(A,1), 'alpha_mean': round(am,3), 'T_sabine': round(T_sab,2), 'T_eyring': round(T_ey,2)}
T_mid = np.mean([out['bands'][b]['T_eyring'] for b in (500, 1000)])
out['T_mid_eyring'] = round(float(T_mid),2)
# Barron 수정이론 (좌석별)
src = np.array([-14.0, -1.2+1.5, float(np.median(seats[:,2]))])
rcv = seats + np.array([0, 0.75, 0])
r = np.linalg.norm(rcv - src, axis=1); T = T_mid
k = 31200*T/V; dcy = np.exp(-0.04*r/T)
d_ = 100/r**2; e80 = k*dcy*(1-np.exp(-1.11/T)); l80 = k*dcy*np.exp(-1.11/T); e50 = k*dcy*(1-np.exp(-0.69/T))
C80 = 10*np.log10((d_+e80)/l80); D50 = (d_+e50)/(d_+e80+l80); G = 10*np.log10(d_+e80+l80)
delay_ms = r/C*1000
np.savez('out/seat_metrics_v0.npz', seats=seats, rcv=rcv, src=src, r=r, C80=C80, D50=D50, G=G, delay_ms=delay_ms)
out['source'] = src.round(2).tolist()
for nm, a in (('r_m', r), ('C80_dB', C80), ('D50', D50), ('G_dB', G), ('delay_ms', delay_ms)):
    out[nm] = {'min': round(float(a.min()),2), 'median': round(float(np.median(a)),2), 'max': round(float(a.max()),2)}
json.dump(out, open('out/acoustics_v0.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'assumptions'}, ensure_ascii=False, indent=1))
