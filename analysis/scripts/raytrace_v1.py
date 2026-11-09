"""v1 형상 기반 음향: 복셀 격자 위 확률적 레이 추적(초기 0~200ms) + 통계적 잔향 꼬리
- 직접음: 해석해 1/(4πr²) + 가시선 판정(팽창 전 원 격자, 귀 주변 1.2m·음원 주변 0.5m 제외 — 관객 머리·바로 앞 등받이는 모델에 없음). 가려지면 -10dB(회절 근사)
- 수음 구 부피: 구 안의 비점유 격자 비율로 보정(바닥·좌석과 겹친 부분 제외)
- 반사음: 레이(음원 전방향) — 경면 반사 + 산란계수 s 비율로 램버트 확산, 반사마다 (1-α)
- 수음: 좌석 귀 위치 반지름 R 구 안의 통과 길이 × 에너지 / 구 부피 (반사 1회 이상 레이만)
- 200ms 이후: 150~200ms 에너지 수준과 추적 결과로 맞춘 감쇠율로 지수 꼬리 적분
- 1kHz 단일 대역. 흡음률은 가정값(시나리오로 민감도 확인). 결과는 좌석 간 상대 비교용.
사용: python -I scripts/raytrace_v1.py --alpha-wall 0.15 --rays 300000 --seed 1 --tag a15s1
"""
import sys, io, json, time, argparse, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ap = argparse.ArgumentParser()
ap.add_argument('--alpha-wall', type=float, default=0.15); ap.add_argument('--alpha-aud', type=float, default=0.90)
ap.add_argument('--scatter', type=float, default=0.3); ap.add_argument('--rays', type=int, default=300000)
ap.add_argument('--batch', type=int, default=50000); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--R', type=float, default=1.0); ap.add_argument('--tag', default='base')
ap.add_argument('--src', type=float, nargs=3, default=[-17.0, 0.4, -3.86])
a = ap.parse_args()
C, DS, TMAX, M_AIR = 343.0, 0.1, 0.200, 0.001
NB = int(TMAX*1000)
G_ = np.load('out/grid.npz'); occ, nrm, aud, lo, H = G_['occ'], G_['nrm'], G_['aud'], G_['lo'], float(G_['H']); occ_raw_f = G_['occ_raw'].ravel()
shape = np.array(occ.shape); s12 = shape[1]*shape[2]
occ_f = occ.ravel(); aud_f = aud.ravel(); nrm_f = nrm.reshape(-1, 3)
seats = np.load('out/seat_metrics_v0.npz')['seats']; NS = len(seats)
# 귀 높이 = 쿠션 +0.75m. 단, 쿠션 높이 5~6.6m의 22석은 점군상 바로 위 0.15m에 발코니 밑면이 있어
# 사람이 앉을 수 없는 위치(설계 모델과 실측 불일치) → 'excluded'로 표시하고 지도·통계에서 뺀다
ear = seats + np.array([0, 0.75, 0]); excluded = (seats[:,1] > 5.0) & (seats[:,1] < 6.6); ear_capped = excluded
src = np.array(a.src)
def cell(p):
    ijk = np.floor((p - lo)/H).astype(np.int64)
    ok = ((ijk >= 0) & (ijk < shape)).all(1)
    return np.where(ok, ijk[:,0]*s12 + ijk[:,1]*shape[2] + ijk[:,2], -1)
assert not occ_f[cell(src[None])[0]], '음원이 점유 격자 안에 있음'
# 수음 조회표: 격자 칸 → 반경 R 안 좌석들(최대 K)
K = 16; rr = int(np.ceil(a.R/H)) + 1
off = np.stack(np.meshgrid(*[np.arange(-rr, rr+1)]*3, indexing='ij'), -1).reshape(-1, 3)
off = off[np.linalg.norm(off*H, axis=1) <= a.R + H]
ec = np.floor((ear - lo)/H).astype(np.int64)
allc = (ec[:,None,:] + off[None]).reshape(-1, 3); sid = np.repeat(np.arange(NS), len(off))
cid = allc[:,0]*s12 + allc[:,1]*shape[2] + allc[:,2]
o = np.argsort(cid, kind='stable'); cid, sid = cid[o], sid[o]
ucell, start, cnt = np.unique(cid, return_index=True, return_counts=True)
rank = np.arange(len(cid)) - np.repeat(start, cnt)
keep = rank < K
table = np.full((len(ucell), K), -1, np.int32)
table[np.repeat(np.arange(len(ucell)), cnt)[keep], rank[keep]] = sid[keep]
VS0 = 4/3*np.pi*a.R**3
# 좌석별 유효 구 부피 = VS0 × (구 안 비점유 격자 비율)
free = np.zeros(NS)
offc = off[np.linalg.norm(off*H, axis=1) <= a.R]
for i in range(NS):
    cc = ec[i] + offc; cc = cc[((cc >= 0) & (cc < shape)).all(1)]
    free[i] = (~occ_f[cc[:,0]*s12 + cc[:,1]*shape[2] + cc[:,2]]).mean()
VS = VS0*np.maximum(free, 0.2)
hist = np.zeros((NS, NB)); lat = np.zeros((NS, NB))
rng = np.random.default_rng(a.seed)
def rand_dirs(n):
    v = rng.normal(size=(n, 3)); return v/np.linalg.norm(v, axis=1, keepdims=True)
t0 = time.time(); nbatch = a.rays // a.batch; lost = 0.0
for b in range(nbatch):
    n = a.batch; p = np.repeat(src[None], n, 0); d = rand_dirs(n)
    E = np.full(n, 1.0/a.rays); dist = np.zeros(n); nref = np.zeros(n, np.int16); alive = np.ones(n, bool)
    for it in range(int(1.6*C*TMAX/DS)):
        ia = np.nonzero(alive)[0]
        if len(ia) == 0: break
        pn = p[ia] + d[ia]*DS; c = cell(pn)
        out = c < 0
        if out.any(): lost += E[ia[out]].sum(); alive[ia[out]] = False
        ia, pn, c = ia[~out], pn[~out], c[~out]
        hit = occ_f[c]
        # 반사
        ih = ia[hit]
        if len(ih):
            nv = nrm_f[c[hit]].astype(np.float64); nl = np.linalg.norm(nv, axis=1)
            nv = np.where(nl[:,None] > 0.3, nv/np.maximum(nl,1e-9)[:,None], -d[ih])
            dn = (d[ih]*nv).sum(1); nv = np.where((dn > 0)[:,None], -nv, nv); dn = -np.abs(dn)
            spec = d[ih] - 2*dn[:,None]*nv
            diff = nv + rand_dirs(len(ih)); diff /= np.linalg.norm(diff, axis=1, keepdims=True)
            usediff = rng.random(len(ih)) < a.scatter
            d[ih] = np.where(usediff[:,None], diff, spec)
            al = np.where(aud_f[c[hit]], a.alpha_aud, a.alpha_wall)
            E[ih] *= (1 - al); nref[ih] += 1
        # 전진
        im = ia[~hit]; p[im] = pn[~hit]; dist[im] += DS; E[im] *= np.exp(-M_AIR*DS)
        alive[ia] &= dist[ia] < C*TMAX
        # 수음 (반사 1회 이상, 전진한 레이)
        cm = c[~hit]; sel = nref[im] > 0
        im, cm = im[sel], cm[sel]
        if len(im):
            k = np.searchsorted(ucell, cm); k = np.minimum(k, len(ucell)-1); okc = ucell[k] == cm
            im, k = im[okc], k[okc]
            if len(im):
                cand = table[k]; valid = cand >= 0
                dd = np.linalg.norm(p[im][:,None,:] - ear[np.maximum(cand,0)], axis=2)
                inside = valid & (dd < a.R)
                ri, kk = np.nonzero(inside)
                if len(ri):
                    s_ = cand[ri, kk]; rid = im[ri]
                    tb = np.minimum((dist[rid]/C*1000).astype(int), NB-1)
                    w = E[rid]*DS/VS[s_]
                    np.add.at(hist, (s_, tb), w); np.add.at(lat, (s_, tb), w*d[rid,2]**2)
    print(f' batch {b+1}/{nbatch}  {time.time()-t0:.0f}s  alive-left {alive.sum()}', flush=True)
# 직접음 + 가시선
r = np.linalg.norm(ear - src, axis=1); blocked = np.zeros(NS, bool)
for i in range(NS):
    L = r[i]; tt = np.arange(0.5, L-1.2, 0.1)
    pts = src + (ear[i]-src)*(tt/L)[:,None]; cc = cell(pts); cc = cc[cc >= 0]
    blocked[i] = occ_raw_f[cc].any()
Ed = np.exp(-M_AIR*r)/(4*np.pi*r**2) * np.where(blocked, 0.1, 1.0)
tdir = (r/C*1000).astype(int)
# 감쇠율: 전 좌석 합산 반사 에너지 60~190ms 로그 기울기
tot = hist.sum(0); tt = np.arange(NB); w = (tt >= 60) & (tt < 190) & (tot > 0)
slope = np.polyfit(tt[w]/1000, 10*np.log10(tot[w]), 1)[0]; T_fit = -60/slope
e_w = hist[:, 150:200].sum(1); tail = (e_w/0.05)*(T_fit/13.82)
E_tot = Ed + hist.sum(1) + tail
early80 = Ed + np.array([hist[i, :max(0, min(NB, tdir[i]+80))].sum() for i in range(NS)])
early50 = Ed + np.array([hist[i, :max(0, min(NB, tdir[i]+50))].sum() for i in range(NS)])
late80 = np.maximum(E_tot - early80, 1e-3*E_tot)
C80 = 10*np.log10(early80/late80); D50 = early50/E_tot; G = 10*np.log10(E_tot/(1/(4*np.pi*100)))
lat5_80 = np.array([lat[i, min(NB-1, tdir[i]+5):min(NB, tdir[i]+80)].sum() for i in range(NS)])
LF = lat5_80/early80
ref_hits = (hist > 0).sum(1)
np.savez(f'out/rt_v1_{a.tag}.npz', ear=ear, excluded=excluded, hist=hist, lat=lat, Ed=Ed, blocked=blocked, r=r, C80=C80, D50=D50, G=G, LF=LF, T_fit=T_fit, seats=seats)
summ = {'args': vars(a), 'excluded_seats': int(excluded.sum()), 'sphere_free_frac': [round(float(x),3) for x in (free.min(), np.median(free), free.max())], 'T_fit_s': round(float(T_fit),3), 'lost_energy_frac_escaped': round(float(lost),4),
        'blocked_direct': int(blocked[~excluded].sum()), 'seats_zero_reflections': int((ref_hits == 0).sum()),
        'runtime_s': round(time.time()-t0)}
for nm, v in (('C80', C80[~excluded]), ('D50', D50[~excluded]), ('G', G[~excluded]), ('LF', LF[~excluded])):
    summ[nm] = [round(float(x),3) for x in (v.min(), np.percentile(v,10), np.median(v), np.percentile(v,90), v.max())]
json.dump(summ, open(f'out/rt_v1_{a.tag}.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1))
