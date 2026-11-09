"""v2 형상 기반 음향: v1 레이 추적 + 대역 3개(250/1k/4k) + 다중 음원·지향성(확성 PA 모드) + 시드 반복
모드
 - natural : 무대 앞 중앙 배우 1명(무지향), 무확성 생음
 - pa      : 프로시니엄 좌·우 메인 + 상단 센터 스피커(가정 배치), 객석 중심 조준, 에너지 지향 패턴 ((1+cos)/2)^q
흡음률·산란·지향성·스피커 위치는 모두 가정값 → 결과는 좌석 간 상대 비교용. 실측 임펄스응답 검증 없음.
산출: out/v2/<mode>_<band>_s<seed>.npz (좌석별 반사 에너지 히스토그램 1ms x 200, 측면 성분, 직접음, 도착 시각, T_fit)
"""
import sys, io, os, json, time, argparse
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ap = argparse.ArgumentParser()
ap.add_argument('--modes', nargs='+', default=['natural', 'pa'])
ap.add_argument('--bands', nargs='+', type=int, default=[250, 1000, 4000])
ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4])
ap.add_argument('--rays', type=int, default=300000)
ap.add_argument('--batch', type=int, default=50000)
ap.add_argument('--R', type=float, default=1.0)
ap.add_argument('--outdir', default='out/v2')
ap.add_argument('--aud', type=float, default=None, help='모든 대역의 객석 흡음률 덮어쓰기(민감도 시험용)')
ap.add_argument('--wall', type=float, default=None, help='모든 대역의 벽·천장 흡음률 덮어쓰기(민감도 시험용)')
a = ap.parse_args()

BAND = {  # 가정값 (문헌 대푯값 수준)
    250:  dict(aud=0.72, wall=0.20, scatter=0.2, m_air=0.00025, q=1),
    1000: dict(aud=0.90, wall=0.15, scatter=0.3, m_air=0.0011,  q=3),
    4000: dict(aud=0.85, wall=0.12, scatter=0.5, m_air=0.0058,  q=5)}
for _b in BAND.values():
    if a.aud is not None: _b['aud'] = a.aud
    if a.wall is not None: _b['wall'] = a.wall
C, DS, TMAX = 343.0, 0.1, 0.200
NB = int(TMAX * 1000)

G_ = np.load('out/grid.npz')
occ, nrm, aud, lo, H = G_['occ'], G_['nrm'], G_['aud'], G_['lo'], float(G_['H'])
occ_raw_f = G_['occ_raw'].ravel()
shape = np.array(occ.shape); s12 = shape[1] * shape[2]
occ_f = occ.ravel(); aud_f = aud.ravel(); nrm_f = nrm.reshape(-1, 3)

seats = np.load('out/seat_metrics_v0.npz')['seats']; NS = len(seats)
ear = seats + np.array([0, 0.75, 0])
excluded = (seats[:, 1] > 5.0) & (seats[:, 1] < 6.6)   # 설계 모델-실측 불일치 22석 (v1 참고)
AIM = ear[~excluded].mean(0)
SRC = {'natural': [dict(pos=[-17.0, 0.4, -3.86], aim=None)],
       'pa': [dict(pos=[-19.8, 6.0, -13.0], aim=AIM),    # 좌 메인
              dict(pos=[-19.8, 6.0, 5.6], aim=AIM),      # 우 메인
              dict(pos=[-20.0, 7.5, -3.7], aim=AIM)]}    # 센터


def cell(p):
    ijk = np.floor((p - lo) / H).astype(np.int64)
    ok = ((ijk >= 0) & (ijk < shape)).all(1)
    return np.where(ok, ijk[:, 0] * s12 + ijk[:, 1] * shape[2] + ijk[:, 2], -1)


for srcs in SRC.values():
    for s in srcs:
        assert not occ_f[cell(np.array([s['pos']]))[0]], f"음원 {s['pos']}이 점유 격자 안"

# 수음 조회표·유효 구 부피 (v1과 동일)
K = 16; rr = int(np.ceil(a.R / H)) + 1
off = np.stack(np.meshgrid(*[np.arange(-rr, rr + 1)] * 3, indexing='ij'), -1).reshape(-1, 3)
off = off[np.linalg.norm(off * H, axis=1) <= a.R + H]
ec = np.floor((ear - lo) / H).astype(np.int64)
allc = (ec[:, None, :] + off[None]).reshape(-1, 3); sid = np.repeat(np.arange(NS), len(off))
cid = allc[:, 0] * s12 + allc[:, 1] * shape[2] + allc[:, 2]
o = np.argsort(cid, kind='stable'); cid, sid = cid[o], sid[o]
ucell, start, cnt = np.unique(cid, return_index=True, return_counts=True)
rank = np.arange(len(cid)) - np.repeat(start, cnt); keep = rank < K
table = np.full((len(ucell), K), -1, np.int32)
table[np.repeat(np.arange(len(ucell)), cnt)[keep], rank[keep]] = sid[keep]
offc = off[np.linalg.norm(off * H, axis=1) <= a.R]; free = np.zeros(NS)
for i in range(NS):
    cc = ec[i] + offc; cc = cc[((cc >= 0) & (cc < shape)).all(1)]
    free[i] = (~occ_f[cc[:, 0] * s12 + cc[:, 1] * shape[2] + cc[:, 2]]).mean()
VS = 4 / 3 * np.pi * a.R ** 3 * np.maximum(free, 0.2)


def directivity(d, aim_dir, q):
    """에너지 지향 패턴. 구면 평균이 1이 되도록 (q+1)을 곱한다."""
    if aim_dir is None:
        return np.ones(len(d))
    return ((1 + d @ aim_dir) / 2) ** q * (q + 1)


def los_blocked(src):
    r = np.linalg.norm(ear - src, axis=1); bl = np.zeros(NS, bool)
    for i in range(NS):
        tt = np.arange(0.5, r[i] - 1.2, 0.1)
        pts = src + (ear[i] - src) * (tt / r[i])[:, None]
        cc = cell(pts); cc = cc[cc >= 0]
        bl[i] = occ_raw_f[cc].any()
    return r, bl


LOS = {mode: [los_blocked(np.array(s['pos'])) for s in srcs] for mode, srcs in SRC.items()}
os.makedirs(a.outdir, exist_ok=True)


def run(mode, band, seed):
    P = BAND[band]; rng = np.random.default_rng(seed * 1000 + band)

    def rand_dirs(n):
        v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1, keepdims=True)

    hist = np.zeros((NS, NB)); lat = np.zeros((NS, NB)); Ed = np.zeros(NS); tdir_E = np.zeros(NS); lost = 0.0
    srcs = SRC[mode]; nsrc = len(srcs); per = a.rays // nsrc
    for si, s in enumerate(srcs):
        src = np.array(s['pos'], float)
        aim = None if s['aim'] is None else (np.asarray(s['aim']) - src) / np.linalg.norm(np.asarray(s['aim']) - src)
        nb = max(1, int(np.ceil(per / a.batch)))
        for b in range(nb):
            n = min(a.batch, per - b * a.batch)
            p = np.repeat(src[None], n, 0); d = rand_dirs(n)
            E = directivity(d, aim, P['q']) / (per * nsrc)
            dist = np.zeros(n); nref = np.zeros(n, np.int16); alive = np.ones(n, bool)
            for it in range(int(1.6 * C * TMAX / DS)):
                ia = np.nonzero(alive)[0]
                if len(ia) == 0:
                    break
                pn = p[ia] + d[ia] * DS; c = cell(pn); out = c < 0
                if out.any():
                    lost += E[ia[out]].sum(); alive[ia[out]] = False
                ia, pn, c = ia[~out], pn[~out], c[~out]
                hit = occ_f[c]; ih = ia[hit]
                if len(ih):
                    nv = nrm_f[c[hit]].astype(np.float64); nl = np.linalg.norm(nv, axis=1)
                    nv = np.where(nl[:, None] > 0.3, nv / np.maximum(nl, 1e-9)[:, None], -d[ih])
                    dn = (d[ih] * nv).sum(1); nv = np.where((dn > 0)[:, None], -nv, nv); dn = -np.abs(dn)
                    spec = d[ih] - 2 * dn[:, None] * nv
                    diff = nv + rand_dirs(len(ih)); diff /= np.linalg.norm(diff, axis=1, keepdims=True)
                    d[ih] = np.where((rng.random(len(ih)) < P['scatter'])[:, None], diff, spec)
                    E[ih] *= 1 - np.where(aud_f[c[hit]], P['aud'], P['wall']); nref[ih] += 1
                im = ia[~hit]; p[im] = pn[~hit]; dist[im] += DS; E[im] *= np.exp(-P['m_air'] * DS)
                alive[ia] &= dist[ia] < C * TMAX
                cm = c[~hit]; sel = nref[im] > 0; im, cm = im[sel], cm[sel]
                if len(im):
                    k = np.minimum(np.searchsorted(ucell, cm), len(ucell) - 1); okc = ucell[k] == cm
                    im, k = im[okc], k[okc]
                    if len(im):
                        cand = table[k]
                        dd = np.linalg.norm(p[im][:, None, :] - ear[np.maximum(cand, 0)], axis=2)
                        ri, kk = np.nonzero((cand >= 0) & (dd < a.R))
                        if len(ri):
                            s_ = cand[ri, kk]; rid = im[ri]
                            tb = np.minimum((dist[rid] / C * 1000).astype(int), NB - 1)
                            w = E[rid] * DS / VS[s_]
                            np.add.at(hist, (s_, tb), w); np.add.at(lat, (s_, tb), w * d[rid, 2] ** 2)
        r, bl = LOS[mode][si]; dsr = (ear - src) / r[:, None]
        e_d = directivity(dsr, aim, P['q']) / nsrc * np.exp(-P['m_air'] * r) / (4 * np.pi * r ** 2) * np.where(bl, 0.1, 1.0)
        Ed += e_d; tdir_E += e_d * (r / C * 1000)
    tdir = tdir_E / np.maximum(Ed, 1e-30)   # 직접음 도착 시각(ms), 음원이 여럿이면 에너지 가중 평균
    tot = hist[~excluded].sum(0); tt = np.arange(NB); w = (tt >= 60) & (tt < 190) & (tot > 0)
    T_fit = -60 / np.polyfit(tt[w] / 1000, 10 * np.log10(tot[w]), 1)[0]
    np.savez_compressed(f'{a.outdir}/{mode}_{band}_s{seed}.npz', hist=hist.astype(np.float32), lat=lat.astype(np.float32),
                        Ed=Ed, tdir=tdir, T_fit=T_fit, lost=lost)
    return T_fit, lost


t0 = time.time(); log = {}
for mode in a.modes:
    for band in a.bands:
        for seed in a.seeds:
            T, lost = run(mode, band, seed)
            log[f'{mode}_{band}_s{seed}'] = {'T_fit': round(float(T), 3), 'lost': round(float(lost), 4)}
            print(f'{mode:8s} {band:5d}Hz seed{seed}  T_fit {T:.2f}s  escaped {lost:.3f}  ({time.time() - t0:.0f}s)', flush=True)
np.savez(f'{a.outdir}/meta.npz', seats=seats, ear=ear, excluded=excluded, free=free,
         los_natural=LOS['natural'][0][1], los_pa=np.array([l[1] for l in LOS['pa']]), r_natural=LOS['natural'][0][0])
json.dump({'bands': BAND,
           'sources': {k: [{'pos': s['pos'], 'aim': None if s['aim'] is None else np.round(s['aim'], 2).tolist()} for s in v]
                       for k, v in SRC.items()},
           'runs': log, 'rays': a.rays, 'R': a.R},
          open(f'{a.outdir}/meta.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
