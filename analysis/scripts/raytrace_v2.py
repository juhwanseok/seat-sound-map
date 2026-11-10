"""v2 형상 기반 음향: v1 레이 추적 + 대역 3개(250/1k/4k) + 다중 음원·지향성(확성 PA 모드) + 시드 반복
모드
 - natural : 무대 앞 중앙 배우 1명(무지향), 무확성 생음
 - pa      : 기관 공개 음향장비 목록 구성(센터 클러스터 + 프런트필 6 + 사이드필 4 + 발코니 아래 4), 위치·출력 비율·지연은 가정
 - pa_simple: r01~r02의 단순 가정(좌·우 메인 + 센터)
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
# 확성(pa): 기관 대관 자료 음향장비 목록(f13.xls)의 구성을 따른다 — CENTER 12(한 덩어리 센터 클러스터로 모델링),
# FRONT FILL 6, SIDE FILL 4, UNDER BALCONY 4. 목록에 설치 위치가 없으므로 위치·조준·출력 비율·지연은 가정값.
# 우퍼(저역 전용), 서라운드·천장(효과음용)은 대사·노래 재생에 쓰지 않는다고 보고 뺐다.
# 지연: 보조 스피커는 센터와의 거리만큼 늦추고(파면 정렬), 발코니 아래는 +10ms 더 늦춰 센터 소리가 먼저 들리게 한다(선행음 효과).
_fl = ~excluded & (seats[:, 1] < 5.0)
_L, _Rt = ear[_fl & (seats[:, 2] < -9)].mean(0), ear[_fl & (seats[:, 2] > 3)].mean(0)
_CTR = np.array([-20.0, 7.5, -3.7])
def _dly(pos, extra=0.0): return float(np.linalg.norm(np.array(pos) - _CTR) / 343.0 * 1000 + extra)
SRC = {'natural': [dict(pos=[-17.0, 0.4, -3.86], aim=None)],
       'pa_simple': [dict(pos=[-19.8, 6.0, -13.0], aim=AIM),    # r01~r02 단순 가정: 좌·우 메인 + 센터
                     dict(pos=[-19.8, 6.0, 5.6], aim=AIM),
                     dict(pos=[-20.0, 7.5, -3.7], aim=AIM)],
       'pa': ([dict(pos=_CTR.tolist(), aim=AIM, share=0.70, delay=0.0, name='center')]
              + [dict(pos=[-20.6, -0.9, z], aim=[-23.0, -1.2, z], share=0.01, q=2, delay=_dly([-20.6, -0.9, z]), name='front_fill')
                 for z in (-12.0, -8.0, -4.5, -1.0, 2.5, 5.5)]
              + [dict(pos=[-19.8, y, z], aim=(_L if z < 0 else _Rt), share=0.03, q=2, delay=_dly([-19.8, y, z]), name='side_fill')
                 for z in (-13.0, 5.6) for y in (2.5, 4.5)]
              + [dict(pos=[-36.8, 5.2, z], aim=[-40.0, 3.0, z], share=0.03, q=3, delay=_dly([-36.8, 5.2, z], 10.0), name='under_balcony')
                 for z in (-12.0, -6.0, 0.0, 6.0)])}


def _variant(ub=0.03, ceil=0.0, keep_center=False):
    """장비 목록 구성(pa)의 변형 — 스피커 배치 가정 민감도용 (r03 할 일 #6).
    ub: 발코니 아래 1대당 출력 비율(0이면 끔). ceil: 2층 천장 보조 스피커 1대당 출력 비율(0이면 없음, 위치는 가정).
    keep_center=False면 센터가 나머지를 맡아 합계 1을 유지한다. True면 센터를 그대로 둔다(스피커를 끈 경우)."""
    out = [dict(s) for s in SRC['pa'] if s['name'] != 'under_balcony' or ub > 0]
    for s in out:
        if s['name'] == 'under_balcony':
            s['share'] = ub
    if ceil > 0:   # 2층 객석 위 천장(높이 약 14m) 아래 0.8m, 2층 객석 가운데 열을 향함
        out += [dict(pos=[-39.0, 13.2, z], aim=[-42.0, 10.5, z], share=ceil, q=3, delay=_dly([-39.0, 13.2, z], 10.0), name='ceiling_2f')
                for z in (-10.0, -5.0, 0.0, 5.0)]
    if not keep_center:
        out[0]['share'] = round(1.0 - sum(s['share'] for s in out[1:]), 4)
    return out


SRC.update(pa_noub=_variant(ub=0, keep_center=True), pa_ub1=_variant(ub=0.01), pa_ub5=_variant(ub=0.05), pa_ceil=_variant(ceil=0.03))


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


LOS = {mode: [los_blocked(np.array(s['pos'], float)) for s in SRC[mode]] for mode in sorted(set(a.modes) | {'natural'})}
os.makedirs(a.outdir, exist_ok=True)


def run(mode, band, seed):
    P = BAND[band]; rng = np.random.default_rng(seed * 1000 + band)

    def rand_dirs(n):
        v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1, keepdims=True)

    hist = np.zeros((NS, NB)); lat = np.zeros((NS, NB)); Ed = np.zeros(NS); tdir_E = np.zeros(NS); lost = 0.0
    srcs = SRC[mode]; nsrc = len(srcs)
    for si, s in enumerate(srcs):
        src = np.array(s['pos'], float)
        aim = None if s['aim'] is None else (np.asarray(s['aim']) - src) / np.linalg.norm(np.asarray(s['aim']) - src)
        share = s.get('share', 1.0 / nsrc); q = s.get('q', P['q']); delay = s.get('delay', 0.0)   # 출력 비율·지향성·지연(ms)
        per = max(8000, int(a.rays * share)) if 'share' in s else a.rays // nsrc
        nb = max(1, int(np.ceil(per / a.batch)))
        for b in range(nb):
            n = min(a.batch, per - b * a.batch)
            p = np.repeat(src[None], n, 0); d = rand_dirs(n)
            E = directivity(d, aim, q) * share / per
            dist = np.full(n, C * delay / 1000); nref = np.zeros(n, np.int16); alive = dist < C * TMAX
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
        e_d = directivity(dsr, aim, q) * share * np.exp(-P['m_air'] * r) / (4 * np.pi * r ** 2) * np.where(bl, 0.1, 1.0)
        Ed += e_d; tdir_E += e_d * (r / C * 1000 + delay)
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
         los_natural=LOS['natural'][0][1], r_natural=LOS['natural'][0][0],
         **({'los_pa': np.array([l[1] for l in LOS['pa']])} if 'pa' in LOS else {}))
json.dump({'bands': BAND,
           'sources': {k: [{'pos': [round(float(x), 2) for x in s['pos']], 'aim': None if s['aim'] is None else np.round(s['aim'], 2).tolist(),
                            **{f: (round(float(s[f]), 2) if isinstance(s[f], (int, float)) else s[f]) for f in ('share', 'q', 'delay', 'name') if f in s}}
                           for s in v] for k, v in SRC.items()},
           'runs': log, 'rays': a.rays, 'R': a.R},
          open(f'{a.outdir}/meta.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
