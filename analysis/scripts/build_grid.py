"""점군 → 0.2m 점유 격자 + 표면 법선 + 흡음 영역(객석) → out/grid.npz
좌표: GLB 프레임 (x, y=높이, z). E57 (x, y, z) → (x, z, -y)
"""
import sys, io, numpy as np, pye57
from scipy import ndimage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
H = 0.2
e = pye57.E57('raw/woori_e57/x/woori_e57/WooriArtHall_1mm.e57'); d = e.read_scan_raw(0)
P = np.c_[d['cartesianX'], d['cartesianZ'], -np.asarray(d['cartesianY'])].astype(np.float32)
lo = P.min(0) - 2*H; idx = np.floor((P - lo)/H).astype(np.int32); shape = idx.max(0) + 3
occ = np.zeros(shape, bool); occ[tuple(idx.T)] = True
raw_occ = occ.sum(); occ_raw = occ.copy()
occ = ndimage.binary_dilation(occ, iterations=1)          # 점군 틈(≤0.4m) 메움 → 벽 두께 0.4~0.6m
B = ndimage.gaussian_filter(occ.astype(np.float32), 1.2)
g = np.stack(np.gradient(B), -1); nrm = -g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9)
nrm[~occ] = 0
# 객석(흡음) 영역: 좌석 중심 xz 0.6m 이내 & 좌석 쿠션 높이 -0.6 ~ +1.0m
seats = np.load('out/seat_metrics_v0.npz')['seats']
aud = np.zeros(shape, bool)
si = np.floor((seats - lo)/H).astype(int)
r = 3
for dx in range(-r, r+1):
    for dz in range(-r, r+1):
        if dx*dx + dz*dz > r*r: continue
        for dy in range(-3, 6):
            a = si + np.array([dx, dy, dz]); a = a[(a >= 0).all(1) & (a < shape).all(1)]
            aud[tuple(a.T)] = True
aud &= occ
np.savez_compressed('out/grid.npz', occ=occ, occ_raw=occ_raw, nrm=nrm.astype(np.float16), aud=aud, lo=lo, H=H)
print('grid', shape.tolist(), 'raw occupied', int(raw_occ), 'dilated', int(occ.sum()), 'audience voxels', int(aud.sum()))
