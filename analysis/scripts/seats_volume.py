"""우리금융아트홀: 좌석 위치 추출 + 실내 공기 체적 추정 → out/seats.npy, out/volume.json"""
import sys, io, json, numpy as np, trimesh, pye57
from scipy import ndimage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
s = trimesh.load('raw/woori_glb/x/woori_glb/WooriArtHall.glb', force='scene')
def mesh_of(matname):
    for node in s.graph.nodes_geometry:
        T, g = s.graph[node]; G = s.geometry[g]
        m = getattr(getattr(G,'visual',None),'material',None)
        if getattr(m,'name','') == matname:
            G = G.copy(); G.apply_transform(T); return G
# 좌석: 쿠션(blue_carpet) 연결요소 중심 = 좌석 1개로 가정
cush = mesh_of('PDM_TheaterSeat_01_blue_carpet')
parts = cush.split(only_watertight=False)
c = np.array([p.centroid for p in parts]); a = np.array([p.area for p in parts])
print('cushion parts', len(parts), ' area median', np.median(a).round(3), ' min/max', a.min().round(3), a.max().round(3))
np.save('out/seat_centroids_raw.npy', c)
# 실내 체적: 점군 0.5m 복셀 점유 → 바깥에서 flood fill → 갇힌 빈 칸
e = pye57.E57('raw/woori_e57/x/woori_e57/WooriArtHall_1mm.e57'); d = e.read_scan_raw(0)
P = np.c_[d['cartesianX'], d['cartesianZ'], -np.asarray(d['cartesianY'])]  # GLB 좌표(Y-up)로 변환
res = {}
for h in (0.5, 0.3):
    lo = P.min(0) - 2*h; idx = np.floor((P - lo)/h).astype(int); shape = idx.max(0) + 3
    occ = np.zeros(shape, bool); occ[tuple(idx.T)] = True
    occ = ndimage.binary_dilation(occ, iterations=1)  # 점군 구멍 메움
    outside = ~occ; lab, n = ndimage.label(outside)
    border = set(np.unique(np.r_[lab[0].ravel(), lab[-1].ravel(), lab[:,0].ravel(), lab[:,-1].ravel(), lab[:,:,0].ravel(), lab[:,:,-1].ravel()]))
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n+1))
    inner = [(sz*h**3, k) for k, sz in enumerate(sizes, 1) if k not in border]
    inner.sort(reverse=True)
    res[h] = [round(v,1) for v,_ in inner[:5]]
    print(f'voxel {h}m: grid {shape.tolist()}, enclosed air regions (m^3, top5)', res[h], ' leak-to-outside', len(border))
    if inner:
        k = inner[0][1]; reg = (lab == k); zz = np.argwhere(reg)
        bb = (zz.min(0)*h + lo).round(1).tolist(), (zz.max(0)*h + lo).round(1).tolist()
        print('   largest region bounds', bb)
        res[f'bounds_{h}'] = bb
json.dump({str(k): v for k, v in res.items()}, open('out/volume.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
