"""재질별 면적(GLB), 점군 범위(E57 샘플), DAE 헤더·형상 수"""
import sys, io, re, collections, numpy as np, trimesh, pye57
from xml.etree.ElementTree import iterparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
s = trimesh.load('raw/woori_glb/x/woori_glb/WooriArtHall.glb', force='scene')
rows = []
for node in s.graph.nodes_geometry:
    T, gname = s.graph[node]
    g = s.geometry[gname]
    gg = g.copy(); gg.apply_transform(T)
    m = getattr(getattr(g,'visual',None),'material',None)
    b = gg.bounds
    rows.append((gg.area, getattr(m,'name',gname), len(gg.faces), b[0].round(1).tolist(), b[1].round(1).tolist(), gg.is_watertight))
print('[GLB 재질별 면적, m^2 가정]')
for a,n,f,b0,b1,w in sorted(rows, reverse=True):
    print(f' {a:10.1f}  {n[:34]:34s} faces={f:8d}  {b0} ~ {b1}  wt={w}')
print(' total area', round(sum(r[0] for r in rows),1))
e = pye57.E57('raw/woori_e57/x/woori_e57/WooriArtHall_1mm.e57')
d = e.read_scan_raw(0)
x,y,z = (np.asarray(d[k]) for k in ('cartesianX','cartesianY','cartesianZ'))
print('[E57] n', len(x))
for nm,a in (('x',x),('y',y),('z',z)):
    print(f' {nm}: min {a.min():.2f} p1 {np.percentile(a,1):.2f} p99 {np.percentile(a,99):.2f} max {a.max():.2f}')
p='raw/kspodome_dae/x/kspodome_dae/sketchfab_설계용)/source/temp_export/temp_export.dae'
cnt=collections.Counter(); unit=None; up=None; mats=collections.Counter(); tri=0
for ev,el in iterparse(p, events=('end',)):
    t=el.tag.split('}')[-1]
    if t=='unit': unit=el.attrib
    elif t=='up_axis': up=el.text
    elif t in ('triangles','polylist','polygons'):
        cnt[t]+=1; tri+=int(el.attrib.get('count',0)); mats[el.attrib.get('material','')]+=1
    elif t in ('geometry','node','material','effect','image'): cnt[t]+=1
    if t in ('triangles','polylist','polygons','float_array','p','geometry','mesh','source'): el.clear()
print('[DAE]', 'unit', unit, 'up', up, dict(cnt), 'prims', tri)
print(' materials', len(mats), mats.most_common(30))
