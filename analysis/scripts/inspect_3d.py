"""3D 원본 점검: 형식·단위·크기·형상 구성 요약 (out/inspect_*.txt)"""
import sys, io, json, collections, numpy as np, trimesh, pye57
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
RAW = 'raw'
def glb(path):
    s = trimesh.load(path, force='scene')
    print('[GLB]', path)
    print(' geometries', len(s.geometry), ' nodes', len(s.graph.nodes_geometry))
    b = s.bounds; print(' bounds', b.round(2).tolist(), ' extents', (b[1]-b[0]).round(2).tolist())
    tri = sum(len(g.faces) for g in s.geometry.values() if hasattr(g,'faces'))
    print(' triangles', tri)
    names = collections.Counter()
    for n in s.graph.nodes_geometry:
        names[n.split('_')[0].split('.')[0][:30]] += 1
    print(' top node-name prefixes', names.most_common(25))
    mats = collections.Counter()
    for g in s.geometry.values():
        m = getattr(getattr(g,'visual',None),'material',None)
        mats[getattr(m,'name',None)] += 1
    print(' materials', len(mats), mats.most_common(25))
    return s
def e57(path):
    e = pye57.E57(path)
    print('[E57]', path, ' scans', e.scan_count)
    for i in range(e.scan_count):
        h = e.get_header(i)
        print(f' scan{i} points', h.point_count, ' fields', h.point_fields[:12])
        try:
            print('  bounds x', h.xMinimum, h.xMaximum, ' y', h.yMinimum, h.yMaximum, ' z', h.zMinimum, h.zMaximum)
        except Exception as ex: print('  bounds n/a', ex)
if __name__ == '__main__':
    glb(f'{RAW}/woori_glb/x/woori_glb/WooriArtHall.glb')
    e57(f'{RAW}/woori_e57/x/woori_e57/WooriArtHall_1mm.e57')
