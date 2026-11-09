"""평면도(높이맵)·단면도 + 좌석 위치 → out/plan_view.png, out/section.png"""
import numpy as np, pye57, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
e = pye57.E57('raw/woori_e57/x/woori_e57/WooriArtHall_1mm.e57'); d = e.read_scan_raw(0)
P = np.c_[d['cartesianX'], d['cartesianZ'], -np.asarray(d['cartesianY'])][::5]
c = np.load('out/seat_centroids_raw.npy')
h = 0.25; lo = P.min(0); ix = ((P[:,[0,2]] - lo[[0,2]])/h).astype(int); W = ix.max(0)+1
top = np.full(W, np.nan); low = np.full(W, np.nan)
np.fmax.at(top, (ix[:,0], ix[:,1]), P[:,1]); np.fmin.at(low, (ix[:,0], ix[:,1]), P[:,1])
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
ext = [lo[0], lo[0]+W[0]*h, lo[2], lo[2]+W[1]*h]
im = ax[0].imshow(top.T, origin='lower', extent=ext, cmap='viridis'); plt.colorbar(im, ax=ax[0], label='max y (ceiling) m')
ax[0].scatter(c[:,0], c[:,2], s=2, c='r'); ax[0].set_title('ceiling height + seats (red)'); ax[0].set_xlabel('x'); ax[0].set_ylabel('z')
im = ax[1].imshow(low.T, origin='lower', extent=ext, cmap='magma'); plt.colorbar(im, ax=ax[1], label='min y (floor) m')
ax[1].scatter(c[:,0], c[:,2], s=2, c='c'); ax[1].set_title('floor height + seats'); 
for a in ax: a.grid(alpha=.3); a.set_xticks(range(-48, 6, 4)); a.set_yticks(range(-28, 22, 4))
plt.tight_layout(); plt.savefig('out/plan_view.png', dpi=70)
sel = np.abs(P[:,2] - (-4)) < 0.3   # z≈-4 단면
fig, ax = plt.subplots(figsize=(18, 6)); ax.scatter(P[sel,0], P[sel,1], s=0.2, c='k')
cs = np.abs(c[:,2]-(-4)) < 0.6; ax.scatter(c[cs,0], c[cs,1], s=8, c='r'); ax.set_aspect('equal'); ax.grid(alpha=.3)
ax.set_xticks(range(-48, 6, 2)); ax.set_title('section z=-4 (x vs height), seats red'); plt.tight_layout(); plt.savefig('out/section.png', dpi=70)
print('ok')
