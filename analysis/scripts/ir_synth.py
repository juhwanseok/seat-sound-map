"""좌석별 합성 임펄스응답(IR)과 미리듣기 음원
- 레이 추적 에너지 히스토그램(1ms) + 지수 잔향 꼬리를 대역별 포락선으로 쓰고, 대역 필터를 거친 잡음에 곱한다
  (250 → 500Hz 이하, 1k → 500~2000Hz, 4k → 2000Hz 이상을 담당). 직접음은 도착 시각의 임펄스.
- 건조음(dry)은 저작권 문제가 없도록 직접 합성: 박수 4회 + 피아노풍 아르페지오
- 좌석 간 음량 차이가 들리도록 전 좌석 공통 배율로 정규화한다(좌석마다 맞추지 않음)
산출: out/audio/*.wav, out/audio/ir_check.json
"""
import sys, io, os, json, wave
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
FS = 32000; BANDS = (250, 1000, 4000)
M = np.load('out/v2/metrics.npz'); seats = M['seats']; T_room = dict(zip(BANDS, M['T_room']))
rng = np.random.default_rng(7)
SOS = {250: butter(8, 500, 'lowpass', fs=FS, output='sos'),
       1000: butter(8, [500, 2000], 'bandpass', fs=FS, output='sos'),
       4000: butter(8, 2000, 'highpass', fs=FS, output='sos')}


def synth_ir(mode, i, dur=2.5):
    n = int(dur * FS); ir = np.zeros(n); t_ms = np.arange(n) / FS * 1000; check = {}
    for b in BANDS:
        h = M[f'{mode}_{b}_hist'][i].astype(float); T = T_room[b]
        rho200 = h[150:200].sum() / 50.0                       # 200ms 시점 ms당 에너지
        env_ms = np.where(t_ms < 200, h[np.minimum(t_ms.astype(int), 199)],
                          rho200 * np.exp(-13.82 * (t_ms - 200) / 1000 / T))
        env = env_ms / (FS / 1000)                              # 표본당 에너지
        nz = sosfilt(SOS[b], rng.normal(size=n)); nz /= nz.std()
        part = nz * np.sqrt(env); ir += part
        check[b] = round(float((part ** 2).sum() / (h.sum() + rho200 * 1000 * T / 13.82)), 3)
        # 직접음: 대역마다 필터를 거친 임펄스를 대역 내 에너지가 Ed_b가 되도록 맞춘다
        # (광대역 임펄스 하나로 넣으면 대역 안 에너지가 약 -10dB로 줄어 C80이 낮게 나온다 — 2026-11-09 점검에서 발견)
        imp = np.zeros(n); imp[int(round(M[f'{mode}_{b}_tdir'][i] / 1000 * FS))] = 1.0
        imp = sosfilt(SOS[b], imp); imp /= np.sqrt((imp ** 2).sum())
        ir += imp * np.sqrt(M[f'{mode}_{b}_Ed'][i])
    return ir, check


def dry_signal():
    t = np.arange(int(4.0 * FS)) / FS; x = np.zeros_like(t)
    for c in (0.10, 0.45, 0.80, 1.15):                          # 박수
        k = int(c * FS); L = int(0.04 * FS)
        burst = rng.normal(size=L) * np.exp(-np.arange(L) / (0.006 * FS))
        x[k:k + L] += 0.8 * sosfilt(butter(2, [800, 4000], 'bandpass', fs=FS, output='sos'), burst)
    for j, f in enumerate((261.6, 329.6, 392.0, 523.3, 392.0, 329.6, 261.6)):   # 피아노풍
        k = int((1.7 + 0.28 * j) * FS); tt = np.arange(int(0.9 * FS)) / FS
        note = sum(np.sin(2 * np.pi * f * h * tt) / h ** 1.3 * np.exp(-tt * (3 + h)) for h in range(1, 8))
        x[k:k + len(note)] += 0.25 * note[:len(x) - k]
    return x / np.abs(x).max() * 0.5


def write_wav(path, y):
    y = np.clip(y, -1, 1); d = (y * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS); w.writeframes(d.tobytes())


os.makedirs('out/audio', exist_ok=True)
ok = ~M['excluded']; bal = seats[:, 1] > 6.6
idx = np.arange(len(seats))
def nearest(x, z, mask):
    c = idx[mask]; return int(c[np.argmin((seats[c, 0] - x) ** 2 + (seats[c, 2] - z) ** 2)])
PICK = {'1층_앞_중앙': nearest(-22.5, -3.9, ok & ~bal), '1층_발코니아래_뒤': nearest(-41.5, -3.9, ok & ~bal),
        '2층_뒤_중앙': nearest(-44.5, -3.9, ok & bal), '1층_중간_측면': nearest(-31, 7, ok & ~bal)}
x = dry_signal(); write_wav('out/audio/00_dry.wav', x)
outs = {}; checks = {}
for mode in ('natural', 'pa'):
    for name, i in PICK.items():
        ir, ck = synth_ir(mode, i); outs[(mode, name)] = fftconvolve(x, ir)[:len(x) + FS]; checks[f'{mode}_{name}'] = {'seat': i, 'energy_ratio_synth/target': ck}
peak = max(np.abs(y).max() for y in outs.values())
for (mode, name), y in outs.items():
    write_wav(f'out/audio/{mode}_{name}.wav', y / peak * 0.9)
json.dump({'picked': {k: {'seat': v, 'xyz': seats[v].round(2).tolist()} for k, v in PICK.items()}, 'checks': checks},
          open('out/audio/ir_check.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(checks, ensure_ascii=False, indent=1))
