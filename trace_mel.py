"""Zoom in on ONE step: how 400 raw audio numbers become 128 mel numbers."""
import numpy as np, mlx.core as mx
from mlx_whisper import audio as A

SR = A.SAMPLE_RATE

def bar(v, lo, hi, w=40):
    n = int(max(0, min(1, (v - lo) / (hi - lo + 1e-9))) * w)
    return "#" * n

def rule(t): print(f"\n{'='*70}\n{t}\n{'='*70}")

# ------------------------------------------------------------------
rule("A. What ONE window of 400 numbers looks like")
t = np.arange(400) / SR
tone = 0.5 * np.sin(2 * np.pi * 220 * t)     # 220 Hz = low A note
print("400 samples of a pure 220 Hz tone. First 20:")
print(np.array2string(tone[:20], precision=3, floatmode="fixed", max_line_width=95))
print("\nSame 20 drawn (each row = one sample, position = pressure):")
for v in tone[:20]:
    print("      " + " " * int((v + 0.6) * 30) + "*")
print("\nThat is the wave. 400 of these numbers = 25 ms of sound.")

# ------------------------------------------------------------------
rule("B. FFT: 400 wave numbers -> 201 'how much of each frequency'")
spec = np.abs(np.fft.rfft(tone * np.hanning(400)))
freqs = np.fft.rfftfreq(400, 1 / SR)
print(f"input  : 400 numbers (a wave over time)")
print(f"output : {len(spec)} numbers (a strength per frequency)")
print(f"\nfrequency step: {freqs[1]:.0f} Hz apart, from 0 Hz to {freqs[-1]:.0f} Hz")
print("\nThe 12 strongest of those 201:")
for i in np.argsort(-spec)[:12]:
    print(f"  {freqs[i]:>7.0f} Hz  {spec[i]:>8.2f}  {bar(spec[i], 0, spec.max())}")
print("\n-> One huge spike at ~220 Hz. FFT found the tone. Everything else ~0.")
print("   FFT does not 'know' about tones. It just asks, for each of 201")
print("   frequencies: how much of THIS frequency is in those 400 numbers?")

# ------------------------------------------------------------------
rule("C. Mel filters: squash 201 frequencies down to 128 buckets")
filters = np.array(A.mel_filters(128))
print(f"filter matrix shape: {filters.shape}   (128 buckets x 201 frequencies)")
print("\nEach bucket is a weighted average over a RANGE of frequencies.")
print("Which Hz range does each bucket cover?\n")
print(f"  {'bucket':>7}  {'covers (Hz)':>16}  width")
for b in [0, 1, 5, 20, 50, 80, 110, 127]:
    nz = np.nonzero(filters[b])[0]
    lo, hi = freqs[nz[0]], freqs[nz[-1]]
    print(f"  {b:>7}  {lo:>7.0f} - {hi:<6.0f}  {hi-lo:>6.0f} Hz wide")
print("\n-> Low buckets are NARROW (fine detail where your ear is sensitive).")
print("   High buckets are WIDE (coarse, where your ear can't tell 8000 from 8100).")
print("   That is the mel scale. Buckets are not equal — they match human hearing.")

# ------------------------------------------------------------------
rule("D. Watch the bucket move when the pitch changes")
for hz in [110, 220, 440, 1000, 3000]:
    x = 0.5 * np.sin(2 * np.pi * hz * (np.arange(480000) / SR))
    mel = np.array(A.log_mel_spectrogram(mx.array(x.astype(np.float32)), n_mels=128)
                   .astype(mx.float32))[100]
    top = int(np.argmax(mel))
    print(f"  {hz:>5} Hz tone  ->  loudest bucket = #{top:<4} "
          f"value {mel[top]:+.2f}   {bar(top, 0, 127, 45)}")
print("\n-> Higher pitch, higher bucket number. The 128 numbers are literally")
print("   'how much energy in this pitch range', ordered low pitch -> high pitch.")

# ------------------------------------------------------------------
rule("E. Now a real voice frame — all 128 numbers, drawn")
wav = A.load_audio("sample.aiff")
mel = np.array(A.log_mel_spectrogram(A.pad_or_trim(wav), n_mels=128).astype(mx.float32))
f = 115
row = mel[f]
lo, hi = row.min(), row.max()
print(f"frame {f} = the 25 ms at t={f*0.01:.2f}s of 'Machine learning turns sound into text.'")
print(f"128 numbers, bucket 0 (lowest pitch) at top:\n")
for b in range(0, 128, 2):
    hzs = np.nonzero(filters[b])[0]
    c = freqs[hzs[len(hzs)//2]]
    print(f"  #{b:>3} ~{c:>5.0f}Hz  {row[b]:>+5.2f}  {bar(row[b], lo, hi, 44)}")
print("\n-> The bumps are the voice. Big low bumps = vocal cords buzzing.")
print("   Smaller bumps higher up = the shape of the mouth making a vowel.")
print("   A different vowel moves the bumps. THAT is what the model reads.")
