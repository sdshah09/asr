"""Trace a Whisper transcription stage by stage, printing the real numbers at each step."""

import sys
import numpy as np
import mlx.core as mx
from mlx_whisper import audio as A, load_models, tokenizer as T

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "sample.aiff"
REPO = "mlx-community/whisper-large-v3-turbo"


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------- 1. raw audio
rule("STEP 1 — audio file becomes a list of numbers")

wav = A.load_audio(AUDIO)  # ffmpeg: downmix to mono, resample to 16kHz
print(f"array shape      : {wav.shape}   <- one number per sample")
print(f"dtype            : {wav.dtype}")
print(f"duration         : {len(wav) / A.SAMPLE_RATE:.2f} s")
print(f"sample rate      : {A.SAMPLE_RATE} Hz")
print(f"value range      : {wav.min():+.4f} .. {wav.max():+.4f}  (air pressure, normalised)")
print(f"\nfirst 12 samples : {np.array2string(wav[:12], precision=5, floatmode='fixed')}")
mid = len(wav) // 2
print(f"12 mid-speech    : {np.array2string(wav[mid:mid + 12], precision=5, floatmode='fixed')}")

# ---------------------------------------------------------- 2. pad to 30s
rule("STEP 2 — pad/trim to exactly 30 seconds")

padded = A.pad_or_trim(wav)
print(f"before : {wav.shape[0]:>7,} samples ({len(wav) / A.SAMPLE_RATE:.2f}s)")
print(f"after  : {padded.shape[0]:>7,} samples (30.00s)  <- N_SAMPLES, always")
print(f"added  : {padded.shape[0] - wav.shape[0]:>7,} zeros of silence on the right")

# ------------------------------------------------ 3. log-mel spectrogram
rule("STEP 3 — waveform becomes a picture (log-mel spectrogram)")

model = load_models.load_model(REPO, dtype=mx.float16)
n_mels = model.dims.n_mels
mel = A.log_mel_spectrogram(padded, n_mels=n_mels)

print(f"FFT window     : {A.N_FFT} samples = {A.N_FFT / A.SAMPLE_RATE * 1000:.0f} ms per slice")
print(f"hop length     : {A.HOP_LENGTH} samples = {A.HOP_LENGTH / A.SAMPLE_RATE * 1000:.0f} ms between slices")
print(f"mel filters    : {n_mels}")
print(f"\n480,000 numbers  ->  mel shape {tuple(mel.shape)}  (time frames x mel bins)")
print(f"that is {mel.size:,} numbers, vs {padded.shape[0]:,} raw  ->  {padded.shape[0] / mel.size:.1f}x smaller")
print(f"value range    : {mel.min().item():+.2f} .. {mel.max().item():+.2f}   (log energy, not loudness)")

m = np.array(mel.astype(mx.float32))
f = int(len(wav) / A.HOP_LENGTH * 0.4)  # a frame during speech
print(f"\nOne single frame (frame {f}, t={f * A.HOP_LENGTH / A.SAMPLE_RATE:.2f}s) — {n_mels} numbers,")
print("one per frequency band, low pitch first:\n")
print(np.array2string(m[f][:16], precision=2, floatmode="fixed", max_line_width=100))
print("... (" + str(n_mels - 16) + " more bands, going up in pitch)")

print("\nThe same data drawn as the picture the encoder sees:")
print("(rows = pitch, high at top | cols = time | denser char = more energy)\n")
ramp = " .:-=+*#%@"
seg = m[: int(len(wav) / A.HOP_LENGTH), :]           # only the non-silent part
lo, hi = seg.min(), seg.max()
rows = np.linspace(0, n_mels - 1, 24, dtype=int)[::-1]
cols = np.linspace(0, seg.shape[0] - 1, 96, dtype=int)
for r in rows:
    line = "".join(ramp[min(9, int((seg[c, r] - lo) / (hi - lo + 1e-9) * 9))] for c in cols)
    print(f"  {line}")
print(f"  {'^':<47}{'^':>48}")
print(f"  {'t=0s':<47}{f't={len(wav) / A.SAMPLE_RATE:.1f}s':>48}")

# -------------------------------------------------- 4. encoder (conv + transformer)
rule("STEP 4 — encoder: conv shrinks it, transformer understands it")

feats = model.embed_audio(mel[None])  # add batch dim
print(f"mel in         : {tuple(mel[None].shape)}      (batch, frames, mel bins)")
print(f"encoder out    : {tuple(feats.shape)}      (batch, positions, features)")
print(f"\n  {mel.shape[0]} time frames --conv stride 2--> {feats.shape[1]} positions")
print(f"  {n_mels} mel bins    --learned--------> {feats.shape[2]} features each")
print(f"\nEach of the {feats.shape[1]} positions covers "
      f"{A.N_SAMPLES_PER_TOKEN / A.SAMPLE_RATE * 1000:.0f} ms of audio.")
print(f"Encoder layers : {model.dims.n_audio_layer}, attention heads: {model.dims.n_audio_head}")

fv = np.array(feats[0, 20].astype(mx.float32))
print(f"\nOne position's feature vector (position 20), first 8 of {len(fv)} numbers:")
print(np.array2string(fv[:8], precision=3, floatmode="fixed"))
print("These numbers mean nothing to a human. They are the encoder's private notes")
print("about what sound is happening there. Only the decoder reads them.")

# ------------------------------------------------------ 5. decoder, token by token
rule("STEP 5 — decoder writes the text, one token at a time")

tok = T.get_tokenizer(multilingual=model.is_multilingual,
                      num_languages=model.num_languages,
                      language="en", task="transcribe")
tokens = list(tok.sot_sequence_including_notimestamps)

print("The decoder starts with a prompt made of control tokens:\n")
for t in tokens:
    print(f"  id {t:>6}  {tok.decode([t])!r}")
print("\nThat prompt says: start | language English | transcribe (don't translate) | no timestamps\n")
print(f"{'step':>4}  {'chose id':>8}  {'text':<14}  {'prob':>6}   runners-up")
print("-" * 72)

step = 0
while step < 40:
    logits = model.logits(mx.array([tokens]), feats)[0, -1]
    probs = mx.softmax(logits.astype(mx.float32))
    order = np.array(mx.argsort(-probs)[:4])
    p = np.array(probs)
    nxt = int(order[0])

    alts = "  ".join(f"{tok.decode([int(i)])!r}={p[i]:.3f}" for i in order[1:4])
    label = "<|endoftext|>" if nxt == tok.eot else repr(tok.decode([nxt]))
    print(f"{step:>4}  {nxt:>8}  {label:<14}  {p[nxt]:>6.3f}   {alts}")

    tokens.append(nxt)
    step += 1
    if nxt == tok.eot:
        break

rule("RESULT")
text = tok.decode([t for t in tokens if t < tok.eot and t not in tok.sot_sequence_including_notimestamps])
print(f"transcript: {text!r}")
print(f"\ntotal tokens generated: {step}")
print(f"decoder forward passes : {step}   <- one per token. this is why decoding is the slow part.")
print(f"encoder forward passes : 1       <- runs once, no matter how long the text.")
