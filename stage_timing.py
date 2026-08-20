"""Where does the time actually go? Time each stage separately."""
import sys, time
import mlx.core as mx
from mlx_whisper import audio as A, load_models, tokenizer as T

F = sys.argv[1] if len(sys.argv) > 1 else "realset/The 30-Second Video.mp3"
REPO = "mlx-community/whisper-large-v3-turbo"

t = time.perf_counter
marks = []
def stage(name, fn):
    a = t(); r = fn(); b = t()
    marks.append((name, b - a))
    return r

model = load_models.load_model(REPO, dtype=mx.float16)   # warmup, not counted
tok = T.get_tokenizer(True, num_languages=model.num_languages, language="en", task="transcribe")

print(f"file: {F}\n")

wav   = stage("1. ffmpeg decode (mp3 -> numbers, mono, 16kHz)", lambda: A.load_audio(F))
dur   = len(wav) / A.SAMPLE_RATE
pad   = stage("2. pad to 30s", lambda: A.pad_or_trim(wav))
mel   = stage("3. spectrogram (FFT + mel + log)", lambda: A.log_mel_spectrogram(pad, n_mels=model.dims.n_mels))
mx.eval(mel)
feats = stage("4. ENCODER (conv + 32 transformer layers)", lambda: mx.eval(model.embed_audio(mel[None])) or model.embed_audio(mel[None]))

tokens = list(tok.sot_sequence_including_notimestamps)
a = t()
n = 0
while n < 200:
    logits = model.logits(mx.array([tokens]), feats)[0, -1]
    nxt = int(mx.argmax(logits))
    tokens.append(nxt); n += 1
    if nxt == tok.eot: break
b = t()
marks.append((f"5. DECODER ({n} tokens, one pass each)", b - a))

total = sum(x for _, x in marks)
print(f"{'stage':<48}{'seconds':>9}{'share':>8}")
print("-" * 65)
for name, s in marks:
    print(f"{name:<48}{s:>8.3f}s{s/total*100:>7.1f}%")
print("-" * 65)
print(f"{'TOTAL':<48}{total:>8.3f}s")
print(f"\naudio duration {dur:.1f}s -> RTF {total/dur:.3f}  ({dur/total:.1f}x realtime)")
print(f"decoder cost per token: {marks[-1][1]/n*1000:.0f} ms")
