"""E1.3 — does the smaller vocabulary hurt on rare words?

Two parts:
  A) tokenizer only — how many fragments does each model need per word?
  B) end to end   — synthesize speech, transcribe with both, compare.
"""
import re, subprocess, sys
from pathlib import Path
import numpy as np, torch

WORDS = {
    "common":   ["the", "people", "yesterday", "important"],
    "names":    ["Shaswat", "Anthropic", "Kubernetes", "PostgreSQL"],
    "medical":  ["atorvastatin", "metoprolol", "levothyroxine", "acetaminophen"],
    "jargon":   ["quantization", "autoregressive", "spectrogram", "tokenizer"],
    "borrowed": ["croissant", "jalapeno", "karaoke", "entrepreneur"],
}

SENTENCES = [
    ("medical",  "The patient was prescribed forty milligrams of atorvastatin daily."),
    ("medical",  "She switched from metoprolol to levothyroxine last spring."),
    ("names",    "Shaswat deployed the service to Kubernetes using PostgreSQL."),
    ("jargon",   "The tokenizer splits rare words before the autoregressive decoder runs."),
    ("jargon",   "Quantization reduced the spectrogram model to eight bit weights."),
    ("borrowed", "The entrepreneur ordered a croissant and sang karaoke."),
    ("common",   "Yesterday the people said it was important."),
]

from transformers import AutoTokenizer, AutoProcessor, MoonshineForConditionalGeneration
wt = AutoTokenizer.from_pretrained("openai/whisper-tiny.en")
mp = AutoProcessor.from_pretrained("moonshine-ai/moonshine-tiny")
mt = mp.tokenizer

print("=" * 74)
print("PART A — tokenizer: how many fragments per word?")
print("=" * 74)
print(f"whisper vocab {wt.vocab_size:,}   moonshine vocab {mt.vocab_size:,}\n")
print(f"{'category':<10}{'word':<18}{'whisper':>9}{'moonshine':>11}   moonshine pieces")
print("-" * 74)
tot_w = tot_m = 0
for cat, ws in WORDS.items():
    for w in ws:
        # NOTE: no leading space. Whisper (BPE) folds a space into the word token;
        # Moonshine (SentencePiece) emits a separate '_' token for it, which would
        # add one spurious token to every word and inflate the comparison.
        a = wt.encode(w, add_special_tokens=False)
        b = mt.encode(w, add_special_tokens=False)
        tot_w += len(a); tot_m += len(b)
        pieces = "".join(f"[{mt.decode([t])}]" for t in b)
        flag = "  <-- more" if len(b) > len(a) else ""
        print(f"{cat:<10}{w:<18}{len(a):>9}{len(b):>11}   {pieces}{flag}")
print("-" * 74)
print(f"{'TOTAL':<28}{tot_w:>9}{tot_m:>11}   moonshine needs {tot_m/tot_w:.2f}x the tokens")

if "--tokens-only" in sys.argv:
    sys.exit()

print("\n" + "=" * 74)
print("PART B — end to end: synthesize, transcribe, compare")
print("=" * 74)

D = Path("vocabset"); D.mkdir(exist_ok=True)
from faster_whisper import WhisperModel
from mlx_whisper import audio as A
wm = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=10)
mm = MoonshineForConditionalGeneration.from_pretrained("moonshine-ai/moonshine-tiny").eval()

def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()

def moonshine(p):
    wav = np.array(A.load_audio(str(p)))
    inp = mp(wav, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = mm.generate(**inp, max_new_tokens=200)
    return mp.batch_decode(ids, skip_special_tokens=True)[0]

score = {"whisper": [0, 0], "moonshine": [0, 0]}
for i, (cat, sent) in enumerate(SENTENCES):
    f = D / f"{i}.wav"
    if not f.exists():
        tmp = D / f"{i}.aiff"
        subprocess.run(["say", "-o", str(tmp), sent], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                        "-ar", "16000", "-ac", "1", str(f)], check=True)
        tmp.unlink()
    ref = norm(sent)
    w_out = "".join(s.text for s in wm.transcribe(str(f), beam_size=1, language="en")[0])
    m_out = moonshine(f)
    print(f"\n[{cat}] {sent}")
    for name, out in (("whisper  ", w_out), ("moonshine", m_out)):
        hyp = norm(out)
        wrong = [t for t in ref if t not in hyp]
        key = name.strip()
        score[key][0] += len(ref) - len(wrong); score[key][1] += len(ref)
        tag = "OK" if not wrong else f"MISSED {wrong}"
        print(f"  {name}: {out.strip()[:80]!r}")
        print(f"             {tag}")

print("\n" + "=" * 74)
print(f"{'model':<12}{'words correct':>16}{'accuracy':>11}")
print("-" * 74)
for k, (ok, tot) in score.items():
    print(f"{k:<12}{f'{ok}/{tot}':>16}{ok/tot*100:>10.1f}%")
