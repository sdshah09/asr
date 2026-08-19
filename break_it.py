"""Phase 0 — run the adversarial set and log what breaks.

Usage:  .venv/bin/python break_it.py [dir]   (default: hardset)
"""
import re, subprocess, sys, time
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "hardset")
MODEL, CT, THREADS = "large-v3-turbo", "int8", 10

# phrases Whisper is known to invent from its training data
HALLUCINATION_MARKERS = [
    "thanks for watching", "subscribe", "thank you for watching", "see you next",
    "www.", ".com", "amara.org", "transcription by", "bye bye", "♪",
]


def duration(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(p)], capture_output=True, text=True)
    return float(o.stdout.strip())


def flags(text, dur):
    out = []
    t = text.strip().lower()
    if not t:
        out.append("EMPTY")
        return out
    for m in HALLUCINATION_MARKERS:
        if m in t:
            out.append(f"HALLUCINATION({m!r})")
            break
    words = t.split()
    if len(words) > 6:
        # crude loop detector: is one 4-gram repeated a lot?
        grams = [" ".join(words[i:i+4]) for i in range(len(words)-3)]
        top = max(set(grams), key=grams.count)
        if grams.count(top) > max(3, len(grams) * 0.25):
            out.append(f"LOOPING({top!r} x{grams.count(top)})")
    if dur > 5 and len(words) < 3:
        out.append("SUSPICIOUSLY_SHORT")
    return out


def main():
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL, device="cpu", compute_type=CT, cpu_threads=THREADS)
    files = sorted(p for p in D.iterdir() if p.suffix.lower() in
                   {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff"})
    rows = []
    for f in files:
        dur = duration(f)
        for vad in (False, True):
            t0 = time.perf_counter()
            segs, _ = model.transcribe(str(f), beam_size=1, language="en", vad_filter=vad)
            text = "".join(s.text for s in segs)
            el = time.perf_counter() - t0
            fl = flags(text, dur)
            rows.append((f.stem, "on" if vad else "off", dur, el, el/dur, text.strip(), fl))
            tag = " ".join(fl) or "ok"
            print(f"\n{f.stem}  [vad {'on ' if vad else 'off'}]  {dur:.1f}s audio, "
                  f"{el:.1f}s proc, RTF {el/dur:.2f}   -> {tag}")
            print(f"    {text.strip()[:150]!r}")

    out = D / "RESULTS.md"
    with open(out, "w") as fh:
        fh.write("# Failure log\n\n| case | vad | audio | RTF | flags | transcript |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for name, vad, dur, el, rtf, text, fl in rows:
            t = text.replace("|", "\\|")[:120] or "*(empty)*"
            fh.write(f"| `{name}` | {vad} | {dur:.1f}s | {rtf:.2f} | "
                     f"{' '.join(fl) or 'ok'} | {t} |\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
