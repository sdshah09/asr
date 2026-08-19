"""Phase 1 — honest baseline. One config, measured properly.

Usage:
    .venv/bin/python bench.py                 # uses ./testset, generates it if missing
    .venv/bin/python bench.py path/to/audio   # your own files
    .venv/bin/python bench.py --repeats 5 --model small
"""
import argparse, resource, subprocess, sys, time
from pathlib import Path
from statistics import mean, median, stdev

# ponytail: hardcoded machine cost. change when you deploy on a GPU.
COST_PER_HOUR = 0.0  # local Mac = free. set to your GPU $/hr to get $/audio-hour.

CLIPS = [
    ("short", "Machine learning turns sound into text."),
    ("medium", "The quick brown fox jumps over the lazy dog. "
               "Pack my box with five dozen liquor jugs. "
               "How vexingly quick daft zebras jump."),
    ("long", " ".join(["This is sentence number %d in a longer recording "
                       "used to measure steady state throughput." % i for i in range(1, 16)])),
]


def make_testset(d: Path):
    d.mkdir(exist_ok=True)
    for name, text in CLIPS:
        f = d / f"{name}.aiff"
        if not f.exists():
            subprocess.run(["say", "-o", str(f), text], check=True)
    print(f"generated {len(CLIPS)} synthetic clips in {d}/")
    print("NOTE: synthetic TTS is unrealistically clean. Replace with real audio "
          "before trusting these numbers.\n")


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def peak_rss_mb() -> float:
    # macOS reports bytes, Linux reports kilobytes
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", default="testset")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--vad", action="store_true")
    # ponytail: 10 = M4 Pro performance cores. Default (all 14) thrashes P/E cores.
    ap.add_argument("--cpu-threads", type=int, default=10)
    args = ap.parse_args()

    src = Path(args.audio)
    if src.is_dir():
        if not any(src.glob("*")):
            make_testset(src)
        files = sorted(p for p in src.iterdir()
                       if p.suffix.lower() in {".aiff", ".wav", ".mp3", ".m4a", ".flac", ".ogg"})
    elif src.exists():
        files = [src]
    else:
        make_testset(src)
        files = sorted(src.iterdir())

    if not files:
        sys.exit(f"no audio files in {src}")

    from faster_whisper import WhisperModel

    print("=" * 68)
    print("CONFIG")
    print("=" * 68)
    for k, v in vars(args).items():
        print(f"  {k:<14} {v}")
    print(f"  {'files':<14} {len(files)}")

    t0 = time.perf_counter()
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type, cpu_threads=args.cpu_threads)
    load_s = time.perf_counter() - t0
    print(f"\nmodel load: {load_s:.2f}s   <- pay this ONCE per worker, never per file")

    def run(path):
        t = time.perf_counter()
        segs, info = model.transcribe(path, beam_size=args.beam_size,
                                      vad_filter=args.vad, language="en")
        text = "".join(s.text for s in segs)   # generator: nothing runs until consumed
        return time.perf_counter() - t, text

    print("\nwarmup...", end=" ", flush=True)
    run(str(files[0]))
    print("done\n")

    print("=" * 68)
    print("RESULTS")
    print("=" * 68)
    print(f"{'file':<14}{'audio':>8}{'proc':>9}{'RTF':>8}{'x-real':>9}{'stdev':>8}")
    print("-" * 68)

    tot_audio = tot_proc = 0.0
    all_rtf = []
    for f in files:
        dur = duration(f)
        times, text = [], ""
        for _ in range(args.repeats):
            t, text = run(str(f))
            times.append(t)
        m = median(times)
        sd = stdev(times) if len(times) > 1 else 0.0
        rtf = m / dur
        all_rtf.append(rtf)
        tot_audio += dur
        tot_proc += m
        print(f"{f.stem:<14}{dur:>7.2f}s{m:>8.2f}s{rtf:>8.3f}{1/rtf:>8.1f}x{sd:>7.3f}s")
        print(f"              -> {text.strip()[:60]!r}")

    print("-" * 68)
    agg = tot_proc / tot_audio
    print(f"{'TOTAL':<14}{tot_audio:>7.2f}s{tot_proc:>8.2f}s{agg:>8.3f}{1/agg:>8.1f}x")

    print("\n" + "=" * 68)
    print("BASELINE")
    print("=" * 68)
    print(f"  RTF (aggregate)    : {agg:.3f}     lower is better")
    print(f"  speed              : {1/agg:.1f}x realtime")
    print(f"  audio-hours / hour : {1/agg:.1f}")
    print(f"  RTF spread         : {min(all_rtf):.3f} .. {max(all_rtf):.3f}"
          f"   (mean {mean(all_rtf):.3f})")
    print(f"  peak memory        : {peak_rss_mb():.0f} MB")
    print(f"  model load         : {load_s:.2f}s")
    if COST_PER_HOUR:
        print(f"  cost / audio-hour  : ${COST_PER_HOUR * agg:.4f}")
    else:
        print(f"  cost / audio-hour  : $0 local. OpenAI API charges $0.36 for comparison.")
    print("\n  Next: vary --compute-type / --model / --vad and build the Phase 2 table.")


if __name__ == "__main__":
    main()
