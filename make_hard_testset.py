"""Build audio designed to make Whisper fail. Phase 0: break it deliberately.

Usage:  python3 make_hard_testset.py [outdir]
Needs:  ffmpeg, and macOS `say`
"""
import subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "hardset")
OUT.mkdir(exist_ok=True)
SENT = "The patient was prescribed forty milligrams of atorvastatin daily."
ALT  = "I completely disagree with that assessment of the quarterly numbers."


def sh(*args):
    subprocess.run(args, check=True, capture_output=True)


def say(text, out, voice=None, rate=None):
    cmd = ["say", "-o", str(out)]
    if voice: cmd += ["-v", voice]
    if rate:  cmd += ["-r", str(rate)]
    sh(*cmd, text)


def ff(*args):
    sh("ffmpeg", "-y", "-loglevel", "error", *args)


tmp = OUT / "_tmp"
tmp.mkdir(exist_ok=True)
say(SENT, tmp / "a.aiff")
say(ALT, tmp / "b.aiff", voice="Daniel")

cases = {}

# 1. pure silence — the classic hallucination trigger
ff("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "30",
   str(OUT / "01_silence_30s.wav"))
cases["01_silence_30s"] = "30s of pure silence. Expect: empty. Failure: invented text."

# 2. speech buried in silence — VAD's whole reason to exist
ff("-i", str(tmp / "a.aiff"), "-af", "adelay=20000|20000,apad=whole_dur=60",
   "-ar", "16000", "-ac", "1", str(OUT / "02_speech_in_silence.wav"))
cases["02_speech_in_silence"] = "One sentence at t=20s inside 60s of silence. Tests VAD + hallucination on the empty parts."

# 3. very quiet speech — VAD's silent failure mode
ff("-i", str(tmp / "a.aiff"), "-af", "volume=0.04", "-ar", "16000", "-ac", "1",
   str(OUT / "03_very_quiet.wav"))
cases["03_very_quiet"] = "Same sentence at 4% volume. Failure: VAD drops it entirely and it vanishes from the transcript."

# 4. loud white noise over speech
ff("-i", str(tmp / "a.aiff"), "-f", "lavfi", "-i", "anoisesrc=c=white:a=0.06",
   "-filter_complex", "[1]atrim=0:12[n];[0][n]amix=inputs=2:duration=first",
   "-ar", "16000", "-ac", "1", str(OUT / "04_noisy.wav"))
cases["04_noisy"] = "Speech + white noise. Tests robustness; watch which words go first."

# 5. two speakers talking over each other
ff("-i", str(tmp / "a.aiff"), "-i", str(tmp / "b.aiff"),
   "-filter_complex", "[0][1]amix=inputs=2:duration=longest",
   "-ar", "16000", "-ac", "1", str(OUT / "05_overlapping.wav"))
cases["05_overlapping"] = "Two people at once. Expect: one speaker wins, the other is lost. No diarization."

# 6. speakers on separate channels — the stereo trick
ff("-i", str(tmp / "a.aiff"), "-i", str(tmp / "b.aiff"),
   "-filter_complex", "[0][1]amerge=inputs=2", "-ar", "16000",
   str(OUT / "06_stereo_two_speakers.wav"))
cases["06_stereo_two_speakers"] = "One speaker per channel. Downmix muddles them; split with -map_channel for two clean transcripts."

# 7. fast speech
say(SENT, tmp / "fast.aiff", rate=320)
ff("-i", str(tmp / "fast.aiff"), "-ar", "16000", "-ac", "1", str(OUT / "07_fast_speech.wav"))
cases["07_fast_speech"] = "320 wpm. Tests whether words get dropped or merged."

# 8. music only, no speech
ff("-f", "lavfi", "-i", "sine=frequency=440:duration=15",
   "-f", "lavfi", "-i", "sine=frequency=554:duration=15",
   "-filter_complex", "[0][1]amix=inputs=2", "-ar", "16000", "-ac", "1",
   str(OUT / "08_music_no_speech.wav"))
cases["08_music_no_speech"] = "Tones, no speech. Expect: empty. Failure: lyrics invented. VAD often calls this speech."

# 9. long file — chunk stitching and context drift
ff("-stream_loop", "40", "-i", str(tmp / "a.aiff"), "-ar", "16000", "-ac", "1",
   str(OUT / "09_long_repeated.wav"))
cases["09_long_repeated"] = "Same sentence x40 (~7 min). Tests chunk seams and condition_on_previous_text looping."

# 10. clipped / distorted
ff("-i", str(tmp / "a.aiff"), "-af", "volume=14,alimiter=limit=1",
   "-ar", "16000", "-ac", "1", str(OUT / "10_clipped.wav"))
cases["10_clipped"] = "Massively over-driven. Simulates a mic too close or bad gain staging."

for f in tmp.iterdir():
    f.unlink()
tmp.rmdir()

notes = OUT / "CASES.md"
notes.write_text(
    "# Adversarial cases\n\n"
    f"Reference sentence: `{SENT}`\n"
    f"Second speaker: `{ALT}`\n\n"
    "| file | what it tests |\n|---|---|\n"
    + "".join(f"| `{k}` | {v} |\n" for k, v in cases.items())
)

print(f"built {len(cases)} cases in {OUT}/")
for k, v in cases.items():
    print(f"  {k:<28} {v[:60]}")
print(f"\nwrote {notes}")
