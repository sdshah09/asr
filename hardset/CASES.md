# Adversarial cases

Reference sentence: `The patient was prescribed forty milligrams of atorvastatin daily.`
Second speaker: `I completely disagree with that assessment of the quarterly numbers.`

| file | what it tests |
|---|---|
| `01_silence_30s` | 30s of pure silence. Expect: empty. Failure: invented text. |
| `02_speech_in_silence` | One sentence at t=20s inside 60s of silence. Tests VAD + hallucination on the empty parts. |
| `03_very_quiet` | Same sentence at 4% volume. Failure: VAD drops it entirely and it vanishes from the transcript. |
| `04_noisy` | Speech + white noise. Tests robustness; watch which words go first. |
| `05_overlapping` | Two people at once. Expect: one speaker wins, the other is lost. No diarization. |
| `06_stereo_two_speakers` | One speaker per channel. Downmix muddles them; split with -map_channel for two clean transcripts. |
| `07_fast_speech` | 320 wpm. Tests whether words get dropped or merged. |
| `08_music_no_speech` | Tones, no speech. Expect: empty. Failure: lyrics invented. VAD often calls this speech. |
| `09_long_repeated` | Same sentence x40 (~7 min). Tests chunk seams and condition_on_previous_text looping. |
| `10_clipped` | Massively over-driven. Simulates a mic too close or bad gain staging. |
