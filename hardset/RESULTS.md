# Failure log

| case | vad | audio | RTF | flags | transcript |
|---|---|---|---|---|---|
| `01_silence_30s` | off | 30.0s | 0.09 | SUSPICIOUSLY_SHORT | Thank you. |
| `01_silence_30s` | on | 30.0s | 0.03 | EMPTY | *(empty)* |
| `02_speech_in_silence` | off | 60.0s | 0.10 | ok | The patient was prescribed 40 milligrams of atorvastatin daily. The patient was prescribed 40 milligrams of atorvastatin |
| `02_speech_in_silence` | on | 60.0s | 0.05 | ok | The patient was prescribed 40 milligrams of atorvastatin daily. |
| `03_very_quiet` | off | 3.9s | 0.77 | ok | The patient was prescribed 40 milligrams of atorvastatin daily. |
| `03_very_quiet` | on | 3.9s | 0.75 | ok | The patient was prescribed 40 milligrams of atorvastatin daily. |
| `04_noisy` | off | 3.9s | 0.79 | ok | The patient was prescribed 40 milligrams of atorvastatin daily. |
| `04_noisy` | on | 3.9s | 0.74 | ok | The patient was prescribed 40 milligrams of atorvastatin daily. |
| `05_overlapping` | off | 3.9s | 0.75 | ok | The patient was prescribed 40 mg of atorvastatin daily. |
| `05_overlapping` | on | 3.9s | 0.76 | ok | The patient was prescribed 40 mg of atorvastatin daily. |
| `06_stereo_two_speakers` | off | 3.6s | 0.77 | ok | The patient was prescribed 40 mg of atorvastatin data. |
| `06_stereo_two_speakers` | on | 3.6s | 0.78 | ok | The patient was prescribed 40 mg of atorvastatin data. |
| `07_fast_speech` | off | 2.3s | 1.25 | ok | The patient was prescribed 40 milligrams of atorvastatin daily. |
| `07_fast_speech` | on | 2.3s | 1.21 | ok | The patient was prescribed 40 milligrams of atorvastatin daily. |
| `08_music_no_speech` | off | 15.0s | 0.20 | SUSPICIOUSLY_SHORT | . |
| `08_music_no_speech` | on | 15.0s | 0.00 | EMPTY | *(empty)* |
| `09_long_repeated` | off | 160.9s | 0.21 | ok | The patient was prescribed 40 mg of atorvastatin daily. The patient was prescribed 40 mg of atorvastatin daily. The pati |
| `09_long_repeated` | on | 160.9s | 0.17 | ok | The patient was prescribed 40 mg of atorvastatin daily. The patient was prescribed 40 mg of atorvastatin daily. The pati |
| `10_clipped` | off | 3.9s | 0.75 | ok | The patient was prescribed 40 mg of atorvastatin daily. |
| `10_clipped` | on | 3.9s | 0.72 | ok | The patient was prescribed 40 mg of atorvastatin daily. |
