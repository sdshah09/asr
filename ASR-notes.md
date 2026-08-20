# ASR — Automatic Speech Recognition, from zero

How Whisper turns sound into text, and how to ship it in production.

**How to read this.** Part 1 is the whole idea in 5 minutes. Part 2 is the dictionary — dip in whenever a word confuses you. Part 3 is the machine running, traced end to end. Part 4 is usage. Part 5 is the production engineering path.

---

# Revision card — read this first, or alone

Everything below in one page. If you only re-read one section, make it this one.

## The model in four steps

```
sound  ->  picture  ->  understanding  ->  text
```

1. **Turn sound into a picture.** Audio is a wiggle; wiggles are hard to read. Redraw it as a picture — low sounds at the bottom, high at the top, loud is bright. Every vowel and consonant makes a distinctive shape.
2. **Look at the picture.** The **encoder** scans it and builds an internal sense of what sounds happened. Runs **once**.
3. **Write the words.** The **decoder** types one word-piece at a time, looking at both the sound picture and what it has already written. Runs **once per token**.
4. **Stop** on a special end token.

## The one idea that explains everything else

Whisper does not match sounds to words like a dictionary. **It guesses the next word using the audio AND the sentence so far.**

There is nothing to "copy" — the audio contains no word boundaries, no spelling, no capitals, no punctuation. All of that is reconstructed.

That guessing is why it survives noise. It is also why it invents text on silence. **The robustness and the hallucination are the same feature.**

## Numbers worth memorising

| Number | What |
|---|---|
| **16,000 Hz** | sample rate. Speech lives below 8 kHz; Nyquist says sample at 2x |
| **30 seconds** | the input window. ALWAYS. Shorter audio is zero-padded |
| **480,000** | samples in that window (30 x 16,000) |
| **25 ms / 10 ms** | FFT window size / hop between windows (so they overlap) |
| **3000 x 128** | the spectrogram: time slices x pitch buckets |
| **1500 x 1280** | encoder output: positions x features (conv halves 3000; 20 ms each) |
| **1 vs N** | encoder passes vs decoder passes. Decoder runs once per token |

## Where the time goes (measured, real 30s clip)

```
ffmpeg decode        0.088s   1.8%
spectrogram          0.002s   0.0%    <- free
ENCODER              1.045s  21.6%    <- runs once
DECODER              3.700s  76.5%    <- runs 137 times, 27 ms each
```
**The decoder is 3/4 of the work.** That is why `turbo` cuts the decoder 32 layers -> 4 and leaves the encoder alone.

## Key terms in one line each

| Term | Meaning |
|---|---|
| **transcribe** | turn audio into text |
| **inference** | using a trained model (vs training it) |
| **token** | one piece of text the model writes, often a word-piece |
| **encoder / decoder** | the half that listens / the half that writes |
| **self-attention** | decoder looking at what it already wrote (context) |
| **cross-attention** | decoder looking at the audio (evidence) |
| **spectrogram** | the audio redrawn as a picture |
| **frame / window / slice** | all the same thing: one 25 ms piece of audio |
| **bin / band / bucket** | all the same thing: one pitch range |
| **RTF** | time taken / audio length. Lower is better. Above 1.0 = slower than listening |
| **VAD** | filter that finds speech so you skip silence |
| **quantization** | storing weights less precisely to save space |
| **hallucination** | the model writing words that were never spoken |
| **WER** | word error rate — the accuracy metric |

## What we measured that contradicts common advice

1. **Default thread count cost 4-5x.** CTranslate2 used all 14 logical cores; pinning to the 10 performance cores took an 8s clip from 8-17s (unstable) to a steady 3.0s.
2. **"int8 is faster" is not universal.** On a short synthetic clip float32 won by 24%. On real audio int8 won by 4x — because float32 hallucinated 28 extra seconds of content.
3. **VAD's value is a property of your audio, not the model.** 3.5x faster on silent files; slightly slower on dense narration.
4. **Short files are disproportionately expensive.** RTF 1.25 at 2.3s vs 0.21 at 161s — because every file pays the full 30-second padding cost.
5. **Precision changes the decoding path, not just speed.** Tiny numerical differences flip threshold decisions (is this silence? is this repetitive?) which cascade into completely different output.
6. **The anti-looping safeguard causes instability.** On repetitive audio the compression-ratio check fires, retries with randomness, and output changes every run. Costs 3-4x. `temperature=[0.0]` for reproducibility.

## The five ways it broke

| Failure | What happened |
|---|---|
| **Hallucination** | 30s of silence -> `'Thank you.'` |
| **Duplication** | a sentence present ONCE was transcribed TWICE (chunk seam) |
| **Silent corruption** | stereo downmix turned `'daily'` into `'data'` — replaced, not dropped |
| **Speaker loss** | two overlapping speakers -> one transcribed, one vanished with no warning |
| **Normalization drift** | same sentence -> `'40 milligrams'` or `'40 mg'` depending on acoustics |

All five are silent — nothing errors, the output just looks fine and is wrong.

## The method, in three lines

```python
start = time.perf_counter()
model.transcribe(file)
elapsed = time.perf_counter() - start
```
Change **one** thing. Run again. Compare. Repeat 3x — **if the runs disagree, that disagreement IS the finding.**

## Commands

```bash
.venv/bin/python bench.py realset --vad     # benchmark
.venv/bin/python break_it.py                # adversarial suite
.venv/bin/python stage_timing.py            # where the time goes
.venv/bin/python trace_whisper.py audio.mp3 # watch the model work
.venv/bin/python trace_mel.py               # sound -> picture, in detail
.venv/bin/python quant_demo.py              # quantization on real weights
python3 make_hard_testset.py                # rebuild the adversarial audio
```

## Map of this document

| Part | What is in it |
|---|---|
| **1** | The whole idea in plain language, 5 minutes |
| **2.1** | Sound and recording — samples, rate, channels, ffmpeg |
| **2.2** | Wave to picture — FFT, mel, spectrogram |
| **2.3** | Network basics — tensors, weights, convolution, tokens |
| **2.4** | Transformers and attention |
| **2.5** | Making a prediction — logits, softmax, temperature |
| **2.6** | Whisper specifics — chunks, special tokens, hallucination, VAD |
| **2.7** | Hardware, memory, quantization internals, CTranslate2, ONNX |
| **3** | The pipeline traced with real numbers |
| **4** | Running it — CLI, Python, every flag |
| **5** | Production path, 7 phases, plus all measured results |

---

# Part 1 — The whole thing in plain language

## Whisper does four things

### 1. Turn sound into a picture

Sound is a wiggle in the air. Wiggles are hard to read. So Whisper redraws the audio as a picture — like the bouncing bars on a stereo, but printed left-to-right over time.

Low sounds draw near the bottom. High sounds near the top. Loud is bright, quiet is dark.

Every vowel and consonant makes its own recognizable shape in that picture. "Ahh" looks different from "Sss". Always. That consistency is what makes the whole thing possible.

### 2. Look at the picture

Half the model is essentially an image reader. It scans the picture and builds an internal understanding of what sounds happened and in what order. It does not produce words yet.

### 3. Write out the words

The other half writes text — one word-piece at a time, left to right, like a person typing. At each step it does two things at once: looks back at the sound picture, and looks at what it has already written. Then picks the next piece.

### 4. Stop

When it thinks the audio is finished, it emits a special "done" marker and halts.

```
sound  ->  picture  ->  understanding  ->  text
```

## The one thing worth actually remembering

Whisper is **not** matching sounds to words like a dictionary lookup.

It **guesses the next word** using both the audio *and* the sentence so far. When it wrote "Machine learning turns sound into ___", it barely needed the audio — the sentence already implied "text".

That is why it survives noisy recordings that defeat older systems: when it cannot hear clearly, it leans on what makes sense. It is also why it sometimes confidently writes a word that was never said.

Everything else in this document is detail underneath those four steps.

---

# Part 2 — Glossary: every term, what it means, how it works

## 2.1 — Sound and recording

### Waveform
**What.** The raw audio: a long list of numbers, one per measurement, describing air pressure over time.

**How.** A microphone has a thin diaphragm. Air pressure pushes it in and out. The recorder measures its position thousands of times per second. Positive = pressure high, negative = low, zero = at rest.

**Example.** Twelve real samples from the middle of a spoken word:
```
-0.019  -0.013  -0.007  -0.005  -0.007  -0.009
-0.010  -0.006  -0.006  +0.000  +0.008  +0.012
```
Read it as motion: dropping, bottoming out, climbing back through zero, going positive. That is **one wobble of air** — half a cycle of a sound wave — across 0.75 milliseconds. All audio is ever this.

### Sample
**What.** One single number in the waveform. One measurement at one instant.

**Example.** `-0.019` is one sample. A 3-second clip contains about 46,000 of them.

### Sample rate
**What.** How many samples per second. Whisper uses **16,000 Hz (16 kHz)**.

| Rate | Used for |
|---|---|
| 8,000 Hz | Telephone |
| **16,000 Hz** | **Speech recognition** |
| 44,100 Hz | CD, music |
| 48,000 Hz | Video |

**Example.** 2.89 seconds x 16,000 = 46,251 samples.

### Nyquist rule
**What.** To capture a frequency, you must sample at **twice** that frequency.

**How.** To describe a wave you need at least two points per cycle — one peak, one trough. Sample slower and the wave is indistinguishable from a slower one (why wagon wheels appear to spin backwards on film — the camera under-samples).

**Example.** Speech carries almost all information below 8,000 Hz. 8,000 x 2 = 16,000. That is exactly why speech models use 16 kHz — the cheapest rate that loses nothing. Music needs 44,100 Hz because cymbals reach ~20,000 Hz.

### Amplitude
**What.** How far the wave swings from zero. Corresponds to loudness.

**Example.** Near `0.0000` = silence. Swinging `-0.47` to `+0.77` = someone talking. Pinned at `±1.0` = clipping, recording too loud and distorted.

### Channel
**What.** One microphone's worth of numbers. One stream of samples.

```
mono:    [0.1, 0.3, 0.2, -0.1, ...]

stereo:  L [0.1, 0.3, 0.2, -0.1, ...]
         R [0.1, 0.2, 0.4,  0.0, ...]
```

### Mono / Stereo
**What.** Mono = 1 channel. Stereo = 2 channels, **left** and **right**.

**How stereo works.** A sound from your left reaches your left ear slightly sooner and slightly louder. Your brain reads that difference — microseconds of delay, a few decibels — and concludes "that came from over there." Stereo recording fakes this deliberately. There is no "position" stored anywhere; just two slightly different copies. Identical channels = sound appears dead center = mono wasting twice the space.

**More channels exist.** 6 = 5.1 surround (front L/C/R, rear L/R, subwoofer). 8+ = 7.1, Atmos, studio multitrack. Channels are just parallel streams.

**Why Whisper discards it.** ASR cares *what was said*, not *where the speaker stood*:
```
mono[i] = (left[i] + right[i]) / 2
```
Half the data, all the words.

**The catch.** If two speakers were miked separately onto left and right (interview, two-host podcast), downmixing merges them into one muddy track and Whisper transcribes both as one run-on speaker. Split first:
```bash
ffmpeg -i interview.wav -map_channel 0.0.0 host.wav -map_channel 0.0.1 guest.wav
```
Two clean transcripts, and you know who said what — poor man's diarization, no extra model.

### ffmpeg
**What.** The universal media converter. Command-line only.

**Why Whisper needs it.** Whisper eats exactly one thing: raw numbers, 16,000/sec, one channel. Your files are never that — an `.mp3` is compressed, an `.m4a` from your phone is stereo at 48 kHz. ffmpeg does three jobs in one pass: **decode** the compressed format, **downmix** stereo to mono, **resample** to 16 kHz.

```bash
ffmpeg -i input.m4a -ac 1 -ar 16000 output.wav
```

| Flag | Meaning |
|---|---|
| `-i input.m4a` | input file |
| `-ac 1` | **a**udio **c**hannels → 1 |
| `-ar 16000` | **a**udio **r**ate → 16000 |

**That is Whisper's entire audio preprocessing step.** `mlx_whisper` and `faster-whisper` shell out to it, which is why installing it was required before anything worked.

Also useful: `ffprobe audio.m4a` prints a file's real sample rate, channel count, codec, and duration. Reach for it when transcription behaves oddly and you want to check what you actually fed in.

---

## 2.2 — From wave to picture

### Frequency
**What.** How many times per second a wave repeats. Measured in Hertz. Perceived as **pitch**.

**Example.** 110 Hz = low male voice fundamental. 220 Hz = typical female voice fundamental. 3000 Hz = the hiss in "s". 8000 Hz = about as high as speech carries.

### Fourier transform / FFT
**What.** The math answering: "which frequencies are inside this chunk of wave, and how strong is each?"

**How.** Takes a chunk of samples, returns one strength number per frequency. It has no concept of "a tone" — it mechanically asks a fixed list of questions ("how much 40 Hz? how much 80 Hz?") and reports answers. FFT = Fast Fourier Transform, an efficient algorithm for computing it.

**Example.** Feed it 400 samples of a pure 220 Hz tone. Out come 201 numbers, one per frequency from 0 to 8000 Hz in 40 Hz steps. The strongest:
```
    240 Hz     42.38  #######################################
    200 Hz     42.36  #######################################
    160 Hz      8.58  ########
    280 Hz      8.56  ########
    120 Hz      1.25  #
     80 Hz      0.44
```
Two big values at 200 and 240, everything else near zero. The tone was 220 Hz — exactly between those bins — so it split across both. **The FFT found the pitch.**

### Window / frame
**What.** One short slice of audio that gets one FFT. Whisper uses **400 samples = 25 milliseconds**.

**Why short.** Speech changes fast. Over 25 ms a sound is roughly steady, so "what frequencies are here" has a clean answer. Over 1 second you would average several different sounds into mush.

**Vocabulary note.** "Frame", "window", and "slice" all mean the same thing. One 25 ms piece of audio → one column of the picture.

### Hop length / overlap
**What.** How far the window slides before the next slice. Whisper uses **160 samples = 10 milliseconds**.

**How.** Window is 25 ms but the step is 10 ms, so consecutive windows **overlap by 15 ms**. Deliberate — a sound landing on a boundary is not chopped in half.

**Example.** 480,000 samples ÷ 160 per hop = **3000 windows**. That is where 3000 comes from.

### Bin, band, bucket
**What.** Three words for the same thing: one pitch range in the output. Interchangeable.

**Example.** "Bucket 4" ≈ 120 Hz. "Bucket 87" ≈ 3000 Hz. There are 128, ordered lowest pitch to highest.

### Mel scale
**What.** A re-spacing of the frequency axis to match human hearing.

**How.** Your ear cares enormously about 100 vs 200 Hz, and barely at all about 7900 vs 8000. So instead of 201 evenly-spaced FFT bins, group them into 128 unevenly-sized buckets: narrow at the bottom, wide at the top.

**Example.** Real bucket coverage from Whisper's filters:
```
 bucket       covers (Hz)    width
      0       40 -   40      0 Hz wide
      5      120 -  160     40 Hz wide
     50     1200 - 1240     40 Hz wide
     80     2480 - 2560     80 Hz wide
    110     5080 - 5280    200 Hz wide
    127     7640 - 7960    320 Hz wide
```

### Mel filterbank
**What.** The matrix that does the re-bucketing. Shape **128 x 201** — for each of 128 buckets, a weight for each of 201 FFT bins. Multiply and you get 128 numbers.

### Log
**What.** Taking the logarithm of each value.

**Why.** Loudness is not linear — this is why decibels exist. Log compresses the range so quiet details are not drowned by loud parts.

### Log-mel spectrogram
**What.** The final input to the model. **A picture of the sound, 3000 columns (time) x 128 rows (pitch).**

**How built.** For each of 3000 windows: 400 samples → FFT → 201 frequency strengths → mel filterbank → 128 buckets → log. Stack all 3000 side by side.

**What the numbers mean.** Index = pitch. Value = energy at that pitch. Proof, using pure tones:
```
  110 Hz tone  ->  loudest bucket = #4
  220 Hz tone  ->  loudest bucket = #9
  440 Hz tone  ->  loudest bucket = #18
 1000 Hz tone  ->  loudest bucket = #42
 3000 Hz tone  ->  loudest bucket = #87
```
Pitch up, bucket number up. That is all the 128 numbers mean.

**Example — one real voice frame**, 25 ms of a spoken vowel, every other bucket labeled with its Hz:
```
  #  0 ~   40Hz  -0.45  ######
  #  2 ~   80Hz  -0.03  ################
  #  4 ~  120Hz  +0.93  ########################################
  #  6 ~  160Hz  +1.09  ############################################
  #  8 ~  200Hz  +0.90  #######################################
  # 10 ~  280Hz  +0.31  ########################
  # 12 ~  320Hz  +0.76  ####################################
  # 14 ~  360Hz  +0.71  ##################################
  ...
  # 80 ~ 2520Hz  -0.10  ##############
  #110 ~ 5200Hz  -0.70
  #120 ~ 6600Hz  -0.43  ######
```
Reading it:
- **Buckets 0–3 near zero** (40–100 Hz): nothing there. Human voices produce little below ~85 Hz.
- **Buckets 4–8 spike** (120–200 Hz): the **vocal cords vibrating**. The pitch of the voice. Loudest thing in any voiced sound.
- **Buckets 10–30 wiggle** (280–700 Hz): the **mouth shape** — tongue and lips filtering the buzz into a specific vowel. Move your tongue, these bumps move, different vowel. Same cords, different filter.
- **Above bucket 100 goes negative** (4000 Hz+): little energy. If this frame were "s" or "sh" the picture would flip — quiet at the bottom, loud at the top, because those are hiss, not buzz.

**Example — the whole clip drawn.** 2.9 seconds of "Machine learning turns sound into text." Rows = pitch (high at top), columns = time, denser character = more energy:
```
     *%#-=---+===*===#%%###%%%%#.+-=#+==:.==-+--= +---*%##*==-=:+=.%%###==#*: +--#**++: --:.  .::
    .#*+---:-%%%###%%#++**##*+-=-=-==*******++--: --:=#:+*##%%%%-: +-+=*=:#*: =--+-*+=: -:.   ::-
     *##=:..:=--:..-=****+****##=-:**########*-.. -::=+**++-=====-.**###==**-.-:-***++- -..   .::
  ^                                                                                             ^
  t=0s                                                                                     t=2.9s
```
Bright at the bottom = energy at low frequencies = a human voice. Vertical gaps are word breaks; dense blobs are sustained vowels. **This picture is the model's actual input.** Whisper is, mechanically, an image model reading pictures of sound.

**Honest note.** 3000 x 128 = 384,000 numbers vs 480,000 raw. Barely smaller. The win is not compression, it is **structure**. The raw waveform hides "which frequencies" inside the wiggle; the spectrogram makes it explicit, and explicit is what a network can learn from.

---

## 2.3 — Neural network basics

### Tensor / shape
**What.** A tensor is an array of numbers with any number of dimensions. Its **shape** lists the size of each.

**Example.** `(3000, 128)` = 3000 rows x 128 columns = 384,000 numbers.

### Batch
**What.** Processing several inputs at once for efficiency. The leading `1` in `(1, 3000, 128)` means "one audio clip in this batch."

### Parameter / weight
**What.** One learned number inside the model. Training adjusts them; inference just uses them.

**Example.** "large-v3 has 1.5B parameters" = 1.5 billion learned numbers. More = more capacity to memorize accents, vocabulary, noise patterns — at the cost of memory and speed.

### Convolution
**What.** A small window sliding across data looking for one specific local pattern.

**How.** The window has learned weights. At each position it multiplies and sums, producing one number: "how strongly does my pattern appear here?" Different filters learn different patterns. Standard tool for images — and the spectrogram is an image.

### Stride
**What.** How far the convolution window hops each step.

**Example.** Stride 2 = output is half as long. Whisper's two conv layers turn **3000 frames into 1500 positions**. Each position now covers 20 ms instead of 10 ms.

### Feature / feature vector
**What.** A list of numbers describing one thing, in a form the network finds useful.

**Example.** After the encoder each of the 1500 positions is described by **1280 numbers** — no longer "how loud at each pitch" but the model's own learned notes about what is happening there.
```
position 20, first 8 of 1280 numbers:
[-0.178  -0.443  -0.168  -0.066  0.109  0.163  0.032  -0.390]
```
Meaningless to a human. Nobody can point at -0.443 and say "that is the 'ch' sound." Information is smeared across all 1280 dimensions, and the only reader is the decoder. **This is where the pipeline stops being human-inspectable.**

### Token
**What.** One piece of text. Often a word-piece, not a whole word.

**Example.** "Unhappiness" → `un` + `happi` + `ness`. Common words are single tokens; rare words split.

### Tokenizer / vocabulary
**What.** The lookup table mapping text pieces to ID numbers and back. Whisper's vocabulary is ~51,865 entries including special tokens.

**Example.** Real IDs: `' Machine'` = 22155, `' learning'` = 2539, `'.'` = 13.

### Embedding
**What.** The vector a token ID maps to.

**How.** ID 22155 indexes into a big learned table and pulls out ~1280 numbers. Similar-meaning tokens end up with similar vectors because training pushes them together. This is how a network does arithmetic on words.

---

## 2.4 — Transformers and attention

### Attention
**What.** The mechanism letting every position in a sequence look at every other position and pull in what is relevant.

**Why needed.** In "the bank was steep," the meaning of "bank" depends on "steep".

**How.** Each position emits three vectors:

| Vector | Meaning |
|---|---|
| **Query** | "what am I looking for?" |
| **Key** | "what do I offer?" |
| **Value** | "here is my actual content" |

Every query is compared against every key. Strong match = high score. Scores become weights, and each position takes a weighted blend of everyone's values.

**Analogy.** A room of people. You raise a question (query). Everyone holds up a sign saying what they know (key). You listen mostly to whoever's sign matches, and blend what they say (value).

### Multi-head attention
**What.** Running attention several times in parallel with different learned Q/K/V projections, then combining.

**Why.** One head tracks grammar, another tracks who is speaking, another tracks pitch. Whisper's encoder uses **20 heads** per layer.

### Layer
**What.** One block of attention + a small feed-forward network. Stacking builds increasingly abstract understanding.

**Example.** large-v3 encoder: **32 layers**. Turbo decoder: **4 layers**.

### Transformer
**What.** The architecture made of stacked attention layers. Same family as GPT, BERT, and most modern AI.

### Self-attention
**What.** Attention where a sequence looks at itself.

**Example.** In the decoder, the words written so far attend to each other so output stays grammatical.

### Cross-attention
**What.** Attention where one sequence looks at a *different* sequence.

**Example.** In the decoder, each word being written attends to the **encoder's audio representation**. This is the bridge — what makes the text correspond to the sound instead of being free invention.

### Masking
**What.** Blocking attention from seeing certain positions.

**Example.** The decoder is **causally masked** — while writing word 5 it sees words 1–4 but not 6 onward. Without this, the model could cheat during training by peeking at the answer.

### Encoder
**What.** The half that reads. Spectrogram in, feature vectors out. Bidirectional — every position sees every other, because the whole audio is available at once. Runs **once** per 30-second chunk.

### Decoder
**What.** The half that writes. Self-attention over what it has written, cross-attention into the audio, then a prediction. Runs **once per token**.

### Autoregressive
**What.** Generating one token at a time, feeding each output back as input for the next step.

**Example.** Write "The" → feed back → "quick" → feed both back → "brown" → ... → stop token.

### Forward pass
**What.** One run of data through a model (or half of one).

**Example — the cost asymmetry.** For an 8-token transcript:
```
encoder forward passes : 1
decoder forward passes : 8
```
The encoder ran **once** on all 30 seconds. The decoder ran **eight times** — once per token, each pass re-reading everything written so far. A 10-minute podcast is ~20 encoder passes but *thousands* of decoder passes.

**This single fact explains `turbo`:** cut the decoder from 32 layers to 4, keep the encoder whole. Gut the part that runs thousands of times, preserve the part that runs once. ~8x faster, almost no accuracy loss.

---

## 2.5 — Making a prediction

### Logits
**What.** Raw scores the model outputs — one per vocabulary entry, ~51,865 of them. Unbounded, can be negative.

### Softmax
**What.** Turns logits into probabilities. Makes everything positive and sums to 1.

**Example.** Logits `[8.2, 5.1, 3.0, ...]` → probabilities `[0.795, 0.123, 0.038, ...]`.

### Greedy decoding
**What.** Always take the highest-probability token. Equivalent to `temperature=0`.

### Temperature
**What.** A knob for randomness. 0 = always top choice, deterministic. Higher = flatten probabilities and sample, so runners-up sometimes win. Whisper starts at 0 and raises it only when its own quality checks say output looks broken.

### Beam search / best_of
**What.** Keep several candidate transcripts alive instead of committing token by token, pick the best complete one.

**Why.** A locally-best token can lead to a globally-worse sentence. Costs ~3x compute for a small gain — usually not worth it in production.

---

## 2.6 — Whisper specifics

### 30-second chunk
**What.** Whisper's input window is fixed at exactly 30 seconds — **480,000 samples**. Shorter audio is zero-padded; longer audio is cut into chunks and stitched using predicted timestamps.

**Example.** 2.89 seconds of speech becomes 46,251 real samples plus **433,749 zeros**. All that fake silence is real input the model must process — and silence is where hallucinations come from.

### Special tokens
**What.** Control tokens that configure the task. They live in the same vocabulary as words.

**Example.** The decoder's starting prompt, real IDs:
```
id  50258  '<|startoftranscript|>'
id  50259  '<|en|>'
id  50360  '<|transcribe|>'
id  50364  '<|notimestamps|>'
```
Four numbers = the entire task configuration.

**Why this is elegant.** Change 50360 to the translate token and the *identical weights* produce English text from Japanese audio. Drop `<|notimestamps|>` and it interleaves time markers. Leave the language token blank and the model predicts it — that is what "Detected language: English" means. **Language ID, translation, and timestamps are not separate features. They are one model conditioned on different prefixes.**

### Weak supervision
**What.** Training on huge amounts of approximately-labeled data instead of small amounts of perfect data.

**How.** Whisper was trained on **680,000 hours** of web audio with whatever subtitles came attached — noisy, mislabeled, accented, music in the background.

**Why it matters.** Older models trained on a few thousand clean hours scored beautifully on their own test set and fell apart on a phone call. **Scale plus mess beats small plus clean.**

### Hallucination
**What.** The model confidently writing text that was never spoken.

**How.** It learned that audio often ends with "Thanks for watching, don't forget to subscribe" — so on silence it sometimes produces exactly that. Not broken; correctly predicting what usually followed in training data.

### Model sizes
| Model | Parameters | Note |
|---|---|---|
| tiny | 39M | fast, sloppy |
| base | 74M | |
| small | 244M | |
| medium | 769M | |
| large-v3 | 1.5B | best accuracy |
| **large-v3-turbo** | 1.5B, 4-layer decoder | **usually the right pick** |

### Distillation
**What.** Training a smaller model to imitate a bigger one. `turbo` is large-v3 with the decoder cut 32 → 4 layers and re-trained to match.

### Quantization
**What.** Storing weights at lower precision (int8/int4 instead of float16) to save memory and gain speed, at small accuracy cost.

### VAD (Voice Activity Detection)
**What.** A small, cheap model answering one question over your audio: *is anyone speaking right now?* Output is just time ranges:
```
0.0s  - 1.2s   silence
1.2s  - 4.8s   SPEECH
4.8s  - 9.3s   silence
9.3s  - 12.1s  SPEECH
```
It does not know *what* was said, only *whether* something was said.

**Why it matters so much.** Whisper always processes a full 30-second window and runs the encoder on every chunk regardless of content. **Silence costs exactly as much as speech.** Real audio is full of silence:

| Audio type | Typically silent |
|---|---|
| Podcast (edited) | 5-10% |
| Meeting recording | 40-60% |
| Call center recording | 50-70% |
| Security / ambient audio | 95%+ |

On a meeting recording VAD cuts half your compute for free. Often a bigger win than quantization: int8 buys ~2x, skipping 50% of a file buys 2x *on top of that*, and it costs milliseconds to compute.

**The second benefit: it kills hallucinations.** Whisper hallucinates "Thanks for watching!" because it is asked to transcribe silence, has no acoustic evidence, and falls back on training-data priors. VAD removes the silence before Whisper sees it, so the failure mode cannot occur. If you see garbage on a long recording with gaps, VAD is the fix ~80% of the time.

**How.** `faster-whisper` bundles **Silero VAD** — a ~1 MB net that runs far faster than realtime on CPU. It emits a speech probability per short frame; you threshold it, merge nearby regions, pad the edges so word onsets are not clipped, and hand only those segments to Whisper. Timestamps come back mapped to the original file.

```python
segments, info = model.transcribe(
    "meeting.m4a",
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=500,   # how long a gap must be to count as silence
        speech_pad_ms=400,             # keep this much either side, so words aren't clipped
        threshold=0.5,                 # speech probability cutoff
    ),
)
```

**Where it goes wrong:**
- **Aggressive thresholds clip word onsets.** A quiet "the" at the start of a sentence gets cut. That is what `speech_pad_ms` is for — never set it to 0.
- **Music and noise register as speech.** Silero detects speech-like audio, not voices specifically.
- **Very quiet speakers get dropped entirely.** A soft-spoken person on a far mic can fall below threshold and vanish. **This is a silent failure** — nothing errors, the words just are not there.

That last one is why VAD belongs in the Phase 6 eval harness as its own slice: *WER with VAD on vs off, on quiet audio specifically.* A config that usually helps and occasionally deletes someone's contribution.

### Diarization
**What.** Working out *who* spoke *when*. Whisper does not do this — needs a separate model (pyannote), or the channel-splitting trick above.

### WER (Word Error Rate)
**What.** The standard accuracy metric: (substitutions + insertions + deletions) ÷ total words. Lower is better. Under 5% is good for clean speech.

### RTF (Real-Time Factor)
**What.** `processing_time ÷ audio_duration`. The core production metric. RTF 0.1 = 10x faster than realtime. RTF above 1.0 = slower than just listening to it.

---

## 2.7 — Hardware, memory, and inference engines

### The three parts of a computer that matter here

| Part | Job | Analogy |
|---|---|---|
| **Storage (SSD)** | Holds the model file permanently | The warehouse across town |
| **RAM** | Holds the weights while running | The kitchen pantry |
| **CPU / GPU** | Does the arithmetic | The cook |

The cook can only use what is in the pantry. So you drive to the warehouse **once** and stock the pantry — that is `model load: 1.19s`. You do it once per worker, never per file.

### Memory bandwidth — the usual bottleneck

The cook is incredibly fast; chopping takes a moment. But every ingredient must be carried from the pantry to the counter, and that walk is slow compared to the chopping. **So the cook mostly stands around waiting.**

That is the real situation inside a computer. Fetching a number from RAM takes ~100x longer than the multiply it feeds. The arithmetic units are usually idle, waiting for data. **Memory bandwidth** — GB/s you can move from RAM to chip — is usually what decides your speed, not raw operations per second.

Concretely: a float32 model is 4456 MB. The M4 Pro moves ~273 GB/s, so shifting ~4 GB takes ~15 ms *minimum* before any math happens. That is the floor.

### CPU vs GPU

| | CPU | GPU |
|---|---|---|
| Cores | ~10, very fast, very smart | thousands, slower, simple |
| Good at | branching, sequencing, one thing at a time | same operation on huge arrays |
| Bad at | doing 10,000 things at once | anything with `if` statements |

CPU = 10 expert chefs who can follow complicated recipes. GPU = 5,000 line cooks who each do one simple thing, all at once. Neural networks are "multiply these millions of numbers, no decisions" — line-cook work. Hence GPUs.

This is also why the **encoder** (one big parallel pass over 1500 positions) loves a GPU, while the **decoder** — sequential, each token depending on the last — benefits less. You cannot parallelize a sequence that must be produced in order.

### Unified memory vs VRAM

**Normal PC:** the GPU has its own separate memory (VRAM) on the card. Weights must be copied across the PCIe bus into VRAM before the GPU can touch them. VRAM is very fast (~1000 GB/s on an A100) but small and separate.

**Apple Silicon:** **unified memory** — CPU and GPU share the same physical RAM. No copying, no PCIe bus. Your 24 GB is available to both.

**Note:** `faster-whisper` on Mac is **CPU-only** — no Metal backend. Your GPU cores sit idle during those benchmarks. `mlx-whisper` is the one that uses them. Different tool, different hardware path, same model.

### Quantization — the arithmetic, worked

An `int8` holds only **whole numbers from -128 to 127**. It cannot hold `0.109`. So you do not store the weight — you store **how many units of a fixed size** it is, and remember the unit separately.

**Pick the unit (the scale).** The largest weight in the group should use the full range:
```
weights: [-0.443, 0.109, -0.178, 0.163]
scale = 0.443 / 127 = 0.00348819
```
Read that as: **one int8 tick = 0.00348819.** A ruler with 127 marks between zero and the largest weight.

**Going in — divide, then round:**
```
   weight     / scale   round
   -0.443    -127.000    -127
    0.109      31.248      31     <- .248 thrown away
   -0.178     -51.029     -51     <- .029 thrown away
    0.163      46.729      47     <- rounded UP
```
Stored: `[-127, 31, -51, 47]`. Four bytes instead of sixteen.

**Coming back out — multiply:**
```
   stored     x scale     result     error
     -127   x 0.00348819 = -0.443000  +0.000000
       31   x 0.00348819 =  0.108134  -0.000866
      -51   x 0.00348819 = -0.177898  +0.000102
       47   x 0.00348819 =  0.163945  +0.000945
```

The whole scheme in two lines:
```
store:     q = round(w / scale)      float -> int
retrieve:  w ~= q * scale            int -> float
```
The `~=` is the point. You never get the original back exactly. Largest error above is under 0.001 on weights around 0.1-0.4 — roughly **0.5% off**.

**The scale must be stored too** — one float32 per group (typically per row or per block of 64/128). Negligible: 4 bytes shared by 128 one-byte weights.

**Why outliers hurt.** The scale is calibrated to the largest value in the group. If one freak weight is 100x its neighbours, it eats the whole range and every other weight is squashed into a handful of ticks. Sophisticated schemes use small groups (one outlier only poisons 64 weights) or store outliers separately at full precision. This is also why `large` tolerates int8 better than `tiny` — more redundancy to absorb rounding.

### Why int8 is not automatically faster

**The number of operations is the same either way.** Same matrix, same shape, same count of multiply-adds. Precision does not change how many multiplications happen.

int8 does not do *fewer* operations — it does **extra** ones. To use an int8 weight the chip must first turn it back into a real number:
```
float32:  multiply, add                          (2 steps)
int8:     unpack weight, multiply, add, rescale  (4 steps)
```

**So why does int8 ever win?** Because of the carrying, not the chopping. int8 is **vacuum-packed ingredients** — half the weight to carry from the pantry, but you must unwrap each one before using it.

- **Long walk to the pantry** (GPU, bandwidth-starved): carrying half the weight saves far more than unwrapping costs. **Big win.**
- **Pantry right next to you** (Mac CPU, bandwidth to spare): carrying was never the problem. You just added unwrapping. **Net loss.**

**int8 trades extra compute for less memory traffic.** Good deal when memory traffic is your bottleneck, bad deal when it is not. Same trade, opposite verdict depending on the machine — see the Phase 2 results for the measurement.

Second factor on Apple Silicon: **AMX**, dedicated hardware for float32/float16 matrix math. CTranslate2's float32 path hits it; its int8 ARM path is far less tuned than the x86 AVX-512 equivalent. Both factors point the same way, hence the measured 24% gap.

### RAM sizes your concurrency

The production consequence. You are not optimizing one transcription, you are optimizing **workers per machine**:
```
float32: 4456 MB/worker  ->  24 GB fits ~5 workers
int8:    2041 MB/worker  ->  24 GB fits ~11 workers
```
Twice the throughput on the same box, even though int8 is 24% slower per stream. **Throughput and latency pulling in opposite directions — the Phase 1 tradeoff as a concrete decision.**

GPU selection in Phase 5 is mostly a memory question too: "does the model plus my batch fit in VRAM?" Right-sizing means the cheapest card where it fits, not the fastest card.

### CTranslate2

**What.** An **inference engine** — a C++ library that runs already-trained transformer models fast. Written by OpenNMT, originally for machine translation, hence the name. Whisper is an encoder-decoder transformer, the same shape, so it fit. `faster-whisper` is a thin Python wrapper around it; almost nothing happens in Python.

**Why it exists.** PyTorch is built for **training**, so it must support autograd, dynamic graphs, every experimental layer, and Python in the hot loop. All essential for training; all dead weight for running a finished model.

An inference engine throws that out. Fixed graph, known ahead of time, so it can **fuse operations** (merge steps into one kernel, avoiding memory round-trips), **reuse memory buffers**, **manage the KV cache**, **batch across requests**, and **use hand-tuned quantized kernels**. Typically **3-5x faster than PyTorch** on the same hardware. Nothing about the model changed — only the machinery running it.

**Analogy.** PyTorch is a research workshop: every tool on the wall, build anything, slow because generality costs. CTranslate2 is a factory line built for one product: cannot invent anything new, enormously faster at making the thing it makes.

**The catch — models must be converted.**
```
PyTorch weights  ->  ct2-transformers-converter  ->  model.bin + config.json
```
Quantization happens at conversion time, which is why `compute_type="int8"` loads fast — the int8 weights already exist on disk. HuggingFace hosts pre-converted models (`Systran/faster-whisper-large-v3`), so you never run the converter yourself. It is also why `int16` failed on this machine: CT2 compiles kernels per architecture, and int16 is not implemented for ARM CPU.

**Its siblings.** Every serious deployment uses one of these:

| Engine | Built for |
|---|---|
| **CTranslate2** | encoder-decoder transformers, CPU + CUDA. Whisper's practical home. |
| **ONNX Runtime** | general-purpose, very portable (see below) |
| **TensorRT** | NVIDIA only, fastest there, painful to use |
| **vLLM** | large language models, high-throughput serving |
| **llama.cpp** | LLMs on consumer hardware, aggressive quantization |
| **MLX** | Apple Silicon, uses the GPU — what `mlx-whisper` uses |

### ONNX and ONNX Runtime

Two different things sharing a name.

**ONNX — the format.** *Open Neural Network Exchange.* A vendor-neutral file format describing a trained model: layers, connections, weights. It exists because you train in PyTorch but may need to deploy on an iPhone, Android, a browser, a Windows desktop, or a microcontroller — none of which want to run PyTorch.
```
PyTorch   -+
TensorFlow +->  ONNX file  ->  iOS / Android / browser / server / edge chip
JAX       -+
```
Train anywhere, export once, run anywhere.

**ONNX Runtime — the engine.** Microsoft's engine for executing those files. Same category as CTranslate2: fixed graph, operator fusion, memory reuse, quantization, no Python in the hot loop. Its distinguishing feature is **execution providers** — pluggable backends chosen at load time:

| Provider | Hardware |
|---|---|
| CPU | anything |
| CUDA / TensorRT | NVIDIA |
| DirectML | any Windows GPU (AMD, Intel, NVIDIA) |
| CoreML | Apple |
| ROCm | AMD |
| QNN | Qualcomm mobile chips |
| WebAssembly / WebGPU | browsers |

Same `.onnx` file, same API, different hardware.

**ONNX Runtime vs CTranslate2**

| | ONNX Runtime | CTranslate2 |
|---|---|---|
| Scope | any model type | transformers only |
| Hardware reach | very wide | CPU + CUDA |
| Transformer speed | good | usually better |
| Ecosystem | huge | small, focused |

CT2 is faster for Whisper specifically because it is **specialized** — it knows it is running an encoder-decoder transformer and can hard-code assumptions ONNX Runtime cannot make about an arbitrary graph. Purpose-built beats general-purpose on the thing it was purpose-built for.

Choose ONNX Runtime when portability matters more than peak speed: Whisper inside a desktop app, in a browser, or on hardware CT2 does not support. For Branch B (Linux + NVIDIA), CT2 is the better fit — but you will meet ONNX constantly, it is the lingua franca of model portability.

**Why it is the Branch B stack:** same code on CPU and CUDA (develop on Mac, deploy on GPU, change one argument), quantization built in rather than bolted on, and batching plus KV cache handled for you — most of Phase 2's throughput work already done.

**The general lesson.** Training frameworks and inference engines are different tools. Shipping a model almost always means converting it out of the framework it was trained in. Standard step in every ML deployment, not a Whisper quirk.

---

# Part 3 — The pipeline running, with real numbers

Traced with `trace_whisper.py` on `large-v3-turbo`. Input: 2.89 seconds of *"Machine learning turns sound into text."*

```bash
.venv/bin/python trace_whisper.py your_audio.m4a
```

## Stage 1 — file becomes numbers
```
array shape : (46251,)
duration    : 2.89 s
value range : -0.4706 .. +0.7691
```

## Stage 2 — pad to exactly 30 seconds
```
before : 46,251 samples (2.89s)
after  : 480,000 samples (30.00s)
added  : 433,749 zeros
```

## Stage 3 — waveform becomes a picture
```
FFT window  : 400 samples = 25 ms per slice
hop length  : 160 samples = 10 ms between slices
mel filters : 128

480,000 numbers  ->  mel shape (3000, 128)
```

## Stage 4 — the encoder
```
mel in      : (1, 3000, 128)
encoder out : (1, 1500, 1280)
```
- **Conv halved the time axis**, 3000 → 1500, stride 2. Each position covers 20 ms.
- **Feature axis grew**, 128 → 1280, through 32 transformer layers with 20 heads each.

## Stage 5 — the decoder, token by token

Every step: its choice, its confidence, and what it *almost* said.

```
step  chose id  text            prob   runners-up
   0     22155  ' Machine'     0.795   ' machine'=0.123  ' "'=0.038
   1      2539  ' learning'    0.855   ' Learning'=0.125  '-'=0.018
   2      4523  ' turns'       0.996   ','=0.001   ' terms'=0.000
   3      1626  ' sound'       0.982   ' "'=0.006   ','=0.004
   4       666  ' into'        0.997   ' in'=0.001  ' to'=0.001
   5      2487  ' text'        0.999   ' texts'=0.000
   6        13  '.'            0.669   ','=0.171   '<|endoftext|>'=0.144
   7     50257  <|endoftext|>  0.994
```

The confidence column tells a story:

- **Step 0, 79.5%.** Its only real uncertainty is capital *Machine* vs lowercase (12.3%). It heard the sound perfectly. It cannot hear whether this starts a sentence — **capitalization is not a sound.**
- **Steps 2–5, 98–99.9%.** Nearly certain. Not because audio got clearer, but because acoustic evidence and the internal language model now agree. After "Machine learning turns sound into", almost nothing but "text" fits. **This fusion is why Whisper survives noise.**
- **Step 6, 66.9% — the interesting one.** Torn between `'.'` (66.9%), `','` (17.1%), and stopping (14.4%). **Punctuation was never spoken.** It is inferred from rhythm and grammar, so confidence drops. Odd punctuation comes from this step.
- **Step 7, 99.4%.** After a period followed by silence, ending is obvious.

---

# Part 4 — Running it

Whisper's weights are open (MIT). An API key is only needed for the OpenAI-hosted version.

| Path | Key? | Cost | Notes |
|---|---|---|---|
| OpenAI API | yes | $0.36/audio-hour | network-bound |
| mlx-whisper | no | free | Apple GPU, Mac only |
| faster-whisper | no | free | CPU + NVIDIA, production stack |

**Setup (Mac):**
```bash
brew install ffmpeg
uv tool install mlx-whisper
```

**CLI:**
```bash
mlx_whisper audio.mp3 --model mlx-community/whisper-large-v3-turbo --language en --output-format srt
```

**Python:**
```python
import mlx_whisper
r = mlx_whisper.transcribe("audio.mp3", path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
print(r["text"])
```

Weights cache to `~/.cache/huggingface` (~1.5 GB), then it runs fully offline.

## Decoding flags — what each one patches

| Flag | What it does |
|---|---|
| `temperature` | Randomness. 0 = always top choice. Raised only if output looks broken. |
| `compression_ratio_threshold` | If output compresses too well it is repetitive — model looping "the the the." Retry hotter. |
| `logprob_threshold` | Model's own confidence. Too low = guessing. Retry. |
| `no_speech_threshold` | Dedicated no-speech token. If it fires strongly, output nothing instead of hallucinating. |
| `condition_on_previous_text` | Feeds last chunk's transcript as context so names stay consistent. Downside: one bad chunk propagates. |
| `language` | Skip auto-detection. Always set it if you know — removes a whole class of errors. |
| `beam_size` | 5 costs ~3x the compute of 1 for a small WER gain. Usually leave at 1 in production. |

## Scripts in this repo

| File | What it does |
|---|---|
| `trace_whisper.py` | Traces a full transcription stage by stage with real numbers |
| `trace_mel.py` | Zooms in on waveform → spectrogram: FFT, mel buckets, pure-tone demo |
| `bench.py` | Phase 1 baseline benchmark — RTF, memory, variance |

---

# Part 5 — Branch B: Production ASR engineering

The path for shipping speech systems rather than understanding them more deeply. ~6–8 weeks part-time.

## What the job actually is

You are not making the model smarter. You answer three questions repeatedly:

1. **How many hours of audio per dollar?**
2. **How long until the user sees their transcript?**
3. **Does it still work when 500 files arrive at once?**

## Revised prerequisites — most of the "understand it" path is skippable

| Original item | Verdict for Branch B | Why |
|---|---|---|
| Mel spectrograms with librosa | **Skip** | Already seen traced. Buys nothing for throughput work. |
| nanoGPT / attention | **Skip for now** | You will not implement attention. Revisit for fine-tuning or custom decoding. |
| Whisper paper | **Skim 45 min** | Evaluation section only — tells you where the model is weak, which is what your eval harness must probe. |
| Read `openai/whisper` code | **Replace** | Read `faster-whisper`'s `transcribe.py` instead — VAD, chunking, batching are the knobs you tune. |
| Break it deliberately | **Do this FIRST** | Not a capstone. It IS the job. It becomes your eval test set. |

**Actual reading list, ~3 hours:**

1. **Break it deliberately** (1h) — silence, music, two speakers, accents, a 40-min file, background noise. Write down every failure mode. Most valuable artifact you'll make.
2. **Whisper paper, evaluation section only** (45m).
3. **`faster-whisper/transcribe.py`** (1h) — `transcribe()`, VAD segment logic, temperature-fallback loop.

**What skipping theory costs you.** You can still make it faster, cheaper, more reliable; diagnose p99 latency; choose model/precision/batch size on evidence. You cannot fine-tune on domain data, debug failures mechanistically, or modify decoding beyond exposed flags. Fine trade. If quality (not speed) becomes the blocker, that's the signal to go back and do nanoGPT with a concrete reason.

## The stack

| Layer | Choice | Why |
|---|---|---|
| Inference | **faster-whisper** (CTranslate2) | int8, batching, same code on CPU and CUDA |
| Pre-filter | **Silero VAD** | skip silence — often 30–50% of real audio |
| Serving | **FastAPI** + a queue | async, boring, well-documented |
| Queue | **Redis + RQ** or Celery | survives restarts, gives retries |
| Storage | S3/R2 for audio, Postgres for job state | |
| Metrics | Prometheus + Grafana, or structured logs first | cannot optimize what you don't measure |

**Platform note.** Production ASR runs on Linux + NVIDIA. MLX is Apple-only. Develop locally on CPU, deploy the same code on a GPU box — CTranslate2 runs both.

## Phase 0 — Prerequisites

Python (asyncio, typing), HTTP, Linux CLI, Docker, git. Working knowledge, not expertise. Don't learn these first — learn them when a phase forces you to.

## Phase 1 — Measurement literacy (week 1)

Everything after this is optimization, and optimization without measurement is superstition.

**Learn:**
- **RTF** = `processing_time / audio_duration`. The core metric.
- **Throughput vs latency** — they trade off. Batching improves one, hurts the other.
- **Percentiles, not averages.** p50/p95/p99.
- **Honest benchmarking** — warmup runs, multiple repeats, report variance not the best number.
- **Cost per audio hour** — the number that decides build-vs-buy.

**Read:** Brendan Gregg, the USE method (Utilization, Saturation, Errors).

**Build:** `bench.py` — transcribes a test set, reports RTF, peak RSS, variance.

**Done when:** you can state your baseline RTF and $/audio-hour, and explain the variance.

### Phase 1 results — my machine (M4 Pro, 24 GB, macOS)

Baseline, `bench.py`, large-v3-turbo / int8 / CPU / beam 1 / no VAD / 10 threads, 3 synthetic clips:

| model | precision | threads | VAD | RTF | mem | notes |
|---|---|---|---|---|---|---|
| large-v3-turbo | int8 | 10 | off | **0.542** | 2256 MB | synthetic clips, CPU-only |

```
file             audio     proc     RTF   x-real   stdev
long            81.11s   44.21s   0.545     1.8x  0.815s
medium           8.79s    3.14s   0.357     2.8x  0.101s
short            2.89s    2.91s   1.006     1.0x  0.066s
TOTAL           92.79s   50.26s   0.542     1.8x
```

**Finding 1 — thread count was costing 4-5x, and variance was the tell.**
The first run looked terrible *and unstable*: an 8-second clip took 8s to 17s across repeats. CTranslate2 defaults to all 14 logical cores; the M4 Pro has 10 performance + 4 efficiency cores, so every batch waits on the slowest core while the OS shuffles threads between them.

```
threads=14 (default)  runs: 8.1  16.1  13.3  9.0  17.3  14.2   <- 2x swing
threads=10 (P-cores)  runs: 3.0   3.0   3.0                    <- 5x faster, no variance
threads= 4            runs: 4.9   4.9   4.9
```

**The Phase 1 lesson exactly: the first number you measure is usually wrong, and variance is the signal that tells you so.** Reporting RTF 0.675 and moving on would have made every Phase 2 comparison noise on top of a broken baseline.

**Finding 2 — the short file has RTF 1.006. It costs as much as it lasts.**
2.89s of audio, 2.91s to process. That is the 30-second window showing up as a bill: the encoder always processes 480,000 samples whether you gave it 3 seconds or 30. For short clips you pay almost entirely for padding. **Implication: batching short files matters enormously** — processing voice memos one at a time wastes ~90% of the compute.

**Finding 3 — aggregate RTF hides a 3x spread** (0.357 to 1.006). Quoting "RTF 0.542" describes no file actually run. Report the distribution, not the mean.

**Open question for Phase 2 — optimal thread count depends on file length.**
The long file got *slower* with 10 threads (37.7s -> 44.2s) while short and medium got 3-5x faster. Both runs had low variance, so it is probably real. Hypothesis: long audio splits into many 30s chunks that parallelize across all 14 cores, while short audio is one chunk where E-core scheduling only hurts. Confirm with a proper sweep before assuming.

**Two caveats on these numbers.**
1. **The test set is synthetic.** `say` output has no background noise, no accent, perfect articulation, no crosstalk. Every WER number from it is optimistic and every VAD number is meaningless (no silence to skip). Replace with real recordings before trusting conclusions.
2. **faster-whisper on Mac is CPU-only.** No Metal support — that is why mlx-whisper felt faster. Fine for development; the same code runs on CUDA when deployed. Do not mistake local RTF for production RTF.

## Phase 2 — Inference optimization (week 2)

**Learn:** quantization (float32 → float16 → int8), runtime formats (CTranslate2, ONNX, TensorRT), batching, KV cache, beam size cost, VAD.

**Build:** the comparison table — model size x precision x VAD, measuring RTF, WER, peak memory.

**Done when:** you can defend a config choice with your own numbers instead of vibes.

### Phase 2 results — precision sweep (SUPERSEDED, see below)

`large-v3-turbo`, medium clip (8.79s), beam 1, 10 threads, 4 reps:

```
precision          median     RTF   peakMB  transcript match
int8                3.07s   0.349     2041  identical
int16             unsupported on ARM CPU
float32             2.34s   0.266     4456  identical
```

**float32 is 24% FASTER than int8 here** — the opposite of the general rule. Transcripts byte-identical, so int8 cost nothing in quality; it just bought nothing in speed either.

**Why the rule failed.** "int8 is faster" assumes **memory bandwidth is the bottleneck**. True on NVIDIA GPUs, where compute massively outpaces memory. Not true here: the M4 Pro has ~273 GB/s unified bandwidth that a CPU workload does not saturate, Apple's float32 path (Accelerate + AMX) is heavily optimized, and CTranslate2's int8 ARM kernels carry dequantization overhead without the tuning its x86 AVX-512 paths get. You pay the quantization tax and get none of the bandwidth benefit.

**What to actually use:**

| Where | Choice | Why |
|---|---|---|
| Mac, development | **float32** | 24% faster; 4.4 GB fits fine in 24 GB |
| NVIDIA GPU, production | **int8 / float16** | genuinely bandwidth-bound there |
| Memory-constrained box | **int8** | 2041 MB vs 4456 MB = 2x the workers per machine |

That last row is the real production tradeoff: you optimize **workers per machine**, not one transcription. Halving memory per worker doubles concurrency, which usually beats a 24% single-stream gain.

**The meta-lesson.** A confident, widely-repeated rule was wrong on this hardware. Twenty minutes of measurement beat the heuristic. **Treat every optimization claim in these notes as a hypothesis until `bench.py` confirms it on your hardware with your audio.**

### Phase 0 results — break it deliberately

Ran 10 adversarial cases (`make_hard_testset.py` + `break_it.py`), each with VAD off and on.

**1. Silence hallucinated — confirmed.**
```
01_silence_30s  [vad off]  ->  'Thank you.'
01_silence_30s  [vad on ]  ->  ''            (and 3.5x faster: 2.8s -> 0.8s)
```
30 seconds of digital silence produced words. Training-data prior firing with zero acoustic evidence.

**2. The best finding — VAD off DUPLICATED a sentence.**
```
02_speech_in_silence  [vad off]  ->  'The patient was prescribed 40 milligrams of atorvastatin daily.
                                      The patient was prescribed 40 milligrams of atorvastatin daily.'
02_speech_in_silence  [vad on ]  ->  'The patient was prescribed 40 milligrams of atorvastatin daily.'
```
The audio contains that sentence **once**, at t=20s inside 60s of silence. Whisper emitted it **twice** — the chunk-seam problem made visible: the sentence sits near a 30s boundary, appears in two chunks, and `condition_on_previous_text` carried it forward. **In a medical or legal transcript a duplicated dosage line is a serious error, and nothing about the output looks wrong.**

The `LOOPING` detector in `break_it.py` missed it (2 repetitions, threshold is 3). A real false negative in the harness — fix before Phase 6.

**3. Stereo downmix corrupted a word.**
```
06_stereo_two_speakers  ->  'atorvastatin data'    (should be 'daily')
```
Not dropped — **replaced**. Silent corruption is worse than an obvious failure.

**4. Second speaker vanished entirely.**
```
05_overlapping  ->  'The patient was prescribed 40 mg of atorvastatin daily.'
```
Speaker B's sentence is simply absent. No diarization, no warning, no partial capture.

**5. Normalization is inconsistent — a Phase 6 landmine.**
```
'40 milligrams'   <- cases 02, 03, 04, 07
'40 mg'           <- cases 05, 06, 09, 10
```
Identical content, two spellings, decided by acoustic conditions. If the WER harness does not normalize this, you measure noise and call it signal.

**Where predictions were wrong.** Very quiet speech (4% volume) transcribed perfectly with VAD on — Silero is more sensitive than expected. Noise, clipping and 320 wpm all transcribed perfectly too. Three of ten "adversarial" cases were not adversarial at all. Useful: it locates the real edges rather than the assumed ones.

**RTF is monotonic with file length.**
```
07_fast_speech    2.3s audio  ->  RTF 1.25   <- worst
03/04/05/10       3.9s audio  ->  RTF ~0.75
09_long_repeated  161s audio  ->  RTF 0.21   <- best
```
Every short file pays the full 30-second padding cost; long files amortize it. Same model, same machine, **6x RTF difference driven purely by input length.** If your workload is short clips, batching is not an optimization — it is the difference between viable and not.

### Phase 1 results — real audio

Two 30-second mp3 clips (44.1 kHz stereo, edited explainer videos):

| test set | config | RTF | speed |
|---|---|---|---|
| real audio | int8, vad off | **0.166** | 6.0x |
| real audio | int8, vad on | 0.174 | 5.8x |
| synthetic (2.9-81s) | int8 | 0.542 | 1.8x |

**Real audio is 3x better than the synthetic baseline** — 30-second files fill the window instead of paying for padding.

**VAD made real audio slightly SLOWER** (0.166 -> 0.174). These are dense edited clips with no silence to skip, so VAD costs a little and saves nothing. Compare with the sparse cases above, where it was up to 3.5x faster and fixed both hallucination and duplication. **VAD's value is entirely a property of your audio, not of the model.**

**Two same-length files differed 2.2x in RTF** (0.102 vs 0.228). Both ~30s, but one has far more speech. Decoder cost scales with **token count**, not audio duration — so RTF depends on how much was said, not how long the file is. Another reason a single aggregate RTF is a poor summary.

### Phase 2 results — CORRECTED, on real audio

The earlier "float32 beats int8" conclusion came from **one 8-second synthetic clip**. It did not survive contact with real audio. Keeping both results here because the correction is the lesson.

**Supported compute types on this machine** (ARM CPU):
```
int8           ok
int8_float32   ok
float32        ok
int8_float16   unsupported
float16        unsupported
bfloat16       unsupported
```

**Same real file (30.12s), int8 vs float32:**
```
file            ct        words   median     RTF
synthetic 8.8s  int8         23    2.86s   0.325
synthetic 8.8s  float32      23    2.12s   0.242   <- float32 wins, matches old finding
real 30s        int8        105    6.64s   0.221
real 30s        float32     216   24.36s   0.809   <- float32 4x SLOWER, and 2x the words
```

**Why the word counts differ — float32 hallucinated 28 seconds of content that does not exist.**

The audio is 30.12s. Whisper's window is 30s, so there is a second chunk holding 0.12s of audio and 29.88s of padding — near-pure silence.

```
int8      last segment: [29.00 -> 30.00] "Oh."                  <- stops at the end
float32   last segment: [55.00 -> 58.00] "I'll do it, but..."    <- 28s past the end
```

Invented segments from the float32 run:
```
[30.00 -> 32.00] Let me show you.
[32.00 -> 34.00] I'm Wrestle to win it right now.
[38.00 -> 39.00] Why, you won't be at that place.
[50.00 -> 53.00] It's an angle of art.
[55.00 -> 58.00] I'll do it, but you'll have time to将 it out.
```
That last one contains a Chinese character. Same failure mode as adversarial case 01: silence produces invented text.

**VAD did NOT fix it** (hypothesis tested and rejected): float32 with `vad_filter=True` still ran to 58.00s and produced *more* words (216 vs 165).

**Two independent effects, separated by disabling the fallback:**
```
float32, temperature=[0.0]  (greedy only, no fallback):
  run 0:   5.67s  188 words  ends 58.00s
  run 1:   5.82s  188 words  ends 58.00s     <- deterministic
  run 2:   5.95s  188 words  ends 58.00s

float32, default temperature ladder (fallback enabled):
  run 0:  17.52s  116 words  ends 58.00s
  run 1:  23.71s  196 words  ends 57.98s     <- random every run
  run 2:  25.16s  182 words  ends 56.00s
```

**Effect 1 — the temperature fallback costs 3-4x and destroys reproducibility.** This audio is a novelty song repeating "30 seconds long". The **compression-ratio check** sees the repetition, concludes the model is looping, and retries at higher temperature — which means random sampling. Different dice, different garbage, every run. **The safeguard designed to prevent looping is itself producing unstable output.** For reproducible transcripts, set `temperature=[0.0]` and accept worse handling of genuinely broken audio.

**Effect 2 — float32 hallucinates past the audio end regardless of the fallback.** Greedy-only still ends at 58.00s. int8 never does this on this file.

**Which is actually faster?** Per token, float32 genuinely is faster on this hardware — the original finding holds:
```
float32 greedy:  188 words in 5.67s  ->  30 ms/word
int8:            105 words in 6.50s  ->  62 ms/word
```
But 83 of those words were invented. **Faster at producing the wrong answer.**

**Corrected recommendation: use int8.** Not because it is faster per token — it is not — but because on real audio it stayed correct, stayed under the fallback thresholds, and finished in 6.5s while float32 took 25s producing fabricated content.

**The deeper finding: precision does not only change speed, it changes the decoding path.** Tiny numerical differences flip threshold decisions in the no-speech and compression-ratio logic, which cascade into completely different output. Not a documented tradeoff in any table — it only appears when you run your own audio.

### Where the time actually goes

Per-stage timing on the real 30s clip (`stage_timing.py`):
```
stage                                        seconds   share
1. ffmpeg decode (mp3 -> numbers, mono 16k)   0.088s    1.8%
2. pad to 30s                                 0.000s    0.0%
3. spectrogram (FFT + mel + log)              0.002s    0.0%
4. ENCODER (conv + 32 transformer layers)     1.045s   21.6%
5. DECODER (137 tokens, one pass each)        3.700s   76.5%
------------------------------------------------------------
TOTAL                                         4.836s      RTF 0.161
```
**The decoder is three quarters of the work**, at 27 ms per token. The parts you would expect to be slow — mp3 decode, spectrogram — are free (0.088s and 0.002s). This is the cost asymmetry in production data, and the reason `turbo` shrinks the decoder and leaves the encoder alone.

### Quantization internals, measured

Real weight matrix, encoder block 0: `(5120, 1280)` = 6,553,600 weights (`quant_demo.py`).

**The algorithm is two lines:**
```python
scale = np.abs(w).max(axis=-1, keepdims=True) / 127.0   # one scale per ROW
q     = np.round(w / scale).astype(np.int8)
```

**Storage:**
```
float32 : 26,214,400 bytes  (26.2 MB)
int8    :  6,553,600 bytes  ( 6.6 MB)
+ scales:     20,480 bytes  (one float32 per row -> 0.3% overhead)
                             -> 3.99x smaller
```
One scale **per row**, not per matrix. 5120 rows, 5120 scales. Small groups mean a single outlier only damages its own row.

**Real weights going through it:**
```
    original    / scale  stored int8     x scale       error
    0.000308       0.43            0    0.000000   -0.000308   <- vanished
   -0.027359     -37.98          -38   -0.027375   -0.000016   <- nearly exact
   -0.005474      -7.60           -8   -0.005763   -0.000289
   -0.000093      -0.13            0    0.000000   +0.000093   <- vanished
```
`0.000308 / scale = 0.43` ticks — **less than one tick**, so it rounds to 0. That weight is gone. **Tiny weights disappear; big weights survive almost exactly.**

**Error, and why it works anyway:**
```
weights: mean error 2.56e-04, max 1.80e-03  ->  1.95% average
layer OUTPUT difference:                        1.14%
```
Weights off by 1.95%, output off by only 1.14%. **The errors partially cancel** — a dot product sums thousands of terms, some rounded up, some down. You are summing random errors, which grow far slower than the signal. That is why quantization works at all.

**The matmul — weights are never unpacked back to float:**
```python
# float path
y = x @ W.T

# int8 path
xq  = round(x / x_scale)                    # quantize the input once
acc = xq.astype(int32) @ q.T.astype(int32)  # int8 x int8, accumulate in int32
y   = acc * x_scale * w_scale               # ONE rescale, at the very end
```
The multiply-add runs entirely in integers. You rescale once per output value, not once per multiply. So int8's overhead is smaller than "unpack every weight" would suggest: quantize the input, plus one rescale.

**What CTranslate2 actually runs** — hand-written C++ dispatching on CPU features:

| Hardware | Instruction | Does |
|---|---|---|
| Intel/AMD | `VPDPBUSD` (AVX512-VNNI) | 64 int8 multiply-adds into int32, one instruction |
| ARM (M-series) | `SDOT` (NEON) | 16 int8 multiply-adds, one instruction |
| Older x86 | `VPMADDUBSW` + widening | 2 instructions, slower |

**4x less int8 throughput per instruction on ARM than on AVX512**, while float32 has Apple's AMX matrix units behind it. That is the hardware-level reason int8 does not win on speed here.

Quantized weights are baked into `model.bin` at conversion time — nothing is quantized at load. `compute_type="int8"` selects which file and which kernels.

### int8 vs float32 — the number formats

**int8** — 8 bits, evenly spaced whole numbers: `-128 .. 127`. 256 values, gap of exactly 1.

**int32** — `-2^31 .. 2^31-1`, about ±2.1 billion. Still evenly spaced.

**float32** — 32 bits split three ways:
```
[1 sign] [8 exponent] [23 mantissa]
```
Stored as `mantissa x 2^exponent` — binary scientific notation.
- Range: ±3.4 x 10^38 (vastly bigger than 2^32)
- Precision: ~7 decimal digits
- **Spacing is NOT even.** Near 1.0 consecutive floats are ~0.0000001 apart; near 1,000,000 they are ~0.06 apart. The gaps grow with magnitude.

int8 gives 256 evenly-spaced slots. float32 gives ~4 billion unevenly-spaced ones, **packed densely near zero** — which is exactly where neural network weights live.

## Phase 3 — Scaling out (weeks 3–4)

Ordinary distributed-systems engineering. The model is incidental.

**Learn:** worker pools (load model once per worker), queues, **backpressure** (bounded queues — without them, work arriving faster than you process it means OOM), **idempotency** (file 847 crashes and retries; must not duplicate), **checkpointing**, at-least-once delivery, dead letter queues.

**Build:** batch-process 1,000 files unattended. Kill it halfway, restart, verify resume.

**Done when:** you can `kill -9` a worker mid-run and lose nothing.

## Phase 4 — Serving (weeks 4–5)

**Learn:**
- **The job API pattern** — `POST` returns a job ID immediately; `GET /jobs/{id}` polls. **Never transcribe inside the request handler.** One 10-minute file blocks a worker for a minute; ten concurrent uploads and you're dead.
- **Async vs sync in FastAPI** — CPU-bound work in an `async def` handler blocks the event loop.
- **Timeouts, retries with exponential backoff + jitter.** Without jitter, retries synchronize and you DDoS yourself.
- **Graceful shutdown** — SIGTERM, drain in-flight jobs, then exit.
- **Health vs readiness probes** — "process alive" and "able to serve" are different questions.
- **Rate limiting and upload caps** — someone will upload a 4-hour file.

**Build:** the API, then hit it with 50 concurrent uploads via `hey` or `locust`.

## Phase 5 — Deployment (weeks 5–6)

**Learn:** Docker for GPU (`nvidia-container-toolkit`; expect to lose an afternoon to CUDA/cuDNN mismatches), **cold start** (model load is 5–50s; autoscaling that ignores this times out), **spot vs on-demand** (60–90% cheaper, can vanish with 2 min notice — your Phase 3 checkpointing makes spot safe), GPU selection (T4 / L4 / A10G / A100 — biggest is rarely right).

**Practice cheaply on:** Modal (serverless GPU, pay per second), RunPod, Lambda Labs. Avoid raw EC2 to start.

## Phase 6 — Observability and evals (weeks 6–7)

The phase that separates professionals from hobbyists. Do not skip.

**Learn:** structured logging (JSON with job IDs), metrics vs logs vs traces, the **RED method** (Rate, Errors, Duration), **SLOs** (alert on "p95 under 2x realtime", not CPU graphs), **WER computation** with `jiwer`, and the hard part — **normalization** (is "don't" vs "do not" an error? casing? punctuation? digits vs words? Decide, document, never change silently), **sliced evaluation** (aggregate WER hides everything).

**Build:** eval harness — ~50 files with reference transcripts, WER overall and per slice, one command.

**Done when:** you change a config, run one command, and know within a minute whether you improved things.

## Phase 7 — Cost engineering (ongoing)

Model right-sizing, batch windows, caching identical audio, spot fleets, and the build-vs-buy crossover — below a few hundred audio-hours/month the API wins once you price your own time.

## Reading list — deliberately short

1. faster-whisper + CTranslate2 docs — your actual manual
2. The 12-Factor App — 30 minutes
3. Brendan Gregg, USE method — one article
4. Google SRE Book, ch. 4 (SLOs) and ch. 6 (monitoring) — free online, those two only
5. *Designing Data-Intensive Applications* — optional reference

## Deliberately skip

Kubernetes (until you have more than one service), microservices, model training, custom CUDA kernels, and every "MLOps platform" until you've felt the pain it claims to solve.

## What proves you can do this

Not certificates. One repo containing: a benchmark table with your own numbers, a service that survives load testing, an eval harness with sliced WER, and a README stating RTF and cost per audio hour. **That README is the credential.**

---

# One-page summary

```
audio file
  |
  | ffmpeg: downmix to mono, resample to 16 kHz
  v
480,000 numbers (30 seconds, zero-padded)
  |
  | slice into 3000 windows of 25 ms, stepping 10 ms
  | each window: FFT -> 201 frequencies -> mel filterbank -> 128 buckets -> log
  v
log-mel spectrogram, shape (3000, 128)   <- a picture of the sound
  |
  | 2 conv layers, stride 2
  v
1500 positions
  |
  | ENCODER: 32 transformer layers, 20 heads. Runs ONCE.
  v
1500 x 1280 feature vectors               <- the model's understanding
  |
  | DECODER: self-attention over text so far
  |          + cross-attention into the audio features
  |          starts from <|sot|> <|en|> <|transcribe|> <|notimestamps|>
  |          runs ONCE PER TOKEN
  v
tokens -> text
```
