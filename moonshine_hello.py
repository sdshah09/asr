"""Step 0: get one transcript out of Moonshine. Nothing else."""
import sys
import numpy as np
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration
from mlx_whisper import audio as A          # reuse the ffmpeg loader we already have

F = sys.argv[1] if len(sys.argv) > 1 else "dictation.aiff"
REPO = sys.argv[2] if len(sys.argv) > 2 else "moonshine-ai/moonshine-base"

wav = np.array(A.load_audio(F))              # 16 kHz mono float32 (mlx -> numpy)
print(f"audio: {F}  ({len(wav)/16000:.2f}s, {len(wav):,} samples)")

processor = AutoProcessor.from_pretrained(REPO)
model = MoonshineForConditionalGeneration.from_pretrained(REPO)
model.eval()

inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
print(f"model input shape: {tuple(inputs['input_values'].shape)}   <- note: NOT padded to 30s")

with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=128)

print("\ntranscript:", processor.batch_decode(ids, skip_special_tokens=True)[0])
