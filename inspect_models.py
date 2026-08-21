"""E1.2 — where did Moonshine's parameters go? No timing, just look."""
import numpy as np, torch
from collections import defaultdict

def params_by_part(named):
    tot = defaultdict(int); grand = 0
    for n, p in named:
        c = p.numel() if hasattr(p, "numel") else p.size
        grand += c
        part = "encoder" if "encoder" in n else ("decoder" if "decoder" in n else "other")
        tot[part] += c
    return tot, grand

def show(title, tot, grand, cfg_lines, pos_hits, attn):
    print(f"\n{'='*62}\n{title}\n{'='*62}")
    print(f"total params : {grand:,}")
    for k in ("encoder", "decoder", "other"):
        if tot.get(k):
            print(f"  {k:<9}: {tot[k]:>12,}  ({tot[k]/grand*100:>5.1f}%)")
    print("\nconfig:")
    for l in cfg_lines: print(f"  {l}")
    print(f"\npositional embedding tensors in encoder: {pos_hits if pos_hits else 'NONE FOUND'}")
    print(f"encoder attention span: {attn}")

# ---------------- Whisper ----------------
from transformers import WhisperForConditionalGeneration, AutoProcessor
w = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
wt, wg = params_by_part(w.named_parameters())
wc = w.config
pos = [n for n, _ in w.named_parameters() if "embed_positions" in n and "encoder" in n]
show("WHISPER tiny.en", wt, wg, [
    f"encoder layers {wc.encoder_layers}, heads {wc.encoder_attention_heads}, d_model {wc.d_model}",
    f"decoder layers {wc.decoder_layers}, heads {wc.decoder_attention_heads}",
    f"max_source_positions {wc.max_source_positions}  <- 1500 = 30s of audio, FIXED",
    f"num_mel_bins {wc.num_mel_bins}",
    f"vocab {wc.vocab_size}",
], pos, f"global — all {wc.max_source_positions} positions attend to each other")

# ---------------- Moonshine ----------------
from transformers import MoonshineForConditionalGeneration
m = MoonshineForConditionalGeneration.from_pretrained("moonshine-ai/moonshine-tiny")
mt, mg = params_by_part(m.named_parameters())
mc = m.config
mpos = [n for n, _ in m.named_parameters() if ("embed_positions" in n or "pos_emb" in n) and "encoder" in n]
show("MOONSHINE tiny", mt, mg, [
    f"encoder layers {mc.encoder_num_hidden_layers}, heads {mc.encoder_num_attention_heads}, hidden {mc.hidden_size}",
    f"decoder layers {mc.decoder_num_hidden_layers}, heads {mc.decoder_num_attention_heads}",
    f"max_position_embeddings {getattr(mc,'max_position_embeddings','n/a')}",
    f"vocab {mc.vocab_size}",
    f"rope_theta {getattr(mc,'rope_theta','n/a')}   <- RoPE = relative, no fixed length",
], mpos, "see config above")

# ---------------- side by side ----------------
print(f"\n{'='*62}\nSIDE BY SIDE\n{'='*62}")
print(f"{'':<22}{'whisper tiny.en':>18}{'moonshine tiny':>18}")
print("-"*58)
print(f"{'total params':<22}{wg:>18,}{mg:>18,}")
print(f"{'  encoder':<22}{wt['encoder']:>18,}{mt['encoder']:>18,}")
print(f"{'  decoder':<22}{wt['decoder']:>18,}{mt['decoder']:>18,}")
print(f"{'encoder layers':<22}{wc.encoder_layers:>18}{mc.encoder_num_hidden_layers:>18}")
print(f"{'decoder layers':<22}{wc.decoder_layers:>18}{mc.decoder_num_hidden_layers:>18}")
print(f"{'hidden size':<22}{wc.d_model:>18}{mc.hidden_size:>18}")
print(f"{'vocab':<22}{wc.vocab_size:>18,}{mc.vocab_size:>18,}")
print(f"{'enc % of model':<22}{wt['encoder']/wg*100:>17.1f}%{mt['encoder']/mg*100:>17.1f}%")
print(f"{'dec % of model':<22}{wt['decoder']/wg*100:>17.1f}%{mt['decoder']/mg*100:>17.1f}%")
