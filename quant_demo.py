"""What quantization actually does, on a real Whisper weight matrix."""
import numpy as np, mlx.core as mx
from mlx_whisper import load_models

model = load_models.load_model("mlx-community/whisper-large-v3-turbo", dtype=mx.float32)
W = np.array(model.encoder.blocks[0].mlp1.weight.astype(mx.float32))   # a real layer
print(f"real weight matrix from encoder block 0: {W.shape}  ({W.size:,} weights)\n")

# ---------------------------------------------------------------- the algorithm
def quantize(w, axis=-1):
    scale = np.abs(w).max(axis=axis, keepdims=True) / 127.0    # one scale per row
    q = np.round(w / scale).astype(np.int8)                    # <- the whole thing
    return q, scale.astype(np.float32)

def dequantize(q, scale):
    return q.astype(np.float32) * scale

q, scale = quantize(W)
W2 = dequantize(q, scale)

print("=== STORAGE ===")
print(f"float32 : {W.nbytes:>10,} bytes   ({W.nbytes/1e6:.1f} MB)")
print(f"int8    : {q.nbytes:>10,} bytes   ({q.nbytes/1e6:.1f} MB)")
print(f"+ scales: {scale.nbytes:>10,} bytes   (one float32 per row of {W.shape[1]})")
print(f"total   : {q.nbytes+scale.nbytes:>10,} bytes"
      f"   -> {W.nbytes/(q.nbytes+scale.nbytes):.2f}x smaller\n")

print("=== ONE ROW, BEFORE AND AFTER ===")
r = 0
print(f"scale for row 0 = {scale[r,0]:.8f}   (= max|w| / 127 = {np.abs(W[r]).max():.6f} / 127)")
print(f"{'original':>12}{'/ scale':>11}{'stored int8':>13}{'x scale':>12}{'error':>12}")
for i in range(6):
    w = W[r, i]; qi = q[r, i]; back = W2[r, i]
    print(f"{w:>12.6f}{w/scale[r,0]:>11.2f}{qi:>13d}{back:>12.6f}{back-w:>+12.6f}")

err = np.abs(W2 - W)
print(f"\nover all {W.size:,} weights: mean error {err.mean():.2e}, "
      f"max error {err.max():.2e}")
print(f"relative: {err.mean()/np.abs(W).mean()*100:.2f}% average\n")

# ---------------------------------------------------------------- matmul both ways
print("=== WHAT A MATMUL LOOKS LIKE ===")
x = np.random.randn(1, W.shape[1]).astype(np.float32)

y_f32 = x @ W.T                                    # float path

xq_scale = np.abs(x).max() / 127.0                 # int path: quantize input too
xq = np.round(x / xq_scale).astype(np.int8)
acc = xq.astype(np.int32) @ q.T.astype(np.int32)   # int8 x int8 -> int32 accumulator
y_int = acc.astype(np.float32) * xq_scale * scale.T[0]   # rescale once at the end

print("float32 path:  y = x @ W.T")
print("int8    path:  y = (xq @ qW.T) * x_scale * w_scale")
print("               ^ integer multiply-add, ONE rescale at the end\n")
d = np.abs(y_f32 - y_int)
print(f"output difference: mean {d.mean():.2e}, max {d.max():.2e}")
print(f"relative to output magnitude: {d.mean()/np.abs(y_f32).mean()*100:.3f}%")
