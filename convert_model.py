import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import zipfile
import io
import numpy as np
import h5py

print("Step 1: Reading Keras 3 weights from .keras zip...", flush=True)
z = zipfile.ZipFile("Artifacts/BiGRU_model.keras")
wdata = {}

def collect(name, obj):
    if isinstance(obj, h5py.Dataset):
        wdata[name] = obj[()]

with h5py.File(io.BytesIO(z.read("model.weights.h5")), "r") as f:
    f.visititems(collect)

non_opt = {k: v for k, v in wdata.items() if not k.startswith("optimizer")}
for k, v in sorted(non_opt.items()):
    print(f"  {k}: {v.shape}", flush=True)

def vars_under(prefix):
    res = {}
    for k, v in wdata.items():
        if k.startswith(prefix) and "/vars/" in k:
            idx = int(k.split("/vars/")[-1].split("/")[0])
            res[idx] = v
    return [res[i] for i in sorted(res)]

print("\nStep 2: Building Keras 2.15 model...", flush=True)
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, GRU, Dropout, Dense

model = Sequential([
    Embedding(10000, 300, input_length=50),
    Bidirectional(GRU(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(GRU(64)),
    Dropout(0.3),
    Dense(6, activation="softmax"),
])
model.build((None, 50))
print("  Architecture built.", flush=True)

print("\nStep 3: Loading weights by position...", flush=True)
emb = vars_under("layers/embedding")
print(f"  Embedding: {[v.shape for v in emb]}", flush=True)
model.layers[0].set_weights(emb)

bi0_fwd = vars_under("layers/bidirectional/forward_layer/cell")
bi0_bwd = vars_under("layers/bidirectional/backward_layer/cell")
print(f"  BiGRU-1 fwd: {[v.shape for v in bi0_fwd]}", flush=True)
print(f"  BiGRU-1 bwd: {[v.shape for v in bi0_bwd]}", flush=True)
model.layers[1].set_weights(bi0_fwd + bi0_bwd)

bi1_fwd = vars_under("layers/bidirectional_1/forward_layer/cell")
bi1_bwd = vars_under("layers/bidirectional_1/backward_layer/cell")
print(f"  BiGRU-2 fwd: {[v.shape for v in bi1_fwd]}", flush=True)
print(f"  BiGRU-2 bwd: {[v.shape for v in bi1_bwd]}", flush=True)
model.layers[3].set_weights(bi1_fwd + bi1_bwd)

dns = vars_under("layers/dense")
print(f"  Dense: {[v.shape for v in dns]}", flush=True)
model.layers[5].set_weights(dns)

print("\nStep 4: Sanity check...", flush=True)
dummy = np.zeros((1, 50), dtype=np.int32)
out = model.predict(dummy, verbose=0)
print(f"  Output: {out}, sum={out.sum():.4f} (should be 1.0)", flush=True)

print("\nStep 5: Saving as .h5...", flush=True)
model.save("Artifacts/BiGRU_model_v2.h5")
print("DONE: Artifacts/BiGRU_model_v2.h5 created!", flush=True)
