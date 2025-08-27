#!/usr/bin/env python3
"""
gpu_test.py — quick PyTorch vs DirectML (Intel Iris Xe) sanity + speed check.

Run:
  python gpu_test.py
  python gpu_test.py --batches 30 --batch-size 8 --image-size 160
"""

import argparse
import time
import traceback
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# DirectML (privateuseone) detect
# ---------------------------
HAS_DML = False
DML = None
DML_NAME = None
try:
    import torch_directml as _dml
    if _dml.device_count() > 0:
        HAS_DML = True
        DML = _dml
        try:
            DML_NAME = _dml.device_name(0)
        except Exception:
            DML_NAME = "DirectML Adapter 0"
except Exception:
    HAS_DML = False
    DML = None
    DML_NAME = None


# ---------------------------
# Models
# ---------------------------
class SmallCNN(nn.Module):
    """Tiny CNN: conv -> conv -> GAP -> linear."""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn3   = nn.BatchNorm2d(128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = x.mean(dim=(2,3))        # global avg pool
        x = self.classifier(x)
        return x


class SmallMLP(nn.Module):
    """Tiny MLP to stress matmul/GEMM."""
    def __init__(self, in_dim=4096, hidden=2048, out_dim=1000, depth=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Helpers
# ---------------------------
def device_str(dev):
    if isinstance(dev, torch.device):
        return f"{dev.type}:{dev.index if dev.index is not None else 0}"
    return str(dev)

def _maybe_channels_last(x, memory_format):
    # Only apply channels_last to 4D image-like inputs
    if memory_format is not None and x.dim() == 4:
        return x.contiguous(memory_format=memory_format)
    return x

def do_forward_timing(model, make_input, device, batches=20, autocast=None, memory_format=None):
    # IMPORTANT: on DML keep model parameters contiguous; don't set memory_format on the model
    model = model.to(device).eval()

    # warm-up
    try:
        x = _maybe_channels_last(make_input(device), memory_format)
        with torch.no_grad():
            if autocast:
                with torch.autocast(device_type=autocast["device_type"], dtype=autocast["dtype"]):
                    _ = model(x)
            else:
                _ = model(x)
    except Exception as e:
        return {"ok": False, "error": f"warmup failed: {e}"}

    times = []
    with torch.no_grad():
        for _ in range(batches):
            x = _maybe_channels_last(make_input(device), memory_format)
            t0 = time.perf_counter()
            if autocast:
                with torch.autocast(device_type=autocast["device_type"], dtype=autocast["dtype"]):
                    _ = model(x)
            else:
                _ = model(x)
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return {"ok": True, "times": times, "avg_ms": mean(times) * 1000.0}

def do_train_step(model, make_batch, device, loss_kind="ce"):
    model = model.train().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    try:
        xb, yb = make_batch(device)
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        if loss_kind == "ce":
            loss = nn.CrossEntropyLoss()(out, yb)
        else:
            loss = F.mse_loss(out, yb)
        loss.backward()
        opt.step()
        return {"ok": True, "loss": float(loss.detach().cpu())}
    except Exception as e:
        return {"ok": False, "error": f"train step failed: {e}"}

def pretty_row(cols, widths):
    return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))

def print_table(rows):
    widths = [max(len(str(c)) for c in col) for col in zip(*rows)]
    for i, r in enumerate(rows):
        print(pretty_row(r, widths))
        if i == 0:
            print("-+-".join("-"*w for w in widths))

# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, default=20, help="Batches for forward timing")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p.add_argument("--image-size", type=int, default=224, help="HxW for CNN input")
    p.add_argument("--mlp-dim", type=int, default=4096, help="MLP input dimension")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    backends = [("cpu", torch.device("cpu"))]
    if HAS_DML:
        dml_dev = DML.device(0)
        backends.append(("dml", torch.device("privateuseone", 0)))
    else:
        dml_dev = None

    print("=== Environment ===")
    print("torch:", torch.__version__)
    if HAS_DML:
        print("torch-directml: detected")
        print("DML adapter:", DML_NAME)
        print("DML device string (PyTorch):", torch.tensor([0.0]).to(dml_dev).device)
    else:
        print("torch-directml: NOT detected (or no adapters)")

    B = args.batch_size
    H = W = args.image_size
    C = 3
    MLP_D = args.mlp_dim
    NUM_CLASSES = 1000

    # Define input makers
    def make_cnn_input(device):
        return torch.randn(B, C, H, W, device=device, dtype=torch.float32)

    def make_cnn_batch(device):
        x = make_cnn_input(device)
        y = torch.randint(0, NUM_CLASSES, (B,), device=device)
        return x, y

    def make_mlp_input(device):
        return torch.randn(B, MLP_D, device=device, dtype=torch.float32)

    def make_mlp_batch(device):
        x = make_mlp_input(device)
        y = torch.randn(B, NUM_CLASSES, device=device)
        return x, y

    # Instantiate models
    cnn = SmallCNN(num_classes=NUM_CLASSES)
    mlp = SmallMLP(in_dim=MLP_D, hidden=max(MLP_D//2, 512), out_dim=NUM_CLASSES, depth=3)

    rows = [["Model", "Backend", "Forward avg (ms)", "Train step", "Notes"]]

    # CNN tests (apply channels_last to INPUTS only)
    for name, dev in backends:
        res_fwd = do_forward_timing(
            cnn, make_cnn_input, dev, batches=args.batches,
            autocast=None,
            memory_format=torch.channels_last
        )
        note = "inputs=channels_last"
        if res_fwd["ok"]:
            rows.append(["CNN", name, f"{res_fwd['avg_ms']:.2f}", "—", note])
        else:
            rows.append(["CNN", name, "ERR", "—", f"{note}; {res_fwd['error']}"])

        res_tr = do_train_step(cnn, make_cnn_batch, dev, loss_kind="ce")
        if res_tr["ok"]:
            rows.append(["CNN train", name, "—", f"OK (loss {res_tr['loss']:.3f})", note])
        else:
            rows.append(["CNN train", name, "—", "ERR", f"{note}; {res_tr['error']}"])

    # CNN mixed precision (DML only) using autocast
    if HAS_DML:
        try:
            res_mp = do_forward_timing(
                cnn, make_cnn_input, torch.device("privateuseone", 0),
                batches=args.batches,
                autocast={"device_type": "privateuseone", "dtype": torch.float16},
                memory_format=torch.channels_last
            )
            if res_mp["ok"]:
                rows.append(["CNN (autocast fp16)", "dml", f"{res_mp['avg_ms']:.2f}", "—", "inputs=channels_last"])
            else:
                rows.append(["CNN (autocast fp16)", "dml", "ERR", "—", res_mp["error"]])
        except Exception as e:
            rows.append(["CNN (autocast fp16)", "dml", "ERR", "—", f"{type(e).__name__}: {e}"])

    # MLP tests (no channels_last)
    for name, dev in backends:
        res_fwd = do_forward_timing(mlp, make_mlp_input, dev, batches=args.batches, autocast=None)
        if res_fwd["ok"]:
            rows.append(["MLP", name, f"{res_fwd['avg_ms']:.2f}", "—", ""])
        else:
            rows.append(["MLP", name, "ERR", "—", res_fwd["error"]])

        res_tr = do_train_step(mlp, make_mlp_batch, dev, loss_kind="mse")
        if res_tr["ok"]:
            rows.append(["MLP train", name, "—", f"OK (loss {res_tr['loss']:.3f})", ""])
        else:
            rows.append(["MLP train", name, "—", "ERR", res_tr["error"]])

    print("\n=== Results (lower forward ms is better) ===")
    print_table(rows)

    # Quick recommendation
    print("\n=== Quick pick ===")
    cpu_cnn = next((r for r in rows if r[0]=="CNN" and r[1]=="cpu" and r[2] not in ("ERR","—")), None)
    dml_cnn = next((r for r in rows if r[0]=="CNN" and r[1]=="dml" and r[2] not in ("ERR","—")), None)

    if cpu_cnn and dml_cnn:
        cpu_ms = float(cpu_cnn[2])
        dml_ms = float(dml_cnn[2])
        ratio = cpu_ms / max(dml_ms, 1e-6)
        if ratio > 1.10:
            print(f"DML looks faster for CNN (~{ratio:.2f}×). Prefer DML for conv-heavy inference/training (with small batches).")
        elif ratio < 0.90:
            print(f"CPU looks faster for CNN (~{1/ratio:.2f}×). Prefer CPU or try smaller inputs/batches on DML.")
        else:
            print("CPU and DML are roughly similar for CNN. Choose based on stability and memory headroom.")
    else:
        if not dml_cnn:
            print("Could not complete CNN on DML — likely an unsupported op or environment issue.")
        if not cpu_cnn:
            print("CPU CNN test failed unexpectedly — check your PyTorch install.")

    print("\nTip: On Iris Xe, keep batch sizes modest. For CNNs, making **inputs** channels_last often helps on DML. Mixed precision may or may not help — validate numerics.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Fatal error:\n", traceback.format_exc())
