import json
import os
import re
import subprocess
import sys
import tempfile

def safe_run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=False)

print("=== PyTorch / DirectML ===")
try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    print("torch: not found", e)

try:
    import torch_directml as dml
    print("torch-directml:", getattr(dml, "__version__", "(version attr not exposed)"))
    try:
        count = dml.device_count()
        print("DML device count:", count)
        for i in range(count):
            # device_name may not exist on some wheels
            name = getattr(dml, "device_name", None)
            if callable(name):
                print(f"  [{i}] {name(i)}")
            else:
                print(f"  [{i}] (device name API not available)")
        # Show default + PyTorch device string
        idx = getattr(dml, "default_device", lambda: 0)()
        dev = dml.device(idx)
        import torch as _torch
        t = _torch.tensor([0.0]).to(dev)
        print("PyTorch device string:", t.device)  # typically "privateuseone:0"
    except Exception as e:
        print("DirectML query error:", e)
except Exception as e:
    print("torch-directml: not found", e)

print("\n=== Windows WMI (basic GPU info) ===")
# Uses PowerShell; works on stock Windows without extra Python packages
ps = r"""
$gpus = Get-CimInstance Win32_VideoController |
  Select-Object Name, AdapterCompatibility, DriverVersion, DriverDate, AdapterRAM, VideoProcessor, PNPDeviceID
$gpus | ConvertTo-Json -Depth 3
"""
res = safe_run(["powershell", "-NoProfile", "-Command", ps])
if res.returncode == 0 and res.stdout.strip():
    try:
        data = json.loads(res.stdout)
        if isinstance(data, dict):
            data = [data]
        for i, gpu in enumerate(data):
            print(f"[{i}] Name:               {gpu.get('Name')}")
            print(f"    Vendor:             {gpu.get('AdapterCompatibility')}")
            print(f"    Driver:             {gpu.get('DriverVersion')}  ({gpu.get('DriverDate')})")
            ram = gpu.get("AdapterRAM")
            if isinstance(ram, int):
                print(f"    AdapterRAM:         {ram/1024/1024:.0f} MB")
            print(f"    VideoProcessor:     {gpu.get('VideoProcessor')}")
            print(f"    PNPDeviceID:        {gpu.get('PNPDeviceID')}")
    except Exception as e:
        print("Failed to parse WMI JSON:", e, "\nRaw:", res.stdout[:400])
else:
    print("PowerShell/WMI query failed:", res.stderr or "(no stderr)")

print("\n=== dxdiag (feature levels, VRAM, driver model) ===")
with tempfile.TemporaryDirectory() as td:
    out_txt = os.path.join(td, "dxdiag.txt")
    rr = safe_run(["dxdiag", "/t", out_txt])
    if rr.returncode != 0:
        print("dxdiag failed (are you on Windows with DirectX?)")
    else:
        try:
            text = open(out_txt, "r", encoding="utf-16", errors="ignore").read()
        except UnicodeError:
            text = open(out_txt, "r", encoding="utf-8", errors="ignore").read()

        # Grab display device sections
        sections = re.split(r"\n+-{5,}\n+", text)
        displays = [s for s in sections if "Display Devices" in s or "Display Device" in s]

        def find(pattern, s, flags=re.IGNORECASE):
            m = re.search(pattern, s, flags)
            return m.group(1).strip() if m else None

        # Fallback: scan entire file if we didn’t isolate sections
        scan_targets = displays if displays else [text]

        for si, s in enumerate(scan_targets):
            print(f"\n-- Adapter block {si} --")
            print("Card name:        ", find(r"Card name:\s*(.+)", s) or find(r"Card name\s*:\s*(.+)", s))
            print("Manufacturer:     ", find(r"Manufacturer:\s*(.+)", s))
            print("Chip type:        ", find(r"Chip type:\s*(.+)", s))
            print("DAC type:         ", find(r"DAC type:\s*(.+)", s))
            print("Driver Model:     ", find(r"Driver Model:\s*(.+)", s))
            print("Driver Version:   ", find(r"Driver Version:\s*(.+)", s) or find(r"Driver Version\s*:\s*(.+)", s))
            print("Dedicated Memory: ", find(r"Dedicated Memory:\s*(.+)", s))
            print("Shared Memory:    ", find(r"Shared Memory:\s*(.+)", s))
            # Feature levels line often looks like: "Feature Levels: 12_1,12_0,11_1,11_0, ..."
            print("Feature Levels:   ", find(r"Feature Levels?:\s*([0-9_,\s]+)", s))

print("\nDone.")
