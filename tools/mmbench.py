#!/usr/bin/env python3
"""mmbench.py - one command, comparable LLM results on any machine.

    python tools/mmbench.py --list                # fingerprint + which pinned models fit
    python tools/mmbench.py --download            # fetch missing pinned models, then run
    python tools/mmbench.py                       # run every tier that fits this machine
    python tools/mmbench.py --tiers base          # only the baseline everyone runs
    python tools/mmbench.py --bandwidth 273       # peak GB/s if it can't be auto-detected
    python tools/mmbench.py --compare             # markdown table of all results/*.json

What it does, for each pinned model (same file, same settings, everywhere):
  1. llama-bench -p 512 -n 128 -o json   -> prefill and decode tokens/s (parsed from JSON)
  2. starts llama-server on 127.0.0.1, runs labs/evals/basic.jsonl at temperature 0
     with thinking off (gpt-oss: reasoning effort low), then stops the server
  3. normalizes: effective GB/s = decode t/s x bytes read per token
                 bandwidth efficiency = effective GB/s / peak GB/s
Results go to results/<machine-id>-<date>.json. Runs on macOS, Linux and native
Windows with the standard library only; `pip install gguf` lets it recompute
bytes-per-token from the file itself (otherwise the pinned values are used).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
EVAL_SET = os.path.join(ROOT, "labs", "evals", "basic.jsonl")
GIB = 1024**3
SCHEMA = 1

# ---------------------------------------------------------------------------
# The pinned model set. Change it and results stop being comparable, so bump
# SET_VERSION if you ever do. read_bytes = weights touched per generated token,
# computed by tools/memcalc.py (MoE: shared weights + used experts; embeddings
# that are only looked up are excluded).
# ---------------------------------------------------------------------------
SET_VERSION = "2026-11"
MODELS = [
    {"id": "qwen3.5-4b-q4_0", "tier": "base",
     "repo": "unsloth/Qwen3.5-4B-GGUF", "rev": "e87f176479d0855a907a41277aca2f8ee7a09523",
     "file": "Qwen3.5-4B-Q4_0.gguf", "size": 2583221408,
     "sha256": "298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb",
     "read_bytes": None, "eval": {"think": False, "max_tokens": 256}},
    {"id": "gemma-4-e4b-qat-q4_0", "tier": "small",
     "repo": "google/gemma-4-E4B-it-qat-q4_0-gguf", "rev": "4b4a2c1d584be7264f87aac328a1bc739ce81b6c",
     "file": "gemma-4-E4B_q4_0-it.gguf", "size": 5154941280,
     "sha256": "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee",
     "read_bytes": None, "eval": {"think": False, "max_tokens": 256}},
    {"id": "gpt-oss-20b-mxfp4", "tier": "medium",
     "repo": "ggml-org/gpt-oss-20b-GGUF", "rev": "41e8a94d15d62070b36cf4553a0250717ecf46b8",
     "file": "gpt-oss-20b-MXFP4.gguf", "size": 12109566624,
     "sha256": "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
     "read_bytes": None, "eval": {"think": True, "reasoning_effort": "low", "max_tokens": 1024}},
    {"id": "qwen3.8-27b-iq4_xs", "tier": "large",
     "repo": "unsloth/Qwen3.8-27B-GGUF", "rev": "4af51b3542f775b0a8e4b310bfb3f17e01c306c9",
     "file": "Qwen3.8-27B-UD-IQ4_XS.gguf", "size": 14352911264,
     "sha256": "4b194b34ecec99fdfb74caef84445ccd01c987de87d96ae8ca50dc434b4b0a38",
     "read_bytes": None, "eval": {"think": False, "max_tokens": 256}},
]
TIERS = ["base", "small", "medium", "large"]
# memory needed on top of the file: KV cache at 4K context + compute buffers
RUNTIME_HEADROOM = 1.0 * GIB

SETTINGS = {"n_prompt": 512, "n_gen": 128, "eval_ctx": 4096, "temperature": 0,
            "thinking": "off (gpt-oss: reasoning_effort=low)", "set_version": SET_VERSION}

# Peak memory bandwidth, GB/s, from vendor spec sheets (same as docs/01).
APPLE_BW = {"M1": 68, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
            "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
            "M3": 100, "M3 Pro": 150, "M3 Max": (300, 400), "M3 Ultra": 819,
            "M4": 120, "M4 Pro": 273, "M4 Max": (410, 546), "M5": 153}
GPU_BW = [("RTX 5090", 1792), ("RTX 4090", 1008), ("RTX 3090", 936), ("RTX 4080", 717),
          ("RTX 4070", 504), ("4060 Ti", 288), ("RTX 3060", 360), ("L40S", 864),
          ("L4", 300), ("A100", 2039), ("H100", 3350), ("H200", 4800),
          ("RX 7900 XTX", 960), ("Radeon 8060S", 256)]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def run_cmd(cmd: list[str], timeout: float = 30) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              errors="replace").stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def _sysctl(key: str) -> str:
    return run_cmd(["sysctl", "-n", key])


def cpu_info() -> dict:
    system = platform.system()
    info = {"name": platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count(), "physical_cores": None, "perf_cores": None}
    if system == "Darwin":
        info["name"] = _sysctl("machdep.cpu.brand_string") or info["name"]
        info["physical_cores"] = int(_sysctl("hw.physicalcpu") or 0) or None
        info["perf_cores"] = int(_sysctl("hw.perflevel0.physicalcpu") or 0) or None
    elif system == "Linux":
        cores = set()
        phys = core = None
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    k, v = k.strip(), v.strip()
                    if k == "model name" and info["name"] in ("", platform.machine()):
                        info["name"] = v
                    elif k == "physical id":
                        phys = v
                    elif k == "core id":
                        core = v
                    elif not k and core is not None:
                        cores.add((phys, core))
                        phys = core = None
            if core is not None:
                cores.add((phys, core))
        except OSError:
            pass
        info["physical_cores"] = len(cores) or None
    elif system == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                info["name"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
        out = run_cmd(["powershell", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"])
        info["physical_cores"] = int(out) if out.isdigit() else None
    return info


def ram_bytes() -> int | None:
    system = platform.system()
    if system == "Darwin":
        v = _sysctl("hw.memsize")
        return int(v) if v.isdigit() else None
    if system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return None
    if system == "Windows":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullTotalPhys)
    return None


def os_info() -> str:
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    if system == "Windows":
        return f"Windows {platform.release()} {platform.version()} ({platform.machine()})"
    pretty = ""
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    pretty = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    wsl = " (WSL2)" if "microsoft" in platform.release().lower() else ""
    return f"Linux {platform.release()}{wsl} {pretty} ({platform.machine()})".strip()


def find_binary(name: str, bin_dir: str | None) -> str | None:
    exe = name + (".exe" if platform.system() == "Windows" else "")
    dirs = [bin_dir, os.environ.get("LLAMA_CPP_BIN")]
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, exe)):
            return os.path.join(d, exe)
    found = shutil.which(name)
    if found:
        return found
    for d in ("build/bin", "build/bin/Release"):
        p = os.path.join(ROOT, "vendor", "llama.cpp", d, exe)
        if os.path.isfile(p):
            return p
    return None


def list_devices(server: str) -> list[dict]:
    """`llama-server --list-devices` -> [{'id','name','total_mib','free_mib'}]. Works for
    every backend (Metal, CUDA, ROCm, Vulkan, SYCL) because llama.cpp asks the driver."""
    out = subprocess.run([server, "--list-devices"], capture_output=True, text=True,
                         errors="replace", timeout=60)
    devs = []
    for line in (out.stdout + "\n" + out.stderr).splitlines():
        m = re.match(r"\s*(\w+):\s+(.+?)\s+\((\d+) MiB, (\d+) MiB free\)", line)
        if m:
            devs.append({"id": m.group(1), "name": m.group(2),
                         "total_mib": int(m.group(3)), "free_mib": int(m.group(4))})
    return devs


def server_version(server: str) -> dict:
    out = subprocess.run([server, "--version"], capture_output=True, text=True,
                         errors="replace", timeout=60)
    text = out.stdout + out.stderr
    # "version: 1.0.0-dev (build 1, commit a300a9c)"  or older  "version: 6500 (a300a9c)"
    m = re.search(r"version:\s*(\S+)\s*\((?:build (\d+), commit )?(\w+)\)", text)
    built = re.search(r"built with (.+)", text)
    return {"version": m.group(1) if m else None, "build": m.group(2) if m else None,
            "commit": m.group(3) if m else None,
            "built_with": built.group(1).strip() if built else None}


def apple_gpu_cores() -> int | None:
    out = run_cmd(["system_profiler", "SPDisplaysDataType"], timeout=30)
    m = re.search(r"Total Number of Cores:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def peak_bandwidth(cpu: str, gpu_names: list[str], gpu_cores: int | None) -> tuple[float | None, str]:
    m = re.match(r"Apple (M\d(?: Pro| Max| Ultra)?)$", cpu or "")
    if m and m.group(1) in APPLE_BW:
        v = APPLE_BW[m.group(1)]
        if isinstance(v, tuple):  # binned Max chips have fewer memory channels
            v = v[1] if (gpu_cores or 0) >= 40 else v[0]
        return float(v), f"Apple spec sheet ({m.group(1)}{f', {gpu_cores}-core GPU' if gpu_cores else ''})"
    for g in gpu_names:
        for key, bw in GPU_BW:
            if key in g:
                return float(bw), f"spec sheet ({g})"
    return None, "unknown: pass --bandwidth <GB/s> from your RAM/GPU spec sheet"


def slug(s: str) -> str:
    s = re.sub(r"\((R|TM|tm|r)\)|CPU|Processor|@.*$|\d+-Core|with .*$", "", s)
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def fingerprint(server: str, bandwidth: float | None, threads: int) -> dict:
    cpu = cpu_info()
    ram = ram_bytes()
    devs = list_devices(server)
    ver = server_version(server)
    gpu_cores = apple_gpu_cores() if platform.system() == "Darwin" else None
    # skip the CPU and pseudo-devices such as "BLAS: Accelerate (0 MiB)"
    gpus = [d for d in devs if d["total_mib"] > 0 and not re.match(r"(CPU|BLAS)", d["id"], re.I)]
    backend_map = {"MTL": "Metal", "CUDA": "CUDA", "ROCm": "ROCm", "HIP": "ROCm",
                   "Vulkan": "Vulkan", "SYCL": "SYCL", "OpenCL": "OpenCL"}
    backend = "CPU"
    if gpus:
        prefix = re.match(r"[A-Za-z]+", gpus[0]["id"]).group(0)
        backend = backend_map.get(prefix, prefix)
    bw, bw_src = peak_bandwidth(cpu["name"], [g["name"] for g in gpus], gpu_cores)
    if bandwidth:
        bw, bw_src = bandwidth, "--bandwidth (user)"
    ram_gb = round((ram or 0) / GIB)
    if gpus and backend != "Metal":
        mid = f"{slug(gpus[0]['name'])}-{round(gpus[0]['total_mib'] / 1024)}gb-{slug(cpu['name'])}"
    else:
        mid = slug(cpu["name"]) + (f"-{gpu_cores}gpu" if gpu_cores else "")
    mid = f"{mid}-{ram_gb}gb-{backend.lower()}"
    return {
        "machine_id": mid,
        "os": os_info(),
        "cpu": cpu["name"], "logical_cores": cpu["logical_cores"],
        "physical_cores": cpu["physical_cores"], "perf_cores": cpu["perf_cores"],
        "ram_gib": round((ram or 0) / GIB, 1),
        "gpus": [{"name": g["name"], "vram_gib": round(g["total_mib"] / 1024, 2),
                  **({"gpu_cores": gpu_cores} if gpu_cores else {})} for g in gpus],
        "backend": backend,
        "llama_cpp": ver,
        "peak_bandwidth_gbs": bw, "bandwidth_source": bw_src,
        "threads": threads,
        "python": platform.python_version(),
    }


def memory_budget(fp: dict) -> float:
    """Bytes the model + context can use: GPU memory, or RAM minus room for the OS."""
    if fp["gpus"]:
        return max(g["vram_gib"] for g in fp["gpus"]) * GIB
    ram = fp["ram_gib"] * GIB
    return max(ram - 4 * GIB, ram * 0.7)


def default_threads() -> int:
    """Performance cores on Apple/hybrid chips, physical cores elsewhere."""
    c = cpu_info()
    return c["perf_cores"] or c["physical_cores"] or os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def bytes_per_token(path: str, pinned: int | None) -> tuple[int | None, str]:
    """Weights read per generated token, the same way tools/memcalc.py counts them."""
    try:
        spec = importlib.util.spec_from_file_location("memcalc", os.path.join(ROOT, "tools", "memcalc.py"))
        mc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mc)
        import gguf  # noqa: F401  (memcalc imports it lazily; fail here if missing)
        m = mc.read_gguf(path)
        if m["n_expert"]:
            active = (m["tensor_bytes"] - m["expert_bytes"] - m["lookup_only_bytes"]
                      + m["expert_bytes"] * m["n_expert_used"] / m["n_expert"])
        else:
            active = m["tensor_bytes"] - m["lookup_only_bytes"]
        return int(active), "gguf (memcalc)"
    except ImportError:
        return pinned, "pinned (pip install gguf to recompute)"


def download(model: dict, dest: str) -> None:
    url = f"https://huggingface.co/{model['repo']}/resolve/{model['rev']}/{model['file']}"
    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    log(f"   downloading {model['file']} ({model['size'] / 1e9:.2f} GB) from {model['repo']}")
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if have else "wb") as f:
        if have and r.status != 206:  # server ignored the Range header: start over
            f.truncate(0)
            have = 0
        done, t0, last = have, time.time(), 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 5:
                last = time.time()
                rate = (done - have) / max(last - t0, 1e-6) / 1e6
                log(f"     {done / model['size']:6.1%}  {rate:6.1f} MB/s")
    if os.path.getsize(tmp) != model["size"]:
        raise SystemExit(f"!! {tmp}: size {os.path.getsize(tmp)} != expected {model['size']}")
    os.replace(tmp, dest)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Benchmark (llama-bench -o json)
# ---------------------------------------------------------------------------

def llama_bench(bench: str, path: str, threads: int, reps: int, extra: list[str]) -> dict:
    cmd = [bench, "-m", path, "-p", str(SETTINGS["n_prompt"]), "-n", str(SETTINGS["n_gen"]),
           "-t", str(threads), "-r", str(reps), "-o", "json", *extra]
    shown = " ".join(os.path.basename(c) if c in (bench, path) else c for c in cmd)  # no home paths
    log("   $ " + shown)
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"llama-bench failed ({p.returncode}):\n{p.stderr[-2000:]}")
    rows = json.loads(p.stdout)
    out = {"cmd": shown}
    for r in rows:
        key = "pp" if r["n_gen"] == 0 else "tg" if r["n_prompt"] == 0 else None
        if key:
            out[key] = {"tokens_per_s": round(r["avg_ts"], 2), "stddev": round(r["stddev_ts"], 2),
                        "samples": [round(x, 2) for x in r.get("samples_ts", [])]}
    first = rows[0]
    for k in ("build_commit", "build_number", "backends", "devices", "cpu_info", "gpu_info", "n_cpu_moe",
              "n_gpu_layers", "flash_attn", "n_batch", "n_ubatch", "n_threads", "type_k", "type_v",
              "model_type", "model_size", "model_n_params"):
        if k in first:
            out[k] = first[k]
    return out


# ---------------------------------------------------------------------------
# Eval against a llama-server we start and stop ourselves
# ---------------------------------------------------------------------------

def load_checker():
    sys.path.insert(0, os.path.join(ROOT, "labs"))
    spec = importlib.util.spec_from_file_location("mini_eval", os.path.join(ROOT, "labs", "03_mini_eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check, mod.clean


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(url: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_eval(server: str, path: str, model: dict, threads: int, extra: list[str]) -> dict:
    check, clean = load_checker()
    items = [json.loads(l) for l in open(EVAL_SET, encoding="utf-8") if l.strip()]
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    logf = tempfile.NamedTemporaryFile("w+", prefix="mmbench-server-", suffix=".log", delete=False)
    cmd = [server, "-m", path, "--host", "127.0.0.1", "--port", str(port),
           "-c", str(SETTINGS["eval_ctx"]), "-np", "1", "-t", str(threads), *extra]
    log(f"   $ llama-server -m {os.path.basename(path)} --host 127.0.0.1 --port {port} "
        f"-c {SETTINGS['eval_ctx']} -np 1 -t {threads} {' '.join(extra)}".rstrip())
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                logf.seek(0)
                raise RuntimeError("llama-server exited early:\n" + logf.read()[-2000:])
            try:
                if http_json(base + "/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.time() - t0 > 600:
                raise RuntimeError("llama-server did not become ready in 600 s")
            time.sleep(1)
        load_s = time.time() - t0
        ev = model["eval"]
        params = {"temperature": 0, "max_tokens": ev["max_tokens"]}
        if not ev["think"]:
            params["chat_template_kwargs"] = {"enable_thinking": False}
        if ev.get("reasoning_effort"):
            params["reasoning_effort"] = ev["reasoning_effort"]
        rows = []
        for it in items:
            t1 = time.perf_counter()
            r = http_json(base + "/v1/chat/completions",
                          {"model": "local", "messages": [{"role": "user", "content": it["q"]}], **params})
            secs = time.perf_counter() - t1
            ans = r["choices"][0]["message"].get("content") or ""
            ok = check(it, ans)
            rows.append({"id": it["id"], "cat": it["cat"], "ok": ok, "answer": clean(ans)[:200],
                         "seconds": round(secs, 2),
                         "completion_tokens": r.get("usage", {}).get("completion_tokens", 0)})
            log(f"     [{'PASS' if ok else 'fail'}] {it['id']:<10} {secs:5.1f}s  {clean(ans)[:50]!r}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        logf.close()
        os.unlink(logf.name)
    cats: dict[str, list[int]] = {}
    for r in rows:
        c = cats.setdefault(r["cat"], [0, 0])
        c[0] += r["ok"]
        c[1] += 1
    return {"score": sum(r["ok"] for r in rows), "n": len(rows),
            "categories": {k: f"{v[0]}/{v[1]}" for k, v in cats.items()},
            "total_s": round(sum(r["seconds"] for r in rows), 1),
            "server_load_s": round(load_s, 1), "params": params, "rows": rows}


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def plan(fp: dict, tiers: list[str] | None) -> list[tuple[dict, bool, str]]:
    budget = memory_budget(fp)
    out = []
    for m in MODELS:
        need = m["size"] + RUNTIME_HEADROOM
        fits = need <= budget
        if tiers is not None:
            pick, why = m["tier"] in tiers, "--tiers"
        else:
            pick = fits
            why = "fits" if fits else f"needs ~{need / GIB:.1f} GiB > budget {budget / GIB:.1f} GiB"
        out.append((m, pick, why))
    return out


def main_run(a) -> None:
    server = find_binary("llama-server", a.llama_bin)
    bench = find_binary("llama-bench", a.llama_bin)
    if not (server and bench):
        raise SystemExit("llama-server / llama-bench not found. Build (scripts/build-llama-cpp.sh), "
                         "install (brew/winget), or pass --llama-bin DIR")
    threads = a.threads or default_threads()
    fp = fingerprint(server, a.bandwidth, threads)
    if a.machine_id:
        fp["machine_id"] = a.machine_id
    log(f"Machine : {fp['machine_id']}")
    log(f"  {fp['os']} | {fp['cpu']} | {fp['logical_cores']} cores | {fp['ram_gib']} GiB RAM")
    for g in fp["gpus"]:
        log(f"  GPU {g['name']}: {g['vram_gib']} GiB usable")
    log(f"  backend {fp['backend']} | llama.cpp {fp['llama_cpp']['version']} ({fp['llama_cpp']['commit']})"
        f" | threads {threads}")
    log(f"  peak bandwidth: {fp['peak_bandwidth_gbs'] or '?'} GB/s [{fp['bandwidth_source']}]")
    log(f"  memory budget for models: {memory_budget(fp) / GIB:.1f} GiB\n")

    tiers = a.tiers.split(",") if a.tiers else None
    todo = plan(fp, tiers)
    for m, pick, why in todo:
        log(f"  [{'x' if pick else ' '}] {m['tier']:<6} {m['id']:<22} {m['size'] / 1e9:5.2f} GB  ({why})")
    if a.list:
        return
    log()

    os.makedirs(RESULTS, exist_ok=True)
    date = dt.date.today().strftime("%Y%m%d")
    out_path = os.path.join(RESULTS, f"{fp['machine_id']}-{date}.json")
    result = {"mmbench": SCHEMA, "date": dt.date.today().isoformat(), "fingerprint": fp,
              "settings": SETTINGS, "models": {}}
    if os.path.exists(out_path):  # same machine, same day: keep models not re-run now
        with open(out_path, encoding="utf-8") as f:
            result["models"] = json.load(f).get("models", {})

    bench_extra = a.bench_args.split() if a.bench_args else []
    server_extra = a.server_args.split() if a.server_args else []
    for m, pick, _ in todo:
        if not pick:
            continue
        path = os.path.join(a.models_dir, m["file"])
        log(f"== {m['id']}  ({m['repo']}/{m['file']})")
        if not os.path.exists(path):
            if not a.download:
                log(f"   missing {path}: skip (use --download, or scripts/get-model.sh {m['repo']} {m['file']})")
                continue
            download(m, path)
        size = os.path.getsize(path)
        if size != m["size"]:
            log(f"   !! size {size} != pinned {m['size']}: not the pinned file, skipping")
            continue
        entry = {"repo": m["repo"], "rev": m["rev"], "file": m["file"], "size": size, "tier": m["tier"]}
        if a.verify:
            log("   verifying sha256 ...")
            entry["sha256_ok"] = sha256(path) == m["sha256"]
            if not entry["sha256_ok"]:
                log("   !! sha256 mismatch: skipping")
                continue
        rb, rb_src = bytes_per_token(path, m["read_bytes"])
        entry["read_bytes_per_token"], entry["read_bytes_source"] = rb, rb_src
        if not a.skip_bench:
            b = llama_bench(bench, path, threads, a.reps, bench_extra)
            entry["bench"] = b
            tg = b["tg"]["tokens_per_s"]
            if rb:
                entry["effective_gbs"] = round(tg * rb / 1e9, 1)
                if fp["peak_bandwidth_gbs"]:
                    entry["ceiling_tokens_per_s"] = round(fp["peak_bandwidth_gbs"] * 1e9 / rb, 1)
                    entry["bandwidth_efficiency"] = round(entry["effective_gbs"] / fp["peak_bandwidth_gbs"], 3)
            log(f"   pp512 {b['pp']['tokens_per_s']:8.1f} t/s | tg128 {tg:6.1f} t/s | "
                f"{entry.get('effective_gbs', '?')} GB/s effective"
                + (f" = {entry['bandwidth_efficiency']:.0%} of peak" if "bandwidth_efficiency" in entry else ""))
        if not a.skip_eval:
            e = run_eval(server, path, m, threads, server_extra)
            entry["eval"] = e
            log(f"   eval {e['score']}/{e['n']} in {e['total_s']} s  "
                + " ".join(f"{k}:{v}" for k, v in e["categories"].items()))
        prev = result["models"].get(m["id"], {})
        result["models"][m["id"]] = {**prev, **entry}
        with open(out_path, "w", encoding="utf-8") as f:  # save after every model
            json.dump(result, f, indent=1)
        log()
    log(f"Saved {out_path}")
    log("Compare with other machines:  python tools/mmbench.py --compare")


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def compare(paths: list[str]) -> None:
    runs = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("mmbench"):
            runs.append(d)
    if not runs:
        print("No mmbench results found in results/.")
        return
    order = {m["id"]: i for i, m in enumerate(MODELS)}
    cats = []
    for r in runs:
        for e in r["models"].values():
            for c in e.get("eval", {}).get("categories", {}):
                if c not in cats:
                    cats.append(c)
    print("| model | machine | backend | pp512 t/s | tg128 t/s | effective GB/s | % of peak "
          f"| eval | {' | '.join(cats)} | eval time |")
    print("|---|---|---|---|---|---|---|---|" + "---|" * len(cats) + "---|")
    rows = []
    for r in runs:
        fp = r["fingerprint"]
        for mid, e in r["models"].items():
            rows.append((order.get(mid, 99), -(e.get("bench", {}).get("tg", {}).get("tokens_per_s") or 0), mid, r, e))
    for _, _, mid, r, e in sorted(rows, key=lambda x: (x[0], x[1])):
        fp = r["fingerprint"]
        b = e.get("bench", {})
        ev = e.get("eval", {})
        eff = e.get("bandwidth_efficiency")
        machine = fp["machine_id"] + (" *" if r.get("imported") else "")
        vals = [mid, machine, fp["backend"],
                f"{b['pp']['tokens_per_s']:.1f}" if "pp" in b else "-",
                f"**{b['tg']['tokens_per_s']:.1f}**" if "tg" in b else "-",
                f"{e['effective_gbs']:.0f}" if e.get("effective_gbs") else "-",
                f"{eff:.0%}" if eff else "-",
                f"**{ev['score']}/{ev['n']}**" if ev else "-",
                *[ev.get("categories", {}).get(c, "-") for c in cats],
                f"{ev['total_s']:.0f}s" if ev else "-"]
        print("| " + " | ".join(vals) + " |")
    if any(r.get("imported") for r in runs):
        print("\n\\* imported from measurements made before mmbench existed (see the file's `note`).")
    print()
    for r in runs:
        fp = r["fingerprint"]
        lc = fp.get("llama_cpp", {})
        gpus = ", ".join(f"{g['name']} {g['vram_gib']} GiB" for g in fp.get("gpus", [])) or "no GPU"
        print(f"- **{fp['machine_id']}** ({r['date']}): {fp['os']}; {fp['cpu']}, "
              f"{fp.get('logical_cores')} cores, {fp['ram_gib']} GiB RAM; {gpus}; "
              f"llama.cpp {lc.get('version')} ({lc.get('commit')}); threads {fp.get('threads')}; "
              f"peak {fp.get('peak_bandwidth_gbs') or '?'} GB/s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show fingerprint and the model plan, run nothing")
    ap.add_argument("--compare", nargs="*", metavar="FILE",
                    help="print a markdown table of results (default: results/*.json)")
    ap.add_argument("--tiers", help=f"comma-separated subset of {','.join(TIERS)} (default: all that fit)")
    ap.add_argument("--download", action="store_true", help="download missing pinned models")
    ap.add_argument("--verify", action="store_true", help="check sha256 of each model file (slow)")
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", os.path.join(ROOT, "models")))
    ap.add_argument("--llama-bin", help="directory with llama-bench / llama-server")
    ap.add_argument("--bandwidth", type=float, help="peak memory bandwidth in GB/s (overrides auto-detect)")
    ap.add_argument("--threads", type=int, help="CPU threads (default: performance/physical cores)")
    ap.add_argument("--reps", type=int, default=3, help="llama-bench repetitions (default 3)")
    ap.add_argument("--machine-id", help="override the auto-generated machine id")
    ap.add_argument("--bench-args", help='extra llama-bench flags, e.g. "-fa 1" (breaks comparability)')
    ap.add_argument("--server-args", help="extra llama-server flags for the eval")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    if a.compare is not None:
        compare(a.compare or sorted(glob.glob(os.path.join(RESULTS, "*.json"))))
    else:
        main_run(a)


if __name__ == "__main__":
    main()
