#!/usr/bin/env -S colab run --gpu G4 --keep -s qwen38-g4 --timeout 43200
"""
One-click & Interactive Deployment Script for Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4
on Google Colab G4 GPU (NVIDIA RTX PRO 6000 Blackwell 96GB) with 262K Context & MTP (NEXTN).

Repository: https://github.com/mizorewww/colab-qwen38-g4

Two ways to run:
  Mode A — Interactive Local Wizard (spawns background keep-alive & auto-heal daemon):
    python3 run_qwen38_g4.py

  Mode B — Direct One-Command Remote Execution via colab CLI (holds kernel active up to 12h):
    colab run --gpu G4 --keep -s qwen38-g4 --timeout 43200 run_qwen38_g4.py

  Helper commands (local):
    python3 run_qwen38_g4.py --keepalive [-s qwen38-g4]   # Start/reattach background keep-alive
    python3 run_qwen38_g4.py --status    [-s qwen38-g4]   # Check session & keep-alive daemon status
    python3 run_qwen38_g4.py --stop      [-s qwen38-g4]   # Stop daemon & release Colab VM
"""

import argparse
import glob
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_ID = os.environ.get("MODEL_ID", "dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4")
LOCAL_MODEL_DIR = "/content/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4"
LOCAL_MTP_DRAFT_DIR = "/content/Qwen3.8-Flash-Next-MTP-Draft"
DRIVE_MODEL_DIR = "/content/drive/MyDrive/models/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4"

GITHUB_RELEASE_BASE = os.environ.get(
    "KERNEL_CACHE_BASE_URL",
    "https://github.com/mizorewww/colab-qwen38-g4/releases/download/v1.0.0",
)
API_KEY = os.environ.get("COLAB_SGLANG_API_KEY", "sk-qwen38-g4-cyber-9f8a7b6c5d4e3f2a")
ENABLE_MTP = os.environ.get("ENABLE_MTP", "1") not in ("0", "false", "False", "no")
ENABLE_TUNNEL = os.environ.get("ENABLE_TUNNEL", "1") not in ("0", "false", "False", "no")
WEIGHT_SOURCE = os.environ.get("WEIGHT_SOURCE", "auto")  # auto | hf | drive
BACKUP_TO_DRIVE = os.environ.get("BACKUP_TO_DRIVE", "0") in ("1", "true", "True", "yes")

# Pop transient per-invocation flags so they never leak across persistent Jupyter kernel cells
WATCHDOG_ONLY = os.environ.pop("WATCHDOG_ONLY", "0") in ("1", "true", "True", "yes")
KEEPALIVE_SECONDS = int(os.environ.pop("COLAB_KEEPALIVE_SECONDS", "43200"))

CF_URL_REGEX = re.compile(r"https://(?!api\.)[a-zA-Z0-9-]+\.trycloudflare\.com")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_running_on_colab_vm() -> bool:
    return os.path.exists("/var/colab/hostname") or (
        os.path.isdir("/content") and sys.platform.startswith("linux")
    )


# ============================================================================
# Remote Colab VM Execution Logic
# ============================================================================

def run_cmd(cmd: str, log_path: str = "/content/cmd.log", check: bool = True) -> int:
    with open(log_path, "a") as f:
        p = subprocess.run(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    if check and p.returncode != 0:
        tail = subprocess.getoutput(f"tail -n 30 {log_path}")
        raise RuntimeError(f"Command failed (rc={p.returncode}): {cmd}\n{tail}")
    return p.returncode


def check_sglang_healthy(api_key: str = API_KEY, port: int = 30000) -> bool:
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            return resp.status == 200
    except Exception:
        return False


def step1_install_and_restore_cache() -> float:
    t0 = time.time()
    log("Step 1/4: Installing SGLang >= 0.5.23, ModelOpt, cloudflared & restoring precompiled SM120 kernels...")

    with open("/tmp/overrides.txt", "w") as f:
        f.write("outlines-core>=0.2.11\n")

    run_cmd(
        'export PATH="/tools/node/bin:/usr/local/bin:$PATH" && '
        'if ! command -v cloudflared >/dev/null 2>&1; then '
        '  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && '
        '  chmod +x /usr/local/bin/cloudflared; '
        'fi && '
        'if [ ! -f /usr/bin/ninja ]; then apt-get update -qq && apt-get install -y -qq ninja-build; fi',
        log_path="/content/setup_env.log",
    )
    run_cmd(
        'uv pip install --system --overrides /tmp/overrides.txt "sglang[all]>=0.5.23" nvidia-modelopt hf_transfer',
        log_path="/content/setup_env.log",
    )

    moe_so_dir = "/root/.cache/sglang/.cache/flashinfer/0.7.2/120f/cached_ops/fused_moe_120"
    moe_so_path = f"{moe_so_dir}/fused_moe_120.so"
    os.makedirs(moe_so_dir, exist_ok=True)

    if not os.path.exists(moe_so_path) or os.path.getsize(moe_so_path) < 1000000:
        run_cmd(
            f'curl -fsSL "{GITHUB_RELEASE_BASE}/fused_moe_120.so.xz" -o /tmp/fused_moe_120.so.xz',
            log_path="/content/setup_env.log",
            check=False,
        )
        if os.path.exists("/tmp/fused_moe_120.so.xz") and os.path.getsize("/tmp/fused_moe_120.so.xz") > 1000000:
            run_cmd(f"xz -dc /tmp/fused_moe_120.so.xz > {moe_so_path} && chmod +x {moe_so_path} && rm -f /tmp/fused_moe_120.so.xz")
            log("Restored precompiled fused_moe_120.so (60MB) from GitHub Release!")

    if not os.path.isdir("/root/.cache/sglang/triton"):
        run_cmd(
            f'curl -fsSL "{GITHUB_RELEASE_BASE}/other_kernels.tar.xz" -o /tmp/other_kernels.tar.xz && '
            f'tar -xJf /tmp/other_kernels.tar.xz -C /root && rm -f /tmp/other_kernels.tar.xz',
            log_path="/content/setup_env.log",
            check=False,
        )
        log("Restored precompiled Triton & FlashInfer auxiliary kernels from GitHub Release!")

    # Install ninja wrapper so FlashInfer immediately uses precompiled .so without rebuilding 96 .cuda.o files
    ninja_wrapper = """#!/usr/bin/env python3
import glob, os, sys
if "-C" in sys.argv:
    d = sys.argv[sys.argv.index("-C") + 1]
    if any(os.path.getsize(p) > 10240 for p in glob.glob(os.path.join(d, "*.so"))):
        sys.exit(0)
os.execv("/usr/bin/ninja", ["/usr/bin/ninja"] + sys.argv[1:])
"""
    if os.path.exists("/usr/local/bin/ninja"):
        os.unlink("/usr/local/bin/ninja")
    with open("/usr/local/bin/ninja", "w") as f:
        f.write(ninja_wrapper)
    os.chmod("/usr/local/bin/ninja", 0o755)
    dt = time.time() - t0
    log(f"Step 1/4 complete in {dt:.1f}s!")
    return dt


def prepare_mtp_draft_dir():
    """Create a 3-shard symlink directory containing only MTP draft weights so MTP draft loads in ~1.8s instead of ~202s!"""
    os.makedirs(LOCAL_MTP_DRAFT_DIR, exist_ok=True)
    idx_path = os.path.join(LOCAL_MODEL_DIR, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        return
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    mtp_wm = {k: v for k, v in wm.items() if "mtp" in k}
    mtp_shards = sorted(set(mtp_wm.values()))

    for fname in os.listdir(LOCAL_MODEL_DIR):
        if fname.startswith(".") or fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        src = os.path.join(LOCAL_MODEL_DIR, fname)
        dst = os.path.join(LOCAL_MTP_DRAFT_DIR, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    for shard in mtp_shards:
        src = os.path.join(LOCAL_MODEL_DIR, shard)
        dst = os.path.join(LOCAL_MTP_DRAFT_DIR, shard)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    with open(os.path.join(LOCAL_MTP_DRAFT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": idx.get("metadata", {}), "weight_map": mtp_wm}, f, indent=2)


def copy_from_drive_to_nvme():
    log(f"Copying weights from Google Drive ({DRIVE_MODEL_DIR}) to local NVMe ({LOCAL_MODEL_DIR}) with 16 threads...")
    all_files = [f for f in os.listdir(DRIVE_MODEL_DIR) if not f.startswith(".")]

    def copy_one(fname):
        src = os.path.join(DRIVE_MODEL_DIR, fname)
        dst = os.path.join(LOCAL_MODEL_DIR, fname)
        if os.path.isdir(src):
            return
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return
        with open(src, "rb") as fsrc, open(dst + ".tmp", "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
        os.replace(dst + ".tmp", dst)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(copy_one, all_files))


def download_from_hf_to_nvme():
    log(f"Downloading weights to local NVMe {LOCAL_MODEL_DIR} via HuggingFace Xet (~400 MB/s)...")
    with open("/content/hf_download.log", "w") as f:
        p = subprocess.Popen(
            f'export PATH="/tools/node/bin:/usr/local/bin:$PATH" && '
            f'HF_XET_HIGH_PERFORMANCE=1 hf download {MODEL_ID} --local-dir {LOCAL_MODEL_DIR}',
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    # Monitor hf_download.log; hf-xet can hang at C++ atexit after printing "Downloaded", so kill once all 206 shards are on disk
    last_prog = ""
    while p.poll() is None:
        dl_txt = subprocess.getoutput("cat /content/hf_download.log 2>/dev/null | tr '\r' '\n'")
        lines = [l for l in dl_txt.splitlines() if l.strip()]
        if lines and lines[-1] != last_prog:
            last_prog = lines[-1]
            if "Fetching" in last_prog or "Downloaded" in last_prog:
                log(f"  [hf] {last_prog[:100]}")
        if "Downloaded" in dl_txt and len(glob.glob(os.path.join(LOCAL_MODEL_DIR, "*.safetensors"))) >= 206:
            subprocess.run(f"kill -9 -{p.pid} 2>/dev/null || pkill -9 -f 'hf download' || true", shell=True)
            break
        time.sleep(5)


def step2_prepare_weights() -> float:
    t0 = time.time()
    log("Step 2/4: Preparing model weights on local NVMe (/content/...)...")
    if os.path.islink(LOCAL_MODEL_DIR):
        os.unlink(LOCAL_MODEL_DIR)
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    existing_shards = glob.glob(os.path.join(LOCAL_MODEL_DIR, "*.safetensors"))
    if len(existing_shards) >= 206 and os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        prepare_mtp_draft_dir()
        dt = time.time() - t0
        log(f"All {len(existing_shards)} safetensors shards + 3-shard MTP draft dir ready on local NVMe ({dt:.1f}s)!")
        return dt

    drive_shards = glob.glob(os.path.join(DRIVE_MODEL_DIR, "*.safetensors")) if os.path.isdir(DRIVE_MODEL_DIR) else []

    if WEIGHT_SOURCE == "drive" and len(drive_shards) >= 206:
        copy_from_drive_to_nvme()
    else:
        if os.path.isdir(DRIVE_MODEL_DIR):
            for fname in os.listdir(DRIVE_MODEL_DIR):
                if not fname.startswith(".") and not fname.endswith(".safetensors"):
                    src = os.path.join(DRIVE_MODEL_DIR, fname)
                    dst = os.path.join(LOCAL_MODEL_DIR, fname)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
        download_from_hf_to_nvme()

    prepare_mtp_draft_dir()

    dt = time.time() - t0
    log(f"Step 2/4 complete: 206 shards + 3-shard MTP draft dir ready on local NVMe in {dt:.1f}s!")
    return dt


def maybe_start_drive_backup_in_background():
    """Start Google Drive backup AFTER SGLang finishes loading weights so Drive FUSE
    writeback never competes with SGLang for NVMe I/O or host RAM page cache.
    """
    if not BACKUP_TO_DRIVE or not os.path.isdir("/content/drive/MyDrive"):
        return
    drive_shards = glob.glob(os.path.join(DRIVE_MODEL_DIR, "*.safetensors")) if os.path.isdir(DRIVE_MODEL_DIR) else []
    if len(drive_shards) >= 206:
        return
    if subprocess.getoutput("pgrep -f 'cp -rn /content/Qwen3.8'").strip():
        return

    log(f"Starting low-priority background backup of weights to Google Drive ({DRIVE_MODEL_DIR})...")
    os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
    subprocess.Popen(
        [
            "nohup", "ionice", "-c", "3", "nice", "-n", "19",
            "cp", "-rn", f"{LOCAL_MODEL_DIR}/.", f"{DRIVE_MODEL_DIR}/",
        ],
        stdout=open("/content/drive_backup.log", "w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


def step3_start_tunnel(force_restart: bool = False) -> str:
    if not ENABLE_TUNNEL:
        log("Step 3/4: Skipping Cloudflare Tunnel (using Colab SSH port-forwarding).")
        return "http://127.0.0.1:30000"

    log("Step 3/4: Starting Cloudflare Quick Tunnel (trycloudflare.com)...")
    if force_restart:
        subprocess.run("pkill -9 -f '[c]loudflared' || true", shell=True)
        time.sleep(1)

    if not subprocess.getoutput("pgrep -f '[c]loudflared'").strip():
        cf_sh = """#!/bin/bash
export PATH="/tools/node/bin:/usr/local/bin:$PATH"
exec cloudflared --no-autoupdate tunnel --url http://localhost:30000
"""
        with open("/content/start_cf.sh", "w") as f:
            f.write(cf_sh)
        subprocess.Popen(
            ["nohup", "bash", "/content/start_cf.sh"],
            stdout=open("/content/cf_tunnel.log", "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    tunnel_url = None
    for _ in range(30):
        log_txt = subprocess.getoutput("cat /content/cf_tunnel.log 2>/dev/null")
        m = CF_URL_REGEX.search(log_txt)
        if m:
            tunnel_url = m.group(0)
            break
        time.sleep(1)

    log(f"Step 3/4 complete: Cloudflare Tunnel active at {tunnel_url}")
    return tunnel_url


def build_sglang_script() -> str:
    if ENABLE_MTP:
        spec_flags = f"""  --speculative-draft-model-path {LOCAL_MTP_DRAFT_DIR} \\
  --speculative-algorithm NEXTN \\
  --speculative-num-steps 3 \\
  --speculative-eagle-topk 1 \\
  --speculative-num-draft-tokens 4 \\
  --speculative-draft-kv-cache-dtype fp8_e4m3 \\
  --enable-linear-replayssm-spec \\
  --max-running-requests 4 \\
  --max-mamba-cache-size 20 \\
  --chunked-prefill-size 4096 \\
  --mem-fraction-static 0.95 \\"""
    else:
        spec_flags = """  --max-running-requests 8 \\
  --max-mamba-cache-size 24 \\
  --chunked-prefill-size 8192 \\
  --mem-fraction-static 0.93 \\"""

    return f"""#!/bin/bash
set -ex
export PATH="/tools/node/bin:/usr/local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
export FLASHINFER_JIT_MAX_JOBS=4

exec sglang serve \\
  --model-path {LOCAL_MODEL_DIR} \\
  --served-model-name {MODEL_ID} \\
  --tp-size 1 \\
  --quantization modelopt_fp4 \\
  --attention-backend flashinfer \\
  --kv-cache-dtype fp8_e4m3 \\
  --mamba-ssm-dtype bfloat16 \\
  --ple-offload-embedding \\
  --mamba-radix-cache-strategy extra_buffer_lazy \\
{spec_flags}
  --disable-flashinfer-autotune \\
  --weight-loader-drop-cache-after-load \\
  --cuda-graph-max-bs-decode 4 \\
  --disable-prefill-cuda-graph \\
  --reasoning-parser qwen3 \\
  --tool-call-parser qwen3_coder \\
  --model-loader-extra-config '{{"enable_multithread_load": true}}' \\
  --api-key {API_KEY} \\
  --host 0.0.0.0 \\
  --port 30000
"""


def step4_start_sglang_and_wait(force_restart: bool = False) -> float:
    mode_label = "MTP Speculative Decoding (NEXTN)" if ENABLE_MTP else "Standard High-Capacity (No-MTP)"
    sglang_sh = build_sglang_script()

    existing_sh = ""
    if os.path.exists("/content/start_sglang.sh"):
        with open("/content/start_sglang.sh") as f:
            existing_sh = f.read()

    if not force_restart and existing_sh == sglang_sh and check_sglang_healthy():
        log(f"Step 4/4: SGLang Server [{mode_label}] is already running and healthy!")
        return 0.1

    log(f"Step 4/4: Launching SGLang Server [{mode_label} + 262K Context]...")
    subprocess.run("pkill -9 -f '[s]glang' || true", shell=True)

    with open("/content/start_sglang.sh", "w") as f:
        f.write(sglang_sh)
    subprocess.Popen(
        ["nohup", "bash", "/content/start_sglang.sh"],
        stdout=open("/content/sglang.log", "w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )

    t0 = time.time()
    last_line = ""
    while time.time() - t0 < 2400:
        sgl_txt = subprocess.getoutput("cat /content/sglang.log 2>/dev/null | tr '\r' '\n'")
        lines = sgl_txt.splitlines()
        if "The server is fired up and ready to roll!" in sgl_txt and check_sglang_healthy():
            dt = time.time() - t0
            log(f"Step 4/4 complete: SGLang server is READY in {dt:.1f}s!")
            return dt

        if lines and lines[-1] != last_line:
            last_line = lines[-1]
            log(f"  [sglang] {last_line[:120]}")
        if any("Rank 0 scheduler died" in l or "SIGQUIT received" in l for l in lines[-15:]):
            raise RuntimeError("SGLang scheduler died during initialization:\n" + "\n".join(lines[-35:]))
        time.sleep(10)

    raise TimeoutError("Timed out waiting for SGLang server to become ready.")


def run_remote_watchdog_loop(duration_sec: int):
    """Hold the Jupyter kernel BUSY with periodic iopub heartbeats so Google Colab
    does not reap the VM after ~20m of kernel/WebSocket inactivity, while also
    auto-healing SGLang and Cloudflare Tunnel if either process crashes.
    """
    if duration_sec <= 0:
        return

    t_start = time.time()
    last_heartbeat = 0.0
    sglang_failures = 0

    while time.time() - t_start < duration_sec:
        # 1. Monitor SGLang health
        if check_sglang_healthy():
            sglang_failures = 0
        else:
            sglang_failures += 1
            proc_alive = bool(subprocess.getoutput("pgrep -f '[s]glang'").strip())
            if not proc_alive or sglang_failures >= 3:
                log("[watchdog] SGLang process exited or failed health checks; restarting...")
                step4_start_sglang_and_wait(force_restart=True)
                sglang_failures = 0

        # 2. Monitor Cloudflare Tunnel health (if enabled)
        tunnel_status = "SSH"
        if ENABLE_TUNNEL:
            cf_alive = bool(subprocess.getoutput("pgrep -f '[c]loudflared'").strip())
            if not cf_alive:
                log("[watchdog] cloudflared exited; restarting Quick Tunnel...")
                new_url = step3_start_tunnel(force_restart=True)
                if new_url:
                    print(f"TUNNEL_BASE_URL={new_url}/v1", flush=True)
            log_txt = subprocess.getoutput("cat /content/cf_tunnel.log 2>/dev/null")
            m = CF_URL_REGEX.search(log_txt)
            tunnel_status = m.group(0) if m else "reconnecting"

        # 3. Emit periodic iopub stream output to keep Colab kernel & proxy WebSocket active
        now = time.time()
        if now - last_heartbeat >= 60:
            gpu_info = subprocess.getoutput(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                "--format=csv,noheader,nounits 2>/dev/null"
            ).strip()
            gpu_summary = f" | GPU: {gpu_info}" if gpu_info else ""
            sglang_status = "OK" if sglang_failures == 0 else f"UNHEALTHY({sglang_failures})"
            log(f"[keepalive] SGLang={sglang_status} | Tunnel={tunnel_status}{gpu_summary}")
            last_heartbeat = now

        remaining = duration_sec - (time.time() - t_start)
        if remaining <= 0:
            break
        time.sleep(min(20.0, remaining))


def run_remote_vm_main():
    if WATCHDOG_ONLY:
        run_remote_watchdog_loop(KEEPALIVE_SECONDS)
        return

    t_start = time.time()
    dt_env = step1_install_and_restore_cache()
    dt_weights = step2_prepare_weights()
    tunnel_url = step3_start_tunnel()
    dt_sglang = step4_start_sglang_and_wait()
    maybe_start_drive_backup_in_background()
    total_dt = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"DEPLOYMENT SUCCESSFUL (total: {total_dt:.1f}s | env: {dt_env:.1f}s | weights: {dt_weights:.1f}s | sglang+warmup: {dt_sglang:.1f}s)")
    print(f"TUNNEL_BASE_URL={tunnel_url}/v1")
    print(f"API_KEY={API_KEY}")
    print(f"MODEL_ID={MODEL_ID}")
    print("=" * 72, flush=True)

    if KEEPALIVE_SECONDS > 0:
        log(
            f"Keep-alive watchdog active for {KEEPALIVE_SECONDS // 3600}h "
            f"(holding Jupyter kernel busy so Colab does not auto-stop the VM)."
        )
        log("Keep this process running, or press Ctrl+C to detach (use 'colab stop' when done).")
        try:
            run_remote_watchdog_loop(KEEPALIVE_SECONDS)
        except KeyboardInterrupt:
            log("Detached from keep-alive watchdog (VM is still running; stop via 'colab stop').")


# ============================================================================
# Local Keep-Alive & SSH Tunnel Daemon (prevents ~20m Colab idle disconnect)
# ============================================================================

CACHE_DIR = Path.home() / ".cache" / "colab-qwen38-g4"


def get_daemon_paths(session_name: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_name)
    return (
        CACHE_DIR / f"keepalive-{safe_name}.pid",
        CACHE_DIR / f"keepalive-{safe_name}.log",
    )


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_local_keepalive_daemon(session_name: str, quiet: bool = False) -> bool:
    pid_path, _ = get_daemon_paths(session_name)
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except Exception:
        pid_path.unlink(missing_ok=True)
        return False

    stopped = False
    if is_pid_alive(pid):
        cmdline = subprocess.getoutput(f"ps -p {pid} -o command=").strip()
        if "run_qwen38_g4.py" in cmdline or "keepalive-daemon" in cmdline or "python" in cmdline.lower():
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(25):
                    if not is_pid_alive(pid):
                        break
                    time.sleep(0.1)
                if is_pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)
                stopped = True
                if not quiet:
                    print(f"✓ Stopped previous keep-alive daemon for '{session_name}' (PID {pid})")
            except OSError:
                pass
    pid_path.unlink(missing_ok=True)
    return stopped


def check_colab_session_state(session_name: str) -> str:
    """Return 'active', 'absent', or 'unknown' (e.g. transient local network error)."""
    try:
        res = subprocess.run(
            ["colab", "sessions"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if res.returncode != 0:
            return "unknown"
        out = (res.stdout or "") + "\n" + (res.stderr or "")
        return "active" if f"[{session_name}]" in out else "absent"
    except Exception:
        return "unknown"


def ensure_colab_ssh_identity() -> Path:
    """Find an Ed25519/ECDSA SSH key accepted by Colab SSH, or create a dedicated Ed25519 key."""
    ssh_dir = Path.home() / ".ssh"
    for key_name in ("id_ed25519", "id_ecdsa"):
        priv = ssh_dir / key_name
        pub = ssh_dir / f"{key_name}.pub"
        if priv.exists() and pub.exists():
            return priv

    colab_key_dir = Path.home() / ".config" / "colab-cli"
    colab_key_dir.mkdir(parents=True, exist_ok=True)
    priv = colab_key_dir / "id_ed25519"
    pub = colab_key_dir / "id_ed25519.pub"
    if not (priv.exists() and pub.exists()):
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(priv), "-q"],
            check=True,
        )
    return priv


def is_local_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_local_keepalive_daemon(
    session_name: str,
    tunnel_mode: str,
    ssh_port: int,
    api_key: str,
    enable_mtp: str,
    update_pi: bool,
):
    pid_path, _ = get_daemon_paths(session_name)
    pid_path.write_text(f"{os.getpid()}\n")

    stop_event = threading.Event()
    child_lock = threading.Lock()
    children: dict[str, subprocess.Popen | None] = {"exec": None, "ssh": None}

    def terminate_children():
        with child_lock:
            for key, proc in list(children.items()):
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                children[key] = None

    def handle_signal(signum, frame):
        stop_event.set()
        terminate_children()
        pid_path.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"[daemon] Started keep-alive & auto-heal daemon (PID {os.getpid()}) for session '{session_name}' (mode={tunnel_mode}).")

    def ssh_forward_worker():
        identity = ensure_colab_ssh_identity()
        colab_bin = shutil.which("colab") or "colab"
        proxy_cmd = (
            f"{shlex.quote(colab_bin)} ssh --proxy-mode "
            f"-s {shlex.quote(session_name)} -i {shlex.quote(str(identity))}"
        )
        while not stop_event.is_set():
            # Refresh token before connecting SSH WebSocket. Only connect when the session is
            # confirmed active: `colab ssh --proxy-mode -s NAME` auto-creates a missing session.
            state = check_colab_session_state(session_name)
            if state == "absent":
                break
            if state != "active":
                stop_event.wait(5.0)
                continue
            ssh_cmd = [
                "ssh",
                "-N",
                "-L", f"127.0.0.1:{ssh_port}:127.0.0.1:30000",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ServerAliveInterval=20",
                "-o", "ServerAliveCountMax=3",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", f"ProxyCommand={proxy_cmd}",
                "-i", str(identity),
                "root@colab-runtime",
            ]
            log(f"[daemon] Connecting Colab SSH port-forward 127.0.0.1:{ssh_port} -> VM:30000...")
            proc = subprocess.Popen(
                ssh_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with child_lock:
                children["ssh"] = proc
            for line in proc.stdout:
                if line.strip():
                    log(f"  [ssh] {line.strip()}")
            proc.wait()
            if stop_event.is_set():
                break
            log(f"[daemon] SSH tunnel disconnected (rc={proc.returncode}); reconnecting in 3s...")
            stop_event.wait(3.0)

    if tunnel_mode == "ssh":
        threading.Thread(target=ssh_forward_worker, daemon=True).start()

    # Run 15-minute (900s) kernel watchdog slices and refresh the 1h Colab runtimeProxyInfo
    # token via `colab sessions` between slices so neither the kernel nor token ever expires.
    absent_streak = 0
    enable_tunnel_flag = "0" if tunnel_mode == "ssh" else "1"
    try:
        while not stop_event.is_set():
            state = check_colab_session_state(session_name)
            if state == "absent":
                absent_streak += 1
                if absent_streak >= 2:
                    log(f"[daemon] Session '{session_name}' is no longer active on Colab. Exiting daemon.")
                    break
                stop_event.wait(5.0)
                continue
            absent_streak = 0

            cmd = [
                "colab", "exec", "-s", session_name,
                "--timeout", "1200",
                "--env", "WATCHDOG_ONLY=1",
                "--env", "COLAB_KEEPALIVE_SECONDS=900",
                "--env", f"COLAB_SGLANG_API_KEY={api_key}",
                "--env", f"ENABLE_MTP={enable_mtp}",
                "--env", f"ENABLE_TUNNEL={enable_tunnel_flag}",
                "-f", os.path.abspath(__file__),
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with child_lock:
                children["exec"] = proc

            for line in proc.stdout:
                text = line.rstrip("\n")
                if text:
                    print(text, flush=True)
                if text.startswith("TUNNEL_BASE_URL=") and update_pi and tunnel_mode == "cf":
                    new_base_url = text.split("=", 1)[1].strip()
                    try:
                        update_pi_models_json(new_base_url, api_key)
                        log(f"[daemon] Auto-updated ~/.pi/agent/models.json -> {new_base_url}")
                    except Exception as e:
                        log(f"[daemon] Failed to update models.json: {e}")

            proc.wait()
            if stop_event.is_set():
                break
            stop_event.wait(2.0)
    finally:
        stop_event.set()
        terminate_children()
        # Run `colab sessions` once more to prune any stale local state written by `colab exec` exit
        try:
            subprocess.run(["colab", "sessions"], capture_output=True, timeout=15)
        except Exception:
            pass
        pid_path.unlink(missing_ok=True)


def spawn_background_keepalive_daemon(
    session_name: str,
    tunnel_mode: str,
    ssh_port: int,
    api_key: str,
    enable_mtp: str,
    update_pi: bool,
) -> tuple[int, Path]:
    stop_local_keepalive_daemon(session_name, quiet=True)
    _, log_path = get_daemon_paths(session_name)

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--keepalive-daemon",
        "--session", session_name,
        "--tunnel-mode", tunnel_mode,
        "--ssh-port", str(ssh_port),
        "--api-key", api_key,
        "--enable-mtp", enable_mtp,
    ]
    if update_pi:
        cmd.append("--update-pi")

    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    return proc.pid, log_path


# ============================================================================
# Local Interactive Wizard & Helper CLI (when invoked locally)
# ============================================================================

def prompt_choice(prompt_text: str, default: str) -> str:
    if not sys.stdin.isatty() or "--yes" in sys.argv or "-y" in sys.argv:
        return default
    try:
        val = input(f"{prompt_text} [{default}]: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)


def update_pi_models_json(base_url: str, api_key: str):
    pi_models_path = Path.home() / ".pi" / "agent" / "models.json"
    pi_models_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(pi_models_path.read_text()) if pi_models_path.exists() else {"providers": {}}
    data.setdefault("providers", {})["colab-g4"] = {
        "name": "Colab G4 SGLang",
        "baseUrl": base_url,
        "api": "openai-completions",
        "apiKey": api_key,
        "models": [
            {
                "id": MODEL_ID,
                "name": "Qwen 3.8 Flash Next Cybersecurity NVFP4 (Colab G4)",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 262144,
                "maxTokens": 65536,
                "compat": {
                    "supportsStore": False,
                    "supportsDeveloperRole": False,
                },
                "thinkingLevelMap": {
                    "off": None,
                    "minimal": "low",
                    "low": "low",
                    "medium": "medium",
                    "high": "xhigh",
                    "xhigh": "xhigh",
                    "max": "xhigh",
                },
                "samplingParams": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                },
                "samplingParamsByThinkingLevel": {
                    "off": {
                        "temperature": 0.7,
                        "top_p": 0.8,
                    }
                },
            }
        ],
    }
    pi_models_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"✓ Updated {pi_models_path} (provider: colab-g4 -> {base_url})")


def run_local_interactive_wizard():
    print("=" * 72)
    print(" Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4 · Google Colab G4 (96GB) Wizard")
    print("=" * 72)

    if not shutil.which("colab"):
        print("Error: 'colab' CLI not found. Install it first via:")
        print("  uv tool install google-colab-cli")
        sys.exit(1)

    session_name = prompt_choice("1. Colab session name", "qwen38-g4")

    print("\n2. Weight loading & Google Drive strategy:")
    print("   [1] Fast Direct NVMe (Recommended for first-time users: ~400 MB/s via HF Xet, ~6.5 min, no Drive quota needed)")
    print("   [2] Mount Google Drive (Reuse cached weights from Drive, or download & back up to Drive if first time)")
    drive_choice = prompt_choice("   Select strategy (1 or 2)", "1")

    print("\n3. Inference acceleration mode:")
    print("   [1] Enable MTP Speculative Decoding (NEXTN: 120~176 tok/s, 472K FP8 KV cache pool, 262K context)")
    print("   [2] High-Capacity Mode (No-MTP: ~81 tok/s, 946K FP8 KV cache pool, higher concurrency)")
    mtp_choice = prompt_choice("   Select mode (1 or 2)", "1")
    enable_mtp = "0" if mtp_choice == "2" else "1"

    print("\n4. Endpoint tunnel & keep-alive strategy:")
    print("   [1] Cloudflare Quick Tunnel + Background Keep-Alive (https://*.trycloudflare.com/v1)")
    print("   [2] Local SSH Port-Forward + Background Keep-Alive (http://127.0.0.1:30000/v1 via Colab SSH, lower latency)")
    tunnel_choice = prompt_choice("   Select tunnel mode (1 or 2)", "1")
    tunnel_mode = "ssh" if tunnel_choice == "2" else "cf"
    ssh_port = 30000
    if tunnel_mode == "ssh" and is_local_port_open(ssh_port):
        ssh_port_str = prompt_choice("   Local port 30000 is in use; choose local forward port", "30001")
        ssh_port = int(ssh_port_str)

    default_key = os.environ.get("COLAB_SGLANG_API_KEY", f"sk-qwen38-{secrets.token_hex(8)}")
    api_key = prompt_choice("\n5. SGLang API Key", default_key)

    # Stop any previous local keep-alive daemon before touching the remote kernel
    was_daemon_running = stop_local_keepalive_daemon(session_name)

    sess_out = subprocess.getoutput("colab sessions")
    if f"[{session_name}]" not in sess_out:
        print(f"\n-> Allocating Google Colab G4 GPU instance '{session_name}'...")
        subprocess.run(["colab", "new", "-s", session_name, "--gpu", "G4"], check=True)
    else:
        print(f"\n-> Reusing existing Colab G4 session '{session_name}'.")
        if was_daemon_running:
            # Clear the running watchdog cell in ipykernel (does not affect nohup sglang/cloudflared)
            subprocess.run(["colab", "restart-kernel", "-s", session_name], capture_output=True)

    weight_source = "hf"
    backup_to_drive = "0"
    if drive_choice == "2":
        print(f"\n-> Mounting Google Drive on session '{session_name}' (follow browser prompt if asked)...")
        subprocess.run(["colab", "drivemount", "-s", session_name], check=True)
        reuse_choice = prompt_choice(
            "   If Drive does not have the 126 GiB weights yet, back them up to Drive in background? (y/n)",
            "y",
        )
        weight_source = "auto"
        backup_to_drive = "1" if reuse_choice.lower().startswith("y") else "0"

    print(f"\n-> Executing deployment on remote Colab G4 VM '{session_name}'...")
    cmd = [
        "colab", "exec", "-s", session_name,
        "--timeout", "3600",
        "--env", "WATCHDOG_ONLY=0",
        "--env", "COLAB_KEEPALIVE_SECONDS=0",
        "--env", f"COLAB_SGLANG_API_KEY={api_key}",
        "--env", f"ENABLE_MTP={enable_mtp}",
        "--env", f"ENABLE_TUNNEL={'0' if tunnel_mode == 'ssh' else '1'}",
        "--env", f"WEIGHT_SOURCE={weight_source}",
        "--env", f"BACKUP_TO_DRIVE={backup_to_drive}",
        "-f", os.path.abspath(__file__),
    ]

    tunnel_base_url = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("TUNNEL_BASE_URL="):
            tunnel_base_url = line.split("=", 1)[1].strip()
    proc.wait()

    if proc.returncode != 0:
        sys.exit(proc.returncode)

    if tunnel_mode == "ssh":
        tunnel_base_url = f"http://127.0.0.1:{ssh_port}/v1"

    pi_choice = prompt_choice("\n6. Update ~/.pi/agent/models.json with this endpoint? (y/n)", "y")
    should_update_pi = pi_choice.lower().startswith("y")
    if should_update_pi and tunnel_base_url:
        update_pi_models_json(tunnel_base_url, api_key)

    pid, log_path = spawn_background_keepalive_daemon(
        session_name=session_name,
        tunnel_mode=tunnel_mode,
        ssh_port=ssh_port,
        api_key=api_key,
        enable_mtp=enable_mtp,
        update_pi=should_update_pi,
    )

    if tunnel_mode == "ssh":
        print(f"\n-> Waiting for Colab SSH local port-forward (127.0.0.1:{ssh_port})...")
        for _ in range(25):
            if check_sglang_healthy(api_key=api_key, port=ssh_port):
                print(f"✓ SSH Tunnel ready at {tunnel_base_url}")
                break
            time.sleep(1)

    print(f"✓ Background keep-alive & auto-heal daemon active (PID {pid})")
    print(f"  Log: {log_path}")
    print(f"  To stop instance & billing when done: python3 {os.path.basename(__file__)} --stop -s {session_name}")
    if should_update_pi:
        print("\nTest with pi:")
        print(f'  pi --model "colab-g4/{MODEL_ID}" -p "看一下本机配置"')


def handle_local_cli():
    parser = argparse.ArgumentParser(description="Colab G4 SGLang Deployment & Keep-Alive Manager")
    parser.add_argument("-s", "--session", default="qwen38-g4", help="Colab session name (default: qwen38-g4)")
    parser.add_argument("-y", "--yes", action="store_true", help="Accept default wizard choices non-interactively")
    parser.add_argument("--keepalive", action="store_true", help="Start/reattach background keep-alive daemon for an existing session")
    parser.add_argument("--status", action="store_true", help="Show Colab session and local keep-alive daemon status")
    parser.add_argument("--stop", action="store_true", help="Stop local keep-alive daemon and terminate Colab VM")
    parser.add_argument("--tunnel-mode", choices=["cf", "ssh"], default="cf", help="Tunnel mode for --keepalive")
    parser.add_argument("--ssh-port", type=int, default=30000, help="Local port for --tunnel-mode ssh")
    parser.add_argument("--api-key", default=None, help="SGLang API key")
    parser.add_argument("--enable-mtp", default="1", help="MTP setting (1 or 0)")
    parser.add_argument("--update-pi", action="store_true", help="Auto-update ~/.pi/agent/models.json if tunnel URL changes")
    parser.add_argument("--keepalive-daemon", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Load existing API key from ~/.pi/agent/models.json if not explicitly supplied
    resolved_api_key = args.api_key or API_KEY
    pi_models_path = Path.home() / ".pi" / "agent" / "models.json"
    if not args.api_key and pi_models_path.exists():
        try:
            pi_cfg = json.loads(pi_models_path.read_text())
            resolved_api_key = pi_cfg.get("providers", {}).get("colab-g4", {}).get("apiKey") or resolved_api_key
        except Exception:
            pass

    if args.keepalive_daemon:
        run_local_keepalive_daemon(
            session_name=args.session,
            tunnel_mode=args.tunnel_mode,
            ssh_port=args.ssh_port,
            api_key=resolved_api_key,
            enable_mtp=args.enable_mtp,
            update_pi=args.update_pi,
        )
        return

    if args.stop:
        stop_local_keepalive_daemon(args.session)
        subprocess.run(["colab", "stop", "-s", args.session])
        return

    if args.status:
        pid_path, log_path = get_daemon_paths(args.session)
        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            alive = is_pid_alive(pid)
            print(f"[keepalive] Daemon PID {pid}: {'RUNNING' if alive else 'DEAD'} (log: {log_path})")
        else:
            print(f"[keepalive] No local daemon running for '{args.session}'.")
        subprocess.run(["colab", "status", "-s", args.session])
        return

    if args.keepalive:
        if check_colab_session_state(args.session) != "active":
            print(f"Error: Colab session '{args.session}' is not active. Run `python3 run_qwen38_g4.py` to create one.")
            sys.exit(1)
        pid, log_path = spawn_background_keepalive_daemon(
            session_name=args.session,
            tunnel_mode=args.tunnel_mode,
            ssh_port=args.ssh_port,
            api_key=resolved_api_key,
            enable_mtp=args.enable_mtp,
            update_pi=True,
        )
        print(f"✓ Attached background keep-alive daemon (PID {pid}) to session '{args.session}'")
        print(f"  Log: {log_path}")
        return

    run_local_interactive_wizard()


if __name__ == "__main__":
    if is_running_on_colab_vm():
        run_remote_vm_main()
    else:
        handle_local_cli()
