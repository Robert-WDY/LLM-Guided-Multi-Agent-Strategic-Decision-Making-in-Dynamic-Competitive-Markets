"""CPU-only experiment service, limited to the lifetime of its owner."""

from contextlib import contextmanager, ExitStack
import atexit
from threading import Lock
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / ".tmp/independent-models"
PORTS = {"smol": 19943, "qwen": 19944}
CPU_ARGS = [
    "--device",
    "none",
    "--gpu-layers",
    "0",
    "--no-kv-offload",
    "--no-op-offload",
]


def cpu_executable(model_dir=MODELS):
    source = shutil.which("llama-server")
    cached = model_dir / "runtime" / "pricing-worker.exe"
    if os.name != "nt":
        if not source:
            raise RuntimeError("llama-server is missing")
        return Path(source)
    src = cached if cached.is_file() else Path(source) if source else None
    if not src:
        raise RuntimeError("llama-server is missing")
    folder = model_dir / "cpu-runtime"
    folder.mkdir(exist_ok=True, parents=True)
    # Isolate dynamically discovered backends, including at hardware discovery time.
    deps = [src, *src.parent.glob("ggml-cpu*.dll")]
    deps.extend(
        src.parent / f
        for f in (
            "ggml.dll",
            "ggml-base.dll",
            "llama.dll",
            "llama-common.dll",
            "llama-server-impl.dll",
            "mtmd.dll",
            "libomp.dll",
        )
        if (src.parent / f).is_file()
    )
    for f in deps:
        dest = folder / ("pricing-worker.exe" if f == src else f.name)
        if not dest.exists():
            try:
                os.link(f, dest)
            except OSError:
                shutil.copy2(f, dest)
        elif not os.path.samefile(f, dest) and f.read_bytes() != dest.read_bytes():
            raise RuntimeError(f"CPU runtime differs from source: {dest.name}")
    return folder / "pricing-worker.exe"


def bind_lifetime(process):
    """Windows closes this non-inherited job handle even if the controller is killed."""
    from ctypes import wintypes as wt

    size = ctypes.c_size_t

    class Limits(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_longlong),
            ("job_time", ctypes.c_longlong),
            ("flags", wt.DWORD),
            ("min_ws", size),
            ("max_ws", size),
            ("processes", wt.DWORD),
            ("affinity", size),
            ("priority", wt.DWORD),
            ("schedule", wt.DWORD),
        ]

    class Extended(ctypes.Structure):
        _fields_ = [
            ("basic", Limits),
            ("io", ctypes.c_ulonglong * 6),
            ("memory", size * 4),
        ]

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    create = api.CreateJobObjectW
    create.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    create.restype = wt.HANDLE
    configure = api.SetInformationJobObject
    configure.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    configure.restype = wt.BOOL
    assign = api.AssignProcessToJobObject
    assign.argtypes = [wt.HANDLE, wt.HANDLE]
    assign.restype = wt.BOOL
    close = api.CloseHandle
    close.argtypes = [wt.HANDLE]
    close.restype = wt.BOOL
    job = create(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = Extended()
    limits.basic.flags = 0x2000
    if not configure(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not assign(
        job, int(process._handle)
    ):
        error = ctypes.get_last_error()
        close(job)
        raise ctypes.WinError(error)
    return lambda: close(job)


@contextmanager
def service(name, port, *, slots=8, context=8192, threads=8, model_dir=MODELS):
    exe = cpu_executable(model_dir)
    endpoint = f"http://127.0.0.1:{port}/health"
    opener = build_opener(ProxyHandler({}))
    import socket

    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} already has an owner")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    env.pop("GGML_BACKEND_PATH", None)
    with ExitStack() as stack:
        log = stack.enter_context((model_dir / f"{name}-{port}-cpu.log").open("ab"))
        proc = subprocess.Popen(
            [
                str(exe),
                "-m",
                str(model_dir / (name + ".gguf")),
                "--alias",
                name,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-c",
                str(context),
                "-np",
                str(slots),
                "-t",
                str(threads),
                "-tb",
                str(threads),
                "--cache-ram",
                "1024",
                *CPU_ARGS,
            ],
            env=env,
            cwd=exe.parent,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        def stop():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        stack.callback(stop)
        if os.name == "nt":
            stack.callback(bind_lifetime(proc))
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError(f"{name} stopped; inspect its CPU runtime log")
            try:
                with opener.open(endpoint, timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError(f"{name} did not become ready")
        yield proc


_active = {}
_service_lock = Lock()
_owned_services = ExitStack()


def start_on_cache_miss(name, port):
    """Numeric stages use threads. Concurrent misses must share the same child."""
    with _service_lock:
        if not _active:
            atexit.register(_owned_services.close)
        if name not in _active:
            proc = _owned_services.enter_context(
                service(name, port, slots=2, context=2048, threads=2)
            )
            _active[name] = proc
        if _active[name].poll() is not None:
            raise RuntimeError(f"{name} CPU server exited during generation")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, choices=("smol", "qwen", "both"))
    parser.add_argument("script", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    cmd = parser.parse_args()
    with ExitStack() as stack:
        for name in PORTS if cmd.models == "both" else [cmd.models]:
            stack.enter_context(service(name, PORTS[name]))
        subprocess.run(
            [sys.executable, str(cmd.script.resolve()), *cmd.args], cwd=ROOT, check=True
        )
