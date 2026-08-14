from __future__ import annotations

import ctypes
import logging
import os
import re
import shutil
# Subprocess calls below use fixed executables and validated argument enums.
import subprocess  # nosec B404
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import psutil

from voice_assistant.config import ApplicationConfig, PcControlConfig


logger = logging.getLogger(__name__)

WM_CLOSE = 0x0010
KEYEVENTF_KEYUP = 0x0002
MEDIA_KEYS = {
    "play_pause": 0xB3,
    "stop": 0xB2,
    "next": 0xB0,
    "previous": 0xB1,
}
POWER_ACTIONS = frozenset({"shutdown", "restart", "logout", "sleep", "hibernate"})
PROTECTED_PROCESS_NAMES = frozenset(
    {
        "system",
        "registry",
        "smss.exe",
        "csrss.exe",
        "wininit.exe",
        "services.exe",
        "lsass.exe",
        "winlogon.exe",
        "svchost.exe",
        "dwm.exe",
        "explorer.exe",
        "fontdrvhost.exe",
        "sihost.exe",
        "taskhostw.exe",
    }
)


class PcControlError(RuntimeError):
    pass


class ApplicationNotFoundError(PcControlError):
    pass


class ProtectedProcessError(PcControlError):
    pass


class FeatureUnavailableError(PcControlError):
    pass


class WindowsPcController:
    """On-demand Windows actions constrained by trusted local configuration."""

    def __init__(self, config: PcControlConfig) -> None:
        self._config = config
        self._start_menu_cache: tuple[Path, ...] | None = None

    def open_application(self, application: ApplicationConfig) -> dict[str, Any]:
        if self.is_application_running(application)["running"]:
            return {"opened": False, "already_running": True}
        target = self._resolve_launch_target(application)
        logger.info(
            "Resolved application %s using trusted target %s.",
            application.identifier,
            target.name,
        )
        try:
            # The target comes only from trusted local allowlist discovery.
            os.startfile(str(target))  # nosec
        except (OSError, AttributeError) as exc:
            raise PcControlError(f"Could not launch {application.identifier}.") from exc
        return {"opened": True, "already_running": False, "resolved_target": target.name}

    def is_application_running(self, application: ApplicationConfig) -> dict[str, Any]:
        processes = self._matching_processes(application)
        return {"running": bool(processes), "process_count": len(processes)}

    def close_application(self, application: ApplicationConfig) -> dict[str, Any]:
        processes = self._matching_processes(application)
        if not processes:
            return {"closed": False, "already_closed": True, "still_running": False}
        self._assert_processes_are_safe(processes)
        target_pids = {process.pid for process in processes}
        windows_signalled = self._post_close_to_process_windows(target_pids)
        if windows_signalled:
            _, remaining = psutil.wait_procs(
                processes,
                timeout=self._config.graceful_close_timeout_seconds,
            )
        else:
            remaining = processes
        return {
            "closed": not remaining,
            "already_closed": False,
            "still_running": bool(remaining),
            "windows_signalled": windows_signalled,
        }

    def force_close_application(self, application: ApplicationConfig) -> dict[str, Any]:
        processes = self._matching_processes(application)
        if not processes:
            return {"closed": False, "already_closed": True}
        self._assert_processes_are_safe(processes)
        for process in processes:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                continue
            except (psutil.AccessDenied, OSError) as exc:
                raise PcControlError(
                    f"Access was denied while closing {application.identifier}."
                ) from exc
        _, remaining = psutil.wait_procs(processes, timeout=2.0)
        if remaining:
            raise PcControlError(f"{application.identifier} did not close.")
        return {"closed": True, "already_closed": False}

    def get_system_volume(self) -> dict[str, Any]:
        with self._endpoint_volume() as endpoint:
            percent = round(float(endpoint.GetMasterVolumeLevelScalar()) * 100)
            muted = bool(endpoint.GetMute())
        return {"percent": percent, "muted": muted, "device_scope": "windows_default_output"}

    def set_system_volume(self, percent: int) -> dict[str, Any]:
        bounded = max(0, min(100, int(percent)))
        with self._endpoint_volume() as endpoint:
            endpoint.SetMasterVolumeLevelScalar(bounded / 100.0, None)
        return self.get_system_volume()

    def change_system_volume(self, delta_percent: int) -> dict[str, Any]:
        current = self.get_system_volume()
        return self.set_system_volume(int(current["percent"]) + int(delta_percent))

    def set_system_mute(self, muted: bool) -> dict[str, Any]:
        with self._endpoint_volume() as endpoint:
            endpoint.SetMute(1 if muted else 0, None)
        return self.get_system_volume()

    def send_media_key(self, action: str) -> dict[str, Any]:
        if action not in MEDIA_KEYS:
            raise ValueError(f"Unsupported media action: {action}")
        self._require_windows()
        key = MEDIA_KEYS[action]
        try:
            ctypes.windll.user32.keybd_event(key, 0, 0, 0)
            ctypes.windll.user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0)
        except (AttributeError, OSError) as exc:
            raise PcControlError("Windows could not send the media key.") from exc
        return {"action": action, "sent": True}

    def lock_computer(self) -> dict[str, Any]:
        self._require_windows()
        try:
            succeeded = bool(ctypes.windll.user32.LockWorkStation())
        except (AttributeError, OSError) as exc:
            raise PcControlError("Windows could not lock the workstation.") from exc
        if not succeeded:
            raise PcControlError("Windows rejected the lock request.")
        return {"locked": True}

    def perform_power_action(self, action: str) -> dict[str, Any]:
        if action not in POWER_ACTIONS:
            raise ValueError(f"Unsupported power action: {action}")
        self._require_windows()
        if action in {"sleep", "hibernate"}:
            try:
                succeeded = bool(
                    ctypes.windll.powrprof.SetSuspendState(
                        action == "hibernate", False, False
                    )
                )
            except (AttributeError, OSError) as exc:
                raise PcControlError(f"Windows could not {action}.") from exc
            if not succeeded:
                raise PcControlError(f"Windows rejected the {action} request.")
            return {"action": action, "started": True}

        shutdown_exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "shutdown.exe"
        arguments = {
            "shutdown": ("/s", "/t", "10"),
            "restart": ("/r", "/t", "10"),
            "logout": ("/l",),
        }[action]
        try:
            # The executable and arguments come from fixed internal mappings.
            subprocess.Popen(  # nosec B603
                [str(shutdown_exe), *arguments],
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise PcControlError(f"Windows could not start {action}.") from exc
        return {"action": action, "started": True}

    def take_screenshot(self) -> dict[str, Any]:
        try:
            from PIL import ImageGrab
        except ImportError as exc:
            raise FeatureUnavailableError("Screenshot support is unavailable.") from exc
        directory = self._config.screenshot_directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now().astimezone()
        path = directory / f"screenshot_{created_at.strftime('%Y%m%d_%H%M%S_%f')}.png"
        try:
            image = ImageGrab.grab(all_screens=True)
            image.save(path, format="PNG")
        except (OSError, ValueError) as exc:
            raise PcControlError("Windows could not capture the screenshot.") from exc
        return {
            "path": str(path),
            "created_at": created_at.isoformat(),
            "width": image.width,
            "height": image.height,
        }

    def get_system_status(self) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        system_drive = Path(f"{os.environ.get('SystemDrive', 'C:')}\\")
        disk = psutil.disk_usage(str(system_drive))
        uptime_seconds = max(0, round(time.time() - psutil.boot_time()))
        return {
            "cpu_percent": round(psutil.cpu_percent(interval=0.15), 1),
            "memory_used_gb": _bytes_to_gb(memory.used),
            "memory_total_gb": _bytes_to_gb(memory.total),
            "memory_percent": round(memory.percent, 1),
            "disk_used_gb": _bytes_to_gb(disk.used),
            "disk_free_gb": _bytes_to_gb(disk.free),
            "disk_total_gb": _bytes_to_gb(disk.total),
            "disk_percent": round(disk.percent, 1),
            "uptime_seconds": uptime_seconds,
        }

    def get_top_processes(self, sort_by: str, limit: int) -> dict[str, Any]:
        if sort_by not in {"cpu", "memory"}:
            raise ValueError("sort_by must be cpu or memory.")
        bounded_limit = max(1, min(10, int(limit)))
        processes: list[psutil.Process] = []
        for process in psutil.process_iter(["pid", "name"]):
            try:
                process.cpu_percent(None)
                processes.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        if sort_by == "cpu":
            time.sleep(0.15)

        logical_cpus = max(1, psutil.cpu_count(logical=True) or 1)
        grouped: dict[str, dict[str, Any]] = {}
        for process in processes:
            try:
                memory_bytes = process.memory_info().rss
                name = process.name() or "Unknown"
                if name.casefold() in {"system idle process", "idle"}:
                    continue
                item = grouped.setdefault(
                    name.casefold(),
                    {
                        "name": name,
                        "process_count": 0,
                        "cpu_percent": 0.0,
                        "memory_mb": 0.0,
                    },
                )
                item["process_count"] += 1
                item["cpu_percent"] += process.cpu_percent(None) / logical_cpus
                item["memory_mb"] += memory_bytes / (1024**2)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        results = list(grouped.values())
        for item in results:
            item["cpu_percent"] = round(min(100.0, item["cpu_percent"]), 1)
            item["memory_mb"] = round(item["memory_mb"], 1)
        key = "cpu_percent" if sort_by == "cpu" else "memory_mb"
        results.sort(key=lambda item: float(item[key]), reverse=True)
        return {"sort_by": sort_by, "processes": results[:bounded_limit]}

    def get_gpu_status(self) -> dict[str, Any]:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            system_candidate = (
                Path(os.environ.get("SystemRoot", r"C:\Windows"))
                / "System32"
                / "nvidia-smi.exe"
            )
            if system_candidate.is_file():
                executable = str(system_candidate)
        if executable is None:
            raise FeatureUnavailableError("NVIDIA monitoring is unavailable.")
        command = [
            executable,
            "--query-gpu=name,utilization.gpu,temperature.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            # This fixed read-only query contains no user-provided arguments.
            completed = subprocess.run(  # nosec B603
                command,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=True,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise FeatureUnavailableError("NVIDIA monitoring is unavailable.") from exc
        line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise PcControlError("NVIDIA returned a malformed status response.")
        try:
            used_mb = float(parts[3])
            total_mb = float(parts[4])
            return {
                "gpu_name": parts[0],
                "gpu_utilization_percent": round(float(parts[1]), 1),
                "temperature_c": round(float(parts[2]), 1),
                "vram_used_mb": round(used_mb),
                "vram_total_mb": round(total_mb),
                "vram_percent": round((used_mb / total_mb) * 100, 1) if total_mb else 0.0,
            }
        except ValueError as exc:
            raise PcControlError("NVIDIA returned a malformed status response.") from exc

    def _resolve_launch_target(self, application: ApplicationConfig) -> Path:
        if application.configured_path is not None:
            configured = application.configured_path.expanduser().resolve()
            if configured.is_file():
                return configured

        for executable_name in application.executable_names:
            found = shutil.which(executable_name)
            if found:
                return Path(found)
            registry_path = self._app_path_from_registry(executable_name)
            if registry_path is not None:
                return registry_path

        desired_names = {
            _normalized_name(name)
            for name in (*application.start_menu_names, *application.aliases)
        }
        for shortcut in self._start_menu_shortcuts():
            if _normalized_name(shortcut.stem) in desired_names:
                return shortcut
        raise ApplicationNotFoundError(application.identifier)

    def _start_menu_shortcuts(self) -> tuple[Path, ...]:
        if self._start_menu_cache is not None:
            return self._start_menu_cache
        roots = [
            Path(os.environ.get("ProgramData", r"C:\ProgramData"))
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs",
            Path(os.environ.get("APPDATA", ""))
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs",
        ]
        shortcuts: list[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            try:
                shortcuts.extend(root.rglob("*.lnk"))
            except OSError:
                logger.debug("Could not inspect Start Menu root: %s", root)
        self._start_menu_cache = tuple(shortcuts)
        return self._start_menu_cache

    @staticmethod
    def _app_path_from_registry(executable_name: str) -> Path | None:
        if os.name != "nt":
            return None
        try:
            import winreg
        except ImportError:
            return None
        key_path = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{executable_name}"
        access_modes = (winreg.KEY_READ, winreg.KEY_READ | winreg.KEY_WOW64_32KEY)
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for access in access_modes:
                try:
                    with winreg.OpenKey(hive, key_path, 0, access) as key:
                        raw_path, _ = winreg.QueryValueEx(key, None)
                    path = Path(str(raw_path).strip().strip('"'))
                    if path.is_file():
                        return path
                except (FileNotFoundError, OSError):
                    continue
        return None

    @staticmethod
    def _matching_processes(application: ApplicationConfig) -> list[psutil.Process]:
        allowed = {name.casefold() for name in application.process_names}
        if not allowed:
            return []
        matches: list[psutil.Process] = []
        for process in psutil.process_iter(["name"]):
            try:
                if str(process.info.get("name") or "").casefold() in allowed:
                    matches.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return matches

    @staticmethod
    def _assert_processes_are_safe(processes: list[psutil.Process]) -> None:
        for process in processes:
            try:
                name = (process.name() or "").casefold()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if name in PROTECTED_PROCESS_NAMES:
                raise ProtectedProcessError(name)

    def _post_close_to_process_windows(self, target_pids: set[int]) -> int:
        self._require_windows()
        signalled = 0
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        @callback_type
        def callback(hwnd: int, _: int) -> bool:
            nonlocal signalled
            process_id = ctypes.c_ulong()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            if process_id.value in target_pids:
                ctypes.windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
                signalled += 1
            return True

        try:
            ctypes.windll.user32.EnumWindows(callback, 0)
        except (AttributeError, OSError) as exc:
            raise PcControlError("Windows could not request a graceful close.") from exc
        return signalled

    @contextmanager
    def _endpoint_volume(self) -> Iterator[Any]:
        self._require_windows()
        try:
            from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
            from ctypes import POINTER, cast
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        except ImportError as exc:
            raise FeatureUnavailableError("Windows volume support is unavailable.") from exc
        CoInitialize()
        try:
            device = AudioUtilities.GetSpeakers()
            endpoint = getattr(device, "EndpointVolume", None)
            if endpoint is None:
                interface = device.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                )
                endpoint = cast(interface, POINTER(IAudioEndpointVolume))
            yield endpoint
        except Exception as exc:
            raise PcControlError("Windows volume control failed.") from exc
        finally:
            CoUninitialize()

    @staticmethod
    def _require_windows() -> None:
        if os.name != "nt":
            raise FeatureUnavailableError("This PC-control feature requires Windows.")


def _bytes_to_gb(value: int) -> float:
    return round(value / (1024**3), 1)


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())
