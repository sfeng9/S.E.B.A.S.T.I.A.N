from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_assistant.config import load_assistant_config
from voice_assistant.integrations.pc_control import PcControlError, WindowsPcController


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run safe, explicit PC-control diagnostics."
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="Capture one local screenshot in the configured directory.",
    )
    parser.add_argument(
        "--volume-cycle",
        action="store_true",
        help="Test volume 30, 40, 30, mute, and unmute, then restore the original state.",
    )
    parser.add_argument(
        "--app-cycle",
        action="append",
        default=[],
        metavar="APP_ID",
        help="Open and gracefully close one configured app only if it was initially closed.",
    )
    args = parser.parse_args()

    config = load_assistant_config()
    controller = WindowsPcController(config.pc_control)
    applications = {
        application.identifier: application
        for application in config.pc_control.applications
    }

    _print_result("system", controller.get_system_status())
    _run_optional("volume", controller.get_system_volume)
    _run_optional("gpu", controller.get_gpu_status)
    _run_optional("top_memory", lambda: controller.get_top_processes("memory", 5))
    _run_optional("top_cpu", lambda: controller.get_top_processes("cpu", 5))

    if args.screenshot:
        _run_optional("screenshot", controller.take_screenshot)
    if args.volume_cycle:
        _volume_cycle(controller)
    for app_id in args.app_cycle:
        application = applications.get(app_id.casefold())
        if application is None:
            print(f"app_cycle[{app_id}]: not in the configured allowlist")
            continue
        _app_cycle(controller, application)
    return 0


def _volume_cycle(controller: WindowsPcController) -> None:
    original = controller.get_system_volume()
    results: list[dict[str, Any]] = []
    try:
        results.append(controller.set_system_volume(30))
        results.append(controller.change_system_volume(10))
        results.append(controller.change_system_volume(-10))
        results.append(controller.set_system_mute(True))
        results.append(controller.set_system_mute(False))
    finally:
        controller.set_system_volume(int(original["percent"]))
        controller.set_system_mute(bool(original["muted"]))
    _print_result("volume_cycle", {"steps": results, "restored": original})


def _app_cycle(controller: WindowsPcController, application: Any) -> None:
    initial = controller.is_application_running(application)
    if initial["running"]:
        _print_result(
            f"app_cycle[{application.identifier}]",
            {"skipped": True, "reason": "already_running"},
        )
        return
    opened = controller.open_application(application)
    time.sleep(3.0)
    running = controller.is_application_running(application)
    closed = controller.close_application(application) if running["running"] else None
    _print_result(
        f"app_cycle[{application.identifier}]",
        {"opened": opened, "running_after_open": running, "graceful_close": closed},
    )


def _run_optional(label: str, function: Any) -> None:
    try:
        _print_result(label, function())
    except PcControlError as exc:
        _print_result(label, {"available": False, "error": type(exc).__name__})


def _print_result(label: str, value: Any) -> None:
    print(f"{label}: {json.dumps(value, ensure_ascii=True, sort_keys=True)}")


if __name__ == "__main__":
    raise SystemExit(main())
