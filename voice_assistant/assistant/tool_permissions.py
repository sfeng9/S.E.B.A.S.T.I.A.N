from __future__ import annotations

from enum import Enum


class ToolPermission(str, Enum):
    READ_ONLY = "read_only"
    SAFE_ACTION = "safe_action"
    CONFIRM_REQUIRED = "confirm_required"
    DISABLED = "disabled"
