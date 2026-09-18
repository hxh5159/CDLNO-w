"""Opt-in child-process bootstrap used by :mod:`monitor.run`.

Python imports ``sitecustomize`` during startup when this directory is on
``PYTHONPATH``.  The environment guard keeps ordinary project processes
unchanged.
"""

from __future__ import annotations

import json
import os


def _install() -> None:
    path = os.environ.get("LINEARNO_MONITOR_CONFIG")
    if not path:
        return
    try:
        from monitor.runtime import install_runtime_monitor
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        install_runtime_monitor(config)
    except Exception as exc:
        # Do not prevent the benchmark from running because diagnostics are
        # unavailable.  run.py records the child return code and log.
        try:
            error_path = os.path.join(os.path.dirname(path), "bootstrap_error.txt")
            with open(error_path, "w", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")
        except Exception:
            pass


_install()

