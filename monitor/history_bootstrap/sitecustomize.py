"""Dedicated opt-in bootstrap, separate from the existing pure monitor."""
import os
import json
import atexit
from pathlib import Path

path=os.environ.get('LINEARNO_HISTORY_MONITOR_CONFIG')
if path:
    try:
        from monitor.history_diagnostics import HistoryDiagnostics
        config=json.loads(Path(path).read_text())
        observer=HistoryDiagnostics(config['output'],every=config['every'],max_snapshots=config['max_snapshots'],
                                    max_points=config['max_points'],plots=config['plots'])
        observer.__enter__()
        atexit.register(observer.__exit__,None,None,None)
    except Exception as exc:
        Path(path).with_name('bootstrap_error.txt').write_text(type(exc).__name__+': '+str(exc))
