#!/usr/bin/env python3
"""Run an existing command with bounded optional research/pure diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--every',type=int,default=100)
    p.add_argument('--max-snapshots',type=int,default=8)
    p.add_argument('--max-points',type=int,default=256)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or min(a.every,a.max_snapshots,a.max_points)<1:
        p.error('command and positive bounds required')
    output=Path(a.output).expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    cfg=dict(output=str(output),every=a.every,max_snapshots=a.max_snapshots,max_points=a.max_points,plots=not a.no_plots,command=command)
    config=output/'runtime_config.json';config.write_text(json.dumps(cfg,indent=2))
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ)
    if env.get('LINEARNO_MONITOR_CONFIG'):
        p.error('do not combine the old pure monitor and history monitor')
    env['LINEARNO_HISTORY_MONITOR_CONFIG']=str(config)
    env['PYTHONPATH']=os.pathsep.join([str(root/'monitor/history_bootstrap'),str(root),env.get('PYTHONPATH','')])
    result=subprocess.run(command,env=env)
    count=len((output/'diagnostics.jsonl').read_text().splitlines()) if (output/'diagnostics.jsonl').exists() else 0
    status=dict(command_returncode=result.returncode,snapshots=count,
                diagnostics_status='PASS' if count and not (output/'bootstrap_error.txt').exists() else 'NOT_CAPTURED')
    (output/'run_summary.json').write_text(json.dumps(status,indent=2))
    return result.returncode if result.returncode else (0 if status['diagnostics_status']=='PASS' else 2)


if __name__=='__main__':raise SystemExit(main())
