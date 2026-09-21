#!/usr/bin/env python3
"""Generate LF7 v2 commands without executing training or creating run dirs."""

import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from linearno_loop.v2.matrix import configuration_matrix


def flags(config,*,multiplier=None):
    loop=config['loop_spec'];values=config['profile_spec']['values']
    result=['--linearno-loop','1','--linearno-loop-topology',loop['topology_preset'],
        '--linearno-loop-residual-mode',loop['residual_mode'],
        '--linearno-loop-core-ffn-mode',loop['core_ffn_mode'],
        '--linearno-profile',loop['profile'],'--seed',str(values['runtime']['seed'])]
    if loop['topology_preset']=='custom':
        result.extend(['--linearno-loop-prefix-blocks',str(loop['prefix_blocks']),
            '--linearno-loop-core-blocks',str(loop['recurrent_core_blocks']),
            '--linearno-loop-repeats',str(loop['loop_repeats']),
            '--linearno-loop-suffix-blocks',str(loop['suffix_blocks'])])
    # Mx1 is the v2 default. Mx2 is always explicit in the control matrix.
    if multiplier is not None:result.extend(['--linearno-loop-rank-multiplier',str(multiplier)])
    return result


def command_row(item,*,control=False):
    config=item['config'];task=config['loop_spec']['task'];run=item['run_id']
    base=['bash',f'tran_evaluate/linearno_loop/{task}.sh']
    training=[*base,'train','--then-eval',*flags(config,multiplier=2 if control else None),
              '--gpu','GPU','--experiment-dir',f'RUN_ROOT/{task}/{run}']
    resume=[*base,'resume','--then-eval','--gpu','GPU','--experiment-dir',f'RUN_ROOT/{task}/{run}']
    evaluation=[*base,'eval','--gpu','GPU','--experiment-dir',f'RUN_ROOT/{task}/{run}']
    return dict(task=task,run_id=run,config_hash=config['config_hash'],status='PLANNED_NOT_RUN',
                train_then_eval=shlex.join(training),resume_then_eval=shlex.join(resume),
                eval=shlex.join(evaluation))


def matrix():
    source=configuration_matrix();primary=[command_row(item) for item in source['runs']]
    controls=[command_row(item,control=True) for item in source['controls']]
    custom=command_row(source['custom'])
    return dict(schema='loop-linearno-ffn-command-matrix-v1',safe_preview=True,
        note='Commands are text only. Replace GPU/RUN_ROOT; this generator never launches them.',
        paired_seeds=source['paired_seeds'],primary=primary,rank_x2_controls=controls,custom=custom)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output');args=parser.parse_args();value=matrix()
    encoded=json.dumps(value,indent=2)+'\n'
    if args.output:
        path=Path(args.output)
        if path.exists():parser.error('output exists; refusing to overwrite')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(encoded)
    else:print(encoded,end='')


if __name__=='__main__':main()
