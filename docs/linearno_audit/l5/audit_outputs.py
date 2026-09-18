"""Read-only checks of completed synthetic records and optimizer step counters."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
from linearno.temporal_worker import args_for
from model_dict import get_model
from cdlno.linearno.checkpoint import inspect_checkpoint
from cdlno.linearno.schema import unpack_state

artifacts=Path(sys.argv[1])
rows=[]
for task in ('ns','plasticity'):
    path=artifacts/f'{task}-split'
    cls=get_model(args_for(task)).Model
    metadata,_=inspect_checkpoint(path,'final',cls)
    state=metadata['resume_state']
    optim=unpack_state(state['optimizer'])
    steps={int(v['step'].item()) for v in optim['state'].values() if 'step' in v}
    assert steps=={120 if task=='plasticity' else 6},steps
    assert state['global_step']==next(iter(steps))
    assert unpack_state(state['scheduler'])['last_epoch']==6
    history=[json.loads(line) for line in (path/'train_history.jsonl').read_text().splitlines()]
    evaluation=json.loads((path/'eval_results.json').read_text())['evaluations'][-1]
    events=json.loads((path/'train_results.json').read_text())['visualization_events']
    assert [r['epoch'] for r in history]==[1,2,3]
    assert evaluation['status']=='completed'
    for key,value in evaluation['metrics'].items():
        assert value==history[-1]['metrics'][key],(task,key,value,history[-1]['metrics'][key])
    assert events and all(row['status']=='completed' for row in events)
    rows.append(dict(task=task,optimizer_steps=sorted(steps),scheduler_steps=6,
        final_validation_equals_fresh_eval=True,metrics=evaluation['metrics'],
        history_epochs=[r['epoch'] for r in history],visualization=events))
print(json.dumps(dict(status='PASS',synthetic=True,rows=rows),indent=2))
