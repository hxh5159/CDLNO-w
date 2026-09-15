#!/usr/bin/env python3
"""K0 source/CLI inventory; no dataset or exp/main module is imported."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import make_regression_fixtures as fixtures

ROOT=Path(__file__).resolve().parents[2]
AUDIT=Path(__file__).resolve().parent


def digest(b):return hashlib.sha256(b).hexdigest()


def cli_worker(project, artifacts, result):
    rows=[]
    for task in (tuple(fixtures.STEMS) if project=='standard' else (project,)):
        entry=Path.cwd()/('exp_'+fixtures.STEMS[task]+'.py' if project=='standard' else 'main.py')
        parser=fixtures.parser_only(entry)
        original=parser.parse_args([])
        resolved=fixtures.parse_task(task)
        row=dict(task=task,raw_parser_defaults=vars(original),CDLNO_train_defaults=vars(resolved),modes=[])
        if project=='standard':
            from model_dict import get_model
            row['CDLNO_registry']=get_model(resolved).__name__
            row['legacy_registry']={key:get_model(SimpleNamespace(model=key)).__name__ for key in (
                'Transolver_Irregular_Mesh','Transolver_Structured_Mesh_2D','Transolver_Structured_Mesh_3D')}
            try:get_model(original)
            except KeyError:row['bare_old_default_registry']='KeyError: old parser alias is not registered; official launchers override model'
        else:
            row['legacy_explicit']=vars(fixtures.parse_task(task,family='Transolver'))
        for mode in fixtures.MODES:
            folder=artifacts/f'cdlno_{task}_{mode}'/'run'
            key='--cdlno-run-dir' if task in fixtures.STEMS else '--run_dir'
            train=fixtures.parse_task(task,['--front-latent-mode',mode])
            evaluation=fixtures.parse_task(task,[key,str(folder)],evaluation=True)
            assert train.front_latent_mode==evaluation.front_latent_mode==mode
            row['modes'].append(dict(mode=mode,train_arguments=vars(train),eval_arguments_omitted_mode=vars(evaluation)))
        rows.append(row)
    fixtures.write_json(result,rows)


def inventory(artifacts):
    baseline=json.loads((AUDIT/'baseline.json').read_text())
    entries=[f"PDE-Solving-StandardBenchmark/exp_{stem}.py" for stem in fixtures.STEMS.values()]
    entries += [f'{project}/{entry}.py' for project in ('Car-Design-ShapeNetCar','Airfoil-Design-AirfRANS')
                for entry in ('main','main_evaluation','train')]
    inventory={}
    for name in entries:
        source=(ROOT/name).read_text();tree=ast.parse(source)
        functions={}
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                functions[node.name]=dict(line=node.lineno,end_line=node.end_lineno,
                                          ast_sha256=digest(ast.dump(node,include_attributes=False).encode()))
        constructors=[];operations=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                fn=ast.unparse(node.func)
                if fn in ('parser.add_argument','torch.load','torch.save') or any(
                    word in fn for word in ('optimizer','scheduler','myloss','normalizer','loadmat','np.load','backward','Model','DataLoader','load_train')):
                    operations.append(dict(line=node.lineno,call=ast.unparse(node)))
        inventory[name]=dict(file_sha256=digest(source.encode()),functions=functions,operations=operations,
                             top_level_statements=[dict(line=node.lineno,type=type(node).__name__,
                                                        code=ast.unparse(node) if not isinstance(node,(ast.FunctionDef,ast.ClassDef)) else node.name)
                                                   for node in tree.body])
    full=Path(baseline['baseline'])/'entry-ast.json';fixtures.write_json(full,inventory)
    short={name:{k:v for k,v in value.items() if k!='top_level_statements'} for name,value in inventory.items()}
    fixtures.write_json(AUDIT/'entry-contracts.json',short)
    presets={name:json.loads((ROOT/name).read_text()) for name in baseline['tracked_files']
             if '/configs/CDLNO/' in name and name.endswith('.json')}
    import yaml
    presets['Airfoil-Design-AirfRANS/params.yaml']=yaml.safe_load((ROOT/'Airfoil-Design-AirfRANS/params.yaml').read_text())
    fixtures.write_json(AUDIT/'presets.json',presets)
    statuses=[]
    for project,cwd in fixtures.PROJECTS.items():
        result=AUDIT/f'cli-{project}.json'
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--project',project,'--artifacts',str(artifacts),'--result',str(result)]
        env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/cwd),PYTHONDONTWRITEBYTECODE='1')
        r=subprocess.run(command,cwd=ROOT/cwd,env=env,text=True,capture_output=True)
        result.with_suffix('.log').write_text(r.stdout+r.stderr)
        statuses.append(dict(project=project,returncode=r.returncode,command=command,result=str(result)))
    fixtures.write_json(AUDIT/'static-results.json',dict(entry_ast=str(full),entries=len(entries),presets=len(presets),cli=statuses))
    return int(any(r['returncode'] for r in statuses))


def freeze():
    baseline=json.loads((AUDIT/'baseline.json').read_text())
    changed=[]
    for name,meta in baseline['tracked_files'].items():
        p=ROOT/name
        if not p.is_file() or digest(p.read_bytes())!=meta['sha256']:changed.append(name)
    extras=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    unexpected=[n for n in extras if not n.startswith('docs/kcdno_audit/') and n not in (
        'docs/KCDNO_REFERENCE_AUDIT.md','docs/KCDNO_IMPLEMENTATION_STATUS.md')]
    result=dict(tracked_files_checked=len(baseline['tracked_files']),changed=changed,unexpected_new=unexpected,
                new_K0_files=extras,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    fixtures.write_json(AUDIT/'freeze.json',result)
    print(json.dumps(result))
    return int(bool(changed or unexpected))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',choices=tuple(fixtures.PROJECTS));p.add_argument('--artifacts',type=Path)
    p.add_argument('--result',type=Path);p.add_argument('--freeze',action='store_true')
    args=p.parse_args()
    if args.freeze:sys.exit(freeze())
    if args.project:cli_worker(args.project,args.artifacts,args.result)
    else:sys.exit(inventory(args.artifacts.resolve()))
