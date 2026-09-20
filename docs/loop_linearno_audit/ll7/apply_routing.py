"""Exact additive LL7 guards; also emit reversible legacy fingerprint projection."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
changes={}
def change(name,before,after,count=1):
 p=ROOT/name;source=p.read_text();assert source.count(before)==count,(name,before)
 backup=OUT/'before'/name
 if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);backup.write_text(source)
 p.write_text(source.replace(before,after))
 changes.setdefault(name,[]).append(dict(before=before,after=after,count=count))
for name,task,field in [('Airfoil-Design-AirfRANS/cdlno_entry.py','airfrans','model'),('Car-Design-ShapeNetCar/models/cdlno_run.py','car','cfd_model')]:
 old='    selected, _ = selector.parse_known_args(tokens)\n'
 change(name,old,old+f"    from cdlno.linearno_loop.industrial_entry import intercept as intercept_loop\n    loop_args = intercept_loop(parser, tokens, task='{task}', evaluation=evaluation, selected_model=selected.{field})\n    if loop_args is not None:\n        return loop_args\n")
for task in ('air','car'):
 name=f'cdlno/linearno/{task}_entry.py';old='def run_cli(args):\n'
 change(name,old,old+f"    if getattr(args, 'linearno_family', None) == 'linearno_loop':\n        from cdlno.linearno_loop.{task}_entry import run_cli as loop_run_cli\n        return loop_run_cli(args)\n")
change('cdlno/linearno/car_entry.py',
 "        key = str(p.relative_to(ROOT)); source = p.read_text(); sources[key] = source\n",
 "        key = str(p.relative_to(ROOT)); source = p.read_text()\n        from cdlno.linearno_loop.provenance import legacy_source\n        source = legacy_source(key, source); sources[key] = source\n")
change('Airfoil-Design-AirfRANS/train.py',"    torch.save(model, osp.join(path, 'model'))\n",
 "    if hasattr(linearno_run, 'export_final'):\n        linearno_run.export_final(model)\n    else:\n        torch.save(model, osp.join(path, 'model'))\n")
change('cdlno/linearno_history/car_entry.py','        torch.save(model,Path(path)/f\'model_{hparams["nb_epochs"]}.pth\')\n',
 '        if hasattr(self, \'export_final\'):\n            self.export_final(model)\n        else:\n            torch.save(model,Path(path)/f\'model_{hparams["nb_epochs"]}.pth\')\n')
name='cdlno/linearno_loop/provenance.py'
change(name,'def legacy_source(relative,text):\n','def legacy_source(relative,text):\n    from .ll7_projection import project\n    text = project(relative,text)\n')
change(name,"    paths={*(root/'cdlno/linearno_loop').glob('*.py'),", "    from .ll7_projection import LL6_FILES, project\n    paths={*(root/'cdlno/linearno_loop'/name for name in LL6_FILES),")
change(name,"hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)","hashlib.sha256(project(str(p.relative_to(root)),p.read_text()).encode()).hexdigest() for p in sorted(paths)")
change(name,"after=ast.dump(ast.parse(p.read_text())))","after=ast.dump(ast.parse(project(name,p.read_text()))))")
# All old industrial/pure/history and LL6 fingerprints retain exact old bytes.
files=[Path(r['path']).name for r in json.loads((OUT/'start-manifest.json').read_text())['files'] if r['path'].startswith('cdlno/linearno_loop/') and r['path'].endswith('.py')]
(ROOT/'cdlno/linearno_loop/ll7_projection.py').write_text('"""Exact LL7 checkpoint/routing additions; no blanket old-source hash bypass."""\n'+
 'LL6_FILES = '+repr(tuple(sorted(files)))+'\n\nREPLACEMENTS = '+repr(changes)+'''\n\ndef project(relative,text):
    for change in reversed(REPLACEMENTS.get(relative,())):
        if text.count(change['after'])!=change['count']:
            raise ValueError('unrecognized LL7 source change: '+relative)
        text=text.replace(change['after'],change['before'])
    return text
''')
print('Changed',len(changes),'existing sources')
