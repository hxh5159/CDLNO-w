import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from loop_linearno.support import ROOT, metadata


class IsolationTests(unittest.TestCase):
    def test_fresh_process_all_schema_paths_forbid_tensor_model_task_imports(self):
        # Serialize real backend states outside the isolated process; the consumer
        # must inspect them with only the standard library and existing pure profiles.
        m=metadata()
        code=r'''
import importlib.abc, json, random, sys
blocked=('torch','numpy','timm','einops','torch_geometric','model','models',
         'cdlno.linearno.schema','cdlno.linearno_loop','cdlno.linearno.attention',
         'cdlno.linearno.airfrans','cdlno.linearno.shapenet')
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==x or fullname.startswith(x+'.') for x in blocked) or fullname.startswith('exp_'):
            raise AssertionError('forbidden import '+fullname)
sys.meta_path.insert(0,Guard())
before=random.getstate()
from linearno_loop.schema import read_metadata,restore_config
from linearno_loop.config import route_intent
from linearno_loop.matrix import configuration_matrix
saved=read_metadata(sys.argv[1]); recovered=restore_config(saved)
assert recovered['config']==saved['resolved_config']
assert len(configuration_matrix()['runs'])==288
assert route_intent('linearno',{}) is None
assert random.getstate()==before
assert not any(n==x or n.startswith(x+'.') for x in blocked for n in sys.modules)
print('stdlib-only metadata/config/matrix; no model import; Python RNG unchanged')
'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metadata.json';path.write_text(json.dumps(m))
            env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
            result=subprocess.run([sys.executable,'-B','-c',code,str(path)],cwd=directory,
                                  env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_resolution_inspection_and_preview_leave_all_global_rng_unchanged(self):
        import numpy as np
        import torch
        from cdlno.linearno.schema import pack_state
        from linearno_loop.schema import validate_metadata,restore_config
        from linearno_loop.matrix import configuration_matrix
        def states():
            return pack_state({'python':random.getstate(),'numpy':np.random.get_state(),
                               'torch_cpu':torch.get_rng_state()})
        before=states();m=metadata()
        validate_metadata(m);restore_config(m);configuration_matrix()
        self.assertEqual(before,states())

    def test_preexisting_bytes_after_exact_LL6_routing_projection_and_classifications(self):
        from cdlno.linearno_loop.provenance import legacy_source
        initial=json.loads((ROOT/'docs/loop_linearno_audit/ll1/start-manifest.json').read_text())
        categories={}
        for label,args in [('tracked',[]),('untracked',['--others','--exclude-standard']),
                           ('ignored',['--others','--ignored','--exclude-standard'])]:
            raw=subprocess.check_output(['git','ls-files','-z',*args],cwd=ROOT)
            categories.update({os.fsdecode(p):label for p in raw.split(b'\0') if p})
        allowed={'docs/LOOP_LINEARNO_IMPLEMENTATION_STATUS.md'}
        for row in initial['files']:
            if row['path'] in allowed:continue
            # Source inventory includes explicitly frozen untracked files;
            # interpreter caches are runtime artifacts, never source fixtures.
            parts=Path(row['path']).parts
            if ('__pycache__' in parts or '.pytest_cache' in parts or
                    row['path'].endswith(('.pyc','.pyo'))):continue
            with self.subTest(path=row['path']):
                path=ROOT/row['path']
                self.assertTrue(path.is_file())
                raw=path.read_bytes()
                from ll9r_test_source_projection import project_test_source
                raw=project_test_source(row['path'],raw)
                if row['path']=='tests/history_entry_projection.py':
                    addition=b'    from loop_entry_projection import strip_loop\n    tree = strip_loop(tree)\n'
                    self.assertEqual(raw.count(addition),1)
                    raw=raw.replace(addition,b'')
                else:
                    # Strip only reviewed exact insertions, not whole files or
                    # arbitrary AST regions. Every original byte stays checked.
                    from cdlno.linearno_loop.provenance import REPLACEMENTS
                    from cdlno.linearno_loop.ll7_projection import REPLACEMENTS as LL7_REPLACEMENTS
                    if row['path'] in (REPLACEMENTS.keys() | LL7_REPLACEMENTS.keys()):raw=legacy_source(row['path'],raw.decode()).encode()
                self.assertEqual(len(raw),row['size'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
                self.assertEqual(categories[row['path']],row['classification'])


if __name__=='__main__':unittest.main()
