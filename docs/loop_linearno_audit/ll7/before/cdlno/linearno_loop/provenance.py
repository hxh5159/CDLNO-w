"""Exact LL6 routing projection preserves ALL old resume source bytes.

Loop provenance always hashes real sources including the routing/projection.
Only these exact additions are stripped for old pure/history fingerprints;
missing or changed snippets fail closed. No saved hash is bypassed.
"""
import ast
import hashlib
from pathlib import Path
import subprocess
from cdlno.linearno.profiles import digest
from linearno_loop.schema import REFERENCE_PINS

REPLACEMENTS = {'PDE-Solving-StandardBenchmark/cdlno_entry.py': [{'before': '    selected, _ = selector.parse_known_args(tokens)\n', 'after': '    selected, _ = selector.parse_known_args(tokens)\n    from cdlno.linearno_loop.standard_entry import intercept as intercept_loop\n    loop_args = intercept_loop(parser, task, tokens, selected.model)\n    if loop_args is not None:\n        return loop_args\n', 'count': 1}], 'PDE-Solving-StandardBenchmark/model_dict.py': [{'before': 'def get_model(args):\n', 'after': "def get_model(args):\n    if hasattr(args, '_linearno_loop_config'):\n        from cdlno.linearno_loop.standard_entry import model_module\n        return model_module(args)\n", 'count': 1}], 'PDE-Solving-StandardBenchmark/linearno_entry.py': [{'before': 'def StandardRun(args, model):\n', 'after': "def StandardRun(args, model):\n    if getattr(args, 'linearno_family', None) == 'linearno_loop':\n        from cdlno.linearno_loop.standard_entry import LoopStandardRun\n        return LoopStandardRun(args, model)\n", 'count': 1}], 'cdlno/experiment.py': [{'before': "('linearno', 'linearno_history')", 'after': "('linearno', 'linearno_history', 'linearno_loop')", 'count': 5}, {'before': "'linearno_history':'LinearNO history'}[self.family]", 'after': "'linearno_history':'LinearNO history', 'linearno_loop':'Looped LinearNO'}[self.family]", 'count': 1}, {'before': "            self.config['linearno_profile'] = _json(args._linearno_config)\n", 'after': "            self.config['linearno_profile'] = _json(args._linearno_config)\n            if self.family == 'linearno_loop':\n                self.config['loop_config'] = _json(args._linearno_loop_config)\n", 'count': 1}], 'cdlno/linearno_history/provenance.py': [{'before': 'def baseline_source(relative, text):\n', 'after': 'def baseline_source(relative, text):\n    from cdlno.linearno_loop.provenance import legacy_source\n    text = legacy_source(relative, text)\n', 'count': 1}, {'before': '    actual={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ROUTING_REPLACEMENTS}', 'after': '    from cdlno.linearno_loop.provenance import legacy_source\n    actual={p:hashlib.sha256(legacy_source(p,(root/p).read_text()).encode()).hexdigest() for p in ROUTING_REPLACEMENTS}', 'count': 1}], 'cdlno/linearno_history/checkpoint.py': [{'before': '    return digest({str(p.relative_to(root)): legacy.sha256(p) for p in paths})', 'after': '    from cdlno.linearno_loop.provenance import legacy_source\n    import hashlib\n    return digest({str(p.relative_to(root)): hashlib.sha256(legacy_source(str(p.relative_to(root)),p.read_text()).encode()).hexdigest() for p in paths})', 'count': 1}]}


def legacy_source(relative,text):
    for change in reversed(REPLACEMENTS.get(relative,())):
        if text.count(change['after'])!=change['count']:
            raise ValueError('unrecognized loop routing source: '+relative)
        text=text.replace(change['after'],change['before'])
    return text


def provenance():
    from cdlno.linearno.standard_entry import provenance as base_provenance
    base=base_provenance();root=Path(__file__).resolve().parents[2]
    paths={*(root/'cdlno/linearno_loop').glob('*.py'),*(root/'linearno_loop').glob('*.py'),
           *(root/'cdlno/linearno').glob('*.py'),*(root/name for name in REPLACEMENTS)}
    paths.add(root/'cdlno/linearno_history/fair_run.py')
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    normalized={}
    for p in sorted(paths):
        name=str(p.relative_to(root))
        try:old=subprocess.check_output(['git','show','HEAD:'+name],cwd=root,stderr=subprocess.DEVNULL).decode()
        except subprocess.CalledProcessError:old=None
        normalized[name]=dict(before=ast.dump(ast.parse(old)) if old else None,after=ast.dump(ast.parse(p.read_text())))
    base.update(**REFERENCE_PINS,residual_scaling_version='2606.18524v1',code_version='loop-linearno-LL6-v1',
        config_schema_version=1,metadata_schema_version=1,source_sha256=digest(dict(base=base['source_sha256'],sources=sources)),
        normalized_patch_sha256=digest(normalized))
    return base
