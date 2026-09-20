"""Explicit source-scope separation for legacy resume fingerprints.

Only the exact reviewed routing additions are projected out of the BASELINE
source fingerprint. Every original byte remains hashed. Changed/unrecognized
routing snippets fail closed; research provenance additionally hashes all actual
routing sources and the complete research package. No stored hash is ignored.
"""
import hashlib
from pathlib import Path
from cdlno.linearno.profiles import digest
from .checkpoint import source_hash

ROUTING_REPLACEMENTS = {'PDE-Solving-StandardBenchmark/cdlno_entry.py': [{'before': "    if selected.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):", 'after': "    if selected.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):\n        from cdlno.linearno_history.standard_entry import intercept as intercept_history\n        history_args = intercept_history(parser, task, tokens, selected.model)\n        if history_args is not None:\n            return history_args", 'count': 1}], 'PDE-Solving-StandardBenchmark/model_dict.py': [{'before': 'def get_model(args):\n', 'after': "def get_model(args):\n    if hasattr(args, '_linearno_history_config'):\n        from cdlno.linearno_history.standard_entry import model_module\n        return model_module(args)\n", 'count': 1}], 'PDE-Solving-StandardBenchmark/linearno_entry.py': [{'before': '"""Six Standard-task LinearNO integration. No exp imports or data side effects."""\nfrom cdlno.linearno.standard_entry import (\n    model_kwargs, start, finish, normalizer, verify_data, StandardRun,\n)\n', 'after': '"""Six Standard-task LinearNO integration. No exp imports or data side effects."""\nfrom cdlno.linearno.standard_entry import (\n    model_kwargs, start, finish, normalizer, verify_data, StandardRun,\n)\n\n# Only the explicitly selected research family uses the new native Run subclass.\n_PureStandardRun = StandardRun\n\ndef StandardRun(args, model):\n    from cdlno.linearno_history.standard_entry import select_run\n    return select_run(args, model, _PureStandardRun)\n', 'count': 1}], 'cdlno/experiment.py': [{'before': "if getattr(args, 'linearno_family', None) == 'linearno':\n            self.family, self.run_key = 'linearno', 'linearno_run_dir'", 'after': "if getattr(args, 'linearno_family', None) in ('linearno', 'linearno_history'):\n            self.family, self.run_key = args.linearno_family, 'linearno_run_dir'", 'count': 1}, {'before': "self.family == 'linearno'", 'after': "self.family in ('linearno', 'linearno_history')", 'count': 4}, {'before': "'linearno':'LinearNO'}[self.family]", 'after': "'linearno':'LinearNO', 'linearno_history':'LinearNO history'}[self.family]", 'count': 1}], 'cdlno/linearno/standard_entry.py': [{'before': '        sources[relative] = text', 'after': '        from cdlno.linearno_history.provenance import baseline_source\n        text = baseline_source(relative, text)\n        sources[relative] = text', 'count': 1}], 'cdlno/linearno/air_entry.py': [{'before': 'def parse_args(parser, tokens, *, evaluation=False):\n', 'after': "def parse_args(parser, tokens, *, evaluation=False):\n    from cdlno.linearno_history.industrial import intercept\n    history_args = intercept(parser, tokens, task='airfrans', evaluation=evaluation)\n    if history_args is not None:\n        return history_args\n", 'count': 1}, {'before': 'def run_cli(args):\n', 'after': "def run_cli(args):\n    if getattr(args, '_linearno_history_adapter', False):\n        from cdlno.linearno_history.air_entry import run_cli as history_run_cli\n        return history_run_cli(args)\n", 'count': 1}, {'before': '        source = path.read_text()\n        sources[key] = _sha256(path)', 'after': '        source = path.read_text()\n        from cdlno.linearno_history.provenance import baseline_source\n        source = baseline_source(key, source)\n        sources[key] = hashlib.sha256(source.encode()).hexdigest()', 'count': 1}], 'cdlno/linearno/car_entry.py': [{'before': 'def parse_args(parser, tokens, *, evaluation=False):\n', 'after': "def parse_args(parser, tokens, *, evaluation=False):\n    from cdlno.linearno_history.industrial import intercept\n    history_args = intercept(parser, tokens, task='car', evaluation=evaluation)\n    if history_args is not None:\n        return history_args\n", 'count': 1}, {'before': 'def run_cli(args):\n', 'after': "def run_cli(args):\n    if getattr(args, '_linearno_history_adapter', False):\n        from cdlno.linearno_history.car_entry import run_cli as history_run_cli\n        return history_run_cli(args)\n", 'count': 1}], 'Airfoil-Design-AirfRANS/train.py': [{'before': 'DataLoader(val_dataset, batch_size=1)', 'after': "DataLoader(val_dataset, batch_size=1, **(linearno_run.loader_kwargs('test') if hasattr(linearno_run, 'loader_kwargs') else {}))", 'count': 1}, {'before': "DataLoader(train_dataset_sampled, batch_size=hparams['batch_size'], shuffle=True)", 'after': "DataLoader(train_dataset_sampled, batch_size=hparams['batch_size'], shuffle=True, **(linearno_run.loader_kwargs('train') if hasattr(linearno_run, 'loader_kwargs') else {}))", 'count': 1}, {'before': 'DataLoader(val_dataset_sampled, batch_size=1, shuffle=True)', 'after': "DataLoader(val_dataset_sampled, batch_size=1, shuffle=True, **(linearno_run.loader_kwargs('test') if hasattr(linearno_run, 'loader_kwargs') else {}))", 'count': 1}]}


def baseline_source(relative, text):
    from cdlno.linearno_loop.provenance import legacy_source
    text = legacy_source(relative, text)
    for change in reversed(ROUTING_REPLACEMENTS.get(relative, ())):
        if text.count(change['after']) != change['count']:
            raise ValueError('unrecognized history routing source; cannot preserve legacy fingerprint: '+relative)
        text=text.replace(change['after'],change['before'])
    return text


def research_provenance(base):
    root=Path(__file__).resolve().parents[2]
    from cdlno.linearno_loop.provenance import legacy_source
    actual={p:hashlib.sha256(legacy_source(p,(root/p).read_text()).encode()).hexdigest() for p in ROUTING_REPLACEMENTS}
    result=dict(base)
    result['source_sha256']=digest(dict(baseline=base['source_sha256'],history=source_hash(),routing=actual))
    result['normalized_patch_sha256']=digest(dict(baseline=base['normalized_patch_sha256'],routing=actual))
    return result
