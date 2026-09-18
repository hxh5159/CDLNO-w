"""Read-only replay of 28 L4 rejection checks on existing synthetic archives."""
import contextlib
import copy
import functools
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from linearno.test_static_integration import args_for, get_model, SHORT
from cdlno.linearno.checkpoint import inspect_checkpoint, read_pair, strict_load
from cdlno.linearno.schema import validate_metadata


def main(artifacts):
    rows = []
    for task in SHORT:
        cls = get_model(args_for(task)).Model
        directory = artifacts / f'{task}-split'
        metadata, manifest = inspect_checkpoint(directory, 'final', cls)
        _, weights = read_pair(manifest, cls)

        @functools.wraps(cls.__init__)
        def forbidden(*args, **kwargs):
            raise AssertionError('constructed before metadata conflict')

        for flag, value in (('--linearno-profile', 'official_release'),
                            ('--linearno-rank', '9'), ('--linearno-variant', 'plain')):
            error = io.StringIO()
            with patch.object(cls, '__init__', new=forbidden), contextlib.redirect_stderr(error):
                try:
                    args_for(task, '--eval', '1', '--experiment-dir', directory, flag, value)
                except SystemExit as exc:
                    assert exc.code == 2 and 'conflicts with checkpoint' in error.getvalue()
                else:
                    raise AssertionError(f'{task}: accepted conflicting {flag}')
            rows.append(dict(task=task, check=flag, status='PASS'))

        model = cls(**metadata['model_spec']['constructor_kwargs'])
        for kind in ('missing', 'extra', 'shape'):
            bad = copy.deepcopy(weights)
            key = next(k for k, value in bad.items() if value.ndim and value.shape[0] > 1)
            if kind == 'missing':
                bad.pop(key)
            elif kind == 'extra':
                bad['unexpected_key'] = bad[key].clone()
            else:
                bad[key] = bad[key][:-1]
            try:
                strict_load(model, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{task}: accepted {kind} state')
            rows.append(dict(task=task, check=kind, status='PASS'))
        bad = copy.deepcopy(metadata)
        bad['family'] = 'CDLNO'
        try:
            validate_metadata(bad, constructor=cls)
        except ValueError as exc:
            assert 'family conflict' in str(exc)
        else:
            raise AssertionError(f'{task}: accepted wrong family')
        rows.append(dict(task=task, check='family', status='PASS'))
    print(json.dumps(dict(status='PASS', count=len(rows), checks=rows), indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
