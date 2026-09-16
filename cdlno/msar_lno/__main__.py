"""Data-free resolved configuration preview. Does not create a model or run directory."""
import argparse
import json

from .config import FAMILY, input_layout
from .metadata import new_run_path
from .options import explicit_arguments, parser_for_family, resolve_training
from .profiles import TASKS


def main():
    base = argparse.ArgumentParser(description=__doc__)
    base.add_argument('--task', required=True, choices=TASKS)
    base.add_argument('--input-tokens', type=int, help='log the legal M1>N expansion, if applicable')
    base.add_argument('--output-root', default='output')
    base.add_argument('--save-name')
    parser = parser_for_family(base, FAMILY)
    import sys
    explicit = explicit_arguments(parser, sys.argv[1:])
    try:
        resolved = resolve_training(explicit['task'], explicit)
        record = resolved.to_dict()
        record['proposed_run_path'] = str(new_run_path(explicit.get('output_root', 'output'), resolved,
                                                      save_name=explicit.get('save_name')))
        if 'input_tokens' in explicit:record['input_layout'] = input_layout(resolved.architecture, explicit['input_tokens'])
    except ValueError as error:parser.error(str(error))
    record['scope'] = 'configuration only; no MSAR model/weights/training/auxiliary computation in M1'
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == '__main__':main()
