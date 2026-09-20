"""Read-only historical comparison base, never the moving checkout HEAD."""
from contextlib import contextmanager
import re
import subprocess
from unittest.mock import patch


@contextmanager
def historical_git_base(commit):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('historical base must be a full immutable SHA')
    original = subprocess.check_output
    def read(command, *args, **kwargs):
        if command[:2] == ['git', 'show'] and command[2].startswith('HEAD:'):
            command = [*command[:2], commit + command[2][4:], *command[3:]]
        return original(command, *args, **kwargs)
    with patch('subprocess.check_output', side_effect=read):
        yield
