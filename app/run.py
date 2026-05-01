#!/usr/bin/env python3

# Windows asyncio fix: Python 3.10 on Windows raises ConnectionResetError
# (WinError 10054) in asyncio ProactorEventLoop when a subprocess pipe closes.
# This is a known CPython bug fixed in 3.11. Patch swallows the spurious error.
import sys as _sys
if _sys.platform == 'win32':
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport as _T
        _orig_ccl = _T._call_connection_lost
        def _patched_ccl(self, exc):
            try:
                _orig_ccl(self, exc)
            except ConnectionResetError:
                pass
        _T._call_connection_lost = _patched_ccl
    except Exception:
        pass

from roop import core
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--execution-provider', default='cuda', help='Execution provider: cpu or cuda')
args = parser.parse_args()
from roop import globals
globals.execution_providers = [args.execution_provider + 'ExecutionProvider']

if __name__ == '__main__':
    core.run()
