#!/usr/bin/env python3

from roop import core
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--execution-provider', default='cuda', help='Execution provider: cpu or cuda')
args = parser.parse_args()
from roop import globals
globals.execution_providers = [args.execution_provider + 'ExecutionProvider']

if __name__ == '__main__':
    core.run()
