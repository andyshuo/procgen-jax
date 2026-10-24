"""Build the C++ roomgen-reference fixture.

    pixi run roomgen-reference     # -> tests/data/roomgen_reference.npz

Compiles ProcGen's own ``roomgen.cpp`` against a grid-only stand-in for
``BasicAbstractGame`` (``tools/reference/roomgen_harness/basic-abstract-game.h``,
whose four accessors mirror ``basic-abstract-game.cpp``), feeds it a
jumper-shaped grid, and records the cellular-automata pass, the best room, an
expanded room and a path.  That is the ground truth for ``procgen_jax.roomgen``.

Requires ``g++``.
"""

import os
import pathlib
import subprocess
import urllib.request

import numpy as np

ROOT = 'https://cdn.jsdelivr.net/gh/openai/procgen@master/procgen/src/'
SOURCES = (
    'roomgen.cpp',
    'roomgen.h',
    'mazegen.cpp',
    'mazegen.h',
    'grid.h',
    'object-ids.h',
    'randgen.cpp',
    'randgen.h',
    'buffer.h',
    'cpp-utils.h',
    'cpp-utils.cpp',
)
CACHE = pathlib.Path('.reference_cache')
HARNESS_DIR = pathlib.Path('tools/reference/roomgen_harness')
HARNESS = pathlib.Path('tools/reference/roomgen_ref.cpp')
OUTPUT = pathlib.Path('tests/data/roomgen_reference.npz')
DIM = 40


def fetch_sources() -> None:
    CACHE.mkdir(exist_ok=True)
    if 'SSL_CERT_FILE' not in os.environ:
        bundle = pathlib.Path('/etc/ssl/certs/ca-certificates.crt')
        if bundle.exists():
            os.environ['SSL_CERT_FILE'] = str(bundle)
    for name in SOURCES:
        target = CACHE / name
        if not target.exists():
            target.write_bytes(urllib.request.urlopen(ROOT + name, timeout=60).read())
    # the harness compiles roomgen against its own stand-in header, so the sources
    # have to sit next to it
    for name in ('roomgen.cpp', 'roomgen.h'):
        (HARNESS_DIR / name).write_bytes((CACHE / name).read_bytes())


def run_reference() -> str:
    binary = CACHE / 'roomgen_ref'
    subprocess.run(
        [
            'g++',
            '-std=c++17',
            '-O1',
            '-I',
            str(HARNESS_DIR),
            '-I',
            str(CACHE),
            '-o',
            str(binary),
            str(HARNESS),
            str(HARNESS_DIR / 'roomgen.cpp'),
            str(CACHE / 'mazegen.cpp'),
            str(CACHE / 'randgen.cpp'),
            str(CACHE / 'cpp-utils.cpp'),
        ],
        check=True,
    )
    return subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout


def parse(text: str) -> list[dict]:
    lines = text.splitlines()
    cases, cursor = [], 0
    while cursor < len(lines):
        if not lines[cursor].startswith('CASE'):
            cursor += 1
            continue
        case = {'seed': int(lines[cursor].split()[1])}
        cursor += 1
        while lines[cursor] != 'END':
            head = lines[cursor].split()
            if head[0] == 'GRID':
                grid = np.array([[int(v) for v in lines[cursor + 1 + y].split()] for y in range(DIM)], np.int32)
                case[head[1]] = grid
                cursor += 1 + DIM
            elif head[0] == 'SET':
                values = [int(v) for v in lines[cursor + 1].split()] if int(head[2]) else []
                mask = np.zeros(DIM * DIM, bool)
                mask[values] = True
                case[head[1]] = mask.reshape(DIM, DIM)
                cursor += 2
            elif head[0] == 'PATH':
                values = [int(v) for v in lines[cursor + 1].split()] if int(head[1]) else []
                case['path'] = values
                cursor += 2
            else:
                cursor += 1
        cases.append(case)
        cursor += 1
    return cases


def main() -> None:
    fetch_sources()
    cases = parse(run_reference())
    length = max(len(case['path']) for case in cases)
    paths = np.full((len(cases), length), -2, np.int32)
    for index, case in enumerate(cases):
        paths[index, : len(case['path'])] = case['path']

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        seeds=np.asarray([case['seed'] for case in cases], np.int64),
        init=np.stack([case['init'] for case in cases]),
        ca=np.stack([case['ca'] for case in cases]),
        best=np.stack([case['best'] for case in cases]),
        expanded=np.stack([case['expanded'] for case in cases]),
        paths=paths,
        path_lengths=np.asarray([len(case['path']) for case in cases], np.int64),
    )
    print(f'{len(cases)} roomgen cases -> {OUTPUT}')


if __name__ == '__main__':
    main()
