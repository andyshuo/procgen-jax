"""Build the C++ reference fixture used by ``tests/test_coinrun_levels.py``.

    pixi run reference      # -> tests/data/coinrun_levels.npz

This downloads the handful of ProcGen sources the level generator needs, compiles
``tools/reference/coinrun_level.cpp`` (which is ``generate_coin_to_the_right``
transcribed verbatim, plus stubs for the asset system) against ProcGen's own
``RandGen``, and records the grids it produces.

The point is that the fixture comes from the *original* code, not from this port:
if the JAX level generator drifts, the parity test fails. Requires ``g++``.
"""

import os
import pathlib
import subprocess
import urllib.request

import numpy as np

ROOT = 'https://cdn.jsdelivr.net/gh/openai/procgen@master/procgen/src/'
SOURCES = ('randgen.cpp', 'randgen.h', 'buffer.h', 'cpp-utils.h', 'cpp-utils.cpp')
CACHE = pathlib.Path('.reference_cache')
HARNESS = pathlib.Path('tools/reference/coinrun_level.cpp')
OUTPUT = pathlib.Path('tests/data/coinrun_levels.npz')


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


def run_reference() -> str:
    binary = CACHE / 'coinrun_level'
    subprocess.run(
        [
            'g++',
            '-std=c++17',
            '-O1',
            '-I',
            str(CACHE),
            '-o',
            str(binary),
            str(HARNESS),
            str(CACHE / 'randgen.cpp'),
            str(CACHE / 'cpp-utils.cpp'),
        ],
        check=True,
    )
    return subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout


def parse(text: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    lines = text.splitlines()
    levels, cursor = {}, 0
    while cursor < len(lines):
        if not lines[cursor].startswith('SEED'):
            cursor += 1
            continue
        seed = int(lines[cursor].split()[1])
        cursor += 2  # SEED, WIDTH/HEIGHT
        assert lines[cursor] == 'GRID'
        cursor += 1
        grid = np.array([[int(v) for v in lines[cursor + y].split()] for y in range(64)], np.int32)
        cursor += 64
        assert lines[cursor].startswith('ENTITIES')
        count = int(lines[cursor].split()[1])
        cursor += 1
        bodies = np.array([[float(v) for v in lines[cursor + i].split()] for i in range(count)], np.float64).reshape(
            -1, 5
        )
        cursor += count
        assert lines[cursor] == 'END'
        cursor += 1
        levels[seed] = (grid, bodies)
    return levels


def main() -> None:
    fetch_sources()
    levels = parse(run_reference())

    seeds = sorted(levels)
    counts = np.array([len(levels[s][1]) for s in seeds], np.int32)
    entities = np.zeros((len(seeds), int(counts.max()), 5), np.float64)
    for index, seed in enumerate(seeds):
        entities[index, : counts[index]] = levels[seed][1]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        seeds=np.array(seeds, np.int64),
        grids=np.stack([levels[s][0] for s in seeds]),
        entities=entities,
        counts=counts,
    )
    printable = ', '.join(str(s) for s in seeds)
    print(f'{len(seeds)} levels from the C++ reference (seeds {printable}) -> {OUTPUT}')


if __name__ == '__main__':
    main()
