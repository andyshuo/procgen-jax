"""Build the C++ maze-reference fixture.

    pixi run maze-reference      # -> tests/data/maze_reference.npz

Compiles ProcGen's own ``mazegen.cpp`` (against its own ``RandGen``) with
``tools/reference/mazegen_ref.cpp`` and records the grids it produces for a few
seeds, maze sizes and modes.  This is the ground truth for porting ``MazeGen`` to
JAX: ``generate_maze``, ``generate_maze_no_dead_ends`` and
``generate_maze_with_doors`` all have to match it cell for cell.

Requires ``g++``.
"""

import os
import pathlib
import subprocess
import urllib.request

import numpy as np

ROOT = 'https://cdn.jsdelivr.net/gh/openai/procgen@master/procgen/src/'
SOURCES = (
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
HARNESS = pathlib.Path('tools/reference/mazegen_ref.cpp')
OUTPUT = pathlib.Path('tests/data/maze_reference.npz')


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
    binary = CACHE / 'mazegen_ref'
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
            str(CACHE / 'mazegen.cpp'),
            str(CACHE / 'randgen.cpp'),
            str(CACHE / 'cpp-utils.cpp'),
        ],
        check=True,
    )
    return subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout


def parse(text: str) -> list[tuple[str, int, int, int, np.ndarray]]:
    """``(mode, seed, maze_dim, doors, grid)`` for every recorded case."""
    lines = text.splitlines()
    cases, cursor = [], 0
    while cursor < len(lines):
        if not lines[cursor].startswith('CASE'):
            cursor += 1
            continue
        parts = lines[cursor].split()
        mode, seed, dim, doors = parts[1], int(parts[3]), int(parts[5]), int(parts[7])
        cursor += 2  # CASE, GRID
        size = dim + 2
        grid = np.array([[int(v) for v in lines[cursor + y].split()] for y in range(size)], np.int32)
        cursor += size + 1  # rows, END
        cases.append((mode, seed, dim, doors, grid))
    return cases


def main() -> None:
    fetch_sources()
    cases = parse(run_reference())

    # Cases come in two sizes; pad to the largest so they stack.
    size = max(case[4].shape[0] for case in cases)
    grids = np.zeros((len(cases), size, size), np.int32)
    for index, case in enumerate(cases):
        grid = case[4]
        grids[index, : grid.shape[0], : grid.shape[1]] = grid

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        modes=np.asarray([case[0] for case in cases]),
        seeds=np.asarray([case[1] for case in cases], np.int64),
        dims=np.asarray([case[2] for case in cases], np.int64),
        doors=np.asarray([case[3] for case in cases], np.int64),
        grids=grids,
    )
    print(f'{len(cases)} maze cases from the C++ reference -> {OUTPUT}')


if __name__ == '__main__':
    main()
