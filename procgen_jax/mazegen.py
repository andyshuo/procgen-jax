"""ProcGen's maze generator, transcribed from ``procgen/src/mazegen.cpp``.

``MazeGen`` builds a rectangular maze on a ``(maze_dim + 2)`` grid (a one-cell
border of walls) with Kruskal's algorithm, then optionally punches extra openings.
It has its own quirks that matter for parity:

* walls are not shuffled; the C++ removes a *random element of a shrinking
  vector*, so the draw is ``randn(remaining)`` and the order of the rest is kept;
* the union-find only ever compares representatives, so a plain parent array
  reproduces the same decisions;
* the first cell is written as ``0`` rather than ``SPACE``.

The generator consumes an :class:`~procgen_jax.rng.Mt19937` state, so the grids
are identical to the original for the same seed and parameters.
"""

import jax
import jax.numpy as jnp

from procgen_jax import rng
from procgen_jax.engine import WALL_OBJ

#: ``MazeGen`` works in ProcGen's own id space, where empty cells are 100
#: (this repo's tile grid uses 0, which is why the constant is imported explicitly).
from procgen_jax.objects import PROCGEN_AGENT as AGENT_OBJ
from procgen_jax.objects import PROCGEN_DOOR as DOOR_OBJ
from procgen_jax.objects import PROCGEN_GOAL as EXIT_OBJ
from procgen_jax.objects import PROCGEN_KEY as KEY_OBJ
from procgen_jax.objects import PROCGEN_SPACE as SPACE

MAZE_OFFSET = 1
#: ``object-ids.h``: ``INVALID_OBJ``, returned for lookups outside the grid.
INVALID_OBJ = -1


def wall_list(maze_dim: int):
    """``MazeGen``'s wall list, in the order the original builds it.

    Each entry is ``(x1, y1, x2, y2)``: the two cells a wall separates.
    """
    walls = []
    for i in range(1, maze_dim, 2):
        for j in range(0, maze_dim, 2):
            if 0 < i < maze_dim - 1:
                walls.append((i - 1, j, i + 1, j))
    for i in range(0, maze_dim, 2):
        for j in range(1, maze_dim, 2):
            if 0 < j < maze_dim - 1:
                walls.append((i, j - 1, i, j + 1))
    return walls


def _find(parents, node):
    """Root of ``node`` (the C++ looks up per-set index maps; only equality is used)."""
    return jax.lax.while_loop(lambda n: parents[n] != n, lambda n: parents[n], node)


def _neighbours(index, array_dim):
    """``get_neighbors`` order: left, down, up, right, as ``(indices, valid)``."""
    x = index % array_dim
    y = index // array_dim
    offsets = jnp.asarray([-1, -array_dim, array_dim, 1], jnp.int32)
    nbr = index + offsets
    # ``get_obj`` rejects the whole one-cell border (``x <= 0`` or ``x >= array_dim - 1``),
    # so a neighbour only counts if it lands strictly inside it.
    valid = jnp.asarray([x > MAZE_OFFSET, y > MAZE_OFFSET, y < array_dim - 2, x < array_dim - 2])
    return nbr, valid


def generate_maze(state: rng.Mt19937, maze_dim: int):
    """Kruskal's algorithm. Returns ``(state, grid)`` with the maze at +1 offset."""
    array_dim = maze_dim + 2
    size = array_dim * array_dim
    walls = wall_list(maze_dim)
    total = len(walls)

    wall_x1 = jnp.asarray([w[0] for w in walls], jnp.int32)
    wall_y1 = jnp.asarray([w[1] for w in walls], jnp.int32)
    wall_x2 = jnp.asarray([w[2] for w in walls], jnp.int32)
    wall_y2 = jnp.asarray([w[3] for w in walls], jnp.int32)

    grid = jnp.full((size,), WALL_OBJ, jnp.int32)
    grid = grid.at[MAZE_OFFSET * array_dim + MAZE_OFFSET].set(0)
    parents = jnp.arange(maze_dim * maze_dim, dtype=jnp.int32)
    remaining = jnp.asarray(total, jnp.int32)
    positions = jnp.arange(total)

    def carve(grid, parents, state, count, wall_x1, wall_y1, wall_x2, wall_y2):
        state, pick = rng.randn(state, count)
        x1, y1 = wall_x1[pick], wall_y1[pick]
        x2, y2 = wall_x2[pick], wall_y2[pick]
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        root_a = _find(parents, maze_dim * y1 + x1)
        root_b = _find(parents, maze_dim * y2 + x2)
        can = (grid[(mid_y + MAZE_OFFSET) * array_dim + mid_x + MAZE_OFFSET] == WALL_OBJ) & (root_a != root_b)

        opened = jnp.stack(
            [
                (y1 + MAZE_OFFSET) * array_dim + x1 + MAZE_OFFSET,
                (mid_y + MAZE_OFFSET) * array_dim + mid_x + MAZE_OFFSET,
                (y2 + MAZE_OFFSET) * array_dim + x2 + MAZE_OFFSET,
            ]
        )
        grid = jax.lax.cond(
            can,
            lambda g: g.at[opened].set(SPACE),
            lambda g: g,
            grid,
        )
        return grid, state, pick, can, root_a, root_b

    def body(index, carry):
        grid, parents, state, count, positions, wall_x1, wall_y1, wall_x2, wall_y2 = carry
        grid, state, pick, can, root_a, root_b = carve(grid, parents, state, count, wall_x1, wall_y1, wall_x2, wall_y2)
        parents = jnp.where(can, parents.at[root_a].set(root_b), parents)

        # ``walls.erase(walls.begin() + pick)`` keeps the relative order.
        keep = positions < pick
        shifted = jnp.concatenate(
            [
                jnp.stack([wall_x1, wall_y1, wall_x2, wall_y2], axis=1)[1:],
                jnp.zeros((1, 4), jnp.int32),
            ]
        )
        current = jnp.stack([wall_x1, wall_y1, wall_x2, wall_y2], axis=1)
        updated = jnp.where(keep[:, None], current, shifted)
        count = count - 1
        return (
            grid,
            parents,
            state,
            count,
            positions,
            updated[:, 0],
            updated[:, 1],
            updated[:, 2],
            updated[:, 3],
        )

    carry = (grid, parents, state, remaining, positions, wall_x1, wall_y1, wall_x2, wall_y2)
    grid, _, state, _, _, _, _, _, _ = jax.lax.fori_loop(0, total, body, carry)
    return state, grid.reshape(array_dim, array_dim)


def generate_maze_no_dead_ends(state: rng.Mt19937, maze_dim: int):
    """``generate_maze`` then punch out one wall behind every dead end."""
    array_dim = maze_dim + 2
    size = array_dim * array_dim
    state, grid = generate_maze(state, maze_dim)
    flat = grid.reshape(size)

    def body(index, carry):
        flat, state = carry
        nbr, valid = _neighbours(index, array_dim)
        in_range = jnp.clip(nbr, 0, size - 1)
        values = flat[in_range]
        is_space = (values == SPACE) & valid
        is_wall = (values == WALL_OBJ) & valid
        dead_end = (flat[index] == SPACE) & (is_space.sum() == 1)

        state, pick = jax.lax.cond(
            dead_end & is_wall.any(),
            lambda s: rng.randn(s, is_wall.sum()),
            lambda s: (s, jnp.zeros((), jnp.int32)),
            state,
        )
        order = jnp.argsort(jnp.where(is_wall, jnp.arange(4), 4))
        target = in_range[order[pick]]
        flat = jax.lax.cond(dead_end & is_wall.any(), lambda g: g.at[target].set(SPACE), lambda g: g, flat)
        return flat, state

    flat, state = jax.lax.fori_loop(0, size, body, (flat, state))
    return state, flat.reshape(array_dim, array_dim)


def _neighbour_tables(flat, array_dim):
    """Per index: neighbour values, neighbour indices and their validity."""
    size = array_dim * array_dim
    positions = jnp.arange(size, dtype=jnp.int32)
    x = positions % array_dim
    y = positions // array_dim
    offsets = jnp.asarray([-1, -array_dim, array_dim, 1], jnp.int32)
    nbr = jnp.clip(positions[:, None] + offsets[None, :], 0, size - 1)
    valid = jnp.stack([x > MAZE_OFFSET, y > MAZE_OFFSET, y < array_dim - 2, x < array_dim - 2], axis=1)
    return flat[nbr], nbr, valid, positions


def _expand_to_type(flat, array_dim, s0, s1, kind):
    """``expand_to_type``: BFS out of ``s0`` to the first neighbour of ``kind``.

    The original walks a ``std::set`` frontier in ascending order and returns as
    soon as a cell has a ``kind`` neighbour -- so the cells *after* that one in the
    same level never contribute their space neighbours to ``s1``. That partial
    side effect is what ``positions <= winner`` reproduces here.
    """
    size = array_dim * array_dim
    values, nbr, valid, positions = _neighbour_tables(flat, array_dim)
    rows = jnp.arange(size)
    big = size

    reaches = (values == kind) & valid
    first = jnp.where(reaches, jnp.arange(4)[None, :], 4).argmin(axis=1)
    target = jnp.where(reaches.any(axis=1), nbr[rows, first], big)
    spaces = (values == SPACE) & valid

    def body(_, carry):
        curr, s1, found, found_index = carry
        candidates = jnp.where(curr, target, big)
        best = candidates.min()
        hit = best < big
        winner = jnp.where(hit, candidates.argmin(), size - 1)
        contributing = curr & (positions <= winner) & ~found

        # dynamic boolean indexing is not allowed under jit: flatten and scatter,
        # using one extra slot as the "not a neighbour" sink.
        targets = jnp.where((spaces & contributing[:, None]).reshape(-1), nbr.reshape(-1), size)
        spawned = jnp.zeros((size + 1,), bool).at[targets].set(True)[:size]
        fresh = spawned & ~s0 & ~s1 & ~found
        s1 = s1 | fresh
        return fresh, s1, found | hit, jnp.where(hit, best, found_index)

    _, s1, _, found = jax.lax.fori_loop(0, size, body, (s0, s1, jnp.asarray(False), jnp.asarray(-1, jnp.int32)))
    return found, s1


def generate_maze_with_doors(state: rng.Mt19937, maze_dim: int, num_doors: int):
    """``generate_maze_with_doors``: forks become doors, then keys and an exit.

    Returns ``(state, grid)``. Doors are numbered ``DOOR_OBJ + 1 ...``, keys
    ``KEY_OBJ + 1 ...`` and the exit is ``EXIT_OBJ``; the agent cell is
    ``AGENT_OBJ``.
    """
    array_dim = maze_dim + 2
    size = array_dim * array_dim
    state, grid = generate_maze(state, maze_dim)
    flat = grid.reshape(size)
    values, nbr, valid, positions = _neighbour_tables(flat, array_dim)

    # ---- forks become doors ------------------------------------------------- #
    space_neighbours = ((values == SPACE) & valid).sum(axis=1)
    forks = (flat == SPACE) & (space_neighbours > 2)
    fork_list = jnp.sort(jnp.where(forks, positions, size + 1))[: int(forks.sum())]
    picks = min(num_doors, int(forks.sum()))

    chosen = jnp.zeros((size,), bool)
    remaining = fork_list
    for _ in range(picks):
        state, index = rng.randn(state, remaining.shape[0])
        chosen = chosen.at[remaining[index]].set(True)
        shifted = jnp.concatenate([remaining[1:], jnp.full((1,), size + 1, jnp.int32)])
        keep = jnp.arange(remaining.shape[0]) < index
        remaining = jnp.where(keep, remaining, shifted)[: remaining.shape[0] - 1]
    flat = jnp.where(chosen, DOOR_OBJ, flat)

    # ---- a start cell that is not next to a door ---------------------------- #
    def next_to_door(cell):
        return ((flat[nbr[cell]] == DOOR_OBJ) & valid[cell]).any()

    space_list = jnp.sort(jnp.where(flat == SPACE, positions, size + 1))[: int((flat == SPACE).sum())]

    def pick_agent(_, carry):
        state, agent, done = carry
        state, index = jax.lax.cond(
            done,
            lambda st: (st, jnp.zeros((), jnp.int32)),
            lambda st: rng.randn(st, space_list.shape[0]),
            state,
        )
        candidate = space_list[jnp.clip(index, 0, space_list.shape[0] - 1)]
        fresh = ~done & ~next_to_door(candidate)
        return state, jnp.where(fresh, candidate, agent), done | fresh

    agent_cell = jnp.asarray(0, jnp.int32)
    state, agent_cell, _ = jax.lax.fori_loop(
        0,
        size,
        pick_agent,
        (state, agent_cell, jnp.asarray(False)),
    )
    flat = flat.at[agent_cell].set(AGENT_OBJ)

    # ---- keys and the exit --------------------------------------------------- #
    s0 = jnp.zeros((size,), bool).at[agent_cell].set(True)
    for door_num in range(picks + 1):
        # ``std::set<int> s1;`` is declared inside the loop in the C++, so each
        # round starts from an empty reachable set even though ``s0`` accumulates.
        s1 = jnp.zeros((size,), bool)
        found_door = jnp.asarray(-1, jnp.int32)
        if door_num < picks:
            found_door, s1 = _expand_to_type(flat, array_dim, s0, s1, DOOR_OBJ)
            flat = flat.at[found_door].set(DOOR_OBJ + door_num + 1)
            s0 = s0 | s1

        _, s1 = _expand_to_type(flat, array_dim, s0, s1, -999)
        cells = jnp.sort(jnp.where(s1, positions, size + 1))[: int(s1.sum())]
        state, index = rng.randn(state, cells.shape[0])
        key_cell = cells[index]
        flat = flat.at[key_cell].set(EXIT_OBJ if door_num == picks else KEY_OBJ + door_num + 1)
        s0 = s0 | s1 | (positions == found_door)
    return state, flat.reshape(array_dim, array_dim)
