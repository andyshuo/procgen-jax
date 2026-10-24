"""ProcGen's cellular-automata room generator, from ``procgen/src/roomgen.cpp``.

``RoomGenerator`` turns a noisy grid into caves, then reasons about the open space:
it labels rooms, picks the biggest, and can grow or connect them.  It never draws a
random number, so it is a pure function of the grid -- but two quirks have to be
reproduced for the levels to match:

* ``build_room`` never inserts its *seed* cell into the room it returns;
* ``find_path`` does not seed its ``covered`` set with the source either, so the
  source can be discovered a second time, and it walks a FIFO queue whose order sets
  each cell's parent.

Everything works in ProcGen's own id space (``WALL_OBJ`` = 51, ``SPACE`` = 100) on a
flat ``(dim * dim,)`` grid.
"""

import jax
import jax.numpy as jnp

WALL_OBJ = 51
SPACE = 100
#: ``object-ids.h``: outside the grid ``get_obj`` returns ``INVALID_OBJ``.
INVALID_OBJ = -1
INVALID_IDX = -2


def _neighbours(positions, dim):
    """The four orthogonal neighbours in the order the C++ scans them.

    ``(x-1, y)``, ``(x, y-1)``, ``(x, y+1)``, ``(x+1, y)``: the loops run ``i`` then
    ``j`` over ``-1..1``, keeping the pairs where exactly one is zero.
    """
    x = positions % dim
    y = positions // dim
    offsets = jnp.asarray([-1, -dim, dim, 1], dtype=jnp.int32)
    return positions[:, None] + offsets[None, :], jnp.stack([x > 0, y > 0, y < dim - 1, x < dim - 1], axis=1)


def _eight_neighbours(positions, dim):
    x = positions % dim
    y = positions // dim
    offsets = []
    valid = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                continue
            offsets.append(i + j * dim)
            valid.append((x + i >= 0) & (x + i < dim) & (y + j >= 0) & (y + j < dim))
    return positions[:, None] + jnp.asarray(offsets, jnp.int32)[None, :], jnp.stack(valid, axis=1)


def _spread(frontier, is_space, nbr, valid, size):
    """The open cells touching ``frontier``.

    ``fresh[j]`` has to describe the *destination*, so the neighbour indices are
    scattered into a mask (the C++ pushes them onto a queue); an extra slot catches
    the out-of-grid neighbours.
    """
    targets = jnp.where(frontier[:, None] & valid, nbr, size)
    landed = jnp.zeros((size + 1,), bool).at[targets.reshape(-1)].set(True)[:size]
    return landed & is_space


def update(flat: jax.Array, dim: int) -> jax.Array:
    """``update``: one cellular-automata pass.

    A cell becomes a wall when at least five of the nine cells around it (itself
    included) are walls, and open space otherwise.
    """
    grid = flat.reshape(dim, dim)
    padded = jnp.pad(grid, 1, constant_values=INVALID_OBJ)
    is_wall = (padded == WALL_OBJ).astype(jnp.int32)
    window = jnp.zeros((dim, dim), jnp.int32)
    for dy in range(3):
        for dx in range(3):
            window = window + is_wall[dy : dy + dim, dx : dx + dim]
    return jnp.where(window >= 5, WALL_OBJ, SPACE).reshape(-1)


def build_room(flat: jax.Array, dim: int, seed: jax.Array) -> jax.Array:
    """``build_room``: the four-connected room around ``seed``, without the seed."""
    size = dim * dim
    positions = jnp.arange(size, dtype=jnp.int32)
    nbr, valid = _neighbours(positions, dim)
    is_space = flat == SPACE
    room = jnp.zeros((size,), bool)
    frontier = jnp.zeros((size,), bool).at[seed].set(is_space[seed])

    def body(carry):
        frontier, room = carry
        fresh = _spread(frontier, is_space, nbr, valid, size) & ~room
        return fresh, room | fresh

    def cond(carry):
        frontier, _ = carry
        return frontier.any()

    _, room = jax.lax.while_loop(cond, body, (frontier, room))
    return room


def find_best_room(flat: jax.Array, dim: int) -> jax.Array:
    """``find_best_room``: the largest open room, minus the cell that seeded it.

    The C++ scans cells in order and keeps a room only when it is *strictly* bigger,
    so component order decides ties; here the label of a component is its lowest
    cell index, which reproduces that.  ``build_room`` never inserts its seed cell
    directly, but the seed is rediscovered from its neighbours, so a component of
    more than one cell comes back whole -- an isolated cell stays empty.
    """
    size = dim * dim
    positions = jnp.arange(size, dtype=jnp.int32)
    nbr, valid = _neighbours(positions, dim)
    is_space = flat == SPACE
    labels = jnp.where(is_space, positions, size)

    def body(labels):
        gathered = jnp.take(labels, nbr, axis=0, mode='clip')
        best = jnp.where(valid, gathered, size).min(axis=1)
        return jnp.where(is_space, jnp.minimum(labels, best), size)

    def cond(labels):
        gathered = jnp.take(labels, nbr, axis=0, mode='clip')
        best = jnp.where(valid, gathered, size).min(axis=1)
        updated = jnp.where(is_space, jnp.minimum(labels, best), size)
        return (updated != labels).any()

    labels = jax.lax.while_loop(cond, body, labels)

    sizes = jnp.zeros((size,), jnp.int32).at[jnp.clip(labels, 0, size - 1)].add(is_space)
    best_label = jnp.argmax(sizes)
    component = labels == best_label
    # ``build_room`` does not insert its seed up front, but the seed is found again as
    # a neighbour of its neighbours -- so only a one-cell room comes back empty.
    return component & (sizes[best_label] > 1)


def expand_room(flat: jax.Array, dim: int, room: jax.Array, rounds: int) -> jax.Array:
    """``expand_room``: grow the set into open space, eight-connected, for ``rounds``."""
    size = dim * dim
    positions = jnp.arange(size, dtype=jnp.int32)
    nbr, valid = _eight_neighbours(positions, dim)
    is_space = flat == SPACE
    current = room

    for _ in range(rounds):
        fresh = (jnp.take(is_space, nbr, axis=0, mode='clip') & valid & current[:, None]).any(axis=1)
        fresh = fresh & ~room
        room = room | fresh
        current = fresh
    return room


def find_path(flat: jax.Array, dim: int, src: jax.Array, dst: jax.Array):
    """``find_path``: BFS from ``src`` to ``dst``, returning ``(cells, length)``.

    The queue is simulated in order because each cell's parent is whichever cell
    discovered it first, and ``jumper`` widens the path, so which cells end up on it
    matters.  ``covered`` starts empty (the source can be discovered again) and the
    search stops the moment ``dst`` is popped, without expanding it.
    """
    size = dim * dim
    capacity = size + 1  # the last slot is a scatter sink
    positions = jnp.arange(size, dtype=jnp.int32)
    nbr, valid = _neighbours(positions, dim)
    is_space = flat == SPACE

    expanded = jnp.full((capacity,), INVALID_IDX, jnp.int32).at[0].set(src)
    parents = jnp.full((capacity,), INVALID_IDX, jnp.int32).at[0].set(-1)
    covered = jnp.zeros((size,), bool)
    tail = jnp.asarray(1, jnp.int32)

    def body(head, carry):
        expanded, parents, covered, tail, done = carry
        current = expanded[head]
        cell = jnp.clip(current, 0, size - 1)
        room_cell = (current >= 0) & (current < size) & is_space[cell]
        active = room_cell & ~done & (current != dst)
        cells = nbr[cell]
        targets = jnp.clip(cells, 0, size - 1)
        fresh = valid[cell] & jnp.take(is_space, cells, axis=0, mode='clip') & ~covered[targets] & active
        rank = jnp.cumsum(fresh.astype(jnp.int32)) - 1
        slots = jnp.where(fresh, tail + rank, capacity - 1)
        expanded = expanded.at[slots].set(jnp.where(fresh, cells, expanded[slots]))
        parents = parents.at[slots].set(jnp.where(fresh, head, parents[slots]))
        covered = covered.at[targets].set(jnp.where(fresh, True, covered[targets]))
        tail = tail + fresh.sum()
        done = done | (room_cell & (current == dst))
        return expanded, parents, covered, tail, done

    expanded, parents, _, _, _ = jax.lax.fori_loop(
        0, capacity, body, (expanded, parents, covered, tail, jnp.asarray(False))
    )

    found = jnp.argmax(expanded == dst)
    hit = (expanded == dst).any()

    def walk(step, carry):
        index, length = carry
        on_path = index >= 0
        length = length + on_path
        return jnp.where(on_path, parents[jnp.clip(index, 0, capacity - 1)], -1), length

    _, length = jax.lax.fori_loop(0, capacity, walk, (found, jnp.asarray(0, jnp.int32)))

    def write(step, carry):
        index, cells = carry
        on_path = index >= 0
        position = jnp.clip(length - 1 - step, 0, capacity - 1)
        cells = cells.at[position].set(jnp.where(on_path, expanded[jnp.clip(index, 0, capacity - 1)], cells[position]))
        return jnp.where(on_path, parents[jnp.clip(index, 0, capacity - 1)], -1), cells

    _, cells = jax.lax.fori_loop(0, capacity, write, (found, jnp.full((capacity,), INVALID_IDX, jnp.int32)))
    return cells, jnp.where(hit, length, 0)
