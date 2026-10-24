// Golden reference for ProcGen's maze generator.
//
// Builds ``mazegen.cpp`` unchanged against its own ``RandGen`` and prints the grid
// for a handful of seeds and parameters, so the JAX port can be checked cell for
// cell -- including the order in which random numbers are consumed.
//
// ``tools/build_maze_reference.py`` compiles and runs this; the fixture lands in
// ``tests/data/maze_reference.npz``.

#include "mazegen.h"
#include "object-ids.h"
#include "randgen.h"

#include <cstdio>

namespace {

const char *MODE_NAMES[] = {"maze", "no_dead_ends", "with_doors"};

void emit(const char *mode, int seed, int maze_dim, int doors, MazeGen &maze) {
    printf("CASE %s SEED %d DIM %d DOORS %d\n", mode, seed, maze_dim, doors);
    const int array_dim = maze_dim + 2;
    printf("GRID\n");
    for (int y = 0; y < array_dim; y++) {
        for (int x = 0; x < array_dim; x++) printf("%d ", maze.grid.get(x, y));
        printf("\n");
    }
    printf("END\n");
}

void run(int mode, int seed, int maze_dim, int doors) {
    RandGen rand_gen;
    rand_gen.seed(seed);
    MazeGen maze(&rand_gen, maze_dim);

    if (mode == 0) {
        maze.generate_maze();
    } else if (mode == 1) {
        maze.generate_maze_no_dead_ends();
    } else {
        maze.generate_maze_with_doors(doors);
    }
    emit(MODE_NAMES[mode], seed, maze_dim, doors, maze);
}

}  // namespace

int main() {
    const int seeds[] = {0, 1, 7, 42, 12345};
    for (int seed : seeds) {
        run(0, seed, 9, 0);
        run(0, seed, 13, 0);
        run(1, seed, 9, 0);
        run(1, seed, 13, 0);
        run(2, seed, 13, 1);
        run(2, seed, 13, 2);
        run(2, seed, 13, 3);
    }
    return 0;
}
