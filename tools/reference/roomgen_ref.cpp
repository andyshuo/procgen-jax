// Golden reference for ProcGen's RoomGenerator (roomgen.cpp).
//
// Builds the real roomgen.cpp against a minimal grid-only stand-in for
// BasicAbstractGame, feeds it the kind of grid jumper makes (a no-dead-ends maze,
// scaled up three times, then randomly filled), and prints every result the port
// has to match: the cellular-automata grid, the best room, an expanded room and a
// path.

#include <cstdio>

#include "mazegen.h"
#include "object-ids.h"
#include "randgen.h"
#include "roomgen.h"

namespace {

int src_of(const std::set<int> &cells) {
    for (int cell : cells) return cell;
    return 0;
}

const int DIM = 40;
const int MAZE_SCALE = 3;

void print_grid(const char *label, const Grid<int> &grid) {
    printf("GRID %s\n", label);
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) printf("%d ", grid.get(x, y));
        printf("\n");
    }
}

void print_set(const char *label, const std::set<int> &cells) {
    printf("SET %s %d\n", label, (int)cells.size());
    for (int cell : cells) printf("%d ", cell);
    printf("\n");
}

void run(int seed) {
    RandGen rand_gen;
    rand_gen.seed(seed);

    int maze_dim = DIM / MAZE_SCALE;
    MazeGen maze(&rand_gen, maze_dim);
    maze.generate_maze_no_dead_ends();

    BasicAbstractGame game;
    game.grid.resize(DIM, DIM);
    game.grid_size = DIM * DIM;
    for (int i = 0; i < game.grid_size; i++) {
        int obj = maze.grid.get((i % DIM) / MAZE_SCALE + 1, (i / DIM) / MAZE_SCALE + 1);
        float prob = obj == WALL_OBJ ? .8f : .2f;
        game.set_obj(i, rand_gen.rand01() < prob ? WALL_OBJ : SPACE);
    }
    print_grid("init", game.grid);

    RoomGenerator roomgen(&game);
    roomgen.update();
    roomgen.update();
    print_grid("ca", game.grid);

    std::set<int> best;
    roomgen.find_best_room(best);
    print_set("best", best);

    std::set<int> room = best;
    roomgen.expand_room(room, 3);
    print_set("expanded", room);

    // The best room is often the whole open area, so the call above cannot grow at
    // all; start from one cell as well, which does exercise the growth.
    std::set<int> seed_room;
    seed_room.insert(src_of(best));
    roomgen.expand_room(seed_room, 2);
    print_set("grown", seed_room);

    int src = -1;
    for (int cell : best) {
        src = cell;
        break;
    }
    int dst = -1;
    for (int cell : best) dst = cell;
    std::vector<int> path;
    roomgen.find_path(src, dst, path);
    printf("PATH %d\n", (int)path.size());
    for (int cell : path) printf("%d ", cell);
    printf("\n");
    printf("END\n");
}

}  // namespace

int main() {
    const int seeds[] = {0, 1, 7, 42, 12345};
    for (int seed : seeds) {
        printf("CASE %d\n", seed);
        run(seed);
    }
    return 0;
}
