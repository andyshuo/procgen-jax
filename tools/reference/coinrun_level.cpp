// Golden reference for coinrun level generation.
//
// This is ProcGen's ``generate_coin_to_the_right`` transcribed verbatim, plus the
// few engine helpers it touches, built against the *real* ``RandGen`` from
// ``procgen/src/randgen.cpp``.  It is not part of the Python package: it exists so
// that the JAX port can be checked tile for tile, including the order in which
// random numbers are consumed.
//
// ``tools/build_reference.py`` compiles this and stores the output; the test suite
// compares ``procgen_jax.games.coinrun.generate_level`` against it.
//
// Stubs: assets. ``choose_random_theme`` only needs to *consume* one draw from the
// right-sized range, so theme counts are hard-coded and no PNG is ever loaded.

#include "randgen.h"

#include <cstdio>
#include <vector>

namespace {

const int W = 64, H = 64;

// object ids from object-ids.h / coinrun.cpp
const int SPACE = 100, GOAL = 1, SAW = 2, ENEMY = 5, ENEMY1 = 6, WALL_MID = 15, WALL_TOP = 16,
          LAVA_MID = 17, LAVA_TOP = 18, ENEMY_BARRIER = 19, CRATE = 20;

// coinrun.cpp: number of sprites per type, which is what choose_random_theme draws.
const int PLAYER_THEMES = 5, ENEMY_THEMES = 9, CRATE_THEMES = 4, GROUND_THEMES = 6;

RandGen rand_gen;
std::vector<char> grid;
std::vector<int> entity_types;
std::vector<float> entity_x, entity_y, entity_vx;
std::vector<int> entity_themes;

int main_width = W, main_height = H;
float max_jump = 1.5f, gravity = 0.2f, maxspeed = 0.5f;

int get_obj(int x, int y) {
    if (x < 0 || y < 0 || x >= main_width || y >= main_height) return WALL_MID;
    return static_cast<unsigned char>(grid[y * main_width + x]);
}

void set_obj(int x, int y, int v) {
    if (x < 0 || y < 0 || x >= main_width || y >= main_height) return;
    grid[y * main_width + x] = static_cast<char>(v);
}

void fill_elem(int x, int y, int dx, int dy, int elem) {
    for (int j = 0; j < dx; j++)
        for (int k = 0; k < dy; k++) set_obj(x + j, y + k, elem);
}

bool is_wall(int type) { return type == WALL_MID || type == WALL_TOP; }

void fill_block_top(int x, int y, int dx, int dy, char fill, char top) {
    fill_elem(x, y, dx, dy - 1, fill);
    fill_elem(x, y + dy - 1, dx, 1, top);
}

void fill_ground_block(int x, int y, int dx, int dy) {
    fill_block_top(x, y, dx, dy, WALL_MID, WALL_TOP);
}

void fill_lava_block(int x, int y, int dx, int dy) {
    fill_block_top(x, y, dx, dy, LAVA_MID, LAVA_TOP);
}

void add_entity(float x, float y, float vx, int type, int theme) {
    entity_types.push_back(type);
    entity_x.push_back(x);
    entity_y.push_back(y);
    entity_vx.push_back(vx);
    entity_themes.push_back(theme);
}

void create_saw_enemy(int x, int y) { add_entity(x + .5f, y + .5f, 0, SAW, 0); }

void create_enemy(int x, int y) {
    // coinrun.cpp: add_entity(..., .15 * (rand_gen.randn(2) * 2 - 1), ...) then
    // choose_random_theme(ent) -> randn(asset_num_themes[ENEMY1]).
    float vx = .15f * (rand_gen.randn(2) * 2 - 1);
    int theme = rand_gen.randn(ENEMY_THEMES);
    add_entity(x + .5f, y + .5f, vx, ENEMY, theme);
}

void create_crate(int x, int y) {
    int theme = rand_gen.randn(CRATE_THEMES);
    add_entity(x + .5f, y + .5f, 0, CRATE, theme);
}

void init_floor_and_walls() {
    fill_elem(0, 0, main_width, 1, WALL_TOP);
    fill_elem(0, 0, 1, main_height, WALL_MID);
    fill_elem(main_width - 1, 0, 1, main_height, WALL_MID);
    fill_elem(0, main_height - 1, main_width, 1, WALL_MID);
}

// ---- verbatim from coinrun.cpp ---------------------------------------------
void generate_coin_to_the_right() {
    int max_difficulty = 3;
    int dif = rand_gen.randn(max_difficulty) + 1;

    int num_sections = rand_gen.randn(dif) + dif;
    int curr_x = 5;
    int curr_y = 1;

    int pit_threshold = dif;
    int danger_type = rand_gen.randn(3);

    bool allow_pit = true, allow_crate = true, allow_dy = true;

    int w = main_width;

    float _max_dy = max_jump * max_jump / (2 * gravity);
    float _max_dx = maxspeed * 2 * max_jump / gravity;

    int max_dy = (_max_dy - .5);
    int max_dx = (_max_dx - .5);

    bool allow_monsters = true;

    for (int section_idx = 0; section_idx < num_sections; section_idx++) {
        if (curr_x + 15 >= w) {
            break;
        }

        int dy = rand_gen.randn(4) + 1 + int(dif / 3);

        if (!allow_dy) {
            dy = 0;
        }

        if (dy > max_dy) {
            dy = max_dy;
        }

        if (curr_y >= 20) {
            dy *= -1;
        } else if (curr_y >= 5 && rand_gen.randn(2) == 1) {
            dy *= -1;
        }

        int dx = rand_gen.randn(2 * dif) + 3 + int(dif / 3);

        curr_y += dy;

        if (curr_y < 1) {
            curr_y = 1;
        }

        bool use_pit = allow_pit && (dx > 7) && (curr_y > 3) && (rand_gen.randn(20) >= pit_threshold);

        if (use_pit) {
            int x1 = rand_gen.randn(3) + 1;
            int x2 = rand_gen.randn(3) + 1;
            int pit_width = dx - x1 - x2;

            if (pit_width > max_dx) {
                pit_width = max_dx;
                x2 = dx - x1 - pit_width;
            }

            fill_ground_block(curr_x, 0, x1, curr_y);
            fill_ground_block(curr_x + dx - x2, 0, x2, curr_y);

            int lava_height = rand_gen.randn(curr_y - 3) + 1;

            if (danger_type == 0) {
                fill_lava_block(curr_x + x1, 1, pit_width, lava_height);
            } else if (danger_type == 1) {
                for (int ei = 0; ei < pit_width; ei++) {
                    create_saw_enemy(curr_x + x1 + ei, 1);
                }
            } else if (danger_type == 2) {
                for (int ei = 0; ei < pit_width; ei++) {
                    create_enemy(curr_x + x1 + ei, 1);
                }
            }

            if (pit_width > 4) {
                int x3, w1;
                if (pit_width == 5) {
                    x3 = 1 + rand_gen.randn(2);
                    w1 = 1 + rand_gen.randn(2);
                } else if (pit_width == 6) {
                    x3 = 2 + rand_gen.randn(2);
                    w1 = 1 + rand_gen.randn(2);
                } else {
                    x3 = 2 + rand_gen.randn(2);
                    int x4 = 2 + rand_gen.randn(2);
                    w1 = pit_width - x3 - x4;
                }

                fill_ground_block(curr_x + x1 + x3, curr_y - 1, w1, 1);
            }

        } else {
            fill_ground_block(curr_x, 0, dx, curr_y);

            int ob1_x = -1;
            int ob2_x = -1;

            if (rand_gen.randn(10) < (2 * dif) && dx > 3) {
                ob1_x = curr_x + rand_gen.randn(dx - 2) + 1;
                create_saw_enemy(ob1_x, curr_y);
            }

            if (rand_gen.randn(10) < dif && dx > 3 && (max_dx >= 4) && allow_monsters) {
                ob2_x = curr_x + rand_gen.randn(dx - 2) + 1;
                create_enemy(ob2_x, curr_y);
            }

            if (allow_crate) {
                for (int i = 0; i < 2; i++) {
                    int crate_x = curr_x + rand_gen.randn(dx - 2) + 1;

                    if (rand_gen.randn(2) == 1 && ob1_x != crate_x && ob2_x != crate_x) {
                        int pile_height = rand_gen.randn(3) + 1;

                        for (int j = 0; j < pile_height; j++) {
                            create_crate(crate_x, curr_y + j);
                        }
                    }
                }
            }
        }

        if (!is_wall(get_obj(curr_x - 1, curr_y))) {
            set_obj(curr_x - 1, curr_y, ENEMY_BARRIER);
        }

        curr_x += dx;

        set_obj(curr_x, curr_y, ENEMY_BARRIER);
    }

    set_obj(curr_x, curr_y, GOAL);

    fill_ground_block(curr_x, 0, 1, curr_y);
    fill_elem(curr_x + 1, 0, main_width - curr_x - 1, main_height, WALL_MID);
}

// ---- CoinRun::game_reset ---------------------------------------------------
void game_reset(int level_seed) {
    rand_gen.seed(level_seed);

    // BasicAbstractGame::game_reset, keeping only its draws.
    rand_gen.rand01();          // bg_pct_x
    rand_gen.randn(8);          // background_index (platform_backgrounds.size())
    rand_gen.rand01();          // random agent start, x
    rand_gen.rand01();          // random agent start, y

    // CoinRun::game_reset, HardMode branch.
    rand_gen.randn(PLAYER_THEMES);   // choose_random_theme(agent)
    rand_gen.randn(GROUND_THEMES);   // wall_theme

    gravity = 0.2f;
    max_jump = 1.5;
    maxspeed = .5;

    grid.assign(main_width * main_height, SPACE);
    entity_types.clear();
    entity_x.clear();
    entity_y.clear();
    entity_vx.clear();
    entity_themes.clear();

    init_floor_and_walls();
    generate_coin_to_the_right();
}

void emit(int seed) {
    printf("SEED %d\n", seed);
    printf("WIDTH %d HEIGHT %d\n", main_width, main_height);
    printf("GRID\n");
    for (int y = 0; y < main_height; y++) {
        for (int x = 0; x < main_width; x++) printf("%d ", get_obj(x, y));
        printf("\n");
    }
    printf("ENTITIES %zu\n", entity_types.size());
    for (size_t i = 0; i < entity_types.size(); i++) {
        printf("%d %.6f %.6f %.6f %d\n", entity_types[i], entity_x[i], entity_y[i], entity_vx[i],
               entity_themes[i]);
    }
    printf("END\n");
}

}  // namespace

int main(int argc, char **argv) {
    const int seeds[] = {0, 1, 2, 3, 7, 42, 123, 9999, 31337, 2147483647};
    for (int seed : seeds) {
        game_reset(seed);
        emit(seed);
    }
    (void)argc;
    (void)argv;
    return 0;
}
