#pragma once

// A minimal stand-in for BasicAbstractGame so that ProcGen's roomgen.cpp can be
// compiled on its own: RoomGenerator only ever touches the grid, the size and the
// four accessors below, and these mirror basic-abstract-game.cpp exactly.

#include <queue>
#include <set>
#include <vector>

#include "grid.h"
#include "object-ids.h"

class BasicAbstractGame {
  public:
    Grid<int> grid;
    int grid_size = 0;
    int out_of_bounds_object = INVALID_OBJ;

    int to_grid_idx(int x, int y) {
        if (!grid.contains(x, y)) {
            return INVALID_IDX;
        }
        return grid.to_index(x, y);
    }

    void to_grid_xy(int idx, int *x, int *y) { grid.to_xy(idx, x, y); }

    int get_obj(int x, int y) {
        if (!grid.contains(x, y)) {
            return out_of_bounds_object;
        }
        return grid.get(x, y);
    }

    int get_obj(int idx) {
        if (!grid.contains_index(idx)) {
            return out_of_bounds_object;
        }
        return grid.get_index(idx);
    }

    void set_obj(int idx, int elem) { grid.set_index(idx, elem); }
    void set_obj(int x, int y, int elem) { grid.set(x, y, elem); }
};
