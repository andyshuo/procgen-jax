"""Tile ids.

Values follow OpenAI ProcGen's ``object-ids.h`` so that level dumps stay readable,
except that ``SPACE`` is 0 here (ProcGen uses 100) to make "empty grid" cheap.
"""

SPACE = 0
WALL = 1
GOAL = 2
PLAYER = 3

# ProcGen compatibility notes:
#   WALL_OBJ = 51, EXIT_OBJ = 52, AGENT_OBJ = 53, SPACE = 100,
#   DOOR_OBJ = 200, KEY_OBJ = 300
PROCGEN_WALL = 51
PROCGEN_GOAL = 52
PROCGEN_AGENT = 53
PROCGEN_SPACE = 100
PROCGEN_DOOR = 200
PROCGEN_KEY = 300

NUM_OBJECTS = 4
