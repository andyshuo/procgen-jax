"""Build the sprite atlas from ProcGen's bundled Kenney assets.

    pixi run atlas        # -> procgen_jax/data/sprites.npz

The assets live in ``procgen/data/assets/kenney`` (CC0, see ASSET_LICENSES.md in
that repository).  PNG decoding would normally need Pillow, which this project
deliberately does not depend on, so each sprite is piped through ``ffmpeg`` into
raw RGBA and downscaled to a uniform ``SIZE x SIZE`` tile.  The source sprites are
128x128 (tiles, items, enemies) or 128x256 (the player), so squashing them all to
one size loses nothing: the renderer draws the player into a 1 x 1.74 tile rect,
which restores its tall aspect.

The atlas is a single ``(N, SIZE, SIZE, 4)`` uint8 array plus the sprite names,
which is all JAX needs to sample it with a gather.
"""

import os
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

# The pixi environment ships no CA bundle of its own, so point urllib at the
# system one when the caller has not.
if 'SSL_CERT_FILE' not in os.environ and Path('/etc/ssl/certs/ca-certificates.crt').exists():
    os.environ['SSL_CERT_FILE'] = '/etc/ssl/certs/ca-certificates.crt'

SIZE = 32
ASSETS = 'https://cdn.jsdelivr.net/gh/openai/procgen@master/procgen/data/assets/'
ROOT = ASSETS + 'kenney/'

PLAYER_COLOURS = ['Beige', 'Blue', 'Green', 'Pink', 'Yellow']
PLAYER_POSES = ['stand', 'jump', 'walk1', 'walk2']
GROUND_THEMES = ['Dirt', 'Grass', 'Planet', 'Sand', 'Snow', 'Stone']
ENEMIES = ['slimeBlock', 'slimePurple', 'slimeBlue', 'slimeGreen', 'mouse', 'snail', 'ladybug', 'wormGreen', 'wormPink']


def sprite_paths() -> dict[str, str]:
    """Sprite name -> path inside the Kenney asset directory, mirroring coinrun.cpp."""
    paths = {}
    for colour in PLAYER_COLOURS:
        for pose in PLAYER_POSES:
            paths[f'Players/{colour}/{pose}'] = f'Players/128x256/{colour}/alien{colour}_{pose}.png'
    for ground in GROUND_THEMES:
        paths[f'Ground/{ground}/mid'] = f'Ground/{ground}/{ground.lower()}Mid.png'
        paths[f'Ground/{ground}/center'] = f'Ground/{ground}/{ground.lower()}Center.png'
    for enemy in ENEMIES:
        paths[f'Enemies/{enemy}/idle'] = f'Enemies/{enemy}.png'
        paths[f'Enemies/{enemy}/move'] = f'Enemies/{enemy}_move.png'
    paths['Items/coin'] = 'Items/coinGold.png'
    paths['Tiles/saw/idle'] = 'Enemies/sawHalf.png'
    paths['Tiles/saw/move'] = 'Enemies/sawHalf_move.png'
    for crate in ('boxCrate', 'boxCrate_double', 'boxCrate_single', 'boxCrate_warning'):
        paths[f'Tiles/{crate}'] = f'Tiles/{crate}.png'
    paths['Tiles/lava'] = 'Tiles/lava.png'
    paths['Tiles/lava_top'] = 'Tiles/lavaTop_low.png'
    # bigfish: misc_assets live beside ``kenney/``, not inside it, so these carry
    # their own absolute URL (bigfish.cpp's asset_for_type).
    for tile in ('072', '074', '078', '080'):
        paths[f'misc/fish{tile}'] = f'{ASSETS}misc_assets/fishTile_{tile}.png'
    # leaper: the frogger board and its traffic (leaper.cpp's asset_for_type).
    for name in ('roadTile6b', 'terrainTile6', 'elementWood044', 'finish2'):
        paths[f'misc/leaper/{name}'] = f'{ASSETS}misc_assets/{name}.png'
    for name in ('car_yellow_5', 'car_black_1', 'car_blue_2', 'car_green_3', 'car_red_4'):
        paths[f'misc/leaper/{name}'] = f'{ASSETS}misc_assets/{name}.png'
    for name in ('frog1', 'frog2', 'frog4', 'frog6', 'frog7'):
        paths[f'misc/leaper/{name}'] = f'{ASSETS}misc_assets/{name}.png'
    # plunder: ships, cannonballs, crates and the target legend.
    for name in ('ship_1', 'ship_2', 'ship_3', 'ship_4', 'ship_5', 'ship_6'):
        paths[f'misc/plunder/{name}'] = f'{ASSETS}misc_assets/{name}.png'
    for name in ('cannonBall', 'panel_wood', 'target_red2'):
        paths[f'misc/plunder/{name}'] = f'{ASSETS}misc_assets/{name}.png'
    return paths


def image_size(path: Path) -> tuple[int, int]:
    """Source dimensions, which the games need for ``match_aspect_ratio``."""
    out = subprocess.run(
        [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=width,height',
            '-of',
            'csv=p=0',
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    width, height = out.split(',')[:2]
    return int(width), int(height)


def decode(path: Path, size: int = SIZE) -> np.ndarray:
    """PNG -> ``(size, size, 4)`` uint8 RGBA, via ffmpeg (no Pillow)."""
    out = subprocess.run(
        [
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            str(path),
            '-vf',
            f'scale={size}:{size}:flags=neighbor',
            '-f',
            'rawvideo',
            '-pix_fmt',
            'rgba',
            '-',
        ],
        check=True,
        capture_output=True,
    ).stdout
    if len(out) != size * size * 4:
        raise RuntimeError(f'{path}: expected {size * size * 4} bytes, got {len(out)}')
    return np.frombuffer(out, np.uint8).reshape(size, size, 4).copy()


def main() -> None:
    paths = sprite_paths()
    cache = Path('.atlas_cache')
    cache.mkdir(exist_ok=True)

    names = ['_transparent', *paths]
    atlas = np.zeros((len(names), SIZE, SIZE, 4), np.uint8)
    aspects = np.ones(len(names), np.float32)

    for row, name in enumerate(paths, start=1):
        png = cache / (name.replace('/', '_') + '.png')
        if not png.exists():
            url = paths[name]
            png.write_bytes(urllib.request.urlopen(url if url.startswith('http') else ROOT + url, timeout=60).read())
        atlas[row] = decode(png)
        width, height = image_size(png)
        aspects[row] = width / height

    out = Path('procgen_jax/data')
    out.mkdir(exist_ok=True)
    target = out / 'sprites.npz'
    np.savez_compressed(target, pixels=atlas, names=np.asarray(names), aspects=aspects)

    print(f'{len(names)} sprites of {SIZE}x{SIZE} -> {target} ({target.stat().st_size // 1024} KiB)')
    for row, name in enumerate(names):
        if row < 3 or row > len(names) - 3:
            print(f'  {row:2d} {name}')


if __name__ == '__main__':
    main()
