"""Play in a browser window, served from this machine.

    pixi run webplay            # then open http://127.0.0.1:8010

A real window instead of the terminal: the page keeps a 64x64 offscreen canvas and
scales it up with nearest-neighbour, so what you see is the actual observation the
agent receives.  Frames travel as raw RGB bytes (12 KiB per frame) over localhost,
which needs no image encoder and no extra dependency.

The server owns one episode.  Keys: a/d or arrows move, w/up/space jump, s stands
still, r restarts, n/m step the level seed, p pauses.  Restarting with a seed is
also available as ``/?seed=42``.
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jax
import jax.numpy as jnp
import numpy as np

from procgen_jax import (
    BigFishEnv,
    CoinRunEnv,
    DodgeballEnv,
    FruitBotEnv,
    HeistEnv,
    JumperEnv,
    LeaperEnv,
    MazeEnv,
    PlunderEnv,
    StarPilotEnv,
)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>procgen-jax</title>
<style>
  body { margin:0; background:#111; color:#ddd; font:14px monospace;
         display:flex; flex-direction:column; align-items:center; gap:8px; padding:12px; }
  canvas { image-rendering: pixelated; background:#000; }
  #hud { white-space:pre; }
  #help { color:#777; }
</style></head>
<body>
<canvas id="c" width="512" height="512"></canvas>
<div id="hud">loading...</div>
<div id="help">a/d or arrows move &nbsp; w/up/space jump &nbsp; s stand &nbsp;
  p pause &nbsp; r restart &nbsp; n/m seed</div>
<script>
const obs = document.createElement('canvas');
obs.width = obs.height = 64;
const octx = obs.getContext('2d');
const view = document.getElementById('c').getContext('2d');
view.imageSmoothingEnabled = false;
const hud = document.getElementById('hud');

let horizontal = 0, jump = false, paused = false, inFlight = false, seed = null;
// ProcGen: action = (vx + 1) * 3 + (vy + 1); slot 5 is the jump button.
const encode = () => (horizontal + 1) * 3 + (jump ? 2 : 1);

function draw(bytes, info) {
  const data = new Uint8ClampedArray(bytes);
  const img = new ImageData(64, 64);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    img.data[i] = data[j]; img.data[i + 1] = data[j + 1]; img.data[i + 2] = data[j + 2]; img.data[i + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  view.imageSmoothingEnabled = false;
  view.drawImage(obs, 0, 0, 64, 64, 0, 0, 512, 512);
  hud.textContent = `seed=${info.seed}  steps=${info.steps}  return=${info.reward.toFixed(1)}  ` +
                    `pos=(${info.x.toFixed(2)},${info.y.toFixed(2)})` +
                    (info.done ? '  [EPISODE ENDED]' : '') + (paused ? '  [paused]' : '');
}

async function tick() {
  if (!paused && !inFlight) {
    inFlight = true;
    try {
      const r = await fetch(`/frame?a=${encode()}`);
      const info = JSON.parse(r.headers.get('X-Info'));
      draw(await r.arrayBuffer(), info);
      seed = info.seed;
    } catch (e) { hud.textContent = 'server gone: ' + e; }
    inFlight = false;
  }
  setTimeout(tick, 16);
}
tick();

addEventListener('keydown', (e) => {
  const k = e.key;
  if (k === 'a' || k === 'ArrowLeft') horizontal = -1;
  else if (k === 'd' || k === 'ArrowRight') horizontal = 1;
  else if (k === 's') horizontal = 0;
  else if (k === 'w' || k === 'ArrowUp' || k === ' ') { jump = true; e.preventDefault(); }
  else if (k === 'p') paused = !paused;
  else if (k === 'r') fetch('/reset');
  else if (k === 'n') fetch(`/reset?seed=${(seed ?? 0) + 1}`);
  else if (k === 'm') fetch(`/reset?seed=${(seed ?? 1) - 1}`);
});
</script></body></html>
"""


class Session:
    """One episode, stepped on demand."""

    def __init__(self, game: str, seed):
        self.lock = threading.Lock()
        self.game = game
        self.seed = seed
        self.build()

    def build(self):
        if self.game == 'maze':
            env = MazeEnv(world_dim=15)
        elif self.game == 'bigfish':
            env = BigFishEnv(level_seed=self.seed)
        elif self.game == 'leaper':
            env = LeaperEnv(level_seed=self.seed)
        elif self.game == 'plunder':
            env = PlunderEnv(level_seed=self.seed)
        elif self.game == 'heist':
            env = HeistEnv(level_seed=self.seed)
        elif self.game == 'starpilot':
            env = StarPilotEnv(level_seed=self.seed)
        elif self.game == 'dodgeball':
            env = DodgeballEnv(level_seed=self.seed)
        elif self.game == 'fruitbot':
            env = FruitBotEnv(level_seed=self.seed)
        elif self.game == 'jumper':
            env = JumperEnv(level_seed=self.seed)
        else:
            env = CoinRunEnv(level_seed=self.seed)
        self.step = jax.jit(env.step)
        self.reset = jax.jit(env.reset)
        self.state = self.reset(jax.random.PRNGKey(0))

    def restart(self, seed=None):
        self.seed = seed
        self.build()

    def advance(self, action: int):
        with self.lock:
            self.state = self.step(self.state, jnp.asarray(action))
            agent = self.state.env_state.agent
            if hasattr(agent, 'x'):
                x, y = float(agent.x), float(agent.y)
            else:
                x, y = float(agent[0]), float(agent[1])
            return np.asarray(self.state.obs), {
                'seed': self.seed,
                'steps': int(self.state.env_state.steps),
                'reward': float(self.state.reward),
                'x': x,
                'y': y,
                'done': bool(self.state.done),
            }


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def do_GET(self):
            path, _, query = self.path.partition('?')
            params = dict(part.partition('=')[::2] for part in query.split('&') if part)
            if path == '/':
                body = PAGE.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == '/reset':
                seed = params.get('seed')
                session.restart(int(seed) if seed not in (None, '') else None)
                self.send_response(204)
                self.end_headers()
                return
            if path == '/frame':
                frame, info = session.advance(int(params.get('a', 4)))
                body = frame.astype(np.uint8).tobytes()
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('X-Info', json.dumps(info))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--game',
        default='coinrun',
        choices=[
            'coinrun',
            'maze',
            'bigfish',
            'leaper',
            'plunder',
            'heist',
            'starpilot',
            'dodgeball',
            'fruitbot',
            'jumper',
        ],
    )
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()

    session = Session(args.game, args.seed)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    print(f'{args.game} ready on http://{args.host}:{args.port}/  (ctrl-c to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
