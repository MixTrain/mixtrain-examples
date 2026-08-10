"""A tiny world model: a grid, an agent, and coins to collect.

`GridWorldEnv` is the whole research surface — `reset()` and `step()`. The same
code runs with a person or a policy deciding the actions; only the source of the
actions changes.

`PlayableGridWorld` is the view a browser session drives, and `GridWorld` is the
deployed model that serves one.
"""

from collections import deque
from io import BytesIO
from typing import Literal

import numpy as np
from mixtrain import (
    Dataset,
    Environment,
    Image,
    MixModel,
    Policy,
    Sandbox,
    Tensor,
    Trajectory,
)

GRID = 10  # cells per side
CELL = 28  # pixels per cell -> 280x280 frames
COINS = 5  # coins to collect per episode
IDLE_TIMEOUT = 120  # seconds to wait for a browser before releasing the container

MOVES = {
    "ArrowUp": (0, -1),
    "ArrowDown": (0, 1),
    "ArrowLeft": (-1, 0),
    "ArrowRight": (1, 0),
}


class World:
    """The state a session keeps between steps."""

    def __init__(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.x = self.y = GRID // 2
        self.coins: set[tuple[int, int]] = set()
        while len(self.coins) < COINS:
            cell = (int(rng.integers(GRID)), int(rng.integers(GRID)))
            if cell != (self.x, self.y):
                self.coins.add(cell)

    def move(self, action: str) -> float:
        """Take one step; return the reward it earned."""
        dx, dy = MOVES[action]
        self.x = min(GRID - 1, max(0, self.x + dx))
        self.y = min(GRID - 1, max(0, self.y + dy))
        if (self.x, self.y) in self.coins:
            self.coins.discard((self.x, self.y))
            return 1.0
        return 0.0

    @property
    def done(self) -> bool:
        return not self.coins

    def features(self) -> np.ndarray:
        """The state as numbers: where the agent is, and where to go next."""
        if self.coins:
            nearest = min(
                self.coins, key=lambda c: abs(c[0] - self.x) + abs(c[1] - self.y)
            )
            dx, dy = nearest[0] - self.x, nearest[1] - self.y
        else:
            dx = dy = 0
        return np.array(
            [
                self.x / (GRID - 1),
                self.y / (GRID - 1),
                dx / GRID,
                dy / GRID,
                len(self.coins) / COINS,
            ],
            dtype="float32",
        )


def render(world: World) -> bytes:
    """Draw the world as a PNG frame."""
    from PIL import Image as PILImage, ImageDraw

    canvas = PILImage.new("RGB", (GRID * CELL, GRID * CELL), (17, 17, 20))
    draw = ImageDraw.Draw(canvas)
    for i in range(GRID + 1):
        draw.line([(i * CELL, 0), (i * CELL, GRID * CELL)], fill=(30, 30, 36))
        draw.line([(0, i * CELL), (GRID * CELL, i * CELL)], fill=(30, 30, 36))
    for cx, cy in world.coins:
        draw.ellipse(
            [cx * CELL + 6, cy * CELL + 6, (cx + 1) * CELL - 6, (cy + 1) * CELL - 6],
            fill=(240, 200, 60),
        )
    ax, ay = world.x * CELL, world.y * CELL
    draw.rectangle([ax + 4, ay + 4, ax + CELL - 4, ay + CELL - 4], fill=(80, 200, 255))

    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


class GridWorldEnv(Environment):
    """Collect every coin. One cell per step, four discrete moves.

    The action space is declared where the action is consumed — `step`'s
    `Literal` annotation — so there is one place to change it, and a browser
    builds its controls from it.

    An observation carries both what a policy reads (`state`) and what a person
    watches (`images.grid`), which is what makes a recording both trainable and
    replayable.
    """

    #: Nothing else ends an episode that never finds the last coin.
    max_episode_steps = 120

    def reset(self, seed: int | None = None) -> dict:
        self._world = World(seed)
        return self._observation()

    def step(
        self, action: Literal["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]
    ) -> tuple[dict, float, bool, bool, dict]:
        reward = self._world.move(action)
        info = {"coins_left": len(self._world.coins)}
        return self._observation(), reward, self._world.done, False, info

    def _observation(self) -> dict:
        return {
            "state": Tensor.from_numpy(self._world.features()),
            "images": {
                "grid": Image.from_bytes(render(self._world), content_type="image/png")
            },
        }


class PlayableGridWorld(GridWorldEnv):
    """The same world, observed as a bare frame — what a browser renders.

    The interactive panel draws a top-level image observation, so a live session
    observes the frame alone. The state is still recorded, so a played session
    and a headless rollout produce the same columns and land in one dataset.
    """

    def __init__(self):
        self._observed: deque[Tensor] = deque()

    def reset(self, seed: int | None = None) -> Image:
        return self._frame(super().reset(seed))

    def step(
        self, action: Literal["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]
    ) -> tuple[Image, float, bool, bool, dict]:
        # The annotation is re-declared because it *is* the action space: a live
        # session builds its keyboard controls from the method it will call.
        obs, reward, terminated, truncated, info = super().step(action)
        return self._frame(obs), reward, terminated, truncated, info

    def record_step(self, obs, action, reward, terminated, truncated, info) -> dict:
        """Record what a headless rollout records, from a frame-only session.

        A row describes the observation its action was taken *at*, and by the
        time the row is written the next frame already exists — so the state is
        queued as each frame is produced rather than read off the world here.
        """
        row = super().record_step(obs, action, reward, terminated, truncated, info)
        frame = row.pop("observation")
        return {
            "observation.state": self._observed.popleft(),
            "observation.images.grid": frame,
            **row,
        }

    def _frame(self, obs: dict) -> Image:
        """Show the frame; keep the state that goes with it for the recording."""
        self._observed.append(obs["state"])
        return obs["images"]["grid"]


class CoinSeeker(Policy):
    """The baseline: always step toward the nearest coin."""

    def __call__(self, obs: dict) -> str:
        _, _, dx, dy, _ = obs["state"].to_numpy()
        if abs(dx) >= abs(dy):
            return "ArrowRight" if dx > 0 else "ArrowLeft"
        return "ArrowDown" if dy > 0 else "ArrowUp"


class GridWorld(MixModel):
    """Serve one playable session and keep what was played."""

    _sandbox = Sandbox(cpu=1, timeout=IDLE_TIMEOUT + 600)

    def run(self, dataset: str = "grid-demos") -> Trajectory | None:
        """Open an interactive session; record it into `dataset` on disconnect.

        A fresh environment per run keeps one session's world from leaking into
        the next on a warm container.
        """
        traj = PlayableGridWorld().rollout(record=True, idle_timeout=IDLE_TIMEOUT)
        if traj is None:
            print("session ended without recording any steps")
            return None

        print(f"recorded {traj.num_episodes} episode(s)")
        if Dataset.exists(dataset):
            Dataset(dataset).append(traj)
        else:
            traj.save(dataset)
        return Trajectory(dataset, episode=traj.episode_ids)
