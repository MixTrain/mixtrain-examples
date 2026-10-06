"""Collect demonstrations without a browser: a scripted policy, run unattended.

This is the same `rollout()` loop that serves a live session, with a policy
deciding the actions instead of a person. The rollout now sets the bounds: how
many episodes to run, how long each may run, and the seed each one starts from.
"""

from grid_world import CoinSeeker, GridWorldEnv
from mixtrain import Dataset, MixFlow, Sandbox, Trajectory


class CollectDemos(MixFlow):
    """Roll the baseline policy out and save the episodes as a dataset."""

    _sandbox = Sandbox(cpu=2, timeout=1800)

    def run(
        self,
        episodes: int = 20,
        seed: int = 0,
        dataset: str = "grid-demos",
    ) -> dict:
        """Record `episodes` episodes of the baseline policy.

        Args:
            episodes: How many episodes to run
            seed: Episode i is reset with `seed + i`. The seed is recorded, so
                any episode can be re-run exactly.
            dataset: Dataset to save into, or append to when it already exists
        """
        traj = GridWorldEnv().rollout(CoinSeeker(), episodes=episodes, seed=seed)

        episode_rows = traj.collect()
        returns = episode_rows.column("episode_return").to_pylist()
        lengths = episode_rows.column("length").to_pylist()

        if Dataset.exists(dataset):
            Dataset(dataset).append(traj)
        else:
            traj.save(dataset)

        return {
            "dataset": Trajectory(dataset),
            "episodes": traj.num_episodes,
            "mean_return": sum(returns) / len(returns),
            "mean_length": sum(lengths) / len(lengths),
        }
