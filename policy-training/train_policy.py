"""Train a policy from recorded episodes, then run it to score it.

The dataset is whatever `world-model-sim` recorded: played sessions, scripted
demonstrations, or both. Training is plain behavior cloning — predict the action
that was taken at each observed state. The last step is what makes the result
trustworthy: the trained policy drives the environment itself and is scored
against the baseline from the same starting states.
"""

import torch
from grid_world import COINS, CoinSeeker, GridWorldEnv
from mixtrain import Dataset, File, Markdown, MixFlow, Policy, Sandbox, Trajectory
from torch import nn

FEATURES = 5  # observation.state is five numbers wide


class LearnedPolicy(Policy):
    """The trained network as an action source for a rollout."""

    def __init__(self, net: nn.Module, actions: list[str]):
        super().__init__()
        self._net = net.eval()
        self._actions = actions

    def __call__(self, obs: dict) -> str:
        features = torch.from_numpy(obs["state"].to_numpy()).unsqueeze(0)
        with torch.no_grad():
            return self._actions[int(self._net(features).argmax(-1))]


class TrainPolicy(MixFlow):
    """Behavior-clone the recorded episodes and evaluate the result."""

    _sandbox = Sandbox(cpu=4, memory=8192, timeout=3600)

    def run(
        self,
        dataset: str = "grid-demos",
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        episodes: int = 20,
        seed: int = 1000,
    ) -> dict:
        """Train on `dataset`, then score the policy against the baseline.

        Args:
            dataset: Recorded episodes to learn from
            epochs: Passes over the recorded steps
            batch_size: Steps per gradient update
            learning_rate: Adam learning rate
            episodes: Evaluation episodes per policy
            seed: Evaluation episode i starts from `seed + i` for both
                policies, so they are scored on the same starting states
        """
        demos = Trajectory(dataset)
        actions = demos.action_keys
        if not actions:
            raise ValueError(
                f"{dataset!r} recorded no discrete action vocabulary; it was not "
                "produced by this environment"
            )

        states, labels = self._load(demos, batch_size)
        print(f"training on {len(states)} steps from {demos.num_episodes} episodes")

        net = nn.Sequential(
            nn.Linear(FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(actions)),
        )
        accuracy = self._train(net, states, labels, epochs, batch_size, learning_rate)

        torch.save({"state_dict": net.state_dict(), "actions": actions}, "policy.pt")
        weights = File.from_file("policy.pt").save()

        # Let each policy drive the environment from the same starting states.
        # `seed` is recorded per episode, so a later run can revisit exactly the
        # episodes that went badly.
        env = GridWorldEnv()
        learned = env.rollout(LearnedPolicy(net, actions), episodes=episodes, seed=seed)
        baseline = env.rollout(CoinSeeker(), episodes=episodes, seed=seed)

        rollouts = f"{dataset}-eval"
        if Dataset.exists(rollouts):
            Dataset(rollouts).append(learned)
        else:
            learned.save(rollouts)

        learned_score = _score(learned)
        baseline_score = _score(baseline)
        return {
            "weights": weights,
            "rollouts": Trajectory(rollouts),
            "train_accuracy": accuracy,
            **{f"learned_{k}": v for k, v in learned_score.items()},
            **{f"baseline_{k}": v for k, v in baseline_score.items()},
            "report": _report(learned_score, baseline_score, accuracy, episodes),
        }

    def _load(self, demos: Trajectory, batch_size: int):
        """Every recorded step as two tensors: the state, and the action taken.

        `to_torch` streams batches. This corpus is small enough to hold in
        memory, so later epochs never touch the network. `action_encoding` turns
        each recorded action into its index in the vocabulary, and the last row
        of each episode is dropped because it has no action to predict.
        """
        loader = demos.steps().to_torch(batch_size=batch_size, load_files=False)
        batches = list(loader)
        states = torch.cat([batch["observation.state"] for batch in batches])
        labels = torch.cat([batch["action"] for batch in batches]).long()
        return states, labels

    def _train(self, net, states, labels, epochs, batch_size, learning_rate) -> float:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        accuracy = 0.0
        for epoch in range(1, epochs + 1):
            order = torch.randperm(len(states))
            total, correct = 0.0, 0
            for start in range(0, len(states), batch_size):
                index = order[start : start + batch_size]
                logits = net(states[index])
                loss = loss_fn(logits, labels[index])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(index)
                correct += int((logits.argmax(-1) == labels[index]).sum())
            accuracy = correct / len(states)
            if epoch % 10 == 0 or epoch == epochs:
                print(
                    f"epoch {epoch:3d}  loss {total / len(states):.4f}  "
                    f"accuracy {accuracy:.3f}"
                )
        return accuracy


def _score(traj: Trajectory) -> dict:
    """Mean return, mean length, and how often every coin was collected."""
    episodes = traj.collect()
    returns = episodes.column("episode_return").to_pylist()
    lengths = episodes.column("length").to_pylist()
    return {
        "mean_return": sum(returns) / len(returns),
        "mean_length": sum(lengths) / len(lengths),
        "solved": sum(1 for value in returns if value >= COINS) / len(returns),
    }


def _report(learned: dict, baseline: dict, accuracy: float, episodes: int) -> Markdown:
    return Markdown(
        content="\n".join(
            [
                "## Rollout comparison",
                "",
                f"{episodes} episodes each, from the same starting states.",
                "",
                "| Policy | Mean return | Mean length | Solved |",
                "| --- | --- | --- | --- |",
                f"| Learned | {learned['mean_return']:.2f} | "
                f"{learned['mean_length']:.1f} | {learned['solved']:.0%} |",
                f"| Baseline | {baseline['mean_return']:.2f} | "
                f"{baseline['mean_length']:.1f} | {baseline['solved']:.0%} |",
                "",
                f"Training accuracy on the recorded actions: {accuracy:.1%}",
            ]
        )
    )
