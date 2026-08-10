# Policy Training <a href="https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fpolicy-training&amp;type=workflow"><img src="https://mixtrain.ai/assets/run-with-mixtrain.svg" alt="Run with MixTrain" height="40" align="right"></a>

Take the episodes recorded by [`world-model-sim`](../world-model-sim/), train a
policy on them, and check whether it works by letting it drive the environment
itself.

Training is plain behavior cloning: predict the action that was taken at each
observed state. The last step is what makes the result trustworthy — the trained
policy and the scripted baseline are rolled out from the *same* starting states.

## What you'll learn

- How to turn a [trajectory](https://mixtrain.ai/docs/guide/trajectories) into a
  PyTorch `DataLoader` with `to_torch()`
- What `action_encoding` and `drop_actionless` do to recorded episodes
- How to wrap a trained network as a
  [`Policy`](https://mixtrain.ai/docs/guide/policies)
- How to evaluate two policies on identical seeds and save the rollouts

## How it works

```
grid-demos ──to_torch()──▶ behavior cloning ──▶ policy.pt
                                     │
                                     └──rollout(seed=1000)──▶ grid-demos-eval
                                                              vs. the baseline
```

`to_torch()` handles the parts specific to episode data:

- `steps()` gives one row per transition, the shape behavior cloning needs.
- `action_encoding="index"` (the default) turns `"ArrowUp"` into its index in
  the recorded action vocabulary, which `Trajectory.action_keys` reports.
- `drop_actionless=True` (the default) drops the last row of each episode, which
  has no action to predict.

The corpus is small, so it is loaded once and held in memory. Every epoch after
that runs without touching the network.

## Prerequisites

- [mixtrain CLI installed and logged in](https://mixtrain.ai/docs/guide/quickstart)
- A `grid-demos` dataset — run [`world-model-sim`](../world-model-sim/) first

## Run it

```bash
mixtrain workflow create . --name train-grid-policy --entrypoint train_policy.py:TrainPolicy
mixtrain workflow run train-grid-policy --dataset grid-demos --episodes 20
```

The run outputs the trained weights, the evaluation rollouts as a dataset, and a
comparison table:

| Policy | Mean return | Mean length | Solved |
| --- | --- | --- | --- |
| Learned | 5.00 | 24.2 | 100% |
| Baseline | 5.00 | 24.2 | 100% |

Open `grid-demos-eval` to replay what the trained policy did.

## Where to take it

- **Find the hard cases.** Filter the eval rollouts by `episode_return` and pass
  those seeds back in: `env.rollout(policy, seed=failed_seeds)` re-runs exactly
  the starting states that went badly.
- **Swap in a real policy.** The loop is the same for a robotics checkpoint.
  `Policy.from_checkpoint(checkpoint)` loads a LeRobot policy into the process,
  and the rollout, the recording, and the comparison stay as they are.

## Learn more

- [Trajectories guide](https://mixtrain.ai/docs/guide/trajectories)
- [Policies guide](https://mixtrain.ai/docs/guide/policies)
- [Environments guide](https://mixtrain.ai/docs/guide/environments)
