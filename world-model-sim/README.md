# World Model Sim <a href="https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fworld-model-sim&amp;type=model"><img src="https://mixtrain.ai/assets/run-with-mixtrain.svg" alt="Run with MixTrain" height="40" align="right"></a>

A small world you can play in the browser, and a policy you can run through it
unattended. Both record the same thing: episodes of
`(observation, action, reward, terminated, truncated, info)`, saved as a
[trajectory](https://mixtrain.ai/docs/guide/trajectories) you can replay, query,
and train on.

The world is a 10×10 grid with five coins. Collect them all and the episode
terminates; run out of steps and it is truncated.

## What you'll learn

- How to write an [`Environment`](https://mixtrain.ai/docs/guide/environments) —
  `reset()` and `step()` — and declare its action space
- How to serve one as a live interactive session on a warm container
- How to run the same environment unattended with a
  [`Policy`](https://mixtrain.ai/docs/guide/policies)
- How a rollout records episodes, and how to save or grow a dataset from them

## How it works

```
GridWorldEnv ──rollout(record=True)────▶ browser session ──┐
             └─rollout(CoinSeeker(), …)─▶ scripted demos ──┴──▶ grid-demos
```

Both paths record the same columns, so played sessions and scripted
demonstrations land in one dataset:

| Column | What it is |
| --- | --- |
| `observation.state` | Five numbers: where the agent is, where the nearest coin is, how many are left |
| `observation.images.grid` | The frame, which is what replays in the app |
| `action` | One of `ArrowUp`, `ArrowDown`, `ArrowLeft`, `ArrowRight` |
| `reward`, `terminated`, `truncated`, `info` | The rest of the step |
| `seed`, `length`, `episode_return` | Recorded once per episode |

`GridWorldEnv` observes both the state and the frame. `PlayableGridWorld`
observes the frame alone, because the interactive panel only draws a top-level
image. Its `record_step()` adds the state back into the recording, so a session
you played is training data too.

## Prerequisites

- [mixtrain CLI installed and logged in](https://mixtrain.ai/docs/guide/quickstart)

## Run it

**Play it.** Create the model and start a run:

```bash
mixtrain model create . --name grid-world --entrypoint grid_world.py:GridWorld
mixtrain model run grid-world
```

Open the run page. The *Interactive Session* panel connects on its own — click
the frame to take the keyboard, and drive with the arrow keys. Press `Esc` to
release. When you disconnect, the session is recorded into the `grid-demos`
dataset and the run page replays it.

**Collect demonstrations instead.** The same environment, driven by a scripted
policy, with no browser involved:

```bash
mixtrain workflow create . --name collect-grid-demos --entrypoint collect_demos.py:CollectDemos
mixtrain workflow run collect-grid-demos --episodes 20 --seed 0
```

Every episode records the seed it started from, so any of them can be re-run
exactly, by this policy or a different one.

## What to do next

Train on what you recorded. [`policy-training`](../policy-training/) reads
`grid-demos`, clones the behavior in it, and scores the result by running the
trained policy through the same environment.

## Learn more

- [Environments guide](https://mixtrain.ai/docs/guide/environments)
- [Policies guide](https://mixtrain.ai/docs/guide/policies)
- [Trajectories guide](https://mixtrain.ai/docs/guide/trajectories)
