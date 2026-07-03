# Parameter Sweep <a href="https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fparameter-sweep&amp;type=workflow"><img src="https://mixtrain.ai/assets/run-with-mixtrain.svg" alt="Run with MixTrain" height="40" align="right"></a>

Launch multiple runs of another workflow by varying one numeric input.

## What you'll learn

- How to use a `Workflow` input to reference another workflow
- How to submit workflow runs from inside a workflow
- How to return workflow run links as typed outputs

## Prerequisites

- [mixtrain CLI installed and logged in](https://mixtrain.ai/docs/guide/quickstart)
- A target workflow that has at least one numeric input, such as `learning_rate`, `epochs`, or `temperature`

## Run it

Create the sweep workflow:

```bash
mixtrain workflow create parameter_sweep.py --name parameter-sweep
```

Run a sweep over a target workflow:

```bash
mixtrain workflow run parameter-sweep \
  --input '{
    "target_workflow": "train-model",
    "sweep_param": "learning_rate",
    "start": 0.0001,
    "stop": 0.001,
    "step": 0.0001,
    "base_inputs": {
      "epochs": 3
    },
    "notes": "Trying a learning-rate sweep before a longer training run."
  }'
```

The sweep values are inclusive of `stop`. For example, `start=1`, `stop=5`, and `step=2` submits runs with values `1`, `3`, and `5`.

`base_inputs` is passed to every target workflow run. The sweep workflow adds or overrides `sweep_param` for each generated value.

## Output

The workflow returns:

- `notes`: a Markdown summary of the sweep, including any notes you provided
- `runs`: links to the workflow runs it submitted
- `sweep`: the generated values and run numbers
- `count`: the number of submitted runs

The run links are visible in the Mixtrain UI because each item is returned as a typed `Workflow` output with a `run_number`.

## Learn more

- [Workflows guide](https://mixtrain.ai/docs/guide/workflows)
- [Input/Output types](https://mixtrain.ai/docs/guide/types)
