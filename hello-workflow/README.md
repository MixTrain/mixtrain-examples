# Hello Workflow

Run a GPU-backed workflow and verify CUDA availability.

## What you'll learn

- How to define a `MixFlow` workflow
- How to use a GPU in a workflow with `Sandbox`
- How to run a workflow with the CLI

## Prerequisites

- [mixtrain CLI installed and logged in](https://mixtrain.ai/docs/guide/quickstart)

## Run it

```bash
mixtrain workflow create hello_workflow.py --name hello-workflow
mixtrain workflow run hello-workflow
```

Logs are streamed to the CLI and also available on the UI. The run URL will be printed on the CLI:

```
https://app.mixtrain.ai/<workspace>/workflows/hello-workflow/runs/1
```

## Learn more

- [Workflows guide](https://mixtrain.ai/docs/guide/workflows)
