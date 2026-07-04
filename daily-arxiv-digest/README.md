# Daily arXiv Digest

Create a digest of recent arXiv papers every weekday morning, or run
it manually whenever you want an update.

## What you'll learn

- How to schedule a `MixRoutine` with a cron expression
- How scheduled and manual runs use the same routine
- How to pass routine inputs and return a rich Markdown results

## Prerequisites

- [mixtrain CLI installed and logged in](https://mixtrain.ai/docs/guide/quickstart)

## Create the routine

```bash
mixtrain routine create daily_arxiv_digest.py --name daily-arxiv-digest
```

By default, the routine runs at 09:00 UTC every weekday and retrieves the five
newest Machine Learning papers.

## Get a digest now

Use a manual run to generate the default digest immediately:

```bash
mixtrain routine run daily-arxiv-digest
```

You can choose another category from the dropdown when running from the MixTrain UI,
or override it from the CLI:

```bash
mixtrain routine run daily-arxiv-digest \
  --category 'Computer Vision and Pattern Recognition' \
  --keywords 'vision transformer' \
  --author 'Alexey Dosovitskiy' \
  --paper-count 3
```

The routine looks back 24 hours and combines the inputs to find the latest papers's abstracts.

## Learn more

- [Routine Guide](https://mixtrain.ai/docs/guide/routines)
