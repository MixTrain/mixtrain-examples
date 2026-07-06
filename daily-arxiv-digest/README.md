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

You can choose multiple categories from the MixTrain UI or pass a comma-separated
list from the CLI:

```bash
mixtrain routine run daily-arxiv-digest \
  --categories 'cs.CV,cs.CL' \
  --keywords 'vision transformer' \
  --author 'Alexey Dosovitskiy' \
  --paper-count 3
```

The routine combines categories with `OR` and the other filters with `AND`. It
looks back 24 hours, or 72 hours on Mondays to cover the weekend.

## Learn more

- [Routine Guide](https://mixtrain.ai/docs/guide/routines)
