# Daily arXiv Digest

Create a digest of recent arXiv papers every weekday morning, or run
it manually whenever you want an update.

## What you'll learn

- How to schedule a `MixRoutine` with a cron expression
- How routine parameters become configurable inputs
- How scheduled and manual runs use the same routine
- How to return a rich Markdown result

## Prerequisites

- [mixtrain CLI installed and logged in](https://mixtrain.ai/docs/guide/quickstart)

No API key or third-party Python package is required.

## Create the routine

```bash
mixtrain routine create daily_arxiv_digest.py --name daily-arxiv-digest
```

By default, the routine runs at 09:00 UTC every weekday and retrieves the five
newest papers in the `cs.LG` category.

## Get a digest now

Use a manual run to generate the default digest immediately:

```bash
mixtrain routine run daily-arxiv-digest
```

Override the inputs to explore another category or search term:

```bash
mixtrain routine run daily-arxiv-digest \
  --query 'all:"reinforcement learning"' \
  --paper-count 3
```

The query uses [arXiv API search syntax](https://info.arxiv.org/help/api/user-manual.html#51-details-of-query-construction).
Other useful examples include `cat:cs.CV`, `cat:cs.CL`, and `au:"Yann LeCun"`.

## How it works

arXiv returns an Atom feed containing paper metadata and author-written
abstracts. The routine orders results by submission date and formats the latest
papers as a readable Markdown digest. It deliberately does not use an LLM: the
abstracts already summarize the papers, keeping this introductory example fast,
free, and dependency-free.
