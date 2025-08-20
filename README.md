# Text2Cypher benchmarks with DSPy

In this repo, we show how to use [DSPy](https://dspy.ai/), a declarative framework for building modular AI software, to
orchestrate a Graph RAG pipeline based on Text2Cypher. Kuzu is the underlying graph database engine used
for all experiments.

We'll use the underlying `GraphRAG` custom module in DSPy to a) run a suite of Cypher queries on the
LDBC dataset, and b) optimize the module to improve performance on the task.

A custom evaluation suite will be provided to check the performance of the base DSPy module and to
act as a learning target for the DSPy optimizer downstream.

## Background

Experiments and benchmarks for Text2Cypher
are shown on the LDBC-SNB dataset, with the overall goal of using DSPy [optimizers](https://dspy.ai/learn/optimization/overview/)
to improve the performance of smaller, cheaper models on the Text2Cypher task.

An earlier version of this exercise used BAML, as shown in [this repo](https://github.com/kuzudb/text2cypher).
Text2Cypher benchmarks run via BAML showed that `openai/gpt-4.1`, a recent, powerful LLM, is really good at
writing Cypher and with some context engineering to prune the graph schema, it passes all tests with 100% accuracy.
However, `gpt-4.1` isn't viable from a cost perspective to run at scale in production, because it's a truly
general-purpose model, and it's really wasteful to use such a powerful model for such a narrow task like
understanding graph schemas and Cypher queries.

Failure modes analyzed in the test suite showed that the smaller models weren't necessarily bad at writing Cypher --
the real issue was that the LDBC graph schema/nomenclature confused the smaller models, such that they mostly
got the relationship *directions* wrong. This indicates that there's a lot of room *in prompt space* alone
to improve performance, because the models' knowledge of Cypher is already sufficient -- we just need to discover
better prompts to guide them.

The primary goal here is to leverage DSPy's optimization tooling
to discover better prompts (rather than humans doing manual prompt engineering) to improve the performance
of smaller models like `openai/gpt-4.1-mini` or `mistralai/devstral-small` on this Text2Cypher benchmark.

## Setup

We recommend installing [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the
Python dependencies:

```bash
uv sync
```

To add any additional dependencies, use the `uv add` command ad follows:

```bash
uv add <package_name>
```

## Dataset

We use the [LDBC social network benchmark](https://ldbcouncil.org/benchmarks/snb/) (LDBC-1) dataset,
which can be downloaded and loaded into Kuzu as shown below.

```bash
# Download the LDBC-1 dataset
uv run download_dataset.py

# Create the graph in Kuzu
uv run create_graph.py
```

> [!NOTE]: The download script has been tested on macOS and Linux and depends on zstd. If
> you're using Windows to run this, we recommend using [WSL](https://github.com/microsoft/WSL).

## Run example Text2Cypher pipeline

```bash
uv run text2cypher.py
```

## Benchmarks

🚧 WIP (To be added soon).
