# MLflow Discover

A learning project for exploring MLflow's LLM evaluation capabilities using
entity extraction as a benchmark task. This project uses the multilingual
[MultiCoNER 2023](https://multiconer.github.io/dataset) dataset to evaluate and
compare how different LLMs perform at extracting named entities across diverse
languages (English, Spanish, Chinese, Hindi, Bengali, Farsi, and more).

The goal is to understand MLflow's workflow for LLM evaluation—from preparing
datasets to tracking experiments, comparing model performance, and managing the
evaluation lifecycle.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

You can clone and set up the project:

```bash
# Clone the repository
git clone https://github.com/Xowap/mlflow-discover.git
cd mlflow-discover

# Install dependencies and create virtual environment
uv sync
```

The `uv sync` command automatically creates a virtual environment and installs
all dependencies based on the `pyproject.toml` configuration.

## Usage

The project provides a command-line interface (CLI) to manage its workflow. The
first step is to prepare the dataset.

### Generating the Dataset

The MultiCoNER dataset is distributed across numerous `.conll` files in a public
S3 bucket. To make it easier to work with, this project includes a command to
download, parse, and consolidate the entire dataset into a single `.jsonl` file.
This file can then be used as a stable input for MLflow experiments.

**Command**

The `generate-dataset` command streams the raw data, processes it, and saves it
to your local machine.

```bash
uv run mlflow-discover generate-dataset [OPTIONS] OUTPUT_FILE
```

**Arguments:**

- `OUTPUT_FILE`: The path where the output `.jsonl` file will be saved. The
  final file will be approximately 500MB.

**Options:**

- `-f`, `--force`: Overwrite the output file if it already exists, without
  prompting for confirmation.

**Example:**

```bash
uv run mlflow-discover generate-dataset data/multiconer.jsonl
```

This will download and process the entire MultiCoNER dataset (~500MB), creating
a consolidated `data/multiconer.jsonl` file ready for LLM evaluation.

**Output Format**

The output is a [JSON Lines](https://jsonlines.org/) (`.jsonl`) file, where each
line is a JSON object representing a single data sample. Each object has the
following structure:

```json
{
    "text": "The sample sentence text goes here.",
    "entities": [
        { "text": "entity text", "type": "ENTITY_LABEL_1" },
        { "text": "another entity", "type": "ENTITY_LABEL_2" }
    ],
    "lang": "en",
    "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
}
```

- **`text`**: The full, detokenized text of the sample.
- **`entities`**: A list of named entities found in the text. Each entity is a
  dictionary containing its `text` and `type` (label).
- **`lang`**: The two-letter ISO 639-1 language code for the sample (e.g., `en`,
  `zh`, `hi`).
- **`id`**: A unique identifier for the sample from the original dataset.

## Project Goals & Roadmap

This is a learning-focused project to understand MLflow's LLM evaluation
capabilities. Current status and planned features:

- [x] **Dataset Preparation**: Consolidate MultiCoNER data into evaluation-ready
      format with proper detokenization across multiple languages
- [ ] **LLM Evaluation Pipeline**: Create prompts for entity extraction and
      evaluate different LLMs (OpenAI, Anthropic, local models) using MLflow's
      evaluation framework
- [ ] **Custom Metrics**: Implement entity-aware metrics (precision, recall, F1)
      for comparing extraction quality across models
- [ ] **Multilingual Analysis**: Compare model performance across different
      languages and entity types
- [ ] **Experiment Tracking**: Use MLflow to track prompts, parameters, and
      results across evaluation runs
- [ ] **Model Comparison**: Build dashboards to compare LLM performance and make
      informed model selection decisions

## Development

Development dependencies are automatically installed by `uv sync`. This project
uses `ruff` for linting and formatting, `mypy` for type checking, and `prettier`
for Markdown formatting.

Run all quality checks at once:

```bash
make clean
```

Or run individual checks:

```bash
# Format code and Markdown
uv run ruff format .
pnpx prettier -w README.md

# Check and fix linting issues
uv run ruff check --fix .

# Type check
uv run mypy src
```

### Project Structure

```
src/mlflow_discover/
├── cli.py                    # CLI entry point
├── commands/
│   └── generate_dataset.py   # Dataset preparation command
└── utils/
    └── multiconer.py         # MultiCoNER parsing and detokenization
```
