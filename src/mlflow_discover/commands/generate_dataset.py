"""Prepares MultiCoNER data for LLM evaluation experiments."""

from pathlib import Path

import orjson
import rich_click as click
from rich.console import Console

from ..utils.multiconer import parse_conll, stream_conll_lines

console = Console()


@click.command("generate-dataset")
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option("--force", "-f", is_flag=True, help="Overwrite without prompting")
def main(output_file: Path, force: bool):
    """
    Streams the whole dataset into a `.jsonl` file which can later be used to
    feed the dataset into MLflow. The output file should be about 500M.
    """

    # Check if it's a directory
    if output_file.is_dir():
        msg = f"'{output_file}' is a directory, not a file"
        raise click.BadParameter(msg)

    # Creating parent dir if needed
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and prompt for overwrite
    if output_file.exists() and not force:
        click.confirm(
            f"'{output_file}' already exists. Overwrite?",
            abort=True,  # Raises click.Abort if user says no
        )

    lines = stream_conll_lines("multiconer", "multiconer2023/")

    with output_file.open("wb") as f:
        for utterance in parse_conll(lines):
            text = utterance.detokenize()
            entities = [
                dict(
                    text=x.detokenize(utterance.domain),
                    type=x.label,
                )
                for x in utterance.entities
            ]
            line = dict(
                text=text, entities=entities, lang=utterance.domain, id=utterance.id
            )
            f.write(orjson.dumps(line) + b"\n")
