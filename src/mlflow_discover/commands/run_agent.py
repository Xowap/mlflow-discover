"""run-agent command"""

import json

import rich_click as click
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from ..agent import ExtracT800

console = Console()


@click.command("run-agent")
@click.argument("sentence")
@click.option("--model", "-m", help="Name of the ollama model to use")
def main(sentence: str, model: str) -> None:
    """Run the agent on the provided sentence to see how it performs on a
    one-shot basis"""

    kwargs = {
        "model_name": model,
    }

    if not kwargs["model_name"]:
        del kwargs["model_name"]

    agent = ExtracT800(**kwargs)  # type: ignore
    extracted = agent.extract_entities(sentence)

    panel = Panel(
        JSON(json.dumps(extracted)),
        title="Extracted Entities",
    )
    console.print(panel)
