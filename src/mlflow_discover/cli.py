"""CLI for learning MLflow's LLM evaluation capabilities."""

import rich_click as click

from .commands.generate_dataset import main as main_generate_dataset
from .commands.run_agent import main as main_run_agent


@click.group()
@click.pass_context
def cli(_ctx):
    """Explore MLflow's LLM evaluation using entity extraction tasks."""
    pass


cli.add_command(main_generate_dataset)
cli.add_command(main_run_agent)


if __name__ == "__main__":
    cli()
