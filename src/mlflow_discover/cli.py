"""CLI for learning MLflow's LLM evaluation capabilities."""

import rich_click as click

from .commands.generate_dataset import main as main_generate_dataset


@click.group()
@click.pass_context
def cli(_ctx):
    """Explore MLflow's LLM evaluation using entity extraction tasks."""
    pass


cli.add_command(main_generate_dataset)


if __name__ == "__main__":
    cli()
