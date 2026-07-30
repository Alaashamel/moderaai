from __future__ import annotations

import click

from .classifier import DEFAULT_MODEL_PATH, LexicalClassifier, train_model


@click.group()
@click.version_option()
def main():
    """moderaai — lightweight text moderation (safe / toxic / spam)."""


@main.command()
def train():
    """Train the reference classifier and save it to disk."""
    model = train_model()
    clf = LexicalClassifier(model=model)
    clf.save()
    click.echo(f"Model trained and saved to {DEFAULT_MODEL_PATH}")


@main.command()
@click.argument("text", nargs=-1, required=True)
def check(text: tuple[str, ...]):
    """Classify a piece of text from the command line."""
    clf = LexicalClassifier.load_or_train()
    result = clf.predict(" ".join(text))
    click.echo(f"label: {result.label}")
    for label, score in sorted(result.scores.items(), key=lambda kv: -kv[1]):
        click.echo(f"  {label}: {score:.3f}")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
def serve(host: str, port: int):
    """Run the HTTP API with uvicorn."""
    import uvicorn

    uvicorn.run("moderaai.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
