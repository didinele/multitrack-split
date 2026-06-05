import typer

app = typer.Typer(help="Multitrack Splitter: detect and segment songs from long multitrack live recordings.")

@app.command()
def split():
    from .gui import launch
    launch()


if __name__ == "__main__":
    app()
