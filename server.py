"""ElectroTrace server launcher."""
from electrotrace.server_app import app, run

__all__ = ["app", "run"]


if __name__ == "__main__":
    run()
