from typer.testing import CliRunner

from ntn import __version__
from ntn.cli import app


def test_version_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
