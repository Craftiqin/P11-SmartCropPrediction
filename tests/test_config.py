import pytest
from pydantic import ValidationError

from src.config import Settings


def test_default_production_url(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    assert Settings().api_base_url == "https://api.craftiq.in"


def test_environment_url(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:8000/")
    assert Settings().api_base_url == "http://127.0.0.1:8000"


@pytest.mark.parametrize(
    "value",
    ["localhost:8000", "ftp://example.com", "https://user:pass@example.com", "https://example.com/path"],
)
def test_invalid_base_url(monkeypatch, tmp_path, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("API_BASE_URL", value)
    with pytest.raises(ValidationError):
        Settings()

