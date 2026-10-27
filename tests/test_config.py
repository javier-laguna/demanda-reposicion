from demanda.config import Config, env_file


def test_mlflow_no_sigue_la_variable_estandar(monkeypatch, tmp_path):
    # `MLFLOW_TRACKING_URI` puede apuntar a un servidor remoto en el shell; el backtest no debe seguirla
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    assert Config(_env_file=None).mlflow_tracking_uri == f"sqlite:///{tmp_path.resolve() / 'mlflow.db'}"

    monkeypatch.setenv("DEMANDA_MLFLOW_TRACKING_URI", "sqlite:///otro.db")
    assert Config(_env_file=None).mlflow_tracking_uri == "sqlite:///otro.db"


def test_demanda_env_file_cambia_el_env_que_se_lee(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    otro = tmp_path / "otro.env"
    otro.write_text("LLM_MODEL=openai:modelo-de-prueba\n")
    monkeypatch.setenv("DEMANDA_ENV_FILE", str(otro))
    assert env_file() == otro
    assert Config().llm_model == "openai:modelo-de-prueba"
    assert Config(_env_file=None).llm_model == "anthropic:claude-opus-5-5"  # un _env_file explícito gana
    monkeypatch.setenv("DEMANDA_ENV_FILE", str(tmp_path / "no_existe.env"))
    assert Config().llm_model == "anthropic:claude-opus-5-5"


def test_el_suite_no_lee_el_env_real():
    # conftest apunta DEMANDA_ENV_FILE a un archivo que no existe
    assert not env_file().exists()
