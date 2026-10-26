from demanda.config import Config


def test_mlflow_no_sigue_la_variable_estandar(monkeypatch, tmp_path):
    # `MLFLOW_TRACKING_URI` puede apuntar a un servidor remoto en el shell; el backtest no debe seguirla
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    assert Config(_env_file=None).mlflow_tracking_uri == f"sqlite:///{tmp_path.resolve() / 'mlflow.db'}"

    monkeypatch.setenv("DEMANDA_MLFLOW_TRACKING_URI", "sqlite:///otro.db")
    assert Config(_env_file=None).mlflow_tracking_uri == "sqlite:///otro.db"
