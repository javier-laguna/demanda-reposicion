"""`python -m demanda.mcp_server`: el servidor MCP por stdio sobre los artefactos de la config."""

from demanda.config import get_config
from demanda.mcp_server.servidor import servir

if __name__ == "__main__":
    servir(get_config().artifacts_dir)
