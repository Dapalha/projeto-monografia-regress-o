"""Interface web do classificador de fraude.

Uso: coloque modelos_finais.pkl na pasta do projeto (ou indique o caminho em
MODELOS_PKL) e execute  python app.py
"""
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

from fraude.interface import criar_interface  # noqa: E402

MODELOS_PKL = Path(os.environ.get("MODELOS_PKL", Path(__file__).with_name("modelos_finais.pkl")))

if __name__ == "__main__":
    criar_interface(joblib.load(MODELOS_PKL)).launch(server_name="0.0.0.0")
