"""
Genera data/senales.json a partir de la metodología Trading Latino,
reutilizando las funciones de trading_latino_chile.py, para que el
dashboard web (index.html) lo muestre.

Se corre automáticamente todos los días vía GitHub Actions (ver
.github/workflows/actualizar.yml), o manualmente en cualquier momento
desde la pestaña "Actions" del repo (botón "Run workflow").

Requisitos: pip install yfinance pandas numpy
"""

import json
import os
from datetime import datetime, timezone

from trading_latino_chile import (
    analizar_universo,
    TICKERS_IPSA,
    ADX_KEY_LEVEL,
    EMA_FAST,
    EMA_SLOW,
)


def main():
    tabla = analizar_universo()

    filas = []
    if not tabla.empty:
        for _, fila in tabla.iterrows():
            filas.append({
                "ticker": fila["ticker"],
                "precio": float(fila["precio"]),
                "tendencia": fila["tendencia"],
                "adx": float(fila["adx"]) if pd_notna(fila["adx"]) else None,
                "senal": fila["señal"],
            })

    data = {
        "actualizado": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "parametros": {
            "ema_fast": EMA_FAST,
            "ema_slow": EMA_SLOW,
            "adx_key_level": ADX_KEY_LEVEL,
        },
        "total_tickers": len(TICKERS_IPSA),
        "senales": filas,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/senales.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_compra = sum(1 for f in filas if f["senal"] == "COMPRA")
    n_venta = sum(1 for f in filas if f["senal"] == "VENTA")
    print(f"Guardado data/senales.json: {len(filas)} tickers "
          f"({n_compra} COMPRA, {n_venta} VENTA)")


def pd_notna(x):
    """Evita importar pandas solo para este chequeo puntual."""
    return x is not None and x == x  # x == x es False solo para NaN


if __name__ == "__main__":
    main()
