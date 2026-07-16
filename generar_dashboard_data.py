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
    nombre_mes_anterior,
)


MONTO_FIJO_CLP = 1_000_000
CSV_CARTERA = "data/cartera_simulada.json"


def calcular_resultado(tipo, monto_clp, precio_entrada, precio_referencia):
    unidades = monto_clp / precio_entrada
    valor_referencia = unidades * precio_referencia
    if tipo == "COMPRA":
        ganancia_clp = valor_referencia - monto_clp
        ganancia_pct = (precio_referencia / precio_entrada - 1) * 100
    else:
        ganancia_clp = monto_clp - valor_referencia
        ganancia_pct = (precio_entrada / precio_referencia - 1) * 100
    return ganancia_clp, ganancia_pct


def actualizar_cartera_simulada(filas, fecha_senal):
    """
    Registra automáticamente $1.000.000 CLP en cada señal nueva de
    COMPRA/VENTA (sin depender de que alguien abra el dashboard), y cierra
    sola cualquier posición abierta cuando aparece la señal contraria para
    ese mismo ticker. Se guarda en el repo (data/cartera_simulada.json),
    no en el navegador, así que queda disponible desde cualquier
    dispositivo y persiste aunque nadie entre a la página en varios días.
    """
    if os.path.exists(CSV_CARTERA):
        with open(CSV_CARTERA, "r", encoding="utf-8") as f:
            posiciones = json.load(f)
    else:
        posiciones = []

    precio_actual_por_ticker = {f["ticker"]: f["precio"] for f in filas}
    senal_actual_por_ticker = {f["ticker"]: f["senal"] for f in filas}

    # 1) Registrar señales nuevas de hoy (evitando duplicar si ya se
    #    registró este ticker para esta misma fecha de señal).
    ya_registrados_hoy = {(p["ticker"], p["fecha_senal"]) for p in posiciones}
    for f in filas:
        if f["senal"] not in ("COMPRA", "VENTA"):
            continue
        clave = (f["ticker"], fecha_senal)
        if clave in ya_registrados_hoy:
            continue
        posiciones.append({
            "id": f"{f['ticker']}-{fecha_senal}",
            "ticker": f["ticker"],
            "tipo": f["senal"],
            "precio_entrada": f["precio"],
            "monto_clp": MONTO_FIJO_CLP,
            "fecha_entrada": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "fecha_senal": fecha_senal,
            "cerrada": False,
        })

    # 2) Cerrar posiciones abiertas si hoy apareció la señal contraria.
    for p in posiciones:
        if p.get("cerrada"):
            continue
        contraria = "VENTA" if p["tipo"] == "COMPRA" else "COMPRA"
        precio_hoy = precio_actual_por_ticker.get(p["ticker"])
        if precio_hoy is not None and senal_actual_por_ticker.get(p["ticker"]) == contraria:
            ganancia_clp, ganancia_pct = calcular_resultado(
                p["tipo"], p["monto_clp"], p["precio_entrada"], precio_hoy
            )
            p["cerrada"] = True
            p["fecha_cierre"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
            p["precio_cierre"] = precio_hoy
            p["ganancia_clp_final"] = round(ganancia_clp, 2)
            p["ganancia_pct_final"] = round(ganancia_pct, 4)

    os.makedirs("data", exist_ok=True)
    with open(CSV_CARTERA, "w", encoding="utf-8") as f:
        json.dump(posiciones, f, ensure_ascii=False, indent=2)

    nuevas = sum(1 for p in posiciones if p["fecha_senal"] == fecha_senal)
    cerradas_hoy = sum(1 for p in posiciones if p.get("fecha_cierre", "").startswith(fecha_senal))
    print(f"Cartera simulada: {len(posiciones)} posiciones en total "
          f"({nuevas} nuevas hoy, {cerradas_hoy} cerradas hoy)")


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
                "retorno_10d": float(fila["retorno_10d_%"]) if pd_notna(fila["retorno_10d_%"]) else None,
                "retorno_1m": float(fila["retorno_1m_%"]) if pd_notna(fila["retorno_1m_%"]) else None,
                "retorno_mes_ant": float(fila["retorno_mes_ant_%"]) if pd_notna(fila["retorno_mes_ant_%"]) else None,
                "watchlist_score": int(fila["watchlist_score"]) if pd_notna(fila["watchlist_score"]) else 0,
                "watchlist_ema_pct": float(fila["watchlist_ema_pct"]) if pd_notna(fila["watchlist_ema_pct"]) else None,
                "watchlist_en_squeeze": bool(fila["watchlist_en_squeeze"]),
                "watchlist_adx_subiendo": bool(fila["watchlist_adx_subiendo"]),
            })

    data = {
        "actualizado": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "parametros": {
            "ema_fast": EMA_FAST,
            "ema_slow": EMA_SLOW,
            "adx_key_level": ADX_KEY_LEVEL,
        },
        "total_tickers": len(TICKERS_IPSA),
        "mes_anterior_label": nombre_mes_anterior(),
        "senales": filas,
    }

    os.makedirs("data", exist_ok=True)
    with open("data/senales.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_compra = sum(1 for f in filas if f["senal"] == "COMPRA")
    n_venta = sum(1 for f in filas if f["senal"] == "VENTA")
    print(f"Guardado data/senales.json: {len(filas)} tickers "
          f"({n_compra} COMPRA, {n_venta} VENTA)")

    fecha_senal = data["actualizado"][:10]  # YYYY-MM-DD
    actualizar_cartera_simulada(filas, fecha_senal)


def pd_notna(x):
    """Evita importar pandas solo para este chequeo puntual."""
    return x is not None and x == x  # x == x es False solo para NaN


if __name__ == "__main__":
    main()
