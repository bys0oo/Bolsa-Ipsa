"""
Analizador de acciones de la Bolsa de Santiago (IPSA) usando la metodología
de Trading Latino (Jaime Merino): EMA 10/55 (filtro de tendencia) +
Squeeze Momentum Indicator (LazyBear) + ADX (pendiente).

Fuente de datos: Yahoo Finance (yfinance), delay ~15 min.

USO:
    python trading_latino_chile.py

Requisitos:
    pip install yfinance pandas numpy
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# 1) UNIVERSO DE ACCIONES A ANALIZAR (tickers de Yahoo Finance, sufijo .SN)
# ---------------------------------------------------------------------------
# Lista aproximada de componentes del IPSA. Revísala/ajústala: la composición
# del índice cambia con cada rebalanceo (S&P DJI). Puedes agregar cualquier
# acción chilena adicional (no tiene que estar en el IPSA) con su nemotécnico
# + ".SN".
TICKERS_IPSA = [
    "SQM-B.SN", "CHILE.SN", "BSANTANDER.SN", "FALABELLA.SN", "CENCOSUD.SN",
    "CCU.SN", "COPEC.SN", "ENELCHILE.SN", "ENELAM.SN", "CMPC.SN",
    "ITAUCL.SN", "BCI.SN", "PARAUCO.SN", "RIPLEY.SN", "SMU.SN",
    "VAPORES.SN", "CAP.SN", "LTM.SN", "ANDINA-B.SN", "CONCHATORO.SN",
    "SONDA.SN", "ENTEL.SN", "ILC.SN", "COLBUN.SN", "AGUAS-A.SN",
    "ECL.SN", "MALLPLAZA.SN", "QUINENCO.SN", "ORO-BLANCO.SN", "SK.SN",
]

# ---------------------------------------------------------------------------
# 2) PARÁMETROS DE LA ESTRATEGIA (valores por defecto de Trading Latino)
# ---------------------------------------------------------------------------
EMA_FAST = 10
EMA_SLOW = 55
ADX_PERIOD = 14
ADX_KEY_LEVEL = 23        # "punto 23" que usa Jaime Merino como referencia
BB_LENGTH = 20
BB_MULT = 2.0
KC_LENGTH = 20
KC_MULT = 1.5
TIMEFRAME_INTERVAL = "1d"  # diario. Se puede cambiar a "1wk", "4h" no existe en yfinance nativo
LOOKBACK_PERIOD = "1y"        # usado para la señal "actual" (último día)
LOOKBACK_HISTORIAL = "10y"    # usado para el historial completo de señales pasadas
# Nota: no todas las acciones chilenas tienen 10 años de historia en Yahoo
# Finance (algunas se listaron después, o Yahoo simplemente no tiene tanto
# histórico cargado para .SN). En esos casos yfinance devuelve lo que tenga
# disponible, sin error.


# ---------------------------------------------------------------------------
# 3) INDICADORES
# ---------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(high, low, close):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD):
    """ADX de Wilder + DI+/DI-. Devuelve (adx, plus_di, minus_di)."""
    high, low, close = df["High"], df["Low"], df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(high, low, close)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx, plus_di, minus_di


def _linreg_last(y: np.ndarray) -> float:
    """Valor proyectado de una regresión lineal sobre la última barra de la ventana."""
    if np.any(np.isnan(y)):
        return np.nan
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return intercept + slope * (len(y) - 1)


def compute_squeeze_momentum(
    df: pd.DataFrame,
    bb_length: int = BB_LENGTH,
    bb_mult: float = BB_MULT,
    kc_length: int = KC_LENGTH,
    kc_mult: float = KC_MULT,
):
    """
    Squeeze Momentum Indicator (LazyBear), la base del indicador que usa
    Trading Latino. Devuelve:
      - momentum: histograma (positivo/negativo, con su pendiente)
      - squeeze_on: True cuando BB está dentro de KC (baja volatilidad, "squeeze")
      - squeeze_off: True cuando BB sale de KC (liberación / posible impulso)
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    basis = close.rolling(bb_length).mean()
    dev = bb_mult * close.rolling(bb_length).std(ddof=0)
    upper_bb = basis + dev
    lower_bb = basis - dev

    ma = close.rolling(kc_length).mean()
    tr = true_range(high, low, close)
    rng_ma = tr.rolling(kc_length).mean()
    upper_kc = ma + rng_ma * kc_mult
    lower_kc = ma - rng_ma * kc_mult

    squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    squeeze_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)

    highest_high = high.rolling(kc_length).max()
    lowest_low = low.rolling(kc_length).min()
    midpoint = (highest_high + lowest_low) / 2
    midpoint = (midpoint + ma) / 2

    val = close - midpoint
    momentum = val.rolling(kc_length).apply(_linreg_last, raw=True)

    return momentum, squeeze_on, squeeze_off


# ---------------------------------------------------------------------------
# 4) LÓGICA DE SEÑAL (regla de Trading Latino)
# ---------------------------------------------------------------------------
def build_signal(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["ema_fast"] = ema(df["Close"], EMA_FAST)
    df["ema_slow"] = ema(df["Close"], EMA_SLOW)
    df["adx"], df["plus_di"], df["minus_di"] = compute_adx(df)
    df["momentum"], df["sqz_on"], df["sqz_off"] = compute_squeeze_momentum(df)

    df["adx_slope"] = df["adx"].diff()
    df["mom_up"] = df["momentum"] > 0
    df["mom_prev_up"] = df["momentum"].shift(1) > 0
    df["mom_flip_up"] = df["mom_up"] & ~df["mom_prev_up"]      # oscilador pasa a alcista
    df["mom_flip_down"] = ~df["mom_up"] & df["mom_prev_up"]    # oscilador pasa a bajista

    last = df.iloc[-1]

    trend_up = last["ema_fast"] > last["ema_slow"]
    adx_falling = last["adx_slope"] < 0
    adx_strong = last["adx"] > ADX_KEY_LEVEL
    # Filtro real de Trading Latino: la pendiente negativa del ADX solo
    # cuenta como "fin de tendencia" si el ADX efectivamente estuvo por
    # sobre el nivel 23 (tendencia con fuerza real). Una pendiente negativa
    # con ADX débil (ej. 9, 14) es solo ruido, no agotamiento de tendencia.
    adx_valido = adx_strong and adx_falling

    signal = "ESPERAR"
    if trend_up and last["mom_flip_up"] and adx_valido:
        signal = "COMPRA"
    elif (not trend_up) and last["mom_flip_down"] and adx_valido:
        signal = "VENTA"
    elif trend_up and last["mom_up"]:
        signal = "TENDENCIA ALCISTA (sin nueva señal)"
    elif (not trend_up) and (not last["mom_up"]):
        signal = "TENDENCIA BAJISTA (sin nueva señal)"

    cercania = evaluar_cercania_compra(df)

    return {
        "precio": round(last["Close"], 2),
        "ema10": round(last["ema_fast"], 2),
        "ema55": round(last["ema_slow"], 2),
        "tendencia": "ALCISTA" if trend_up else "BAJISTA",
        "adx": round(last["adx"], 1),
        "adx_fuerte": bool(adx_strong),
        "adx_bajando": bool(adx_falling),
        "momentum": "ALCISTA" if last["mom_up"] else "BAJISTA",
        "squeeze": "EN SQUEEZE (baja volatilidad)" if last["sqz_on"] else (
            "LIBERADO" if last["sqz_off"] else "-"),
        "señal": signal,
        "watchlist_score": cercania["score"],
        "watchlist_ema_pct": cercania["distancia_ema_%"],
        "watchlist_en_squeeze": cercania["en_squeeze_reciente"],
        "watchlist_adx_subiendo": cercania["adx_subiendo"],
    }


# ---------------------------------------------------------------------------
# 4.1) WATCHLIST: acciones que aún NO gatillan COMPRA pero se están
# acercando, usando versiones relajadas de los mismos 3 filtros de Trading
# Latino. Esto NO reemplaza la señal real (que exige los 3 confirmados a
# la vez); es solo una alerta temprana para hacerles seguimiento.
# ---------------------------------------------------------------------------
N_CERCANIA = 5  # barras hacia atrás para medir si se está "acercando"


def evaluar_cercania_compra(df: pd.DataFrame) -> dict:
    """
    Puntaje 0-3 según cuántas de estas condiciones (relajadas) se cumplen:
      1) EMA10 se está acercando a cruzar hacia arriba a la EMA55 (todavía
         por debajo, pero la distancia porcentual se viene achicando).
      2) Estuvo en squeeze (baja volatilidad / compresión) en las últimas
         N_CERCANIA barras -> presión acumulada, posible impulso próximo.
      3) El ADX viene subiendo desde niveles bajos (aún no supera
         ADX_KEY_LEVEL, pero la tendencia del indicador es al alza).
    Requiere que las columnas ema_fast/ema_slow/adx/sqz_on ya estén
    calculadas en df (build_signal las agrega antes de llamar a esto).
    """
    if len(df) < N_CERCANIA + 2 or df[["ema_fast", "ema_slow", "adx"]].iloc[-1].isna().any():
        return {"score": 0, "distancia_ema_%": None, "en_squeeze_reciente": False,
                "adx_subiendo": False}

    last = df.iloc[-1]
    antes = df.iloc[-1 - N_CERCANIA]

    dist_ema_hoy = (last["ema_fast"] - last["ema_slow"]) / last["ema_slow"] * 100
    dist_ema_antes = (antes["ema_fast"] - antes["ema_slow"]) / antes["ema_slow"] * 100
    # Todavía bajo la EMA55 (no hay cruce aún), pero la brecha se acorta.
    cond_ema = bool(-2.0 <= dist_ema_hoy < 0 and dist_ema_hoy > dist_ema_antes)

    cond_squeeze = bool(df["sqz_on"].iloc[-N_CERCANIA:].fillna(False).any())

    adx_hoy, adx_antes = last["adx"], antes["adx"]
    cond_adx = bool(pd.notna(adx_hoy) and pd.notna(adx_antes)
                    and adx_hoy < ADX_KEY_LEVEL and adx_hoy > adx_antes)

    score = int(cond_ema) + int(cond_squeeze) + int(cond_adx)
    return {
        "score": score,
        "distancia_ema_%": round(float(dist_ema_hoy), 2),
        "en_squeeze_reciente": cond_squeeze,
        "adx_subiendo": cond_adx,
    }


# ---------------------------------------------------------------------------
# 4.5) HISTORIAL DE SEÑALES (todas las fechas pasadas, no solo el último día)
# ---------------------------------------------------------------------------
HORIZONTES_DIAS = (5, 10, 20)  # días de trading después de la señal, para medir rendimiento


def historial_señales(df: pd.DataFrame, ticker: str) -> list:
    """
    Recorre TODO el historial descargado y devuelve una lista de eventos
    (uno por fila) cada vez que se cumplió la regla de COMPRA o VENTA de
    Trading Latino, con fecha, precio, ADX de ese momento, y además el
    RETORNO FUTURO a 5/10/20 días de trading después de la señal (para
    poder medir objetivamente si la señal "funcionó" o no, en vez de
    revisarlo caso por caso a ojo).
    """
    df = df.copy()
    df["ema_fast"] = ema(df["Close"], EMA_FAST)
    df["ema_slow"] = ema(df["Close"], EMA_SLOW)
    df["adx"], df["plus_di"], df["minus_di"] = compute_adx(df)
    df["momentum"], df["sqz_on"], df["sqz_off"] = compute_squeeze_momentum(df)

    df["adx_slope"] = df["adx"].diff()
    df["trend_up"] = df["ema_fast"] > df["ema_slow"]
    df["mom_up"] = df["momentum"] > 0
    df["mom_prev_up"] = df["momentum"].shift(1) > 0
    df["mom_flip_up"] = df["mom_up"] & ~df["mom_prev_up"]
    df["mom_flip_down"] = ~df["mom_up"] & df["mom_prev_up"]
    df["adx_falling"] = df["adx_slope"] < 0
    df["adx_strong"] = df["adx"] > ADX_KEY_LEVEL
    # Mismo filtro que en build_signal: pendiente negativa SOLO cuenta si
    # el ADX venía de estar sobre el nivel 23 (tendencia real agotándose),
    # no cualquier caída de ADX débil que es solo ruido.
    df["adx_valido"] = df["adx_strong"] & df["adx_falling"]

    compra = df["trend_up"] & df["mom_flip_up"] & df["adx_valido"]
    venta = (~df["trend_up"]) & df["mom_flip_down"] & df["adx_valido"]

    df = df.reset_index()  # para poder usar posición entera (iloc) y calcular retornos futuros
    close = df["Close"]

    def _evento(pos, tipo):
        fecha = df.loc[pos, df.columns[0]]  # primera columna = el índice de fechas original
        precio_hoy = close.iloc[pos]
        ev = {
            "ticker": ticker,
            "fecha": fecha.date() if hasattr(fecha, "date") else fecha,
            "tipo": tipo,
            "precio": round(precio_hoy, 2),
            "adx": round(df.loc[pos, "adx"], 1),
        }
        for h in HORIZONTES_DIAS:
            pos_futuro = pos + h
            if pos_futuro < len(df):
                precio_futuro = close.iloc[pos_futuro]
                retorno = (precio_futuro - precio_hoy) / precio_hoy * 100
                ev[f"retorno_{h}d_%"] = round(retorno, 2)
            else:
                ev[f"retorno_{h}d_%"] = None  # aún no hay suficientes días futuros (señal muy reciente)
        # "Acierto" evaluado con el horizonte intermedio (10 días): para COMPRA
        # se espera que el precio suba, para VENTA que baje.
        r10 = ev.get("retorno_10d_%")
        if r10 is not None:
            # OJO: forzar bool() nativo de Python. numpy.bool_ guardado en una
            # columna "object" hace que pandas .sum()/.mean() sume como OR
            # lógico en vez de contar 1 por acierto, dando un % falso casi 0.
            ev["acierto_10d"] = bool(r10 > 0) if tipo == "COMPRA" else bool(r10 < 0)
        else:
            ev["acierto_10d"] = None
        return ev

    eventos = [_evento(pos, "COMPRA") for pos in df.index[compra.values]]
    eventos += [_evento(pos, "VENTA") for pos in df.index[venta.values]]

    eventos.sort(key=lambda e: e["fecha"])
    return eventos


def construir_historial_universo(tickers=TICKERS_IPSA, period=LOOKBACK_HISTORIAL,
                                   interval=TIMEFRAME_INTERVAL):
    """Descarga cada ticker UNA vez y devuelve el historial completo de señales."""
    todos_eventos = []
    for tk in tickers:
        try:
            data = yf.download(tk, period=period, interval=interval, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if data.empty or len(data) < KC_LENGTH + 5:
                print(f"[!] {tk}: sin datos suficientes para historial, se omite")
                continue
            años_reales = (data.index[-1] - data.index[0]).days / 365
            print(f"    {tk}: {len(data)} barras (~{años_reales:.1f} años de historia disponible)")
            eventos = historial_señales(data, tk)
            todos_eventos.extend(eventos)
        except Exception as e:
            print(f"[!] {tk}: error en historial -> {e}")
        time.sleep(0.3)

    if not todos_eventos:
        return pd.DataFrame()

    hist = pd.DataFrame(todos_eventos)
    hist = hist.sort_values(["ticker", "fecha"]).reset_index(drop=True)
    return hist


CSV_HISTORIAL = "historial_señales_ipsa.csv"


def acumular_historial_csv(nuevo: pd.DataFrame, path: str = CSV_HISTORIAL) -> pd.DataFrame:
    """
    Fusiona el historial recién calculado con el CSV ya existente en disco,
    en vez de sobreescribirlo. Así el registro va CRECIENDO cada vez que
    corres el script, sin depender de que Yahoo Finance siga teniendo 10
    años completos hacia atrás en cada corrida.

    - Si una señal (mismo ticker+fecha+tipo) ya existía, se actualiza con
      los valores recién calculados (útil porque una señal reciente puede
      no haber tenido aún retorno_10d_%/20d_% la primera vez que se detectó,
      por falta de días futuros, y sí lo tiene semanas/meses después).
    - Si una señal antigua ya no aparece en el recálculo (porque se salió
      de la ventana de 10 años de Yahoo), se conserva igual: nunca se borra
      nada que ya estaba guardado.
    """
    if os.path.exists(path):
        anterior = pd.read_csv(path)
        anterior["fecha"] = pd.to_datetime(anterior["fecha"]).dt.date
        combinado = pd.concat([anterior, nuevo], ignore_index=True)
    else:
        combinado = nuevo.copy()

    # Si hay duplicados (misma señal calculada dos veces), nos quedamos con
    # la ÚLTIMA (la recién calculada), porque puede traer retornos futuros
    # más completos que la versión guardada anteriormente.
    combinado = combinado.drop_duplicates(subset=["ticker", "fecha", "tipo"], keep="last")
    combinado = combinado.sort_values(["ticker", "fecha"]).reset_index(drop=True)
    combinado.to_csv(path, index=False)
    return combinado


# ---------------------------------------------------------------------------
# 4.7) BENCHMARK NEUTRAL ("comprar cualquier día", sin ninguna señal)
# ---------------------------------------------------------------------------
def calcular_benchmark(tickers=TICKERS_IPSA, period=LOOKBACK_HISTORIAL,
                        interval=TIMEFRAME_INTERVAL, horizontes=HORIZONTES_DIAS):
    """
    Calcula el retorno promedio a 5/10/20 días tomando TODOS los días de
    todas las acciones (no solo los días con señal). Sirve como punto de
    comparación neutral: si el retorno promedio de las señales COMPRA no es
    claramente mejor que este número, el filtro técnico no está aportando
    nada por sobre simplemente tener la acción en cualquier momento (y ese
    "cualquier momento" ya refleja si el mercado en general subió mucho en
    el período analizado).
    """
    retornos = {h: [] for h in horizontes}
    for tk in tickers:
        try:
            data = yf.download(tk, period=period, interval=interval, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if data.empty or len(data) < max(horizontes) + 5:
                continue
            close = data["Close"]
            for h in horizontes:
                fut = close.shift(-h)
                ret = (fut - close) / close * 100
                retornos[h].extend(ret.dropna().tolist())
        except Exception as e:
            print(f"[!] {tk}: error en benchmark -> {e}")
        time.sleep(0.3)

    resultado = {}
    for h in horizontes:
        valores = retornos[h]
        if valores:
            resultado[h] = {"promedio": float(np.mean(valores)), "n": len(valores)}
        else:
            resultado[h] = {"promedio": None, "n": 0}
    return resultado


# ---------------------------------------------------------------------------
# 5) DESCARGA + BARRIDO DE TODO EL UNIVERSO (señal del último día)
# ---------------------------------------------------------------------------
DIAS_HABILES_MES = 21
DIAS_HABILES_10 = 10  # mismo horizonte usado para medir acierto en el backtest

MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def nombre_mes_anterior() -> str:
    """Ej.: si hoy es cualquier día de julio, devuelve 'Junio 2026'."""
    hoy = pd.Timestamp.now(tz="UTC")
    primer_dia_mes_actual = hoy.replace(day=1)
    primer_dia_mes_anterior = primer_dia_mes_actual - pd.DateOffset(months=1)
    nombre = MESES_ES[primer_dia_mes_anterior.month - 1].capitalize()
    return f"{nombre} {primer_dia_mes_anterior.year}"


def analizar_universo(tickers=TICKERS_IPSA, period=LOOKBACK_PERIOD, interval=TIMEFRAME_INTERVAL):
    resultados = []
    for tk in tickers:
        try:
            data = yf.download(tk, period=period, interval=interval, progress=False, auto_adjust=True)
            # yfinance reciente devuelve columnas MultiIndex (campo, ticker)
            # incluso para un solo ticker; hay que aplanarlas o data["Close"]
            # sale como DataFrame de 1 columna en vez de Series.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if data.empty or len(data) < KC_LENGTH + 5:
                print(f"[!] {tk}: sin datos suficientes, se omite")
                continue
            info = build_signal(data)
            info["ticker"] = tk
            precio_hoy = float(data["Close"].iloc[-1])

            # Retorno a 10 días hábiles: mismo horizonte que usamos para medir
            # acierto en el backtest histórico, para que el ranking sea
            # directamente comparable con esas estadísticas.
            if len(data) > DIAS_HABILES_10:
                precio_hace_10d = float(data["Close"].iloc[-DIAS_HABILES_10 - 1])
                info["retorno_10d_%"] = round((precio_hoy / precio_hace_10d - 1) * 100, 2)
            else:
                info["retorno_10d_%"] = None

            # Retorno del último mes MÓVIL (~21 días hábiles): una ventana que
            # se desliza todos los días, no se reinicia nunca.
            if len(data) > DIAS_HABILES_MES:
                precio_hace_1m = float(data["Close"].iloc[-DIAS_HABILES_MES - 1])
                info["retorno_1m_%"] = round((precio_hoy / precio_hace_1m - 1) * 100, 2)
            else:
                info["retorno_1m_%"] = None

            # Retorno del MES CALENDARIO ANTERIOR completo (ej. si hoy es
            # julio, calcula el retorno de junio completo: cierre de junio
            # vs. cierre de mayo). Se reinicia cada 1° de mes.
            fecha_hoy_dt = data.index[-1]
            primer_dia_mes_actual = fecha_hoy_dt.replace(day=1)
            primer_dia_mes_anterior = primer_dia_mes_actual - pd.DateOffset(months=1)
            datos_mes_anterior = data[(data.index >= primer_dia_mes_anterior) & (data.index < primer_dia_mes_actual)]
            datos_antes_mes_anterior = data[data.index < primer_dia_mes_anterior]
            if not datos_mes_anterior.empty and not datos_antes_mes_anterior.empty:
                precio_cierre_mes_ant = float(datos_mes_anterior["Close"].iloc[-1])
                precio_base_mes_ant = float(datos_antes_mes_anterior["Close"].iloc[-1])
                info["retorno_mes_ant_%"] = round((precio_cierre_mes_ant / precio_base_mes_ant - 1) * 100, 2)
            else:
                info["retorno_mes_ant_%"] = None

            resultados.append(info)
        except Exception as e:
            print(f"[!] {tk}: error -> {e}")
        time.sleep(0.3)  # evitar rate-limit de Yahoo

    if not resultados:
        print("No se obtuvieron resultados.")
        return pd.DataFrame()

    cols = ["ticker", "precio", "tendencia", "ema10", "ema55",
            "adx", "adx_bajando", "momentum", "squeeze", "señal",
            "retorno_10d_%", "retorno_1m_%", "retorno_mes_ant_%",
            "watchlist_score", "watchlist_ema_pct", "watchlist_en_squeeze",
            "watchlist_adx_subiendo"]
    out = pd.DataFrame(resultados)[cols]

    orden_prioridad = {"COMPRA": 0, "VENTA": 1}
    out["_orden"] = out["señal"].map(orden_prioridad).fillna(2)
    out = out.sort_values("_orden").drop(columns="_orden").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 4.8) RIGOR ESTADÍSTICO: mediana, aviso de outliers y desglose por período
# (mismas funciones que se agregaron al script de cripto, para tener el
# mismo nivel de exigencia en ambos activos)
# ---------------------------------------------------------------------------
def reportar_rendimiento(evaluables: pd.DataFrame, etiqueta: str,
                          col_tipo="tipo", col_retorno="retorno_10d_%", col_acierto="acierto_10d"):
    """
    Imprime acierto, promedio, MEDIANA y sensibilidad a outliers para
    COMPRA y VENTA. La mediana es clave: si está muy lejos del promedio,
    el resultado probablemente está siendo inflado/distorsionado por 1-2
    operaciones puntuales, no por un patrón repetible.
    """
    if evaluables.empty:
        print(f"  ({etiqueta}: sin señales evaluables)")
        return
    for tipo in ["COMPRA", "VENTA"]:
        sub = evaluables[evaluables[col_tipo] == tipo].sort_values(col_retorno, ascending=False)
        if len(sub) == 0:
            continue
        win_rate = sub[col_acierto].astype(bool).mean() * 100
        prom = sub[col_retorno].mean()
        mediana = sub[col_retorno].median()
        aviso = ""
        if len(sub) >= 3:
            sin_top2 = sub.iloc[2:][col_retorno].mean()
            if (prom > 0) != (sin_top2 > 0) or abs(prom - sin_top2) > abs(prom) * 0.5:
                aviso = (f"  [!] ADVERTENCIA: quitando las 2 mejores señales, el promedio "
                         f"pasa a {sin_top2:+.2f}% -> resultado poco robusto, posible "
                         f"distorsión por outliers")
        print(f"  {etiqueta} | {tipo}: {len(sub)} señales | acierto: {win_rate:.1f}% | "
              f"promedio: {prom:+.2f}% | mediana: {mediana:+.2f}%{aviso}")


def analizar_por_periodos(historial: pd.DataFrame, cortes=("2022-01-01",)):
    """
    Divide el historial en tramos de tiempo y reporta rendimiento por
    separado en cada tramo. Dice si el patrón es consistente en distintos
    ciclos de mercado, en vez de solo mirar un promedio general que puede
    estar dominado por un único período.
    """
    historial = historial.copy()
    historial["fecha"] = pd.to_datetime(historial["fecha"])
    evaluables = historial.dropna(subset=["acierto_10d"]).copy()
    if evaluables.empty:
        return

    cortes_dt = [pd.Timestamp(c) for c in cortes]
    bordes = [evaluables["fecha"].min()] + cortes_dt + [evaluables["fecha"].max() + pd.Timedelta(days=1)]

    print("\n=== RENDIMIENTO POR PERÍODO (para revisar si el patrón es consistente en el tiempo) ===")
    for i in range(len(bordes) - 1):
        ini, fin = bordes[i], bordes[i + 1]
        tramo = evaluables[(evaluables["fecha"] >= ini) & (evaluables["fecha"] < fin)]
        etiqueta = f"{ini.date()} a {fin.date()}"
        if tramo.empty:
            print(f"  ({etiqueta}: sin señales evaluables)")
            continue
        reportar_rendimiento(tramo, etiqueta)


if __name__ == "__main__":
    print("Analizando acciones IPSA con metodología Trading Latino (señal actual)...\n")
    tabla = analizar_universo()
    if not tabla.empty:
        print(tabla.to_string(index=False))
        tabla.to_csv("señales_ipsa.csv", index=False)
        print("\nGuardado en señales_ipsa.csv")

    print("\n" + "=" * 70)
    print("Construyendo historial de señales (hasta 10 años, según disponibilidad)...")
    print("=" * 70)
    historial_nuevo = construir_historial_universo()
    if not historial_nuevo.empty:
        resumen = historial_nuevo.groupby(["ticker", "tipo"]).size().unstack(fill_value=0)
        print("\nResumen: cantidad de señales detectadas en esta corrida (últimos 10 años)")
        print(resumen.to_string())

        # Fusiona con lo que ya estaba guardado en disco (si existe), sin
        # borrar nunca señales antiguas que ya no estén en la ventana de 10y.
        historial = acumular_historial_csv(historial_nuevo)
        print(f"\nHistorial ACUMULADO en disco: {len(historial)} señales en total "
              f"(de las cuales {len(historial_nuevo)} vienen de esta corrida)")

        print("\nÚltimas 20 señales (todo el historial acumulado, orden cronológico real):")
        ultimas_cronologicas = historial.sort_values("fecha").tail(20)
        print(ultimas_cronologicas.to_string(index=False))

        print(f"\nGuardado/actualizado en {CSV_HISTORIAL}")
        print("Ábrelo en Excel/Sheets y filtra por 'ticker' para revisar una")
        print("acción específica y comparar esas fechas contra un gráfico.")

        # ------------------------------------------------------------------
        # RENDIMIENTO OBJETIVO: % de aciertos y retorno promedio
        # Se calcula sobre el HISTORIAL ACUMULADO completo (no solo lo nuevo
        # de esta corrida), para que la estadística vaya siendo cada vez más
        # sólida con el tiempo en vez de reiniciarse cada vez.
        # (solo señales con al menos 10 días hábiles ya transcurridos)
        # ------------------------------------------------------------------
        evaluables = historial.dropna(subset=["acierto_10d"]).copy()
        evaluables["acierto_10d"] = evaluables["acierto_10d"].astype(bool)
        if not evaluables.empty:
            print("\n" + "=" * 70)
            print("RENDIMIENTO (evaluado a 10 días hábiles después de cada señal,")
            print("sobre TODO el historial acumulado hasta ahora)")
            print("=" * 70)
            reportar_rendimiento(evaluables, "TOTAL")

            print(f"\n  TOTAL: {len(evaluables)} señales evaluables | "
                  f"acierto global: {evaluables['acierto_10d'].mean()*100:.1f}%")
            print("\n  (Un acierto del 50% no significa que el sistema no sirva:")
            print("   lo que importa también es cuánto se gana cuando acierta vs.")
            print("   cuánto se pierde cuando falla. Revisa 'retorno_5d_%',")
            print("   'retorno_10d_%' y 'retorno_20d_%' en el CSV para ver la")
            print("   distribución completa, no solo el promedio.)")

            # --------------------------------------------------------------
            # BENCHMARK: ¿le gana la señal a simplemente tener la acción
            # en cualquier día al azar? Esto es la prueba real de si el
            # filtro técnico aporta algo por sobre la tendencia general
            # del mercado en el período analizado.
            # --------------------------------------------------------------
            print("\n" + "=" * 70)
            print("BENCHMARK: retorno promedio 'comprando cualquier día' (sin señal)")
            print("=" * 70)
            benchmark = calcular_benchmark()
            b10 = benchmark.get(10, {})
            if b10.get("promedio") is not None:
                print(f"  Cualquier día al azar, 10 días después: "
                      f"{b10['promedio']:+.2f}% (sobre {b10['n']} observaciones)")

                retorno_compra_10d = evaluables[evaluables["tipo"] == "COMPRA"]["retorno_10d_%"].mean()
                retorno_venta_10d = evaluables[evaluables["tipo"] == "VENTA"]["retorno_10d_%"].mean()

                print(f"\n  COMPARACIÓN a 10 días:")
                print(f"    Benchmark (cualquier día):        {b10['promedio']:+.2f}%")
                if pd.notna(retorno_compra_10d):
                    diff_compra = retorno_compra_10d - b10["promedio"]
                    veredicto = "SÍ le gana al benchmark" if diff_compra > 0 else "NO le gana al benchmark"
                    print(f"    Señal COMPRA:                     {retorno_compra_10d:+.2f}%  "
                          f"({diff_compra:+.2f} pts vs benchmark → {veredicto})")
                if pd.notna(retorno_venta_10d):
                    print(f"    Señal VENTA:                      {retorno_venta_10d:+.2f}%  "
                          f"(referencia: si esto es MENOR al benchmark, tiene sentido como")
                    print(f"                                        señal de salida/corto; si es MAYOR,")
                    print(f"                                        significa que igual siguió subiendo)")

                print("\n  Nota: este benchmark mezcla los mismos 10 años para todas las")
                print("  acciones, así que si el IPSA en general tuvo un período muy alcista,")
                print("  el benchmark también va a ser un número positivo grande. Lo que")
                print("  importa es la DIFERENCIA entre la señal y el benchmark, no el")
                print("  número de la señal solo.")

            analizar_por_periodos(historial)
    else:
        print("No se encontraron señales históricas en esta corrida.")
