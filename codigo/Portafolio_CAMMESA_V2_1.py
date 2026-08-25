# -*- coding: utf-8 -*-
"""
Portafolio CAMMESA - V2.1 

Refactor controlado del proyecto original de demanda horaria provincial.
No reemplaza ni modifica el archivo original.

Objetivos:
- corregir la carga del Excel sin eliminar observaciones válidas;
- validar calidad e integridad de la base;
- concentrar el análisis en métricas de alto valor;
- incorporar explícitamente el perfil horario;
- comparar modelos de pronóstico contra un baseline estacional;
- evaluar pronósticos con MAE, RMSE, MAPE, WAPE y sesgo;
- priorizar una salida ejecutiva: 6 gráficos principales, 1 análisis avanzado y 1 anexo metodológico.
- separar métricas protagonistas de evidencia técnica secundaria.
"""

from __future__ import annotations

import argparse
import calendar
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, silhouette_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


COLUMNAS = [
    "AÑO", "MES", "N_MES", "N_DIA", "TIPO_DIA", "DIA", "FECHA", "HORA",
    "BUENOS_AIRES", "CATAMARCA", "CHACO", "CHUBUT", "CORDOBA", "CORRIENTES",
    "ENTRE_RIOS", "FORMOSA", "JUJUY", "LA_PAMPA", "LA_RIOJA", "MENDOZA",
    "MISIONES", "NEUQUEN", "RIO_NEGRO", "SALTA", "SAN_JUAN", "SAN_LUIS",
    "SANTA_CRUZ", "SANTA_FE", "SGO_DEL_ESTERO", "TUCUMAN", "TOTAL"
]

PROVINCIAS = COLUMNAS[8:-1]
ANIOS_COMPLETOS_ANALISIS = [2023, 2024, 2025]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def preparar_carpetas(output_dir: Path) -> tuple[Path, Path]:
    tablas = output_dir / "tablas"
    figuras = output_dir / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    return tablas, figuras


def cargar_datos(ruta_excel: Path) -> pd.DataFrame:
    """Carga el Excel desde la fila correcta de encabezados.

    El archivo original contiene tres filas de título/metadata antes de los encabezados.
    Por eso se usa header=3. Esto evita la lógica anterior que descartaba dos registros
    horarios válidos del 01/01/2023.
    """
    df = pd.read_excel(ruta_excel, header=3)

    if df.shape[1] != len(COLUMNAS):
        raise ValueError(
            f"Se esperaban {len(COLUMNAS)} columnas y se encontraron {df.shape[1]}. "
            "Revisar estructura del Excel."
        )

    df.columns = COLUMNAS
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="raise")
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="raise").astype(int)
    df["N_MES"] = pd.to_numeric(df["N_MES"], errors="raise").astype(int)
    df["HORA"] = pd.to_numeric(df["HORA"], errors="raise").astype(int)

    for col in PROVINCIAS + ["TOTAL"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Normalización textual de tipo de día sin alterar el dato original de fecha/hora.
    tipo = (
        df["TIPO_DIA"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    df["TIPO_DIA"] = np.where(
        tipo.str.contains("no") & tipo.str.contains("hábil|habil", regex=True),
        "No Hábil",
        np.where(
            tipo.str.contains("hábil|habil", regex=True),
            "Hábil",
            df["TIPO_DIA"].astype(str).str.strip(),
        ),
    )

    return df


def auditar_calidad(df: pd.DataFrame, tablas_dir: Path) -> dict:
    duplicados = int(df.duplicated(subset=["FECHA", "HORA"]).sum())
    nulos_demanda = int(df[PROVINCIAS + ["TOTAL"]].isna().sum().sum())
    negativos = int((df[PROVINCIAS + ["TOTAL"]] < 0).sum().sum())
    ceros = int((df[PROVINCIAS] == 0).sum().sum())
    diferencia_total = (df[PROVINCIAS].sum(axis=1) - df["TOTAL"]).abs()

    conteo_horas_fecha = df.groupby("FECHA")["HORA"].nunique()
    dias_incompletos = conteo_horas_fecha[conteo_horas_fecha != 24]

    anomalias_cero = []
    for provincia in PROVINCIAS:
        mask = df[provincia] == 0
        if mask.any():
            aux = df.loc[mask, ["FECHA", "HORA", provincia]].copy()
            aux["PROVINCIA"] = provincia
            aux = aux.rename(columns={provincia: "VALOR"})
            anomalias_cero.append(aux[["FECHA", "HORA", "PROVINCIA", "VALOR"]])

    if anomalias_cero:
        pd.concat(anomalias_cero, ignore_index=True).to_csv(
            tablas_dir / "anomalias_valores_cero.csv", index=False
        )

    resumen = {
        "filas": int(len(df)),
        "fecha_min": str(df["FECHA"].min().date()),
        "fecha_max": str(df["FECHA"].max().date()),
        "timestamps_duplicados": duplicados,
        "nulos_demanda": nulos_demanda,
        "valores_negativos": negativos,
        "valores_cero_provinciales": ceros,
        "dias_con_horas_incompletas": int(len(dias_incompletos)),
        "diferencia_max_total_vs_suma_provincias": float(diferencia_total.max()),
        "registros_por_anio": {str(k): int(v) for k, v in df.groupby("AÑO").size().items()},
    }

    pd.DataFrame([resumen]).to_csv(tablas_dir / "auditoria_calidad.csv", index=False)
    return resumen


def calcular_anual(df: pd.DataFrame, tablas_dir: Path) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    base = df[df["AÑO"].isin(ANIOS_COMPLETOS_ANALISIS)].copy()

    nacional_mwh = base.groupby("AÑO")["TOTAL"].sum()
    nacional_twh = nacional_mwh / 1_000_000
    variacion_nacional = nacional_twh.pct_change() * 100

    provincial = base.groupby("AÑO")[PROVINCIAS].sum()
    participacion = provincial.div(provincial.sum(axis=1), axis=0) * 100

    crecimiento_23_25 = (
        (provincial.loc[2025] - provincial.loc[2023]) / provincial.loc[2023] * 100
    ).sort_values(ascending=False)

    tabla_nacional = pd.DataFrame({
        "DEMANDA_TWH": nacional_twh,
        "VARIACION_INTERANUAL_%": variacion_nacional,
    })
    tabla_nacional.to_csv(tablas_dir / "demanda_nacional_anual.csv")

    participacion.to_csv(tablas_dir / "participacion_provincial_anual.csv")
    crecimiento_23_25.rename("CRECIMIENTO_2023_2025_%").to_csv(
        tablas_dir / "crecimiento_provincial_2023_2025.csv"
    )

    return nacional_twh, participacion, crecimiento_23_25.to_frame()


def promedio_diario_mensual(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    datos = df[df["AÑO"] == anio]
    acumulado = datos.groupby("N_MES")[PROVINCIAS].sum()
    dias_observados = datos.groupby("N_MES")["FECHA"].nunique()
    return acumulado.div(dias_observados, axis=0)


def calcular_mensual(df: pd.DataFrame, tablas_dir: Path) -> dict[int, pd.DataFrame]:
    resultado = {}
    for anio in sorted(df["AÑO"].unique()):
        mensual = promedio_diario_mensual(df, int(anio))
        resultado[int(anio)] = mensual
        mensual.to_csv(tablas_dir / f"promedio_diario_mensual_{int(anio)}.csv")
    return resultado


def analizar_tipo_dia(df: pd.DataFrame, tablas_dir: Path) -> pd.DataFrame:
    diario = (
        df.groupby(["AÑO", "FECHA", "TIPO_DIA"], as_index=False)["TOTAL"]
        .sum()
        .rename(columns={"TOTAL": "CONSUMO_DIARIO_MWH"})
    )

    resumen = (
        diario.groupby(["AÑO", "TIPO_DIA"])["CONSUMO_DIARIO_MWH"]
        .agg(CANTIDAD_DIAS="size", MEDIA="mean", MEDIANA="median", DESVIO="std")
        .reset_index()
    )
    resumen["CV_%"] = resumen["DESVIO"] / resumen["MEDIA"] * 100

    filas = []
    for anio, grupo in resumen.groupby("AÑO"):
        g = grupo.set_index("TIPO_DIA")
        if {"Hábil", "No Hábil"}.issubset(g.index):
            media_h = g.loc["Hábil", "MEDIA"]
            media_nh = g.loc["No Hábil", "MEDIA"]
            reduccion = (media_h - media_nh) / media_h * 100
            filas.append({
                "AÑO": int(anio),
                "MEDIA_HABIL": media_h,
                "MEDIA_NO_HABIL": media_nh,
                "REDUCCION_NO_HABIL_%": reduccion,
            })

    comparacion = pd.DataFrame(filas)
    resumen.to_csv(tablas_dir / "tipo_dia_estadisticas.csv", index=False)
    comparacion.to_csv(tablas_dir / "tipo_dia_comparacion.csv", index=False)
    return comparacion


def analizar_horario(df: pd.DataFrame, tablas_dir: Path) -> tuple[pd.DataFrame, dict]:
    perfil = df.groupby("HORA")["TOTAL"].agg(["mean", "median", "std"]).reset_index()
    perfil = perfil.rename(columns={"mean": "MEDIA", "median": "MEDIANA", "std": "DESVIO"})

    por_tipo = (
        df.groupby(["HORA", "TIPO_DIA"])["TOTAL"]
        .mean()
        .unstack("TIPO_DIA")
        .reset_index()
    )

    hora_valle = int(perfil.loc[perfil["MEDIA"].idxmin(), "HORA"])
    valor_valle = float(perfil["MEDIA"].min())
    hora_pico = int(perfil.loc[perfil["MEDIA"].idxmax(), "HORA"])
    valor_pico = float(perfil["MEDIA"].max())

    idx_pico_abs = df["TOTAL"].idxmax()
    idx_valle_abs = df["TOTAL"].idxmin()

    resumen = {
        "hora_pico_promedio": hora_pico,
        "demanda_hora_pico_promedio_mwh": valor_pico,
        "hora_valle_promedio": hora_valle,
        "demanda_hora_valle_promedio_mwh": valor_valle,
        "amplitud_pico_valle_%": (valor_pico / valor_valle - 1) * 100,
        "pico_absoluto_fecha": str(df.loc[idx_pico_abs, "FECHA"].date()),
        "pico_absoluto_hora": int(df.loc[idx_pico_abs, "HORA"]),
        "pico_absoluto_mwh": float(df.loc[idx_pico_abs, "TOTAL"]),
        "valle_absoluto_fecha": str(df.loc[idx_valle_abs, "FECHA"].date()),
        "valle_absoluto_hora": int(df.loc[idx_valle_abs, "HORA"]),
        "valle_absoluto_mwh": float(df.loc[idx_valle_abs, "TOTAL"]),
    }

    perfil.to_csv(tablas_dir / "perfil_horario_nacional.csv", index=False)
    por_tipo.to_csv(tablas_dir / "perfil_horario_por_tipo_dia.csv", index=False)
    pd.DataFrame([resumen]).to_csv(tablas_dir / "perfil_horario_resumen.csv", index=False)
    return por_tipo, resumen


def validar_clustering(mensuales: dict[int, pd.DataFrame], tablas_dir: Path) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    anios = ANIOS_COMPLETOS_ANALISIS
    matrices = {}
    etiquetas = {}
    filas = []

    for anio in anios:
        mensual = mensuales[anio]
        relativo = mensual.div(mensual.mean(axis=0), axis=1)
        X = relativo.T.astype(float)
        matrices[anio] = X
        Z = linkage(X, method="ward")
        for k in range(2, 7):
            labels = fcluster(Z, t=k, criterion="maxclust")
            etiquetas[(anio, k)] = labels
            filas.append({
                "AÑO": anio,
                "K": k,
                "SILHOUETTE": silhouette_score(X, labels),
            })

    detalle = pd.DataFrame(filas)
    resumen = []
    for k in range(2, 7):
        sil = detalle.loc[detalle["K"] == k, "SILHOUETTE"].mean()
        ari = np.mean([
            adjusted_rand_score(etiquetas[(2023, k)], etiquetas[(2024, k)]),
            adjusted_rand_score(etiquetas[(2024, k)], etiquetas[(2025, k)]),
            adjusted_rand_score(etiquetas[(2023, k)], etiquetas[(2025, k)]),
        ])
        resumen.append({
            "K": k,
            "SILHOUETTE_PROMEDIO": sil,
            "ARI_PROMEDIO": ari,
            "PUNTAJE_COMPUESTO": (sil + ari) / 2,
        })

    calidad = pd.DataFrame(resumen)
    k_final = int(calidad.loc[calidad["PUNTAJE_COMPUESTO"].idxmax(), "K"])

    # Perfil relativo promedio multianual para una clasificación final estable.
    relativos = []
    for anio in anios:
        rel = mensuales[anio].div(mensuales[anio].mean(axis=0), axis=1).copy()
        rel.index = rel.index.astype(int)
        relativos.append(rel)
    promedio_relativo = sum(relativos) / len(relativos)
    X_final = promedio_relativo.T.astype(float)
    Z_final = linkage(X_final, method="ward")
    labels_final = fcluster(Z_final, t=k_final, criterion="maxclust")

    clasificacion = pd.DataFrame({
        "PROVINCIA": X_final.index,
        "CLUSTER": labels_final,
    }).sort_values(["CLUSTER", "PROVINCIA"]).reset_index(drop=True)

    detalle.to_csv(tablas_dir / "clustering_silhouette_por_anio.csv", index=False)
    calidad.to_csv(tablas_dir / "clustering_calidad_k.csv", index=False)
    clasificacion.to_csv(tablas_dir / "clustering_clasificacion_final.csv", index=False)
    promedio_relativo.to_csv(tablas_dir / "clustering_perfil_relativo_promedio.csv")

    return k_final, calidad, clasificacion


def metricas_pronostico(real: np.ndarray, pred: np.ndarray) -> dict:
    real = np.asarray(real, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    mask = np.isfinite(real) & np.isfinite(pred) & (real != 0)
    real = real[mask]
    pred = pred[mask]
    if len(real) == 0:
        return {k: np.nan for k in ["MAE", "RMSE", "MAPE_%", "WAPE_%", "BIAS_%"]}

    error = pred - real
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_%": float(np.mean(np.abs(error / real)) * 100),
        "WAPE_%": float(np.sum(np.abs(error)) / np.sum(np.abs(real)) * 100),
        "BIAS_%": float(np.sum(error) / np.sum(real) * 100),
    }


def serie_mensual_con_fechas(mensuales: dict[int, pd.DataFrame], anios: list[int]) -> pd.DataFrame:
    partes = []
    for anio in anios:
        d = mensuales[anio].copy()
        d.index = pd.to_datetime([f"{anio}-{int(m):02d}-01" for m in d.index])
        partes.append(d)
    return pd.concat(partes).sort_index()


def pronostico_seasonal_naive(serie: pd.Series, fechas: pd.DatetimeIndex) -> np.ndarray:
    hist = serie.sort_index().astype(float)
    salida = []
    for fecha in fechas:
        ref = fecha - pd.DateOffset(years=1)
        if ref not in hist.index:
            raise ValueError(f"No existe lag estacional para {fecha.date()}")
        salida.append(float(hist.loc[ref]))
    return np.asarray(salida)


def pronostico_ets(serie: pd.Series, horizonte: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            serie.astype(float).sort_index(),
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        ajuste = modelo.fit(optimized=True, use_brute=True)
        return np.maximum(np.asarray(ajuste.forecast(horizonte), dtype=float), 0)


def pronostico_sarima(serie: pd.Series, horizonte: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            serie.astype(float).sort_index(),
            order=(1, 0, 0),
            seasonal_order=(1, 1, 0, 12),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        ajuste = modelo.fit(disp=False, maxiter=500)
        return np.maximum(np.asarray(ajuste.forecast(horizonte), dtype=float), 0)


VARIABLES_XGB = [
    "MES", "MES_SIN", "MES_COS", "TENDENCIA", "LAG_1", "LAG_2", "LAG_12", "MEDIA_MOVIL_3"
]


def crear_variables_xgb(serie: pd.Series) -> pd.DataFrame:
    tabla = pd.DataFrame({"CONSUMO": serie.astype(float).sort_index()})
    tabla["MES"] = tabla.index.month
    tabla["MES_SIN"] = np.sin(2 * np.pi * tabla["MES"] / 12)
    tabla["MES_COS"] = np.cos(2 * np.pi * tabla["MES"] / 12)
    tabla["TENDENCIA"] = np.arange(len(tabla))
    tabla["LAG_1"] = tabla["CONSUMO"].shift(1)
    tabla["LAG_2"] = tabla["CONSUMO"].shift(2)
    tabla["LAG_12"] = tabla["CONSUMO"].shift(12)
    tabla["MEDIA_MOVIL_3"] = tabla["CONSUMO"].shift(1).rolling(3).mean()
    return tabla.dropna()


def pronostico_xgboost(serie: pd.Series, fechas: pd.DatetimeIndex) -> np.ndarray:
    serie = serie.astype(float).sort_index()
    tabla = crear_variables_xgb(serie)
    if len(tabla) < 10:
        raise ValueError("Muestra supervisada demasiado pequeña para XGBoost.")

    modelo = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        min_child_weight=2,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(tabla[VARIABLES_XGB], tabla["CONSUMO"])

    historial = serie.copy()
    salida = []
    for fecha in fechas:
        mes = fecha.month
        fila = pd.DataFrame({
            "MES": [mes],
            "MES_SIN": [np.sin(2 * np.pi * mes / 12)],
            "MES_COS": [np.cos(2 * np.pi * mes / 12)],
            "TENDENCIA": [len(historial)],
            "LAG_1": [historial.iloc[-1]],
            "LAG_2": [historial.iloc[-2]],
            "LAG_12": [historial.iloc[-12]],
            "MEDIA_MOVIL_3": [historial.iloc[-3:].mean()],
        })
        pred = max(float(modelo.predict(fila[VARIABLES_XGB])[0]), 0)
        salida.append(pred)
        historial.loc[fecha] = pred
    return np.asarray(salida)


def evaluar_pronosticos(mensuales: dict[int, pd.DataFrame], tablas_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    modelos = ["SeasonalNaive", "ETS", "SARIMA", "XGBoost"]
    folds = [
        {"nombre": "Validacion_2025", "train": [2023, 2024], "test_anio": 2025, "meses": 6},
        {"nombre": "Validacion_2026", "train": [2023, 2024, 2025], "test_anio": 2026, "meses": 6},
    ]

    filas_metricas = []
    predicciones_fold = {}

    for fold in folds:
        train = serie_mensual_con_fechas(mensuales, fold["train"])
        test_completo = serie_mensual_con_fechas(mensuales, [fold["test_anio"]])
        test = test_completo.iloc[: fold["meses"]].copy()
        fechas = test.index

        pred_modelos = {
            m: pd.DataFrame(index=fechas, columns=PROVINCIAS, dtype=float)
            for m in modelos
        }

        for provincia in PROVINCIAS:
            serie = train[provincia].dropna()
            for modelo in modelos:
                try:
                    if modelo == "SeasonalNaive":
                        pred = pronostico_seasonal_naive(serie, fechas)
                    elif modelo == "ETS":
                        pred = pronostico_ets(serie, len(fechas))
                    elif modelo == "SARIMA":
                        pred = pronostico_sarima(serie, len(fechas))
                    else:
                        pred = pronostico_xgboost(serie, fechas)
                    pred_modelos[modelo][provincia] = pred
                except Exception:
                    pred_modelos[modelo][provincia] = np.nan

        for modelo in modelos:
            met = metricas_pronostico(test[PROVINCIAS].to_numpy(), pred_modelos[modelo][PROVINCIAS].to_numpy())
            filas_metricas.append({"FOLD": fold["nombre"], "MODELO": modelo, **met})

            # Métricas por provincia para auditoría, no para saturar el informe.
            prov_rows = []
            for provincia in PROVINCIAS:
                m = metricas_pronostico(test[provincia].to_numpy(), pred_modelos[modelo][provincia].to_numpy())
                prov_rows.append({"PROVINCIA": provincia, "MODELO": modelo, **m})
            pd.DataFrame(prov_rows).to_csv(
                tablas_dir / f"forecast_metricas_provincia_{fold['nombre']}_{modelo}.csv",
                index=False,
            )

        predicciones_fold[fold["nombre"]] = {
            "real": test,
            **pred_modelos,
        }

    metricas = pd.DataFrame(filas_metricas)

    # Resultado conjunto de robustez: promedio simple de los dos folds en las métricas porcentuales.
    robustez = (
        metricas.groupby("MODELO", as_index=False)[["MAPE_%", "WAPE_%", "BIAS_%"]]
        .mean()
        .rename(columns={
            "MAPE_%": "MAPE_PROMEDIO_FOLDS_%",
            "WAPE_%": "WAPE_PROMEDIO_FOLDS_%",
            "BIAS_%": "BIAS_PROMEDIO_FOLDS_%",
        })
        .sort_values("WAPE_PROMEDIO_FOLDS_%")
    )

    metricas.to_csv(tablas_dir / "forecast_metricas_por_fold.csv", index=False)
    robustez.to_csv(tablas_dir / "forecast_robustez_modelos.csv", index=False)

    # Exportar comparación nacional del holdout 2026.
    valid26 = predicciones_fold["Validacion_2026"]
    comp = pd.DataFrame(index=valid26["real"].index)
    comp["REAL"] = valid26["real"][PROVINCIAS].sum(axis=1)
    for modelo in modelos:
        comp[modelo] = valid26[modelo][PROVINCIAS].sum(axis=1)
    comp.index.name = "FECHA_MES"
    comp.to_csv(tablas_dir / "forecast_nacional_2026.csv")

    return metricas, predicciones_fold


def guardar_figura(fig, ruta: Path) -> None:
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generar_figuras(
    df: pd.DataFrame,
    figuras_dir: Path,
    nacional_twh: pd.Series,
    participacion: pd.DataFrame,
    crecimiento: pd.DataFrame,
    mensuales: dict[int, pd.DataFrame],
    tipo_dia: pd.DataFrame,
    perfil_tipo: pd.DataFrame,
    calidad_cluster: pd.DataFrame,
    k_final: int,
    clasificacion_cluster: pd.DataFrame,
    metricas_forecast: pd.DataFrame,
    predicciones_fold: dict[str, dict],
) -> None:
    """Genera solo las visualizaciones con función analítica clara.

    Estructura:
    - principal: 6 gráficos que responden preguntas centrales del informe;
    - analisis_avanzado: clustering interpretativo;
    - anexos_metodologicos: evidencia para justificar decisiones técnicas.
    """
    principal = figuras_dir / "principal"
    avanzado = figuras_dir / "analisis_avanzado"
    anexos = figuras_dir / "anexos_metodologicos"
    for carpeta in [principal, avanzado, anexos]:
        carpeta.mkdir(parents=True, exist_ok=True)

    # 1. Participación provincial 2025: top 10.
    top = participacion.loc[2025].sort_values(ascending=False).head(10).sort_values()
    top5 = participacion.loc[2025].sort_values(ascending=False).head(5).sum()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    top.plot(kind="barh", ax=ax)
    ax.set_title(
        "Participación provincial en la demanda nacional — 2025\n"
        f"Las 5 principales concentran {top5:.1f}% del total"
    )
    ax.set_xlabel("Participación (%)")
    ax.set_ylabel("Provincia")
    for i, valor in enumerate(top.values):
        ax.text(valor + 0.25, i, f"{valor:.1f}%", va="center", fontsize=9)
    guardar_figura(fig, principal / "01_participacion_provincial_2025.png")

    # 2. Cambio provincial 2023-2025: todas las provincias.
    crec = crecimiento.iloc[:, 0].sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    crec.plot(kind="barh", ax=ax)
    ax.axvline(0, linewidth=1)
    ax.set_title("Variación acumulada de la demanda provincial — 2023 a 2025")
    ax.set_xlabel("Variación (%)")
    ax.set_ylabel("Provincia")
    guardar_figura(fig, principal / "02_variacion_provincial_2023_2025.png")

    # 3. Estacionalidad mensual nacional usando promedio diario mensual.
    fig, ax = plt.subplots(figsize=(10, 6))
    for anio in ANIOS_COMPLETOS_ANALISIS:
        serie = mensuales[anio][PROVINCIAS].sum(axis=1)
        ax.plot(serie.index, serie.values, marker="o", label=str(anio))
    ax.set_xticks(range(1, 13), MESES)
    ax.set_title("Estacionalidad mensual de la demanda nacional")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Promedio diario mensual (MWh/día)")
    ax.legend(title="Año")
    guardar_figura(fig, principal / "03_estacionalidad_mensual.png")

    # 4. Perfil horario: total + hábil/no hábil, con pico y valle del promedio total.
    perfil_total = df.groupby("HORA")["TOTAL"].mean()
    hora_valle = int(perfil_total.idxmin())
    hora_pico = int(perfil_total.idxmax())
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(perfil_total.index, perfil_total.values, marker="o", linewidth=2.5, label="Promedio total")
    for col in [c for c in perfil_tipo.columns if c != "HORA" and c in {"Hábil", "No Hábil"}]:
        ax.plot(perfil_tipo["HORA"], perfil_tipo[col], marker="o", linewidth=1.5, label=col)
    ax.scatter([hora_valle], [perfil_total.loc[hora_valle]], s=55, zorder=5)
    ax.scatter([hora_pico], [perfil_total.loc[hora_pico]], s=55, zorder=5)
    ax.annotate(
        f"Valle medio: {hora_valle:02d}:00",
        (hora_valle, perfil_total.loc[hora_valle]),
        xytext=(hora_valle + 1, perfil_total.loc[hora_valle] - 900),
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        f"Pico medio: {hora_pico:02d}:00",
        (hora_pico, perfil_total.loc[hora_pico]),
        xytext=(hora_pico - 6, perfil_total.loc[hora_pico] + 650),
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_xticks(sorted(df["HORA"].unique()))
    ax.set_title("Perfil horario medio de demanda — hábil vs. no hábil")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Demanda media horaria (MWh)")
    ax.legend()
    guardar_figura(fig, principal / "04_perfil_horario_habil_no_habil.png")

    # 5. Error WAPE por modelo y período de validación.
    pivot = metricas_forecast.pivot(index="MODELO", columns="FOLD", values="WAPE_%")
    orden = (
        metricas_forecast.groupby("MODELO")["WAPE_%"].mean().sort_values().index
    )
    pivot = pivot.loc[orden]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Error fuera de muestra de los modelos de pronóstico")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("WAPE (%) — menor es mejor")
    ax.legend(title="Período de validación")
    guardar_figura(fig, principal / "05_error_wape_modelos.png")

    # 6. Real vs. pronosticado 2026: solo real, modelo más robusto y mejor holdout.
    val = predicciones_fold["Validacion_2026"]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    real_nac = val["real"][PROVINCIAS].sum(axis=1)
    ax.plot(real_nac.index, real_nac.values, marker="o", linewidth=3, label="Real")
    for modelo in ["ETS", "XGBoost"]:
        pred_nac = val[modelo][PROVINCIAS].sum(axis=1)
        ax.plot(pred_nac.index, pred_nac.values, marker="o", linewidth=2, label=modelo)
    ax.set_title("Demanda real vs. pronosticada — enero a junio 2026")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Promedio diario mensual (MWh/día)")
    ax.legend(title="Serie")
    guardar_figura(fig, principal / "06_real_vs_pronostico_2026.png")

    # Análisis avanzado: perfil mensual relativo de los clusters finales.
    relativos = []
    for anio in ANIOS_COMPLETOS_ANALISIS:
        rel = mensuales[anio].div(mensuales[anio].mean(axis=0), axis=1).copy()
        rel.index = rel.index.astype(int)
        relativos.append(rel)
    prom_rel = sum(relativos) / len(relativos)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in sorted(clasificacion_cluster["CLUSTER"].unique()):
        provs = clasificacion_cluster.loc[
            clasificacion_cluster["CLUSTER"] == cluster, "PROVINCIA"
        ].tolist()
        curva = prom_rel[provs].mean(axis=1)
        ax.plot(curva.index, curva.values, marker="o", label=f"Patrón {cluster}")
    ax.axhline(1, linestyle="--", linewidth=1)
    ax.set_xticks(range(1, 13), MESES)
    ax.set_title(f"Patrones estacionales provinciales — solución de {k_final} clusters")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Consumo relativo a la media anual")
    ax.legend(title="Cluster")
    guardar_figura(fig, avanzado / "01_perfiles_estacionales_clusters.png")

    # Anexo metodológico: justificación de K.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(calidad_cluster["K"], calidad_cluster["SILHOUETTE_PROMEDIO"], marker="o", label="Silhouette")
    ax.plot(calidad_cluster["K"], calidad_cluster["ARI_PROMEDIO"], marker="o", label="ARI estabilidad")
    ax.plot(calidad_cluster["K"], calidad_cluster["PUNTAJE_COMPUESTO"], marker="o", label="Puntaje compuesto")
    ax.axvline(k_final, linestyle="--", linewidth=1, label=f"K seleccionado = {k_final}")
    ax.set_title("Validación metodológica del número de clusters")
    ax.set_xlabel("Cantidad de clusters")
    ax.set_ylabel("Puntaje")
    ax.legend()
    guardar_figura(fig, anexos / "A01_validacion_numero_clusters.png")

def construir_resumen(
    calidad: dict,
    nacional_twh: pd.Series,
    participacion: pd.DataFrame,
    crecimiento: pd.DataFrame,
    tipo_dia: pd.DataFrame,
    horario: dict,
    k_final: int,
    metricas_forecast: pd.DataFrame,
    tablas_dir: Path,
) -> dict:
    top5_2025 = participacion.loc[2025].sort_values(ascending=False).head(5)
    robustez = (
        metricas_forecast.groupby("MODELO")["WAPE_%"]
        .mean()
        .sort_values()
    )
    val26 = (
        metricas_forecast[metricas_forecast["FOLD"] == "Validacion_2026"]
        .set_index("MODELO")["WAPE_%"]
        .sort_values()
    )

    resumen = {
        "calidad_datos": calidad,
        "demanda_twh": {str(k): float(v) for k, v in nacional_twh.items()},
        "variacion_nacional_2024_vs_2023_%": float((nacional_twh.loc[2024] / nacional_twh.loc[2023] - 1) * 100),
        "variacion_nacional_2025_vs_2024_%": float((nacional_twh.loc[2025] / nacional_twh.loc[2024] - 1) * 100),
        "variacion_nacional_2025_vs_2023_%": float((nacional_twh.loc[2025] / nacional_twh.loc[2023] - 1) * 100),
        "top5_participacion_2025_%": {k: float(v) for k, v in top5_2025.items()},
        "concentracion_top5_2025_%": float(top5_2025.sum()),
        "mayor_crecimiento_2023_2025": {
            "provincia": str(crecimiento.iloc[:, 0].idxmax()),
            "variacion_%": float(crecimiento.iloc[:, 0].max()),
        },
        "mayor_caida_2023_2025": {
            "provincia": str(crecimiento.iloc[:, 0].idxmin()),
            "variacion_%": float(crecimiento.iloc[:, 0].min()),
        },
        "reduccion_no_habil_%": {
            str(int(r["AÑO"])): float(r["REDUCCION_NO_HABIL_%"])
            for _, r in tipo_dia.iterrows()
        },
        "perfil_horario": horario,
        "clusters_seleccionados": int(k_final),
        "modelo_mas_robusto_por_wape_promedio": str(robustez.index[0]),
        "wape_promedio_folds_%": {k: float(v) for k, v in robustez.items()},
        "mejor_modelo_holdout_2026": str(val26.index[0]),
        "wape_holdout_2026_%": {k: float(v) for k, v in val26.items()},
    }

    with open(tablas_dir / "resumen_ejecutivo.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    return resumen



def crear_metricas_principales(resumen: dict, tablas_dir: Path) -> pd.DataFrame:
    """Crea una tabla breve de indicadores protagonistas del informe."""
    reducciones = [
        v for anio, v in resumen["reduccion_no_habil_%"].items()
        if anio in {"2023", "2024", "2025"}
    ]
    reduccion_media = float(np.mean(reducciones))
    h = resumen["perfil_horario"]
    mayor_crec = resumen["mayor_crecimiento_2023_2025"]
    mayor_caida = resumen["mayor_caida_2023_2025"]
    robusto = resumen["modelo_mas_robusto_por_wape_promedio"]
    mejor26 = resumen["mejor_modelo_holdout_2026"]

    filas = [
        {
            "METRICA": "Demanda nacional 2025",
            "VALOR": resumen["demanda_twh"]["2025"],
            "UNIDAD": "TWh",
            "DETALLE": "Nivel anual de referencia más reciente con año completo.",
        },
        {
            "METRICA": "Variación nacional 2025 vs 2024",
            "VALOR": resumen["variacion_nacional_2025_vs_2024_%"],
            "UNIDAD": "%",
            "DETALLE": f"Cambio acumulado 2023-2025: {resumen['variacion_nacional_2025_vs_2023_%']:.2f}%.",
        },
        {
            "METRICA": "Concentración Top 5 provincial 2025",
            "VALOR": resumen["concentracion_top5_2025_%"],
            "UNIDAD": "%",
            "DETALLE": "Participación conjunta de las cinco provincias de mayor demanda.",
        },
        {
            "METRICA": "Heterogeneidad provincial 2023-2025",
            "VALOR": mayor_crec["variacion_%"] - mayor_caida["variacion_%"],
            "UNIDAD": "p.p.",
            "DETALLE": (
                f"Mayor crecimiento: {mayor_crec['provincia']} {mayor_crec['variacion_%']:.2f}%; "
                f"mayor caída: {mayor_caida['provincia']} {mayor_caida['variacion_%']:.2f}%."
            ),
        },
        {
            "METRICA": "Reducción media en días no hábiles (2023-2025)",
            "VALOR": reduccion_media,
            "UNIDAD": "%",
            "DETALLE": "Diferencia media respecto de los días hábiles; patrón estable entre años.",
        },
        {
            "METRICA": "Amplitud del perfil horario medio",
            "VALOR": h["amplitud_pico_valle_%"],
            "UNIDAD": "%",
            "DETALLE": f"Valle medio {h['hora_valle_promedio']:02d}:00; pico medio {h['hora_pico_promedio']:02d}:00.",
        },
        {
            "METRICA": "Pico horario absoluto observado",
            "VALOR": h["pico_absoluto_mwh"],
            "UNIDAD": "MWh",
            "DETALLE": f"{h['pico_absoluto_fecha']} a las {h['pico_absoluto_hora']:02d}:00.",
        },
        {
            "METRICA": "Pronóstico — robustez fuera de muestra",
            "VALOR": resumen["wape_promedio_folds_%"][robusto],
            "UNIDAD": "WAPE %",
            "DETALLE": (
                f"Modelo más robusto: {robusto}. En el holdout 2026 el mejor fue {mejor26} "
                f"con {resumen['wape_holdout_2026_%'][mejor26]:.2f}% WAPE."
            ),
        },
    ]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(tablas_dir / "metricas_principales.csv", index=False)
    return tabla


def escribir_guia_resultados(output_dir: Path, resumen: dict) -> None:
    """Documenta qué salidas se usan en el informe y cuáles son soporte técnico."""
    contenido = f"""# CAMMESA V2.1 — Guía de resultados\n\n## Cuerpo principal\nSe priorizan seis visualizaciones, cada una asociada a una pregunta analítica distinta:\n\n1. `figuras/principal/01_participacion_provincial_2025.png` — ¿Dónde se concentra la demanda?\n2. `figuras/principal/02_variacion_provincial_2023_2025.png` — ¿Qué provincias crecieron o cayeron?\n3. `figuras/principal/03_estacionalidad_mensual.png` — ¿Cómo varía la demanda a lo largo del año?\n4. `figuras/principal/04_perfil_horario_habil_no_habil.png` — ¿Cómo cambia dentro del día y según tipo de día?\n5. `figuras/principal/05_error_wape_modelos.png` — ¿Qué modelo pronostica mejor fuera de muestra?\n6. `figuras/principal/06_real_vs_pronostico_2026.png` — ¿Qué tan cerca estuvo el pronóstico de los datos reales 2026?\n\nLa tabla `tablas/metricas_principales.csv` contiene los ocho KPI protagonistas.\n\n## Análisis avanzado\n`figuras/analisis_avanzado/01_perfiles_estacionales_clusters.png` resume los {resumen['clusters_seleccionados']} patrones provinciales seleccionados por calidad y estabilidad.\n\n## Anexos metodológicos\n`figuras/anexos_metodologicos/A01_validacion_numero_clusters.png` justifica la elección del número de clusters.\nLas tablas restantes conservan controles de calidad, métricas completas de pronóstico y resultados provinciales para auditoría y reproducibilidad.\n\n## Criterio de selección\nNo se generan gráficos separados para demanda anual ni para reducción hábil/no hábil porque esas conclusiones se comunican mejor como KPI y ya están contextualizadas por otras visualizaciones. Se conserva el cálculo completo en tablas.\n"""
    (output_dir / "GUIA_RESULTADOS.md").write_text(contenido, encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis auditado de demanda horaria provincial CAMMESA")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("Demanda Horaria por Provincia.xlsx"),
        help="Ruta al Excel de demanda horaria.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("CAMMESA_V2_1_RESULTADOS"),
        help="Directorio de salida.",
    )
    args = parser.parse_args()

    tablas_dir, figuras_dir = preparar_carpetas(args.output)

    print("[1/8] Cargando y normalizando datos...")
    df = cargar_datos(args.input)

    print("[2/8] Auditando calidad...")
    calidad = auditar_calidad(df, tablas_dir)

    print("[3/8] Calculando análisis anual y mensual...")
    nacional_twh, participacion, crecimiento = calcular_anual(df, tablas_dir)
    mensuales = calcular_mensual(df, tablas_dir)

    print("[4/8] Analizando tipo de día y perfil horario...")
    tipo_dia = analizar_tipo_dia(df, tablas_dir)
    perfil_tipo, horario = analizar_horario(df, tablas_dir)

    print("[5/8] Validando clustering...")
    k_final, calidad_cluster, clasificacion_cluster = validar_clustering(mensuales, tablas_dir)

    print("[6/8] Evaluando pronósticos con backtesting...")
    metricas_forecast, predicciones_fold = evaluar_pronosticos(mensuales, tablas_dir)

    print("[7/8] Generando visualizaciones priorizadas...")
    generar_figuras(
        df, figuras_dir, nacional_twh, participacion, crecimiento, mensuales,
        tipo_dia, perfil_tipo, calidad_cluster, k_final, clasificacion_cluster,
        metricas_forecast, predicciones_fold,
    )

    print("[8/8] Construyendo resumen ejecutivo reproducible...")
    resumen = construir_resumen(
        calidad, nacional_twh, participacion, crecimiento, tipo_dia,
        horario, k_final, metricas_forecast, tablas_dir,
    )
    crear_metricas_principales(resumen, tablas_dir)
    escribir_guia_resultados(args.output, resumen)

    print("\nProceso completado.")
    print(f"Datos: {calidad['filas']:,} registros")
    print(f"Clusters seleccionados por calidad + estabilidad: {k_final}")
    print(f"Modelo más robusto por WAPE promedio: {resumen['modelo_mas_robusto_por_wape_promedio']}")
    print(f"Mejor modelo en holdout 2026: {resumen['mejor_modelo_holdout_2026']}")
    print(f"Resultados: {args.output.resolve()}")


if __name__ == "__main__":
    main()
