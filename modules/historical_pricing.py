"""Backtesting histórico de precios con elasticidades ya calculadas.

Este módulo responde una pregunta histórica: ¿qué habría pasado en un periodo
pasado si el precio efectivo observado hubiera cambiado? No genera demanda base
futura ni recomendaciones prospectivas; solo re-simula periodos reales con las
elasticidades disponibles en ``elasticidades_periodo``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .elasticity import PERIODOS_ELASTICIDAD, build_period_blocks
from .pricing import _ensure_sku_alias, _fast_mode_by_group
from .utils import normalizar_categoria_est_socio, parse_transaction_dates

HISTORICAL_PRICE_SCENARIOS = [
    (-0.20, "bajar precio 20%"),
    (-0.15, "bajar precio 15%"),
    (-0.10, "bajar precio 10%"),
    (-0.05, "bajar precio 5%"),
    (0.00, "mantener precio"),
    (0.05, "subir precio 5%"),
    (0.10, "subir precio 10%"),
    (0.15, "subir precio 15%"),
    (0.20, "subir precio 20%"),
]

PRICING_HISTORICO_ESCENARIOS_COLUMNS = [
    "categoria",
    "departamento",
    "periodo_tipo",
    "periodo",
    "fecha_inicio",
    "fecha_fin",
    "SKU",
    "tipo_elasticidad_usada",
    "elasticidad_usada",
    "r2",
    "p_value",
    "confianza_elasticidad",
    "recomendable_elasticidad",
    "precio_real",
    "precio_lista",
    "precio_efectivo",
    "descuento_efectivo",
    "cambio_precio_pct",
    "unidades_reales",
    "unidades_simuladas",
    "ingreso_real",
    "ingreso_simulado",
    "margen_real",
    "margen_simulado",
    "variacion_unidades",
    "variacion_ingreso",
    "variacion_margen",
    "tipo_escenario",
    "nombre_escenario",
    "recomendacion_historica",
    "confianza",
    "razon_recomendacion",
]


def _mode_or_na(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return pd.NA
    values = df[col].replace(["", " ", "nan", "NaN", "None", "none"], np.nan).dropna()
    if values.empty:
        return pd.NA
    mode = values.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else values.iloc[0]


def _resolve_list_price(ventas: pd.DataFrame) -> pd.Series:
    """Devuelve precio lista por línea usando columnas disponibles o precio máximo por SKU."""
    candidates = [
        "precio_lista",
        "list_price",
        "regular_price",
        "precio_regular",
        "original_price",
        "unit_list_price",
    ]
    for col in candidates:
        if col in ventas.columns:
            price = pd.to_numeric(ventas[col], errors="coerce")
            if price.notna().any():
                return price
    return ventas.groupby("SKU", observed=True)["precio_unitario"].transform("max")


def _assign_periods(ventas: pd.DataFrame, periodo_tipo: str) -> pd.DataFrame:
    """Asigna periodo histórico a cada venta sin calcular elasticidad."""
    if periodo_tipo not in PERIODOS_ELASTICIDAD:
        return ventas.iloc[0:0].copy()

    out = ventas.copy()
    if periodo_tipo == "mensual":
        out["periodo"] = out["mes"].astype(str)
        out["fecha_inicio"] = out["mes"].dt.to_timestamp(how="start").dt.date
        out["fecha_fin"] = out["mes"].dt.to_timestamp(how="end").dt.date
        return out

    if periodo_tipo == "trimestral":
        rows = []
        for bloque in build_period_blocks(out, "trimestral"):
            for mes in bloque["meses"]:
                rows.append(
                    {
                        "mes": mes,
                        "periodo": bloque["periodo"],
                        "fecha_inicio": bloque["fecha_inicio"],
                        "fecha_fin": bloque["fecha_fin"],
                    }
                )
        mapa = pd.DataFrame(rows)
        return out.merge(mapa, on="mes", how="inner") if not mapa.empty else out.iloc[0:0].copy()

    if periodo_tipo == "semestral":
        rows = []
        for bloque in build_period_blocks(out, "semestral"):
            for mes in bloque["meses"]:
                rows.append(
                    {
                        "mes": mes,
                        "periodo": bloque["periodo"],
                        "fecha_inicio": bloque["fecha_inicio"],
                        "fecha_fin": bloque["fecha_fin"],
                    }
                )
        mapa = pd.DataFrame(rows)
        return out.merge(mapa, on="mes", how="inner") if not mapa.empty else out.iloc[0:0].copy()

    if periodo_tipo == "anual":
        out["periodo"] = out["mes"].dt.year.astype(str)
        out["fecha_inicio"] = out.groupby("periodo", observed=True)["mes"].transform("min").dt.to_timestamp(how="start").dt.date
        out["fecha_fin"] = out.groupby("periodo", observed=True)["mes"].transform("max").dt.to_timestamp(how="end").dt.date
        return out

    meses = sorted(out["mes"].dropna().unique())
    if not meses:
        return out.iloc[0:0].copy()
    out["periodo"] = periodo_tipo
    out["fecha_inicio"] = meses[0].to_timestamp(how="start").date()
    out["fecha_fin"] = meses[-1].to_timestamp(how="end").date()
    return out


def _prepare_historical_sales(ventas_nse: pd.DataFrame) -> pd.DataFrame:
    ventas = _ensure_sku_alias(ventas_nse.copy())
    ventas.columns = ventas.columns.astype(str).str.strip()
    ventas["tran_date"] = parse_transaction_dates(ventas["tran_date"])
    ventas["qty"] = pd.to_numeric(ventas["qty"], errors="coerce")
    ventas["net_sale"] = pd.to_numeric(ventas["net_sale"], errors="coerce")
    ventas["SKU"] = ventas["SKU"].astype("string").str.strip().astype(str)

    if "precio_unitario" not in ventas.columns:
        ventas["precio_unitario"] = ventas["net_sale"] / ventas["qty"]
    else:
        ventas["precio_unitario"] = pd.to_numeric(ventas["precio_unitario"], errors="coerce").fillna(ventas["net_sale"] / ventas["qty"])

    if "costo_unitario" not in ventas.columns:
        ventas["costo_unitario"] = pd.to_numeric(ventas.get("costo2", np.nan), errors="coerce")
    else:
        ventas["costo_unitario"] = pd.to_numeric(ventas["costo_unitario"], errors="coerce")

    ventas = ventas.replace([np.inf, -np.inf], np.nan).dropna(subset=["tran_date", "SKU", "qty", "net_sale", "precio_unitario"])
    ventas = ventas[(ventas["qty"] > 0) & (ventas["net_sale"] > 0) & (ventas["precio_unitario"] > 0)].copy()
    if ventas.empty:
        return ventas

    ventas["mes"] = ventas["tran_date"].dt.to_period("M")
    ventas["precio_lista_linea"] = _resolve_list_price(ventas)
    ventas["precio_lista_linea"] = pd.to_numeric(ventas["precio_lista_linea"], errors="coerce").fillna(ventas["precio_unitario"])
    ventas["costo_total_linea"] = ventas["costo_unitario"].fillna(0) * ventas["qty"]
    ventas["margen_total"] = ventas["net_sale"] - ventas["costo_total_linea"]
    if "categoria_est_socio" in ventas.columns:
        ventas["categoria_est_socio"] = ventas["categoria_est_socio"].apply(normalizar_categoria_est_socio)
    return ventas


def _aggregate_real_periods(ventas_periodo: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["SKU", "periodo_tipo", "periodo", "fecha_inicio", "fecha_fin"]
    agg = (
        ventas_periodo.groupby(group_cols, observed=True, sort=False)
        .agg(
            unidades_reales=("qty", "sum"),
            ingreso_real=("net_sale", "sum"),
            costo_total=("costo_total_linea", "sum"),
            margen_real=("margen_total", "sum"),
            precio_lista=("precio_lista_linea", "max"),
        )
        .reset_index()
    )
    agg["precio_real"] = agg["ingreso_real"] / agg["unidades_reales"]
    agg["costo_unitario"] = agg["costo_total"] / agg["unidades_reales"]
    agg = agg.replace([np.inf, -np.inf], np.nan)

    descriptor_map = {
        "subdept_nm": "categoria",
        "dept_nm": "departamento",
    }
    for src, dst in descriptor_map.items():
        if src in ventas_periodo.columns:
            modes = _fast_mode_by_group(ventas_periodo, group_cols, src).rename(columns={src: dst})
            agg = agg.merge(modes, on=group_cols, how="left")
        else:
            agg[dst] = pd.NA
    return agg


def _attach_elasticity(real: pd.DataFrame, elasticidades_tipo: pd.DataFrame, periodo_tipo: str) -> pd.DataFrame:
    elast = elasticidades_tipo.copy()
    elast["SKU"] = elast["SKU"].astype(str)
    base_cols = [
        "periodo_tipo",
        "periodo",
        "elasticidad",
        "r2",
        "p_value",
        "confianza_elasticidad",
        "recomendable_elasticidad",
        "razon_no_recomendable",
    ]

    if periodo_tipo == "categoria_departamento":
        cols = ["categoria", "departamento"] + base_cols
        return real.merge(elast[[c for c in cols if c in elast.columns]], on=["categoria", "departamento", "periodo_tipo", "periodo"], how="inner")

    cols = ["SKU"] + base_cols
    return real.merge(elast[[c for c in cols if c in elast.columns]], on=["SKU", "periodo_tipo", "periodo"], how="inner")


def _recommendation(row: pd.Series) -> tuple[str, str]:
    if pd.isna(row.get("elasticidad_usada")):
        return "Sin recomendación histórica", "No hay elasticidad calculada para este SKU-periodo."
    if not bool(row.get("recomendable_elasticidad", False)):
        reason = row.get("razon_no_recomendable")
        return "Mantener / revisar", f"Elasticidad de baja confianza: {reason}" if pd.notna(reason) else "Elasticidad de baja confianza."
    if row.get("variacion_margen", 0) > 0 and row.get("variacion_ingreso", 0) >= 0:
        return "Escenario históricamente favorable", "Mejora margen histórico simulado sin reducir ingreso."
    if row.get("variacion_margen", 0) > 0:
        return "Favorable solo en margen", "Mejora margen histórico simulado, pero puede afectar ingreso."
    return "No favorable", "No mejora margen histórico simulado frente al precio real."


def simulate_historical_pricing_scenarios(
    ventas_nse: pd.DataFrame,
    elasticidades_periodo: pd.DataFrame,
) -> pd.DataFrame:
    """Construye la tabla interna ``pricing_historico_escenarios``.

    Usa ventas reales del periodo histórico y elasticidades ya calculadas. No
    estima elasticidades nuevas ni proyecta una demanda base futura.
    """
    if ventas_nse is None or ventas_nse.empty or elasticidades_periodo is None or elasticidades_periodo.empty:
        return pd.DataFrame(columns=PRICING_HISTORICO_ESCENARIOS_COLUMNS)

    ventas = _prepare_historical_sales(ventas_nse)
    if ventas.empty:
        return pd.DataFrame(columns=PRICING_HISTORICO_ESCENARIOS_COLUMNS)

    elasticidades = elasticidades_periodo.copy().replace([np.inf, -np.inf], np.nan)
    required = {"periodo_tipo", "periodo", "elasticidad"}
    if not required.issubset(elasticidades.columns):
        return pd.DataFrame(columns=PRICING_HISTORICO_ESCENARIOS_COLUMNS)

    real_parts = []
    for periodo_tipo in sorted(elasticidades["periodo_tipo"].dropna().astype(str).unique()):
        ventas_periodo = _assign_periods(ventas, periodo_tipo)
        if ventas_periodo.empty:
            continue
        ventas_periodo["periodo_tipo"] = periodo_tipo
        real = _aggregate_real_periods(ventas_periodo)
        elast_tipo = elasticidades[elasticidades["periodo_tipo"].astype(str).eq(periodo_tipo)].copy()
        enriched = _attach_elasticity(real, elast_tipo, periodo_tipo)
        if not enriched.empty:
            enriched["tipo_elasticidad_usada"] = np.where(
                periodo_tipo == "categoria_departamento",
                "categoría-departamento",
                "SKU-" + periodo_tipo,
            )
            real_parts.append(enriched)

    if not real_parts:
        return pd.DataFrame(columns=PRICING_HISTORICO_ESCENARIOS_COLUMNS)

    base = pd.concat(real_parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    base = base.dropna(subset=["precio_real", "unidades_reales", "elasticidad"])
    base = base[(base["precio_real"] > 0) & (base["unidades_reales"] > 0)].copy()
    if base.empty:
        return pd.DataFrame(columns=PRICING_HISTORICO_ESCENARIOS_COLUMNS)

    esc = pd.DataFrame(HISTORICAL_PRICE_SCENARIOS, columns=["cambio_precio", "nombre_escenario"])
    esc["tipo_escenario"] = np.select(
        [esc["cambio_precio"] < 0, esc["cambio_precio"] > 0],
        ["bajar precio", "subir precio"],
        default="mantener precio",
    )
    base["_join_key"] = 1
    esc["_join_key"] = 1
    out = base.merge(esc, on="_join_key", how="left").drop(columns="_join_key")

    cambio = pd.to_numeric(out["cambio_precio"], errors="coerce")
    elasticidad = pd.to_numeric(out["elasticidad"], errors="coerce")
    precio_real = pd.to_numeric(out["precio_real"], errors="coerce")
    unidades_reales = pd.to_numeric(out["unidades_reales"], errors="coerce")
    costo_unitario = pd.to_numeric(out["costo_unitario"], errors="coerce").fillna(0)

    out["precio_efectivo"] = precio_real * (1 + cambio)
    out["unidades_simuladas"] = unidades_reales * np.exp(elasticidad * np.log1p(cambio))
    out["ingreso_simulado"] = out["precio_efectivo"] * out["unidades_simuladas"]
    out["margen_simulado"] = (out["precio_efectivo"] - costo_unitario) * out["unidades_simuladas"]
    out["descuento_efectivo"] = np.where(out["precio_lista"] > 0, 1 - (out["precio_efectivo"] / out["precio_lista"]), np.nan)
    out["cambio_precio_pct"] = cambio * 100
    out["variacion_unidades"] = out["unidades_simuladas"] - out["unidades_reales"]
    out["variacion_ingreso"] = out["ingreso_simulado"] - out["ingreso_real"]
    out["variacion_margen"] = out["margen_simulado"] - out["margen_real"]
    out["elasticidad_usada"] = out["elasticidad"]
    out["confianza"] = out.get("confianza_elasticidad", pd.Series([pd.NA] * len(out))).fillna("Sin confianza calculada")

    recs = out.apply(_recommendation, axis=1, result_type="expand")
    out["recomendacion_historica"] = recs[0]
    out["razon_recomendacion"] = recs[1]

    out = out.replace([np.inf, -np.inf], np.nan)
    for col in PRICING_HISTORICO_ESCENARIOS_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[PRICING_HISTORICO_ESCENARIOS_COLUMNS].reset_index(drop=True)
