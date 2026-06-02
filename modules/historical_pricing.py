"""Backtesting histórico de escenarios de precio.

Este módulo separa explícitamente el pricing histórico del pricing futuro:
usa ventas reales de periodos pasados y elasticidades ya calculadas en
``elasticidades_periodo`` para responder qué habría ocurrido si el precio
histórico hubiera cambiado. No estima demanda base futura ni pronósticos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ELASTICIDAD_CAP_MAX, ELASTICIDAD_CAP_MIN
from .promotions import escenarios_con_promociones, evaluar_riesgo_promocion, es_promocion
from .utils import build_quarter_label, parse_transaction_dates

PERIODOS_HISTORICOS = ["mensual", "trimestral", "semestral", "anual"]

ESCENARIOS_HISTORICOS = pd.DataFrame(
    [
        ("bajar_20", "bajar precio 20%", -0.20),
        ("bajar_15", "bajar precio 15%", -0.15),
        ("bajar_10", "bajar precio 10%", -0.10),
        ("bajar_5", "bajar precio 5%", -0.05),
        ("mantener", "mantener precio", 0.00),
        ("subir_5", "subir precio 5%", 0.05),
        ("subir_10", "subir precio 10%", 0.10),
        ("subir_15", "subir precio 15%", 0.15),
        ("subir_20", "subir precio 20%", 0.20),
    ],
    columns=["escenario_id", "nombre_escenario", "cambio_precio_pct"],
)
ESCENARIOS_HISTORICOS["tipo_escenario"] = "simple"
ESCENARIOS_HISTORICOS = escenarios_con_promociones(ESCENARIOS_HISTORICOS)

PRICING_HISTORICO_ESCENARIOS_COLUMNS = [
    "SKU",
    "categoria",
    "departamento",
    "periodo_tipo",
    "periodo",
    "fecha_inicio",
    "fecha_fin",
    "tipo_elasticidad_usada",
    "elasticidad_usada",
    "r2_elasticidad",
    "p_value_elasticidad",
    "precio_real",
    "precio_lista",
    "precio_efectivo",
    "descuento_efectivo",
    "cambio_precio_pct",
    "riesgo_promocion",
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
    "mejor_escenario_historico",
    "recomendacion_historica",
    "confianza",
    "razon_recomendacion",
]


def empty_pricing_historico_escenarios() -> pd.DataFrame:
    """Devuelve la estructura interna de pricing_historico_escenarios."""
    return pd.DataFrame(columns=PRICING_HISTORICO_ESCENARIOS_COLUMNS)


def _ensure_sku(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SKU" not in out.columns and "prod_nbr" in out.columns:
        out["SKU"] = out["prod_nbr"]
    if "prod_nbr" not in out.columns and "SKU" in out.columns:
        out["prod_nbr"] = out["SKU"]
    if "SKU" in out.columns:
        out["SKU"] = out["SKU"].astype("string").str.strip().astype(str)
    if "prod_nbr" in out.columns:
        out["prod_nbr"] = out["prod_nbr"].astype("string").str.strip().astype(str)
    return out


def _mode_or_na(series: pd.Series):
    clean = series.replace(["", " ", "nan", "NaN", "None", "none", "null", "Null"], np.nan).dropna()
    if clean.empty:
        return np.nan
    mode = clean.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else clean.iloc[0]


def _add_period_columns(ventas: pd.DataFrame, periodo_tipo: str) -> pd.DataFrame:
    out = ventas.copy()
    out["tran_date"] = parse_transaction_dates(out["tran_date"])
    out = out.dropna(subset=["tran_date"])
    out["mes"] = out["tran_date"].dt.to_period("M")

    if periodo_tipo == "mensual":
        period = out["tran_date"].dt.to_period("M")
        out["periodo"] = period.astype(str)
        out["fecha_inicio"] = period.dt.to_timestamp(how="start").dt.date
        out["fecha_fin"] = period.dt.to_timestamp(how="end").dt.date
    elif periodo_tipo == "trimestral":
        meses_ordenados = sorted(out["mes"].dropna().unique())
        mapa = {}
        for i in range(0, len(meses_ordenados), 3):
            meses_bloque = meses_ordenados[i : i + 3]
            if len(meses_bloque) < 3:
                continue
            periodo_3m = f"{meses_bloque[0]} a {meses_bloque[-1]}"
            for mes in meses_bloque:
                mapa[mes] = {
                    "periodo": build_quarter_label(periodo_3m),
                    "fecha_inicio": meses_bloque[0].to_timestamp(how="start").date(),
                    "fecha_fin": meses_bloque[-1].to_timestamp(how="end").date(),
                }
        meta = out["mes"].map(mapa)
        out = out[meta.notna()].copy()
        if out.empty:
            return out
        meta_df = pd.DataFrame(out["mes"].map(mapa).tolist(), index=out.index)
        for col in meta_df.columns:
            out[col] = meta_df[col].values
    elif periodo_tipo == "semestral":
        meses_ordenados = sorted(out["mes"].dropna().unique())
        mapa = {}
        for i in range(0, len(meses_ordenados), 6):
            meses_bloque = meses_ordenados[i : i + 6]
            if len(meses_bloque) < 6:
                continue
            periodo = f"{meses_bloque[0]} a {meses_bloque[-1]}"
            for mes in meses_bloque:
                mapa[mes] = {
                    "periodo": periodo,
                    "fecha_inicio": meses_bloque[0].to_timestamp(how="start").date(),
                    "fecha_fin": meses_bloque[-1].to_timestamp(how="end").date(),
                }
        meta = out["mes"].map(mapa)
        out = out[meta.notna()].copy()
        if out.empty:
            return out
        meta_df = pd.DataFrame(out["mes"].map(mapa).tolist(), index=out.index)
        for col in meta_df.columns:
            out[col] = meta_df[col].values
    elif periodo_tipo == "anual":
        year = out["tran_date"].dt.year
        out["periodo"] = year.astype(str)
        out["fecha_inicio"] = pd.to_datetime(year.astype(str) + "-01-01").dt.date
        out["fecha_fin"] = pd.to_datetime(year.astype(str) + "-12-31").dt.date
    else:
        raise ValueError(f"periodo_tipo no soportado para pricing histórico: {periodo_tipo}")

    out["periodo_tipo"] = periodo_tipo
    return out


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def _prepare_historical_financial_base(ventas: pd.DataFrame, periodo_tipo: str) -> pd.DataFrame:
    ventas = _ensure_sku(ventas)
    if ventas.empty or "tran_date" not in ventas.columns or "SKU" not in ventas.columns:
        return pd.DataFrame()

    required = ["qty", "net_sale"]
    if any(col not in ventas.columns for col in required):
        return pd.DataFrame()

    df = _add_period_columns(ventas, periodo_tipo)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["net_sale"] = pd.to_numeric(df["net_sale"], errors="coerce")

    precio_col = _find_first_column(df, ["precio_unitario", "precio", "price", "unit_price"])
    if precio_col:
        df["_precio_unitario"] = pd.to_numeric(df[precio_col], errors="coerce")
    else:
        df["_precio_unitario"] = np.nan
    df["_precio_unitario"] = df["_precio_unitario"].fillna(df["net_sale"] / df["qty"])

    lista_col = _find_first_column(df, ["precio_lista", "list_price", "precio_regular", "regular_price"])
    if lista_col:
        df["_precio_lista_linea"] = pd.to_numeric(df[lista_col], errors="coerce")
    else:
        df["_precio_lista_linea"] = np.nan
    df["_precio_lista_linea"] = df["_precio_lista_linea"].fillna(df["_precio_unitario"])

    costo_col = _find_first_column(df, ["costo_unitario", "costo2", "unit_cost", "costo"])
    if costo_col:
        df["_costo_unitario"] = pd.to_numeric(df[costo_col], errors="coerce")
    else:
        df["_costo_unitario"] = np.nan

    margen_col = _find_first_column(df, ["margen", "margin", "margen_total"])
    if margen_col:
        df["_margen_linea"] = pd.to_numeric(df[margen_col], errors="coerce")
    else:
        df["_margen_linea"] = np.nan
    df["_margen_linea"] = df["_margen_linea"].fillna((df["_precio_unitario"] - df["_costo_unitario"]) * df["qty"])
    df["_costo_total"] = df["_costo_unitario"] * df["qty"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["SKU", "qty", "net_sale", "_precio_unitario"])
    df = df[(df["qty"] > 0) & (df["net_sale"] > 0) & (df["_precio_unitario"] > 0)]
    if df.empty:
        return pd.DataFrame()

    group_cols = ["SKU", "periodo_tipo", "periodo", "fecha_inicio", "fecha_fin"]
    agg = (
        df.groupby(group_cols, observed=True, sort=False)
        .agg(
            unidades_reales=("qty", "sum"),
            ingreso_real=("net_sale", "sum"),
            precio_lista=("_precio_lista_linea", "max"),
            costo_total=("_costo_total", "sum"),
            margen_real=("_margen_linea", "sum"),
            costo_observaciones=("_costo_unitario", "count"),
        )
        .reset_index()
    )
    agg["precio_real"] = agg["ingreso_real"] / agg["unidades_reales"]
    agg["precio_lista"] = agg["precio_lista"].fillna(agg["precio_real"])
    agg["costo_unitario"] = np.where(agg["costo_observaciones"].gt(0), agg["costo_total"] / agg["unidades_reales"], np.nan)

    for source_col, target_col in [("subdept_nm", "categoria"), ("categoria", "categoria"), ("dept_nm", "departamento"), ("departamento", "departamento")]:
        if source_col in df.columns and target_col not in agg.columns:
            modes = df.groupby(group_cols, observed=True, sort=False)[source_col].agg(_mode_or_na).reset_index(name=target_col)
            agg = agg.merge(modes, on=group_cols, how="left")
    if "categoria" not in agg.columns:
        agg["categoria"] = np.nan
    if "departamento" not in agg.columns:
        agg["departamento"] = np.nan

    agg = agg.replace([np.inf, -np.inf], np.nan)
    return agg


def _normalize_elasticidades(elasticidades_periodo: pd.DataFrame) -> pd.DataFrame:
    elas = _ensure_sku(elasticidades_periodo)
    if elas.empty:
        return elas
    for col in ["periodo_tipo", "periodo", "categoria", "departamento"]:
        if col in elas.columns:
            elas[col] = elas[col].astype("string").str.strip().astype(object)
    for col in ["elasticidad", "r2", "p_value"]:
        if col in elas.columns:
            elas[col] = pd.to_numeric(elas[col], errors="coerce")
    return elas


def _build_elasticity_candidates(base: pd.DataFrame, elasticidades_periodo: pd.DataFrame) -> pd.DataFrame:
    elas = _normalize_elasticidades(elasticidades_periodo)
    if base.empty or elas.empty:
        return pd.DataFrame()

    rename = {
        "elasticidad": "elasticidad_usada",
        "r2": "r2_elasticidad",
        "p_value": "p_value_elasticidad",
        "confianza_elasticidad": "confianza",
        "razon_no_recomendable": "razon_elasticidad",
    }
    keep = ["SKU", "categoria", "departamento", "periodo_tipo", "periodo", *rename.keys()]
    keep = [col for col in keep if col in elas.columns]

    candidates: list[pd.DataFrame] = []

    # Elasticidad exacta del SKU en el periodo histórico seleccionado.
    exact = elas[elas["periodo_tipo"].isin(PERIODOS_HISTORICOS)][keep].rename(columns=rename)
    if not exact.empty:
        exact = exact.drop_duplicates(["SKU", "periodo_tipo", "periodo"])
        merged = base.merge(
            exact.drop(columns=[c for c in ["categoria", "departamento"] if c in exact.columns]),
            on=["SKU", "periodo_tipo", "periodo"],
            how="left",
        )
        merged["tipo_elasticidad_usada"] = "elasticidad_sku_periodo"
        candidates.append(merged)

    # Elasticidad global del SKU, replicada sobre cada periodo histórico real.
    global_sku = elas[elas["periodo_tipo"].eq("global_sku")][keep].rename(columns=rename)
    if not global_sku.empty:
        global_sku = global_sku.drop_duplicates(["SKU"])
        global_sku = global_sku.drop(columns=[c for c in ["categoria", "departamento", "periodo_tipo", "periodo"] if c in global_sku.columns])
        merged = base.merge(global_sku, on="SKU", how="left")
        merged["tipo_elasticidad_usada"] = "elasticidad_sku_global"
        candidates.append(merged)

    # Elasticidad por categoría/departamento, replicada sobre cada SKU-periodo real.
    cat_dept = elas[elas["periodo_tipo"].eq("categoria_departamento")][keep].rename(columns=rename)
    if not cat_dept.empty and {"categoria", "departamento"}.issubset(cat_dept.columns):
        cat_dept = cat_dept.drop_duplicates(["categoria", "departamento"])
        cat_dept = cat_dept.drop(columns=[c for c in ["SKU", "periodo_tipo", "periodo"] if c in cat_dept.columns])
        merged = base.merge(cat_dept, on=["categoria", "departamento"], how="left")
        merged["tipo_elasticidad_usada"] = "elasticidad_categoria_departamento"
        candidates.append(merged)

    if not candidates:
        return pd.DataFrame()
    out = pd.concat(candidates, ignore_index=True, sort=False)
    return out.dropna(subset=["elasticidad_usada"])


def _recommendation_reason(row: pd.Series) -> tuple[str, str]:
    confianza = row.get("confianza")
    if es_promocion(row.get("tipo_escenario")) and row.get("riesgo_promocion") == "Alto":
        return "No recomendar", "Promoción riesgosa: elasticidad, demanda base, costo o margen no cumplen los guardrails."
    if pd.isna(row.get("elasticidad_usada")) or str(confianza).lower() in {"no usable", "no recomendable", "baja"}:
        return "No recomendar", "Elasticidad insuficiente o de baja confianza para recomendar este escenario."
    if bool(row.get("mejor_escenario_historico", False)):
        if pd.isna(row.get("margen_simulado", np.nan)):
            return "Mejor escenario histórico", "Maximiza ingreso simulado. No se cuenta con costo unitario, por lo que la recomendación se basa en ingreso y no en margen."
        return "Mejor escenario histórico", "Maximiza margen simulado dentro del backtesting del mismo SKU, periodo y tipo de elasticidad."
    if row.get("variacion_margen", 0) > 0 and row.get("variacion_ingreso", 0) >= 0:
        return "Escenario viable", "Mejora margen sin deteriorar ingreso frente al periodo real observado."
    if row.get("cambio_precio_pct", 0) == 0:
        return "Mantener como referencia", "Escenario base para comparar contra los cambios de precio simulados."
    return "No preferente", "No supera al escenario real o al mejor escenario histórico en margen/ingreso."


def build_pricing_historico_escenarios(
    ventas_historicas: pd.DataFrame,
    elasticidades_periodo: pd.DataFrame,
    periodo_tipos: list[str] | None = None,
) -> pd.DataFrame:
    """Construye la tabla interna ``pricing_historico_escenarios``.

    Usa ventas reales históricas agregadas por SKU-periodo y aplica elasticidades
    ya existentes en ``elasticidades_periodo``. No calcula elasticidad, demanda
    futura ni pronósticos.
    """
    if ventas_historicas is None or ventas_historicas.empty or elasticidades_periodo is None or elasticidades_periodo.empty:
        return empty_pricing_historico_escenarios()

    periodo_tipos = [p for p in (periodo_tipos or PERIODOS_HISTORICOS) if p in PERIODOS_HISTORICOS]
    bases = [_prepare_historical_financial_base(ventas_historicas, periodo_tipo) for periodo_tipo in periodo_tipos]
    bases = [base for base in bases if base is not None and not base.empty]
    if not bases:
        return empty_pricing_historico_escenarios()

    base = pd.concat(bases, ignore_index=True, sort=False)
    candidates = _build_elasticity_candidates(base, elasticidades_periodo)
    if candidates.empty:
        return empty_pricing_historico_escenarios()

    esc = ESCENARIOS_HISTORICOS.copy()
    candidates["_join_key"] = 1
    esc["_join_key"] = 1
    sim = candidates.merge(esc, on="_join_key", how="inner").drop(columns="_join_key")

    cambio = pd.to_numeric(sim["cambio_precio_pct"], errors="coerce")
    elasticidad = pd.to_numeric(sim["elasticidad_usada"], errors="coerce").clip(ELASTICIDAD_CAP_MIN, ELASTICIDAD_CAP_MAX)
    precio_real = pd.to_numeric(sim["precio_real"], errors="coerce")
    precio_lista = pd.to_numeric(sim["precio_lista"], errors="coerce").fillna(precio_real)
    unidades_reales = pd.to_numeric(sim["unidades_reales"], errors="coerce")
    ingreso_real = pd.to_numeric(sim["ingreso_real"], errors="coerce")
    margen_real = pd.to_numeric(sim["margen_real"], errors="coerce")
    costo_unitario = pd.to_numeric(sim["costo_unitario"], errors="coerce")

    promo = es_promocion(sim["tipo_escenario"])
    factor_precio = pd.to_numeric(sim.get("factor_precio", 1 + cambio), errors="coerce").fillna(1 + cambio)
    valid = elasticidad.notna() & precio_real.gt(0) & unidades_reales.gt(0) & factor_precio.gt(0)
    sim["precio_efectivo"] = np.where(valid, precio_real * factor_precio, np.nan)
    sim["descuento_efectivo"] = np.where(
        promo,
        pd.to_numeric(sim.get("descuento_efectivo"), errors="coerce"),
        np.where(precio_lista.gt(0), 1 - (sim["precio_efectivo"] / precio_lista), np.nan),
    )
    cambio_unidades_pct = elasticidad * cambio
    sim["unidades_simuladas"] = np.where(
        valid & promo,
        unidades_reales * (1 + cambio_unidades_pct),
        np.where(valid, unidades_reales * np.exp(elasticidad * np.log1p(cambio)), np.nan),
    )
    sim["unidades_simuladas"] = pd.to_numeric(sim["unidades_simuladas"], errors="coerce").clip(lower=0)
    sim["ingreso_simulado"] = sim["precio_efectivo"] * sim["unidades_simuladas"]
    sim["margen_simulado"] = np.where(
        costo_unitario.notna(),
        (sim["precio_efectivo"] - costo_unitario) * sim["unidades_simuladas"],
        np.nan,
    )

    sim["variacion_unidades"] = sim["unidades_simuladas"] - unidades_reales
    sim["variacion_ingreso"] = sim["ingreso_simulado"] - ingreso_real
    sim["variacion_margen"] = sim["margen_simulado"] - margen_real
    sim["riesgo_promocion"] = evaluar_riesgo_promocion(
        sim["tipo_escenario"],
        elasticidad,
        unidades_reales,
        costo_unitario,
        sim["precio_efectivo"],
        sim["margen_simulado"],
        confianza_elasticidad=sim.get("confianza"),
    )

    group_cols = ["SKU", "periodo_tipo", "periodo", "tipo_elasticidad_usada"]
    sim["mejor_escenario_historico"] = False
    valid_best = sim.dropna(subset=["ingreso_simulado", "unidades_simuladas"])
    valid_best = valid_best[~(es_promocion(valid_best["tipo_escenario"]) & valid_best["riesgo_promocion"].eq("Alto"))].copy()
    if valid_best["margen_simulado"].notna().any():
        valid_best = valid_best[valid_best["margen_simulado"].notna()].copy()
    if not valid_best.empty:
        best_idx = (
            valid_best.sort_values(
                group_cols + ["margen_simulado", "ingreso_simulado", "unidades_simuladas"],
                ascending=[True, True, True, True, False, False, False],
                kind="stable",
            )
            .drop_duplicates(group_cols)
            .index
        )
        sim.loc[best_idx, "mejor_escenario_historico"] = True

    recommendations = sim.apply(_recommendation_reason, axis=1, result_type="expand")
    sim["recomendacion_historica"] = recommendations[0]
    sim["razon_recomendacion"] = recommendations[1]

    sim["cambio_precio_pct"] = sim["cambio_precio_pct"] * 100
    sim["descuento_efectivo"] = sim["descuento_efectivo"] * 100

    defaults = {col: np.nan for col in PRICING_HISTORICO_ESCENARIOS_COLUMNS}
    for col, default in defaults.items():
        if col not in sim.columns:
            sim[col] = default

    out = sim[PRICING_HISTORICO_ESCENARIOS_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["periodo_tipo", "periodo", "SKU", "tipo_elasticidad_usada", "cambio_precio_pct"]).reset_index(drop=True)
