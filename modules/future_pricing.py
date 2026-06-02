"""Simulador futuro de pricing.

Fase 5: usa exclusivamente ``demanda_base_futura`` y elasticidades ya
calculadas para proyectar escenarios simples de cambio de precio a 1 y 3 meses.
No calcula ni recalcula elasticidad dentro de este módulo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ELASTICIDAD_CAP_MAX, ELASTICIDAD_CAP_MIN
from .promotions import escenarios_con_promociones, evaluar_riesgo_promocion, es_promocion
from .utils import parse_transaction_dates

ESCENARIOS_FUTUROS = pd.DataFrame(
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
ESCENARIOS_FUTUROS["tipo_escenario"] = "simple"
ESCENARIOS_FUTUROS = escenarios_con_promociones(ESCENARIOS_FUTUROS)

PRICING_FUTURO_ESCENARIOS_COLUMNS = [
    "SKU",
    "categoria",
    "departamento",
    "horizonte",
    "metodo_proyeccion",
    "fecha_inicio_proyeccion",
    "fecha_fin_proyeccion",
    "tipo_elasticidad_usada",
    "precio_actual",
    "precio_lista",
    "precio_efectivo",
    "descuento_efectivo",
    "cambio_precio_pct",
    "riesgo_promocion",
    "demanda_base",
    "unidades_simuladas",
    "ingreso_base",
    "ingreso_simulado",
    "margen_base",
    "margen_simulado",
    "variacion_unidades",
    "variacion_ingreso",
    "variacion_margen",
    "elasticidad_usada",
    "confianza_elasticidad",
    "confianza_demanda",
    "confianza_final",
    "riesgo",
    "recomendacion",
    "razon_recomendacion",
    "tipo_escenario",
    "nombre_escenario",
]


def empty_pricing_futuro_escenarios() -> pd.DataFrame:
    """Devuelve la estructura interna ``pricing_futuro_escenarios`` sin filas."""
    return pd.DataFrame(columns=PRICING_FUTURO_ESCENARIOS_COLUMNS)


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


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def _mode_or_na(series: pd.Series):
    clean = series.replace(["", " ", "nan", "NaN", "None", "none", "null", "Null"], np.nan).dropna()
    if clean.empty:
        return np.nan
    mode = clean.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else clean.iloc[0]


def _confidence_rank(value: str) -> int:
    return {"alta": 3, "media": 2, "baja": 1}.get(str(value).strip().lower(), 0)


def _rank_confidence(rank: int) -> str:
    if rank >= 3:
        return "Alta"
    if rank == 2:
        return "Media"
    if rank == 1:
        return "Baja"
    return "No usable"


def _normalize_demanda(demanda_base_futura: pd.DataFrame) -> pd.DataFrame:
    demanda = _ensure_sku(demanda_base_futura)
    required = ["SKU", "horizonte", "metodo_proyeccion", "demanda_base", "confianza_demanda"]
    if demanda.empty or any(col not in demanda.columns for col in required):
        return pd.DataFrame()
    demanda = demanda.copy()
    demanda["demanda_base"] = pd.to_numeric(demanda["demanda_base"], errors="coerce")
    for col in ["categoria", "departamento", "horizonte", "metodo_proyeccion", "confianza_demanda"]:
        if col not in demanda.columns:
            demanda[col] = np.nan
    return demanda.replace([np.inf, -np.inf], np.nan)


def _build_price_base(ventas_precios: pd.DataFrame | None, demanda: pd.DataFrame) -> pd.DataFrame:
    """Obtiene precio actual/lista/costo por SKU desde ventas o desde demanda si ya vienen allí."""
    price_cols = ["precio_actual", "precio_lista", "costo_unitario"]
    if ventas_precios is None or ventas_precios.empty:
        available = [col for col in ["SKU", *price_cols] if col in demanda.columns]
        if "SKU" not in available or "precio_actual" not in available:
            return pd.DataFrame()
        out = demanda[available].drop_duplicates("SKU").copy()
        if "precio_lista" not in out.columns:
            out["precio_lista"] = out["precio_actual"]
        if "costo_unitario" not in out.columns:
            out["costo_unitario"] = np.nan
        return out

    ventas = _ensure_sku(ventas_precios)
    if ventas.empty or "SKU" not in ventas.columns:
        return pd.DataFrame()

    qty_col = _first_existing_column(ventas, ["qty", "unidades", "cantidad", "quantity", "units"])
    sale_col = _first_existing_column(ventas, ["net_sale", "venta_neta", "ingreso", "sales", "revenue"])
    price_col = _first_existing_column(ventas, ["precio_actual", "precio_unitario", "precio", "price", "unit_price"])
    if qty_col is None and price_col is None:
        return pd.DataFrame()

    work = ventas.copy()
    work["_qty"] = pd.to_numeric(work[qty_col], errors="coerce") if qty_col else 1.0
    if price_col:
        work["_precio_actual"] = pd.to_numeric(work[price_col], errors="coerce")
    elif sale_col:
        work["_precio_actual"] = pd.to_numeric(work[sale_col], errors="coerce") / work["_qty"]
    else:
        return pd.DataFrame()

    list_col = _first_existing_column(work, ["precio_lista", "list_price", "precio_regular", "regular_price"])
    work["_precio_lista"] = pd.to_numeric(work[list_col], errors="coerce") if list_col else np.nan
    work["_precio_lista"] = work["_precio_lista"].fillna(work["_precio_actual"])

    cost_col = _first_existing_column(work, ["costo_unitario", "costo2", "unit_cost", "costo"])
    work["_costo_unitario"] = pd.to_numeric(work[cost_col], errors="coerce") if cost_col else np.nan

    date_col = _first_existing_column(work, ["tran_date", "fecha", "date", "fecha_transaccion", "fecha_venta"])
    if date_col:
        work[date_col] = parse_transaction_dates(work[date_col])
        latest = work[date_col].max()
        if pd.notna(latest):
            work = work[work[date_col].dt.to_period("M").eq(latest.to_period("M"))].copy()

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["SKU", "_qty", "_precio_actual"])
    work = work[(work["_qty"] > 0) & (work["_precio_actual"] > 0)]
    if work.empty:
        return pd.DataFrame()

    work["_precio_x_qty"] = work["_precio_actual"] * work["_qty"]
    work["_costo_x_qty"] = work["_costo_unitario"] * work["_qty"]
    base = (
        work.groupby("SKU", observed=True, sort=False)
        .agg(
            unidades_precio=("_qty", "sum"),
            precio_x_qty=("_precio_x_qty", "sum"),
            precio_lista=("_precio_lista", "max"),
            costo_x_qty=("_costo_x_qty", "sum"),
        )
        .reset_index()
    )
    base["precio_actual"] = base["precio_x_qty"] / base["unidades_precio"]
    base["precio_lista"] = base["precio_lista"].fillna(base["precio_actual"])
    base["costo_unitario"] = base["costo_x_qty"] / base["unidades_precio"]
    return base[["SKU", "precio_actual", "precio_lista", "costo_unitario"]].replace([np.inf, -np.inf], np.nan)


def _normalize_elasticidades(elasticidades_periodo: pd.DataFrame) -> pd.DataFrame:
    elas = _ensure_sku(elasticidades_periodo)
    if elas.empty or "elasticidad" not in elas.columns:
        return pd.DataFrame()
    for col in ["periodo_tipo", "periodo", "categoria", "departamento", "confianza_elasticidad"]:
        if col in elas.columns:
            elas[col] = elas[col].astype("string").str.strip().astype(object)
    elas["elasticidad"] = pd.to_numeric(elas["elasticidad"], errors="coerce")
    if "fecha_fin" in elas.columns:
        elas["_fecha_fin_ord"] = pd.to_datetime(elas["fecha_fin"], errors="coerce")
    else:
        elas["_fecha_fin_ord"] = pd.NaT
    if "confianza_elasticidad" not in elas.columns:
        elas["confianza_elasticidad"] = "No usable"
    return elas.replace([np.inf, -np.inf], np.nan)


def _build_elasticity_candidates(demanda: pd.DataFrame, elasticidades_periodo: pd.DataFrame) -> pd.DataFrame:
    elas = _normalize_elasticidades(elasticidades_periodo)
    if demanda.empty or elas.empty:
        return pd.DataFrame()

    keep = ["SKU", "categoria", "departamento", "periodo_tipo", "periodo", "elasticidad", "confianza_elasticidad", "_fecha_fin_ord"]
    keep = [col for col in keep if col in elas.columns]
    candidates: list[pd.DataFrame] = []

    global_sku = elas[elas.get("periodo_tipo", pd.Series(index=elas.index, dtype=object)).eq("global_sku")][keep].copy()
    if not global_sku.empty:
        global_sku = global_sku.sort_values(["SKU", "_fecha_fin_ord"], ascending=[True, False], kind="stable").drop_duplicates("SKU")
        global_sku = global_sku.drop(columns=[c for c in ["categoria", "departamento", "periodo_tipo", "periodo", "_fecha_fin_ord"] if c in global_sku.columns])
        merged = demanda.merge(global_sku, on="SKU", how="left")
        merged["tipo_elasticidad_usada"] = "elasticidad_sku_global"
        candidates.append(merged)

    sku_period = elas[elas.get("periodo_tipo", pd.Series(index=elas.index, dtype=object)).isin(["mensual", "trimestral", "semestral", "anual"])][keep].copy()
    if not sku_period.empty:
        sku_period = sku_period.sort_values(["SKU", "_fecha_fin_ord"], ascending=[True, False], kind="stable").drop_duplicates("SKU")
        sku_period = sku_period.drop(columns=[c for c in ["categoria", "departamento", "periodo_tipo", "periodo", "_fecha_fin_ord"] if c in sku_period.columns])
        merged = demanda.merge(sku_period, on="SKU", how="left")
        merged["tipo_elasticidad_usada"] = "elasticidad_sku_reciente"
        candidates.append(merged)

    cat_dept = elas[elas.get("periodo_tipo", pd.Series(index=elas.index, dtype=object)).eq("categoria_departamento")][keep].copy()
    if not cat_dept.empty and {"categoria", "departamento"}.issubset(cat_dept.columns):
        cat_dept = cat_dept.sort_values(["categoria", "departamento", "_fecha_fin_ord"], ascending=[True, True, False], kind="stable").drop_duplicates(["categoria", "departamento"])
        cat_dept = cat_dept.drop(columns=[c for c in ["SKU", "periodo_tipo", "periodo", "_fecha_fin_ord"] if c in cat_dept.columns])
        merged = demanda.merge(cat_dept, on=["categoria", "departamento"], how="left")
        merged["tipo_elasticidad_usada"] = "elasticidad_categoria_departamento"
        candidates.append(merged)

    if not candidates:
        return pd.DataFrame()
    out = pd.concat(candidates, ignore_index=True, sort=False)
    out = out.dropna(subset=["elasticidad"])
    out = out.rename(columns={"elasticidad": "elasticidad_usada"})
    return out


def _risk_and_recommendation(row: pd.Series) -> tuple[str, str, str]:
    reasons: list[str] = []
    risk = "Bajo"
    if row.get("confianza_final") in {"No usable", "Baja"}:
        risk = "Alto"
        reasons.append("confianza final insuficiente")
    if es_promocion(row.get("tipo_escenario")) and row.get("riesgo_promocion") == "Alto":
        risk = "Alto"
        reasons.append("promoción no cumple guardrails de elasticidad, demanda, costo o margen")
    if row.get("elasticidad_usada", 0) >= 0 and row.get("cambio_precio_pct", 0) != 0:
        risk = "Alto"
        reasons.append("elasticidad positiva o atípica")
    if row.get("unidades_simuladas", 0) <= 0:
        risk = "Alto"
        reasons.append("unidades simuladas nulas")
    if pd.notna(row.get("margen_simulado", np.nan)) and row.get("margen_simulado", 0) < 0:
        risk = "Alto"
        reasons.append("margen simulado negativo")
    if abs(row.get("cambio_precio_pct", 0)) >= 15 and row.get("confianza_final") != "Alta":
        risk = "Alto"
        reasons.append("cambio de precio agresivo con confianza no alta")
    elif risk != "Alto" and abs(row.get("cambio_precio_pct", 0)) >= 15:
        risk = "Medio"
        reasons.append("cambio de precio agresivo")

    if risk == "Alto":
        return risk, "No recomendar", "Escenario sospechoso: " + "; ".join(dict.fromkeys(reasons)) + "."
    if row.get("mejor_escenario", False):
        if pd.isna(row.get("margen_simulado", np.nan)):
            return risk, "Recomendar", "Maximiza ingreso simulado futuro. No se cuenta con costo unitario, por lo que la recomendación se basa en ingreso y no en margen."
        return risk, "Recomendar", "Maximiza margen simulado futuro entre escenarios válidos del mismo SKU, horizonte, método y elasticidad."
    if row.get("cambio_precio_pct", 0) == 0:
        return risk, "Mantener como referencia", "Escenario base contra el cual se comparan los cambios de precio futuros."
    if row.get("variacion_margen", 0) > 0 and row.get("variacion_ingreso", 0) >= 0:
        return risk, "Escenario viable", "Mejora margen sin deteriorar ingreso frente a la demanda base futura."
    return risk, "No preferente", "No supera claramente el escenario base en margen e ingreso."


def build_pricing_futuro_escenarios(
    demanda_base_futura: pd.DataFrame,
    elasticidades_periodo: pd.DataFrame,
    ventas_precios: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construye ``pricing_futuro_escenarios`` para horizontes de 1 y 3 meses.

    La fórmula usada es lineal y deliberadamente simple:
    ``cambio_porcentual_unidades = elasticidad * cambio_porcentual_precio`` y
    ``unidades_simuladas = demanda_base * (1 + cambio_porcentual_unidades)``.
    """
    demanda = _normalize_demanda(demanda_base_futura)
    if demanda.empty or elasticidades_periodo is None or elasticidades_periodo.empty:
        return empty_pricing_futuro_escenarios()

    demanda = demanda[demanda["horizonte"].isin(["1 mes", "3 meses"])].copy()
    demanda = demanda[pd.to_numeric(demanda["demanda_base"], errors="coerce").gt(0)].copy()
    if demanda.empty:
        return empty_pricing_futuro_escenarios()

    price_base = _build_price_base(ventas_precios, demanda)
    if price_base.empty:
        return empty_pricing_futuro_escenarios()

    base = demanda.merge(price_base, on="SKU", how="inner", suffixes=("", "_precio"))
    candidates = _build_elasticity_candidates(base, elasticidades_periodo)
    if candidates.empty:
        return empty_pricing_futuro_escenarios()

    candidates["precio_actual"] = pd.to_numeric(candidates["precio_actual"], errors="coerce")
    candidates["precio_lista"] = pd.to_numeric(candidates["precio_lista"], errors="coerce").fillna(candidates["precio_actual"])
    candidates["costo_unitario"] = pd.to_numeric(candidates.get("costo_unitario", np.nan), errors="coerce")
    candidates["elasticidad_usada"] = pd.to_numeric(candidates["elasticidad_usada"], errors="coerce").clip(ELASTICIDAD_CAP_MIN, ELASTICIDAD_CAP_MAX)
    candidates = candidates.replace([np.inf, -np.inf], np.nan)
    candidates = candidates[
        candidates["precio_actual"].gt(0)
        & candidates["precio_lista"].gt(0)
        & candidates["demanda_base"].gt(0)
        & candidates["elasticidad_usada"].notna()
    ].copy()
    if candidates.empty:
        return empty_pricing_futuro_escenarios()

    esc = ESCENARIOS_FUTUROS.copy()
    candidates["_join_key"] = 1
    esc["_join_key"] = 1
    sim = candidates.merge(esc, on="_join_key", how="inner").drop(columns="_join_key")

    cambio = pd.to_numeric(sim["cambio_precio_pct"], errors="coerce")
    promo = es_promocion(sim["tipo_escenario"])
    factor_precio = pd.to_numeric(sim.get("factor_precio", 1 + cambio), errors="coerce").fillna(1 + cambio)
    demand_factor = 1 + sim["elasticidad_usada"] * cambio
    valid = sim["precio_actual"].gt(0) & factor_precio.gt(0) & sim["demanda_base"].gt(0) & np.isfinite(demand_factor)

    sim["precio_efectivo"] = np.where(valid, sim["precio_actual"] * factor_precio, np.nan)
    sim["unidades_simuladas"] = np.where(valid, sim["demanda_base"] * demand_factor, np.nan)
    sim["unidades_simuladas"] = pd.to_numeric(sim["unidades_simuladas"], errors="coerce").clip(lower=0)
    sim["descuento_efectivo"] = np.where(
        promo,
        pd.to_numeric(sim.get("descuento_efectivo"), errors="coerce"),
        np.where(sim["precio_lista"].gt(0), 1 - (sim["precio_efectivo"] / sim["precio_lista"]), np.nan),
    )

    sim["ingreso_base"] = sim["precio_actual"] * sim["demanda_base"]
    sim["ingreso_simulado"] = sim["precio_efectivo"] * sim["unidades_simuladas"]
    sim["margen_base"] = np.where(
        sim["costo_unitario"].notna(),
        (sim["precio_actual"] - sim["costo_unitario"]) * sim["demanda_base"],
        np.nan,
    )
    sim["margen_simulado"] = np.where(
        sim["costo_unitario"].notna(),
        (sim["precio_efectivo"] - sim["costo_unitario"]) * sim["unidades_simuladas"],
        np.nan,
    )
    # Especificación Fase 5: variacion_unidades es la variación RELATIVA respecto a la
    # demanda base, (unidades_simuladas - demanda_base) / demanda_base. demanda_base ya
    # viene filtrada a > 0, por lo que la división es segura.
    sim["variacion_unidades"] = (sim["unidades_simuladas"] - sim["demanda_base"]) / sim["demanda_base"]
    sim["variacion_ingreso"] = sim["ingreso_simulado"] - sim["ingreso_base"]
    sim["variacion_margen"] = sim["margen_simulado"] - sim["margen_base"]

    sim["confianza_elasticidad"] = sim["confianza_elasticidad"].fillna("No usable")
    sim["confianza_demanda"] = sim["confianza_demanda"].fillna("No usable")
    final_rank = sim.apply(lambda r: min(_confidence_rank(r["confianza_elasticidad"]), _confidence_rank(r["confianza_demanda"])), axis=1)
    sim["confianza_final"] = final_rank.map(_rank_confidence)
    missing_cost_promo = sim["costo_unitario"].isna() & es_promocion(sim["tipo_escenario"])
    sim.loc[missing_cost_promo & sim["confianza_final"].eq("Alta"), "confianza_final"] = "Media"
    sim.loc[missing_cost_promo & sim["confianza_final"].eq("Media"), "confianza_final"] = "Baja"
    sim["riesgo_promocion"] = evaluar_riesgo_promocion(
        sim["tipo_escenario"],
        sim["elasticidad_usada"],
        sim["demanda_base"],
        sim["costo_unitario"],
        sim["precio_efectivo"],
        sim["margen_simulado"],
        confianza_demanda=sim["confianza_demanda"],
        confianza_elasticidad=sim["confianza_elasticidad"],
    )

    group_cols = ["SKU", "horizonte", "metodo_proyeccion", "tipo_elasticidad_usada"]
    sim["mejor_escenario"] = False
    eligible = sim[
        sim["confianza_final"].isin(["Alta", "Media"])
        & (~(es_promocion(sim["tipo_escenario"]) & sim["riesgo_promocion"].eq("Alto")))
        & (sim["margen_simulado"].ge(0) | sim["margen_simulado"].isna())
        & sim["ingreso_simulado"].notna()
        & sim["unidades_simuladas"].gt(0)
    ]
    if not eligible.empty:
        best_idx = (
            eligible.assign(_score_margen=eligible["margen_simulado"].fillna(eligible["ingreso_simulado"])).sort_values(
                group_cols + ["_score_margen", "ingreso_simulado", "unidades_simuladas"],
                ascending=[True, True, True, True, False, False, False],
                kind="stable",
            )
            .drop_duplicates(group_cols)
            .index
        )
        sim.loc[best_idx, "mejor_escenario"] = True

    risk_reco = sim.apply(_risk_and_recommendation, axis=1, result_type="expand")
    sim["riesgo"] = risk_reco[0]
    sim["recomendacion"] = risk_reco[1]
    sim["razon_recomendacion"] = risk_reco[2]

    sim["cambio_precio_pct"] = sim["cambio_precio_pct"] * 100
    sim["descuento_efectivo"] = sim["descuento_efectivo"] * 100

    out = sim[PRICING_FUTURO_ESCENARIOS_COLUMNS].replace([np.inf, -np.inf], np.nan)
    # Mantiene la tabla libre de NaN/inf en columnas críticas y evita romper SKUs insuficientes.
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    fill_numeric_cols = [col for col in numeric_cols if col not in {"margen_base", "margen_simulado", "variacion_margen"}]
    out[fill_numeric_cols] = out[fill_numeric_cols].fillna(0.0)
    object_cols = [col for col in out.columns if col not in numeric_cols]
    out[object_cols] = out[object_cols].fillna("Sin dato")
    return out.sort_values(["horizonte", "metodo_proyeccion", "SKU", "tipo_elasticidad_usada", "cambio_precio_pct"]).reset_index(drop=True)
