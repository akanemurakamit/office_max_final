"""Demand Forecast Engine para estimar demanda base futura.

Este módulo calcula únicamente proyecciones de unidades base con el precio actual.
No recalcula elasticidad ni simula escenarios de precio; esos pasos pertenecen a
módulos posteriores del pricing futuro.
"""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd

from .config import (
    DEMANDA_FUTURA_MIN_MESES_VENTANA,
    DEMANDA_FUTURA_METODOS,
    DEMANDA_FUTURA_PESOS_DEFAULT,
    DEMANDA_FUTURA_VOLATILIDAD_CV_ALTA,
)
from .utils import parse_transaction_dates

DEMANDA_BASE_FUTURA_COLUMNS = [
    "SKU",
    "categoria",
    "departamento",
    "horizonte",
    "metodo_proyeccion",
    "fecha_inicio_proyeccion",
    "fecha_fin_proyeccion",
    "demanda_base",
    "promedio_ultimos_3_meses",
    "promedio_ultimos_6_meses",
    "promedio_ultimos_12_meses",
    "promedio_ultimos_24_meses",
    "promedio_mismo_mes_historico",
    "promedio_mismo_trimestre_historico",
    "pesos_usados",
    "confianza_demanda",
    "razon_confianza_demanda",
]

HORIZONTES_DEMANDA_FUTURA = ["1 mes", "3 meses"]


def empty_demanda_base_futura() -> pd.DataFrame:
    """Devuelve la estructura interna de ``demanda_base_futura`` sin filas."""
    return pd.DataFrame(columns=DEMANDA_BASE_FUTURA_COLUMNS)


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


def _mode_or_default(series: pd.Series, default: str = "Sin dato"):
    clean = series.replace(["", " ", "nan", "NaN", "None", "none", "null", "Null"], np.nan).dropna()
    if clean.empty:
        return default
    mode = clean.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else clean.iloc[0]


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def _normalize_method(method: str) -> str:
    value = str(method).strip().lower()
    aliases = {
        "automatico recomendado": "Automático recomendado",
        "automático recomendado": "Automático recomendado",
        "automatico": "Automático recomendado",
        "automático": "Automático recomendado",
        "reciente": "Reciente",
        "estacional": "Estacional",
        "historico amplio": "Histórico amplio",
        "histórico amplio": "Histórico amplio",
        "manual avanzado": "Manual avanzado",
    }
    if value not in aliases:
        raise ValueError(f"Método de proyección no soportado: {method}")
    return aliases[value]


def _projection_dates(last_month: pd.Period, horizonte: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    months = 1 if horizonte == "1 mes" else 3
    start_period = last_month + 1
    end_period = last_month + months
    return start_period.to_timestamp(how="start").date(), end_period.to_timestamp(how="end").date()


def _coerce_to_month_period(series: pd.Series) -> pd.Series:
    """Convierte de forma segura una columna `mes` a periodos mensuales.

    Acepta valores que ya son ``Period``/``PeriodIndex`` o fechas (``2024-11``,
    ``2024-11-01``...). Si la columna trae números de mes sueltos (1..12) u otros
    valores no fechables, devuelve ``NaT`` en lugar de lanzar una excepción, ya que
    un número de mes aislado no identifica un periodo histórico real.
    """
    # Caso 1: ya es un periodo. Se normaliza a frecuencia mensual vía el accessor .dt.
    if isinstance(series.dtype, pd.PeriodDtype):
        if series.dtype.freq.freqstr == "M":
            return series
        return series.dt.asfreq("M")

    # Caso 2: números sueltos (1..12). pd.to_datetime los interpretaría como
    # nanosegundos desde epoch (1970), por eso se descartan: un número de mes
    # aislado no identifica un periodo histórico real.
    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(pd.NaT, index=series.index)

    # Caso 3: fechas/strings fechables -> se convierten con coerción.
    fechas = pd.to_datetime(series, errors="coerce")
    if fechas.notna().any():
        return fechas.dt.to_period("M")

    # Caso 4: no es fechable. No es un periodo real.
    return pd.Series(pd.NaT, index=series.index)


def _prepare_monthly_sales(ventas: pd.DataFrame) -> pd.DataFrame:
    if ventas is None or ventas.empty:
        return pd.DataFrame()

    out = _ensure_sku(ventas)
    if "SKU" not in out.columns:
        raise ValueError("No se encontró columna de SKU. La base debe tener `prod_nbr` o `SKU`.")

    date_col = _first_existing_column(out, ["tran_date", "fecha", "date", "fecha_transaccion", "fecha_venta"])
    if date_col is None and "mes" not in out.columns:
        raise ValueError("No se encontró columna de fecha para calcular demanda futura.")

    qty_col = _first_existing_column(out, ["qty", "unidades", "cantidad", "quantity", "units"])
    if qty_col is None:
        raise ValueError("No se encontró columna de unidades (`qty`) para calcular demanda futura.")

    work = out.copy()
    # Se prioriza SIEMPRE la columna de fecha real para derivar el mes, porque otras
    # etapas (p. ej. el motor de elasticidad) pueden dejar una columna `mes` con números
    # de mes enteros (1..12). Forzar pd.PeriodIndex sobre esos enteros provoca el error
    # "Given date string \"11\" not likely a datetime". Solo se usa `mes` preexistente
    # si no hay columna de fecha, y siempre con conversión segura.
    if date_col is not None:
        work[date_col] = parse_transaction_dates(work[date_col])
        work = work.dropna(subset=[date_col])
        work["mes"] = work[date_col].dt.to_period("M")
    else:
        work["mes"] = _coerce_to_month_period(work["mes"])
        work = work.dropna(subset=["mes"])

    work["qty"] = pd.to_numeric(work[qty_col], errors="coerce")
    work = work.dropna(subset=["SKU", "mes", "qty"])
    work = work[work["qty"] > 0].copy()
    if work.empty:
        return pd.DataFrame()

    categoria_col = _first_existing_column(work, ["categoria", "subdept_nm", "categoría", "category", "subcategoria"])
    departamento_col = _first_existing_column(work, ["departamento", "dept_nm", "department", "depto"])

    agg_spec = {"unidades": ("qty", "sum")}
    monthly = work.groupby(["SKU", "mes"], observed=True, sort=False).agg(**agg_spec).reset_index()

    descriptors = work.groupby("SKU", observed=True, sort=False).agg(
        categoria=(categoria_col, _mode_or_default) if categoria_col else ("SKU", lambda _s: "Sin dato"),
        departamento=(departamento_col, _mode_or_default) if departamento_col else ("SKU", lambda _s: "Sin dato"),
    ).reset_index()
    monthly = monthly.merge(descriptors, on="SKU", how="left")
    monthly["month"] = monthly["mes"].dt.month
    monthly["quarter"] = monthly["mes"].dt.quarter
    return monthly


def _mean_if_enough(values: pd.Series, min_months: int, horizon_multiplier: int = 1) -> tuple[float, bool, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean[clean > 0]
    if len(clean) < min_months:
        return np.nan, False, int(len(clean))
    return float(clean.mean() * horizon_multiplier), True, int(len(clean))


def _sku_components(sku_monthly: pd.DataFrame, last_month: pd.Period, target_start: pd.Period) -> dict:
    series = sku_monthly.set_index("mes").sort_index()["unidades"]
    target_month = target_start.month
    target_quarter = target_start.quarter
    quarter_months = {p.month for p in pd.period_range(target_start, target_start + 2, freq="M")}

    components: dict[str, dict] = {}
    for window in [3, 6, 12, 24]:
        months = pd.period_range(last_month - window + 1, last_month, freq="M")
        values = series.reindex(months)
        value_1m, available_1m, count = _mean_if_enough(values, DEMANDA_FUTURA_MIN_MESES_VENTANA[f"ultimos_{window}_meses"])
        components[f"ultimos_{window}_meses"] = {
            "valor_1m": value_1m,
            "valor_3m": value_1m * 3 if pd.notna(value_1m) else np.nan,
            "disponible": available_1m,
            "meses_disponibles": count,
        }

    same_month = sku_monthly[(sku_monthly["mes"] <= last_month) & (sku_monthly["month"] == target_month)]["unidades"]
    value, available, count = _mean_if_enough(same_month, DEMANDA_FUTURA_MIN_MESES_VENTANA["mismo_mes_historico"])
    components["mismo_mes_historico"] = {
        "valor_1m": value,
        "valor_3m": np.nan,
        "disponible": available,
        "meses_disponibles": count,
    }

    same_quarter = sku_monthly[
        (sku_monthly["mes"] <= last_month)
        & (sku_monthly["quarter"] == target_quarter)
        & (sku_monthly["month"].isin(quarter_months))
    ]["unidades"]
    value, available, count = _mean_if_enough(
        same_quarter,
        DEMANDA_FUTURA_MIN_MESES_VENTANA["mismo_trimestre_historico"],
        horizon_multiplier=3,
    )
    components["mismo_trimestre_historico"] = {
        "valor_1m": np.nan,
        "valor_3m": value,
        "disponible": available,
        "meses_disponibles": count,
    }

    return components


def _compute_all_components_batch(
    monthly: pd.DataFrame,
    anchor_month: pd.Period,
    target_start: pd.Period,
) -> dict[str, dict[str, pd.Series]]:
    """
    Versión vectorizada de _sku_components: calcula todos los componentes de ventana
    para TODOS los SKUs en una sola pasada de operaciones pandas, en lugar de iterar
    SKU por SKU con groupby. Retorna un dict {componente: {campo: Series indexada por SKU}}.

    Esto elimina el cuello de botella del for-loop sobre SKUs en build_demanda_base_futura.
    """
    target_month   = target_start.month
    target_quarter = target_start.quarter
    quarter_months = {p.month for p in pd.period_range(target_start, target_start + 2, freq="M")}

    # Pivot: filas=SKU, columnas=mes -> unidades (NaN si no hay venta ese mes)
    hist = monthly[monthly["mes"] <= anchor_month].copy()
    if hist.empty:
        return {}

    pivot = (
        hist.pivot_table(index="SKU", columns="mes", values="unidades", aggfunc="sum")
        .sort_index(axis=1)
    )

    result: dict[str, dict[str, pd.Series]] = {}
    min_v = DEMANDA_FUTURA_MIN_MESES_VENTANA

    for window, key in [(3, "ultimos_3_meses"), (6, "ultimos_6_meses"), (12, "ultimos_12_meses"), (24, "ultimos_24_meses")]:
        months_in_window = pd.period_range(anchor_month - window + 1, anchor_month, freq="M")
        sub = pivot.reindex(columns=months_in_window)
        count_ser = sub.notna().sum(axis=1)
        mean_ser  = sub.mean(axis=1)  # NaN where all missing
        min_m = min_v[key]
        available_ser = count_ser.ge(min_m)
        valor_1m = mean_ser.where(available_ser)
        result[key] = {
            "valor_1m":        valor_1m,
            "valor_3m":        valor_1m * 3,
            "disponible":      available_ser,
            "meses_disponibles": count_ser,
        }

    # Mismo mes histórico
    same_month_sub = hist[hist["month"] == target_month]
    sm_agg = same_month_sub.groupby("SKU", observed=True)["unidades"].agg(["mean", "count"])
    sm_mean  = pivot.index.to_series().map(sm_agg["mean"])
    sm_count = pivot.index.to_series().map(sm_agg["count"]).fillna(0).astype(int)
    sm_avail = sm_count.ge(min_v["mismo_mes_historico"])
    result["mismo_mes_historico"] = {
        "valor_1m":        sm_mean.where(sm_avail),
        "valor_3m":        pd.Series(np.nan, index=pivot.index),
        "disponible":      sm_avail,
        "meses_disponibles": sm_count,
    }

    # Mismo trimestre histórico
    same_q_sub = hist[hist["month"].isin(quarter_months) & (hist["quarter"] == target_quarter)]
    sq_agg = same_q_sub.groupby("SKU", observed=True)["unidades"].agg(["mean", "count"])
    sq_mean  = pivot.index.to_series().map(sq_agg["mean"])
    sq_count = pivot.index.to_series().map(sq_agg["count"]).fillna(0).astype(int)
    sq_avail = sq_count.ge(min_v["mismo_trimestre_historico"])
    result["mismo_trimestre_historico"] = {
        "valor_1m":        pd.Series(np.nan, index=pivot.index),
        "valor_3m":        (sq_mean * 3).where(sq_avail),
        "disponible":      sq_avail,
        "meses_disponibles": sq_count,
    }

    return result


def _weights_for_method(horizonte: str, metodo: str, pesos_config: dict) -> dict[str, float]:
    if metodo == "Reciente":
        return {"ultimos_3_meses": 1.0} if horizonte == "1 mes" else {"ultimos_6_meses": 1.0}
    if metodo == "Estacional":
        return {"mismo_mes_historico": 1.0} if horizonte == "1 mes" else {"mismo_trimestre_historico": 1.0}
    if metodo == "Histórico amplio":
        return {"ultimos_12_meses": 1.0} if horizonte == "1 mes" else {"ultimos_24_meses": 1.0}
    return deepcopy(pesos_config[horizonte][metodo])


def _redistribute_weights(raw_weights: dict[str, float], components: dict, value_key: str) -> tuple[dict[str, float], list[str]]:
    available = {
        name: float(weight)
        for name, weight in raw_weights.items()
        if weight > 0 and name in components and components[name]["disponible"] and pd.notna(components[name][value_key])
    }
    missing = [name for name, weight in raw_weights.items() if weight > 0 and name not in available]
    total = sum(available.values())
    if total <= 0:
        return {}, missing
    return {name: weight / total for name, weight in available.items()}, missing


def _classify_confidence(
    components: dict,
    weights_used: dict[str, float],
    missing: list[str],
    sku_monthly: pd.DataFrame,
    cv_precomp: float | None = None,
) -> tuple[str, str]:
    """Clasifica confianza de la proyección.

    ``cv_precomp`` permite pasar el coeficiente de variación pre-calculado para
    todos los SKUs a la vez (evita recomputed por SKU en el bucle principal).
    """
    if not weights_used:
        return "No usable", "No hay datos suficientes en ninguna ventana de demanda."

    reasons = []
    if missing:
        reasons.append("Ventanas sin datos suficientes: " + ", ".join(missing))

    recent_ok = any(name in weights_used for name in ["ultimos_3_meses", "ultimos_6_meses"])
    seasonal_ok = any(name in weights_used for name in ["mismo_mes_historico", "mismo_trimestre_historico"])

    volatile = False
    if cv_precomp is not None:
        cv = cv_precomp
        volatile = cv >= DEMANDA_FUTURA_VOLATILIDAD_CV_ALTA
        if volatile:
            reasons.append(f"Demanda muy volátil (CV últimos meses={cv:.2f}).")
    else:
        units = pd.to_numeric(sku_monthly.sort_values("mes")["unidades"].tail(12), errors="coerce").dropna()
        if len(units) >= 3 and units.mean() > 0:
            cv = float(units.std(ddof=0) / units.mean())
            volatile = cv >= DEMANDA_FUTURA_VOLATILIDAD_CV_ALTA
            if volatile:
                reasons.append(f"Demanda muy volátil (CV últimos meses={cv:.2f}).")

    if volatile or (len(weights_used) == 1 and not recent_ok):
        confidence = "Baja"
    elif recent_ok and seasonal_ok and not missing:
        confidence = "Alta"
    elif recent_ok:
        confidence = "Media"
        if not seasonal_ok:
            reasons.append("Historia reciente suficiente, pero poca historia estacional.")
    else:
        confidence = "Baja"
        reasons.append("Pocos datos recientes disponibles.")

    if not reasons:
        reasons.append("Historia reciente y estacional suficiente.")
    return confidence, " ".join(dict.fromkeys(reasons))


def build_demanda_base_futura(
    ventas: pd.DataFrame,
    horizontes: list[str] | None = None,
    metodos: list[str] | None = None,
    pesos_config: dict | None = None,
    mes_inicio_proyeccion: str | pd.Period | None = None,
) -> pd.DataFrame:
    """Construye la tabla interna ``demanda_base_futura``.

    La proyección estima unidades base futuras con el precio actual. No usa ni
    recalcula elasticidad. Para ventanas faltantes, redistribuye el peso entre
    las ventanas disponibles, baja/confirma la confianza y registra la razón.

    ``mes_inicio_proyeccion`` (opcional, "YYYY-MM" o Period mensual) permite elegir
    el primer mes a proyectar. Si no se da, se proyecta a partir del mes siguiente
    al último con datos. El horizonte (1 o 3 meses) define cuántos meses
    consecutivos se cubren desde ese inicio.
    """
    monthly = _prepare_monthly_sales(ventas)
    if monthly.empty:
        return empty_demanda_base_futura()

    requested_horizons = horizontes or HORIZONTES_DEMANDA_FUTURA
    requested_methods = [_normalize_method(m) for m in (metodos or DEMANDA_FUTURA_METODOS)]
    pesos = deepcopy(DEMANDA_FUTURA_PESOS_DEFAULT)
    if pesos_config:
        for horizonte, methods in pesos_config.items():
            pesos.setdefault(horizonte, {})
            for method, weights in methods.items():
                pesos[horizonte][_normalize_method(method)] = weights

    last_month = monthly["mes"].max()
    # Mes de inicio elegido por el usuario (o el mes siguiente al último dato).
    if mes_inicio_proyeccion is not None:
        try:
            target_start_global = pd.Period(str(mes_inicio_proyeccion), freq="M")
        except Exception:
            target_start_global = last_month + 1
    else:
        target_start_global = last_month + 1
    # El "ancla" histórica es el mes anterior al inicio de proyección, para que las
    # ventanas (últimos N meses, mismo mes histórico, etc.) se midan correctamente.
    anchor_month = target_start_global - 1

    # Precomputar todos los componentes para todos los SKUs de una vez (vectorizado).
    batch = _compute_all_components_batch(monthly, anchor_month, target_start_global)

    # Convertir batch a lookup dicts para acceso O(1) por SKU.
    # Estructura: lookup[comp_key][field] = {sku: value}
    lookup: dict[str, dict[str, dict]] = {}
    for comp_key, fields in batch.items():
        lookup[comp_key] = {}
        for field, ser in fields.items():
            if isinstance(ser, pd.Series):
                lookup[comp_key][field] = ser.to_dict()
            else:
                lookup[comp_key][field] = {}

    # Descriptores por SKU — modo vectorizado para evitar Python-level groupby
    def _fast_mode_col(df, group_col, value_col, default="Sin dato"):
        counts = df.groupby([group_col, value_col], observed=True).size().reset_index(name="_n")
        counts = counts.sort_values("_n", ascending=False).drop_duplicates(group_col)
        return counts.set_index(group_col)[value_col].rename(value_col)

    cat_mode  = _fast_mode_col(monthly, "SKU", "categoria")
    dept_mode = _fast_mode_col(monthly, "SKU", "departamento")
    sku_meta  = pd.DataFrame({"categoria": cat_mode, "departamento": dept_mode}).fillna("Sin dato")

    # Precomputar CV (coeficiente de variación) para todos los SKUs — vectorizado.
    # Evita llamar sort_values+tail(12) dentro del bucle de _classify_confidence.
    last12_start = anchor_month - 11
    recent_monthly = monthly[monthly["mes"].between(last12_start, anchor_month)]
    sku_cv_raw = (
        recent_monthly.groupby("SKU", observed=True)["unidades"]
        .agg(lambda s: float(s.std(ddof=0) / s.mean()) if len(s) >= 3 and s.mean() > 0 else 0.0)
    )
    sku_cv_map: dict[str, float] = sku_cv_raw.to_dict()

    sku_monthly_map = {sku: grp for sku, grp in monthly.groupby("SKU", observed=True, sort=False)}

    rows = []
    all_skus = monthly["SKU"].unique()

    for horizonte in requested_horizons:
        if horizonte not in HORIZONTES_DEMANDA_FUTURA:
            raise ValueError(f"Horizonte no soportado: {horizonte}")
        value_key = "valor_1m" if horizonte == "1 mes" else "valor_3m"
        start_date, end_date = _projection_dates(anchor_month, horizonte)

        for metodo in requested_methods:
            raw_weights = _weights_for_method(horizonte, metodo, pesos)

            for sku in all_skus:
                # Reconstruir components dict para este SKU desde el lookup precomputado
                components: dict[str, dict] = {}
                for comp_key, fields in lookup.items():
                    components[comp_key] = {
                        "valor_1m":          fields.get("valor_1m", {}).get(sku, np.nan),
                        "valor_3m":          fields.get("valor_3m", {}).get(sku, np.nan),
                        "disponible":        bool(fields.get("disponible", {}).get(sku, False)),
                        "meses_disponibles": int(fields.get("meses_disponibles", {}).get(sku, 0)),
                    }

                weights_used, missing = _redistribute_weights(raw_weights, components, value_key)
                demanda_base = (
                    sum(
                        components[name][value_key] * weight
                        for name, weight in weights_used.items()
                        if pd.notna(components[name][value_key])
                    )
                    if weights_used else np.nan
                )
                confidence, reason = _classify_confidence(
                    components, weights_used, missing,
                    sku_monthly_map.get(sku, monthly.iloc[:0]),
                    cv_precomp=sku_cv_map.get(sku),
                )

                meta = sku_meta.loc[sku] if sku in sku_meta.index else pd.Series({"categoria": "Sin dato", "departamento": "Sin dato"})
                rows.append(
                    {
                        "SKU": sku,
                        "categoria": meta["categoria"],
                        "departamento": meta["departamento"],
                        "horizonte": horizonte,
                        "metodo_proyeccion": metodo,
                        "fecha_inicio_proyeccion": start_date,
                        "fecha_fin_proyeccion": end_date,
                        "demanda_base": demanda_base,
                        "promedio_ultimos_3_meses":          components.get("ultimos_3_meses", {}).get("valor_1m", np.nan),
                        "promedio_ultimos_6_meses":          components.get("ultimos_6_meses", {}).get("valor_1m", np.nan),
                        "promedio_ultimos_12_meses":         components.get("ultimos_12_meses", {}).get("valor_1m", np.nan),
                        "promedio_ultimos_24_meses":         components.get("ultimos_24_meses", {}).get("valor_1m", np.nan),
                        "promedio_mismo_mes_historico":      components.get("mismo_mes_historico", {}).get("valor_1m", np.nan),
                        "promedio_mismo_trimestre_historico": components.get("mismo_trimestre_historico", {}).get("valor_3m", np.nan),
                        "pesos_usados": json.dumps(weights_used, ensure_ascii=False, sort_keys=True),
                        "confianza_demanda": confidence,
                        "razon_confianza_demanda": reason,
                    }
                )

    if not rows:
        return empty_demanda_base_futura()
    return pd.DataFrame(rows, columns=DEMANDA_BASE_FUTURA_COLUMNS)
