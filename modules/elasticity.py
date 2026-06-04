"""Cálculo de elasticidad multi-periodo con compatibilidad trimestral."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import MIN_OBSERVACIONES, MIN_PRECIOS_DISTINTOS
from .utils import build_quarter_label, parse_transaction_dates


def integrate_promotions(ventas_base: pd.DataFrame, promociones: pd.DataFrame | None) -> pd.DataFrame:
    """Integra promociones a nivel SKU-mes de forma opcional."""
    ventas = ventas_base.copy()
    ventas["num_promociones"] = 0
    ventas["tiene_promocion"] = 0

    if promociones is None or promociones.empty:
        return ventas

    promos = promociones.copy()
    promos.columns = promos.columns.astype(str).str.strip()

    posibles_cols_sku = ["prod_nbr", "SKU", "sku", "producto", "product_id"]
    col_sku = next((col for col in posibles_cols_sku if col in promos.columns), None)

    posibles_cols_fecha_inicio = [
        "tran_date",
        "fecha",
        "fecha_inicio",
        "inicio_promocion",
        "fecha_ini",
        "start_date",
        "Start_Date",
    ]
    posibles_cols_fecha_fin = [
        "fecha_fin",
        "fin_promocion",
        "fecha_final",
        "end_date",
        "End_Date",
        "fecha_termino",
        "fecha_término",
    ]

    col_inicio = next((col for col in posibles_cols_fecha_inicio if col in promos.columns), None)
    col_fin = next((col for col in posibles_cols_fecha_fin if col in promos.columns), None)

    if col_sku is None or col_inicio is None:
        return ventas

    promos[col_sku] = promos[col_sku].astype(str)
    promos[col_inicio] = parse_transaction_dates(promos[col_inicio])

    if col_fin is not None:
        promos[col_fin] = parse_transaction_dates(promos[col_fin])
    else:
        col_fin = col_inicio

    promos[col_fin] = promos[col_fin].fillna(promos[col_inicio])

    registros = []
    for _, promo in promos.dropna(subset=[col_sku, col_inicio]).iterrows():
        fecha_inicio = promo[col_inicio]
        fecha_fin = promo[col_fin] if pd.notna(promo[col_fin]) else fecha_inicio
        if pd.isna(fecha_inicio) or pd.isna(fecha_fin):
            continue
        meses_promo = pd.period_range(fecha_inicio.to_period("M"), fecha_fin.to_period("M"), freq="M")
        for mes_promo in meses_promo:
            registros.append({"prod_nbr": str(promo[col_sku]), "mes": mes_promo})

    promociones_mes = pd.DataFrame(registros)
    if promociones_mes.empty:
        return ventas

    promociones_mes = (
        promociones_mes.groupby(["prod_nbr", "mes"], as_index=False)
        .size()
        .rename(columns={"size": "num_promociones"})
    )
    promociones_mes["tiene_promocion"] = 1

    ventas = ventas.merge(
        promociones_mes,
        on=["prod_nbr", "mes"],
        how="left",
        suffixes=("", "_promo"),
    )

    ventas["num_promociones"] = ventas["num_promociones_promo"].fillna(0)
    ventas["tiene_promocion"] = ventas["tiene_promocion_promo"].fillna(0)
    ventas = ventas.drop(
        columns=[c for c in ["num_promociones_promo", "tiene_promocion_promo"] if c in ventas.columns],
        errors="ignore",
    )

    return ventas


def build_three_month_blocks(ventas_base: pd.DataFrame) -> list[dict]:
    """Crea bloques fijos de 3 meses, igual que el notebook base."""
    meses_ordenados = sorted(ventas_base["mes"].dropna().unique())
    bloques = []
    for i in range(0, len(meses_ordenados), 3):
        meses_bloque = meses_ordenados[i : i + 3]
        if len(meses_bloque) < 3:
            continue
        periodo_3m = f"{meses_bloque[0]} a {meses_bloque[-1]}"
        bloques.append(
            {
                "bloque_id": len(bloques) + 1,
                "mes_inicio": meses_bloque[0],
                "mes_fin": meses_bloque[-1],
                "meses": meses_bloque,
                "periodo_3m": periodo_3m,
                "trimestre": build_quarter_label(periodo_3m),
            }
        )
    return bloques




PERIODOS_ELASTICIDAD = [
    "mensual",
    "trimestral",
    "semestral",
    "anual",
    "global_sku",
    "categoria_departamento",
]

# Columna de nivel socioeconómico usada para estratificar la elasticidad.
NSE_COL = "categoria_est_socio"

ELASTICIDADES_PERIODO_COLUMNS = [
    "SKU",
    "categoria",
    "departamento",
    "categoria_est_socio",   # Segmento NSE — NaN cuando no hay datos suficientes por segmento
    "periodo_tipo",
    "periodo",
    "fecha_inicio",
    "fecha_fin",
    "elasticidad",
    "r2",
    "p_value",
    "num_observaciones",
    "num_precios_distintos",
    "precio_promedio",
    "unidades_promedio",
    "ingreso_promedio",
    "margen_promedio",
    "confianza_elasticidad",
    "recomendable_elasticidad",
    "razon_no_recomendable",
]


def _finite_or_none(value):
    """Devuelve None para NaN/inf y float para valores numéricos finitos."""
    if value is None or pd.isna(value):
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if np.isfinite(value_float) else None


def _period_label(periodo_tipo: str, fecha_inicio, fecha_fin) -> str:
    if periodo_tipo == "mensual":
        return str(pd.Period(fecha_inicio, freq="M"))
    if periodo_tipo == "trimestral":
        return build_quarter_label(f"{pd.Period(fecha_inicio, freq='M')} a {pd.Period(fecha_fin, freq='M')}")
    if periodo_tipo == "semestral":
        return f"{pd.Period(fecha_inicio, freq='M')} a {pd.Period(fecha_fin, freq='M')}"
    if periodo_tipo == "anual":
        return str(pd.Timestamp(fecha_inicio).year)
    if periodo_tipo == "global_sku":
        return "global_sku"
    if periodo_tipo == "categoria_departamento":
        return "categoria_departamento"
    return str(periodo_tipo)


def build_period_blocks(ventas_base: pd.DataFrame, periodo_tipo: str) -> list[dict]:
    """Crea bloques de tiempo para la elasticidad multi-periodo."""
    if periodo_tipo not in PERIODOS_ELASTICIDAD:
        raise ValueError(f"periodo_tipo inválido: {periodo_tipo}. Opciones: {', '.join(PERIODOS_ELASTICIDAD)}")

    if ventas_base.empty:
        return []

    meses_ordenados = sorted(ventas_base["mes"].dropna().unique())
    if not meses_ordenados:
        return []

    if periodo_tipo == "trimestral":
        bloques = []
        for bloque in build_three_month_blocks(ventas_base):
            bloques.append(
                {
                    "periodo_tipo": periodo_tipo,
                    "periodo": bloque["trimestre"],
                    "fecha_inicio": bloque["mes_inicio"].to_timestamp(how="start").date(),
                    "fecha_fin": bloque["mes_fin"].to_timestamp(how="end").date(),
                    "meses": bloque["meses"],
                    **bloque,
                }
            )
        return bloques

    if periodo_tipo == "mensual":
        return [
            {
                "periodo_tipo": periodo_tipo,
                "periodo": str(mes),
                "fecha_inicio": mes.to_timestamp(how="start").date(),
                "fecha_fin": mes.to_timestamp(how="end").date(),
                "meses": [mes],
            }
            for mes in meses_ordenados
        ]

    if periodo_tipo == "semestral":
        bloques = []
        for i in range(0, len(meses_ordenados), 6):
            meses_bloque = meses_ordenados[i : i + 6]
            if len(meses_bloque) < 6:
                continue
            fecha_inicio = meses_bloque[0].to_timestamp(how="start").date()
            fecha_fin = meses_bloque[-1].to_timestamp(how="end").date()
            bloques.append(
                {
                    "periodo_tipo": periodo_tipo,
                    "periodo": _period_label(periodo_tipo, fecha_inicio, fecha_fin),
                    "fecha_inicio": fecha_inicio,
                    "fecha_fin": fecha_fin,
                    "meses": meses_bloque,
                }
            )
        return bloques

    if periodo_tipo == "anual":
        bloques = []
        meses_por_anio: dict[int, list] = {}
        for mes in meses_ordenados:
            meses_por_anio.setdefault(mes.year, []).append(mes)
        for anio, meses_bloque in meses_por_anio.items():
            meses_bloque = sorted(meses_bloque)
            fecha_inicio = meses_bloque[0].to_timestamp(how="start").date()
            fecha_fin = meses_bloque[-1].to_timestamp(how="end").date()
            bloques.append(
                {
                    "periodo_tipo": periodo_tipo,
                    "periodo": str(anio),
                    "fecha_inicio": fecha_inicio,
                    "fecha_fin": fecha_fin,
                    "meses": meses_bloque,
                }
            )
        return bloques

    fecha_inicio = meses_ordenados[0].to_timestamp(how="start").date()
    fecha_fin = meses_ordenados[-1].to_timestamp(how="end").date()
    return [
        {
            "periodo_tipo": periodo_tipo,
            "periodo": periodo_tipo,
            "fecha_inicio": fecha_inicio,
            "fecha_fin": fecha_fin,
            "meses": meses_ordenados,
        }
    ]


def evaluar_confianza_elasticidad(estimacion: dict, metricas: dict | None = None) -> dict:
    """Clasifica confianza y recomendabilidad de una elasticidad estimada."""
    metricas = metricas or {}
    elasticidad = _finite_or_none(estimacion.get("Elasticidad"))
    r2 = _finite_or_none(estimacion.get("R2"))
    p_value = _finite_or_none(estimacion.get("P_Value"))
    n_obs = int(metricas.get("num_observaciones") or estimacion.get("Observaciones_Modelo") or 0)
    n_modelo = int(estimacion.get("Observaciones_Modelo") or 0)
    precios = int(metricas.get("num_precios_distintos") or estimacion.get("Precios_Distintos_Modelo") or 0)

    razones_no_usable = []
    if elasticidad is None:
        razones_no_usable.append(estimacion.get("Motivo_Modelo") or "elasticidad NaN o infinita")
    elif elasticidad > 0:
        razones_no_usable.append("elasticidad positiva")
    elif elasticidad == 0:
        razones_no_usable.append("elasticidad cero sospechosa")
    if n_obs < MIN_OBSERVACIONES or n_modelo < MIN_OBSERVACIONES:
        razones_no_usable.append("datos insuficientes")
    if precios < 3:
        razones_no_usable.append("menos de 3 precios distintos")

    if razones_no_usable:
        return {
            "confianza_elasticidad": "No usable",
            "recomendable_elasticidad": False,
            "razon_no_recomendable": "; ".join(dict.fromkeys(str(r) for r in razones_no_usable if r)),
        }

    razones_baja = []
    if n_obs < 8 or n_modelo < 5:
        razones_baja.append("pocos datos")
    if r2 is None or r2 < 0.15:
        razones_baja.append("bajo R2")
    if p_value is not None and p_value > 0.20:
        razones_baja.append("p-value alto")
    if elasticidad < -10:
        razones_baja.append("comportamiento inestable")

    if razones_baja:
        return {
            "confianza_elasticidad": "Baja",
            "recomendable_elasticidad": False,
            "razon_no_recomendable": "; ".join(razones_baja),
        }

    if n_obs >= 15 and n_modelo >= 10 and precios >= 4 and r2 is not None and r2 >= 0.50 and -5 <= elasticidad < 0:
        return {
            "confianza_elasticidad": "Alta",
            "recomendable_elasticidad": True,
            "razon_no_recomendable": "",
        }

    return {
        "confianza_elasticidad": "Media",
        "recomendable_elasticidad": True,
        "razon_no_recomendable": "",
    }


def _metricas_periodo(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "num_observaciones": 0,
            "num_precios_distintos": 0,
            "precio_promedio": None,
            "unidades_promedio": None,
            "ingreso_promedio": None,
            "margen_promedio": None,
        }

    margen = pd.Series(np.nan, index=df.index, dtype="float64")
    if "margen_total" in df.columns:
        margen = pd.to_numeric(df["margen_total"], errors="coerce")
    elif "margen_unitario" in df.columns:
        margen = pd.to_numeric(df["margen_unitario"], errors="coerce") * pd.to_numeric(df["qty"], errors="coerce")
    elif "costo_unitario" in df.columns:
        margen = pd.to_numeric(df["net_sale"], errors="coerce") - (pd.to_numeric(df["costo_unitario"], errors="coerce") * pd.to_numeric(df["qty"], errors="coerce"))
    elif "costo2" in df.columns:
        margen = pd.to_numeric(df["net_sale"], errors="coerce") - (pd.to_numeric(df["costo2"], errors="coerce") * pd.to_numeric(df["qty"], errors="coerce"))

    valores = {
        "num_observaciones": int(len(df)),
        "num_precios_distintos": int(df["precio_unitario"].round(2).nunique()) if "precio_unitario" in df.columns else 0,
        "precio_promedio": _finite_or_none(pd.to_numeric(df.get("precio_unitario"), errors="coerce").mean()),
        "unidades_promedio": _finite_or_none(pd.to_numeric(df.get("qty"), errors="coerce").mean()),
        "ingreso_promedio": _finite_or_none(pd.to_numeric(df.get("net_sale"), errors="coerce").mean()),
        "margen_promedio": _finite_or_none(margen.mean()),
    }
    return valores


def _empty_elasticidades_periodo() -> pd.DataFrame:
    return pd.DataFrame(columns=ELASTICIDADES_PERIODO_COLUMNS)

def diagnosticar_elasticidad(beta) -> str:
    """Diagnóstico interpretativo de elasticidad."""
    if pd.isna(beta):
        return "Datos insuficientes"
    if beta >= 0:
        return "Relación positiva / revisar datos"
    if beta > -1:
        return "Inelástica"
    if beta == -1:
        return "Elasticidad unitaria"
    return "Elástica"


def preparar_df_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega ventas por día y precio para estimar demanda diaria por nivel de precio.
    """
    cols_req = ["fecha_dia", "precio_modelo", "qty", "net_sale"]
    df_m = df.dropna(subset=cols_req).copy()
    df_m = df_m[(df_m["qty"] > 0) & (df_m["net_sale"] > 0) & (df_m["precio_modelo"] > 0)].copy()

    if df_m.empty:
        return pd.DataFrame(columns=["qty_modelo", "precio_modelo"])

    df_agg = (
        df_m.groupby(["fecha_dia", "precio_modelo"], as_index=False)
        .agg(qty_modelo=("qty", "sum"), venta_modelo=("net_sale", "sum"))
    )

    df_agg["precio_modelo"] = df_agg["venta_modelo"] / df_agg["qty_modelo"]
    df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
    df_agg = df_agg.dropna(subset=["qty_modelo", "precio_modelo"])
    df_agg = df_agg[(df_agg["qty_modelo"] > 0) & (df_agg["precio_modelo"] > 0)].copy()

    return df_agg


def estimar_elasticidad_loglog(
    df: pd.DataFrame,
    fuente: str = "SKU-trimestre",
    min_observaciones: int = MIN_OBSERVACIONES,
    min_precios_distintos: int = MIN_PRECIOS_DISTINTOS,
) -> dict:
    """
    Estima elasticidad con OLS log-log:
    log(qty_modelo) = alfa + beta * log(precio_modelo).
    """
    df_modelo = preparar_df_modelo(df)

    n_modelo = len(df_modelo)
    precios_distintos = df_modelo["precio_modelo"].nunique() if not df_modelo.empty else 0
    qty_distintas = df_modelo["qty_modelo"].nunique() if not df_modelo.empty else 0

    if n_modelo < min_observaciones:
        return {
            "Beta": np.nan,
            "Elasticidad": np.nan,
            "Alfa": np.nan,
            "R2": np.nan,
            "P_Value": np.nan,
            "Observaciones_Modelo": n_modelo,
            "Precios_Distintos_Modelo": precios_distintos,
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": f"Menos de {min_observaciones} observaciones agregadas",
        }

    if precios_distintos < min_precios_distintos:
        return {
            "Beta": np.nan,
            "Elasticidad": np.nan,
            "Alfa": np.nan,
            "R2": np.nan,
            "P_Value": np.nan,
            "Observaciones_Modelo": n_modelo,
            "Precios_Distintos_Modelo": precios_distintos,
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": "Sin variación suficiente de precio",
        }

    if qty_distintas < 2:
        return {
            "Beta": 0.0,
            "Elasticidad": 0.0,
            "Alfa": float(np.log(df_modelo["qty_modelo"].iloc[0])),
            "R2": 0.0,
            "P_Value": 1.0,
            "Observaciones_Modelo": n_modelo,
            "Precios_Distintos_Modelo": precios_distintos,
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": "Cantidad agregada constante; elasticidad aproximada a 0",
        }

    df_modelo["log_qty"] = np.log(df_modelo["qty_modelo"])
    df_modelo["log_precio"] = np.log(df_modelo["precio_modelo"])
    df_modelo = df_modelo.replace([np.inf, -np.inf], np.nan)
    df_modelo = df_modelo.dropna(subset=["log_qty", "log_precio"])

    if len(df_modelo) < min_observaciones:
        return {
            "Beta": np.nan,
            "Elasticidad": np.nan,
            "Alfa": np.nan,
            "R2": np.nan,
            "P_Value": np.nan,
            "Observaciones_Modelo": len(df_modelo),
            "Precios_Distintos_Modelo": df_modelo["precio_modelo"].nunique() if not df_modelo.empty else 0,
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": "Observaciones insuficientes después de logs",
        }

    X = sm.add_constant(df_modelo["log_precio"], has_constant="add")
    y = df_modelo["log_qty"]

    try:
        modelo = sm.OLS(y, X).fit()
        beta = modelo.params.get("log_precio", np.nan)
        alfa = modelo.params.get("const", np.nan)
        r2 = modelo.rsquared if np.isfinite(modelo.rsquared) else np.nan
        p_value = modelo.pvalues.get("log_precio", np.nan)

        if pd.isna(p_value) and len(df_modelo) <= 2:
            motivo = "Modelo estimado, pero p-value no disponible por pocos grados de libertad"
        else:
            motivo = "Modelo estimado correctamente"

        return {
            "Beta": beta,
            "Elasticidad": beta,
            "Alfa": alfa,
            "R2": r2,
            "P_Value": p_value,
            "Observaciones_Modelo": len(df_modelo),
            "Precios_Distintos_Modelo": df_modelo["precio_modelo"].nunique(),
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": motivo,
        }
    except Exception as exc:
        return {
            "Beta": np.nan,
            "Elasticidad": np.nan,
            "Alfa": np.nan,
            "R2": np.nan,
            "P_Value": np.nan,
            "Observaciones_Modelo": len(df_modelo),
            "Precios_Distintos_Modelo": df_modelo["precio_modelo"].nunique(),
            "Fuente_Elasticidad": fuente,
            "Motivo_Modelo": f"Error statsmodels: {str(exc)[:120]}",
        }


def _moda_segura(df: pd.DataFrame, col: str):
    if col in df.columns:
        m = df[col].dropna().mode()
        if not m.empty:
            return m.iloc[0]
    return np.nan


def mejor_estimacion_con_fallback(
    df_sku_trimestre: pd.DataFrame,
    sku: str,
    bloque: dict,
    df_bloque: pd.DataFrame,
    ventas_completa: pd.DataFrame,
) -> dict:
    """
    Intenta SKU-trimestre y luego fallback:
    SKU global, subdepartamento-trimestre, departamento-trimestre, total-trimestre.
    """
    estimacion = estimar_elasticidad_loglog(df_sku_trimestre, fuente="SKU-trimestre")

    if pd.notna(estimacion["Elasticidad"]):
        return estimacion

    df_sku_global = ventas_completa[ventas_completa["prod_nbr"].astype(str) == str(sku)].copy()
    est_sku_global = estimar_elasticidad_loglog(df_sku_global, fuente="SKU-global")
    if pd.notna(est_sku_global["Elasticidad"]):
        return est_sku_global

    if "subdept_nm" in df_sku_trimestre.columns:
        modos = df_sku_trimestre["subdept_nm"].dropna().mode()
        if not modos.empty:
            subdept = modos.iloc[0]
            df_subdept_bloque = df_bloque[df_bloque["subdept_nm"] == subdept].copy()
            est_subdept = estimar_elasticidad_loglog(df_subdept_bloque, fuente="Subdepartamento-trimestre")
            if pd.notna(est_subdept["Elasticidad"]):
                return est_subdept

    if "dept_nm" in df_sku_trimestre.columns:
        modos = df_sku_trimestre["dept_nm"].dropna().mode()
        if not modos.empty:
            dept = modos.iloc[0]
            df_dept_bloque = df_bloque[df_bloque["dept_nm"] == dept].copy()
            est_dept = estimar_elasticidad_loglog(df_dept_bloque, fuente="Departamento-trimestre")
            if pd.notna(est_dept["Elasticidad"]):
                return est_dept

    est_total = estimar_elasticidad_loglog(df_bloque, fuente="Total-trimestre")
    if pd.notna(est_total["Elasticidad"]):
        return est_total

    return estimacion




def _preparar_ventas_elasticidad(
    ventas_nse: pd.DataFrame,
    promociones: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Normaliza ventas para cualquier cálculo de elasticidad."""
    ventas = ventas_nse.copy()
    ventas.columns = ventas.columns.astype(str).str.strip()

    ventas["tran_date"] = parse_transaction_dates(ventas["tran_date"])
    ventas["qty"] = pd.to_numeric(ventas["qty"], errors="coerce")
    ventas["net_sale"] = pd.to_numeric(ventas["net_sale"], errors="coerce")
    ventas["prod_nbr"] = ventas["prod_nbr"].astype(str)

    if "precio_unitario" not in ventas.columns:
        ventas["precio_unitario"] = ventas["net_sale"] / ventas["qty"]
    else:
        ventas["precio_unitario"] = pd.to_numeric(ventas["precio_unitario"], errors="coerce")
        ventas["precio_unitario"] = ventas["precio_unitario"].fillna(ventas["net_sale"] / ventas["qty"])

    if "costo_unitario" not in ventas.columns and "costo2" in ventas.columns:
        ventas["costo_unitario"] = pd.to_numeric(ventas["costo2"], errors="coerce")
    elif "costo_unitario" in ventas.columns:
        ventas["costo_unitario"] = pd.to_numeric(ventas["costo_unitario"], errors="coerce")

    if "costo_unitario" in ventas.columns:
        ventas["margen_unitario"] = ventas["precio_unitario"] - ventas["costo_unitario"]
        ventas["margen_total"] = ventas["margen_unitario"] * ventas["qty"]

    ventas = ventas.replace([np.inf, -np.inf], np.nan)
    ventas = ventas.dropna(subset=["tran_date", "qty", "net_sale", "prod_nbr", "precio_unitario"]).copy()
    ventas = ventas[(ventas["qty"] > 0) & (ventas["net_sale"] > 0) & (ventas["precio_unitario"] > 0)].copy()

    ventas["mes"] = ventas["tran_date"].dt.to_period("M")
    ventas["fecha_dia"] = ventas["tran_date"].dt.date
    ventas["precio_modelo"] = ventas["precio_unitario"].round(2)

    ventas = integrate_promotions(ventas, promociones)
    bloques = build_three_month_blocks(ventas)
    return ventas, bloques


def _row_elasticidad_periodo(
    sku: str,
    df_sku: pd.DataFrame,
    bloque: dict,
    estimacion: dict,
    periodo_tipo: str,
    categoria: str | None = None,
    departamento: str | None = None,
) -> dict:
    metricas = _metricas_periodo(df_sku)
    confianza = evaluar_confianza_elasticidad(estimacion, metricas)
    return {
        "SKU": str(sku),
        "categoria": categoria if categoria is not None else _moda_segura(df_sku, "subdept_nm"),
        "departamento": departamento if departamento is not None else _moda_segura(df_sku, "dept_nm"),
        "periodo_tipo": periodo_tipo,
        "periodo": bloque["periodo"],
        "fecha_inicio": bloque["fecha_inicio"],
        "fecha_fin": bloque["fecha_fin"],
        "elasticidad": _finite_or_none(estimacion.get("Elasticidad")),
        "r2": _finite_or_none(estimacion.get("R2")),
        "p_value": _finite_or_none(estimacion.get("P_Value")),
        **metricas,
        **confianza,
    }


def _fallback_categoria_departamento(
    df_sku: pd.DataFrame,
    df_periodo: pd.DataFrame,
    periodo_tipo: str,
) -> dict | None:
    """Intenta fallback confiable por categoría y departamento para un SKU-periodo."""
    categoria = _moda_segura(df_sku, "subdept_nm")
    departamento = _moda_segura(df_sku, "dept_nm")

    candidatos: list[tuple[str, pd.DataFrame]] = []
    if pd.notna(categoria) and "subdept_nm" in df_periodo.columns:
        candidatos.append(("Categoría/departamento-periodo", df_periodo[df_periodo["subdept_nm"] == categoria].copy()))
    if pd.notna(departamento) and "dept_nm" in df_periodo.columns:
        candidatos.append(("Departamento-periodo", df_periodo[df_periodo["dept_nm"] == departamento].copy()))

    for fuente, df_fallback in candidatos:
        est = estimar_elasticidad_loglog(df_fallback, fuente=f"{fuente}-{periodo_tipo}")
        metricas = _metricas_periodo(df_fallback)
        confianza = evaluar_confianza_elasticidad(est, metricas)
        if confianza["recomendable_elasticidad"]:
            est["Motivo_Modelo"] = f"Fallback confiable: {fuente}"
            return est
    return None


def _calculate_elasticity_period_prepared(ventas: pd.DataFrame, periodo_tipo: str) -> pd.DataFrame:
    """Calcula la tabla interna elasticidades_periodo para un periodo_tipo."""
    if periodo_tipo not in PERIODOS_ELASTICIDAD:
        raise ValueError(f"periodo_tipo inválido: {periodo_tipo}. Opciones: {', '.join(PERIODOS_ELASTICIDAD)}")
    if ventas.empty:
        return _empty_elasticidades_periodo()

    resultados = []
    bloques = build_period_blocks(ventas, periodo_tipo)

    if periodo_tipo == "categoria_departamento":
        for bloque in bloques:
            df_periodo = ventas[ventas["mes"].isin(bloque["meses"])].copy()
            group_cols = [col for col in ["subdept_nm", "dept_nm"] if col in df_periodo.columns]
            if not group_cols:
                continue
            for keys, df_grupo in df_periodo.groupby(group_cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                valores = dict(zip(group_cols, keys))
                categoria = valores.get("subdept_nm", np.nan)
                departamento = valores.get("dept_nm", np.nan)
                est = estimar_elasticidad_loglog(df_grupo, fuente="Categoría/departamento-global")
                resultados.append(
                    _row_elasticidad_periodo(
                        sku=f"{departamento}|{categoria}",
                        df_sku=df_grupo,
                        bloque=bloque,
                        estimacion=est,
                        periodo_tipo=periodo_tipo,
                        categoria=categoria,
                        departamento=departamento,
                    )
                )
        return pd.DataFrame(resultados, columns=ELASTICIDADES_PERIODO_COLUMNS).replace([np.inf, -np.inf], np.nan)

    for bloque in bloques:
        df_periodo = ventas[ventas["mes"].isin(bloque["meses"])].copy()
        for sku, df_sku in df_periodo.groupby("prod_nbr"):
            fuente = f"SKU-{periodo_tipo}"
            est = estimar_elasticidad_loglog(df_sku, fuente=fuente)
            metricas = _metricas_periodo(df_sku)
            confianza = evaluar_confianza_elasticidad(est, metricas)
            if not confianza["recomendable_elasticidad"]:
                fallback = _fallback_categoria_departamento(df_sku, df_periodo, periodo_tipo)
                if fallback is not None:
                    est = fallback

            resultados.append(_row_elasticidad_periodo(sku, df_sku, bloque, est, periodo_tipo))

    out = pd.DataFrame(resultados, columns=ELASTICIDADES_PERIODO_COLUMNS)
    if not out.empty:
        out = out.replace([np.inf, -np.inf], np.nan).sort_values(
            by=["SKU", "periodo_tipo", "fecha_inicio"], kind="stable"
        ).reset_index(drop=True)
    return out


def calculate_elasticity_by_period(
    ventas_nse: pd.DataFrame,
    promociones: pd.DataFrame | None = None,
    periodo_tipo: str = "trimestral",
) -> pd.DataFrame:
    """Función general de elasticidad para mensual/trimestral/semestral/anual/global/fallback."""
    ventas, _ = _preparar_ventas_elasticidad(ventas_nse, promociones)
    return _calculate_elasticity_period_prepared(ventas, periodo_tipo)


def calculate_elasticidades_periodo(
    ventas_nse: pd.DataFrame,
    promociones: pd.DataFrame | None = None,
    periodo_tipos: list[str] | None = None,
) -> pd.DataFrame:
    """Construye la tabla interna elasticidades_periodo para todos los niveles solicitados."""
    periodo_tipos = periodo_tipos or PERIODOS_ELASTICIDAD
    ventas, _ = _preparar_ventas_elasticidad(ventas_nse, promociones)
    tablas = [_calculate_elasticity_period_prepared(ventas, periodo_tipo) for periodo_tipo in periodo_tipos]
    tablas = [tabla for tabla in tablas if tabla is not None and not tabla.empty]
    if not tablas:
        return _empty_elasticidades_periodo()
    elasticidades_periodo = pd.concat(tablas, ignore_index=True)
    elasticidades_periodo = elasticidades_periodo.replace([np.inf, -np.inf], np.nan)
    return elasticidades_periodo[ELASTICIDADES_PERIODO_COLUMNS]

def calculate_elasticity(
    ventas_nse: pd.DataFrame,
    promociones: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Calcula elasticidad trimestral robusta por SKU usando la base limpia + NSE.

    La entrada debe ser la base maestra generada en la vista 1:
    ventas_nse = ventas limpias + cruce con NSE. Así la elasticidad conserva
    categoria_est_socio y estado para las vistas posteriores.
    """
    ventas, bloques = _preparar_ventas_elasticidad(ventas_nse, promociones)
    tablas_periodo = [
        _calculate_elasticity_period_prepared(ventas, periodo_tipo)
        for periodo_tipo in PERIODOS_ELASTICIDAD
    ]
    tablas_periodo = [tabla for tabla in tablas_periodo if tabla is not None and not tabla.empty]
    elasticidades_periodo = (
        pd.concat(tablas_periodo, ignore_index=True)[ELASTICIDADES_PERIODO_COLUMNS]
        if tablas_periodo
        else _empty_elasticidades_periodo()
    )




def _fast_mode_by_group(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """Moda por grupo sin groupby.apply para descriptores de elasticidad."""
    if value_col not in df.columns or df.empty:
        return pd.DataFrame(columns=group_cols + [value_col])
    tmp = df[group_cols + [value_col]].copy()
    tmp[value_col] = tmp[value_col].replace(["", " ", "nan", "NaN", "None", "none", "null", "Null"], np.nan)
    tmp = tmp.dropna(subset=[value_col])
    if tmp.empty:
        return pd.DataFrame(columns=group_cols + [value_col])
    counts = tmp.groupby(group_cols + [value_col], observed=True, sort=False).size().reset_index(name="_n")
    counts = counts.sort_values(group_cols + ["_n"], ascending=[True] * len(group_cols) + [False])
    return counts.drop_duplicates(group_cols, keep="first")[group_cols + [value_col]]


def _attach_fast_modes(base: pd.DataFrame, ventas: pd.DataFrame, group_cols: list[str], descriptoras: list[str]) -> pd.DataFrame:
    out = base.copy()
    for col in descriptoras:
        if col in ventas.columns:
            out = out.merge(_fast_mode_by_group(ventas, group_cols, col), on=group_cols, how="left")
        elif col not in out.columns:
            out[col] = np.nan
    return out


def _apply_nse_fallback_vectorized(
    base_nse: pd.DataFrame,
    base_sku: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fallback NSE → SKU completo (vectorizado).

    Para cada fila de base_nse cuya elasticidad no es estimable (NaN), busca
    la elasticidad del mismo SKU × periodo en base_sku (que agrega todos los
    segmentos NSE). Si existe un valor usable, lo copia y marca el motivo.
    Filas que tampoco tienen fallback en base_sku quedan con NaN y seguirán
    al fallback por categoría/departamento que se aplica después.
    """
    base = base_nse.copy()
    # Evaluate the confidence of each NSE-segment row using the same criteria as the
    # main pipeline. Only fall back when the segment estimate is NOT recomendable.
    eval_tmp = pd.DataFrame({
        "SKU": base.get("prod_nbr", pd.RangeIndex(len(base))),
        "elasticidad": pd.to_numeric(base.get("Elasticidad"), errors="coerce"),
        "r2": pd.to_numeric(base.get("R2"), errors="coerce"),
        "p_value": pd.to_numeric(base.get("P_Value"), errors="coerce"),
        "num_observaciones": pd.to_numeric(base.get("num_observaciones"), errors="coerce").fillna(0),
        "num_precios_distintos": pd.to_numeric(base.get("num_precios_distintos"), errors="coerce").fillna(0),
        "Observaciones_Modelo": pd.to_numeric(base.get("Observaciones_Modelo"), errors="coerce").fillna(0),
        "Motivo_Modelo": base.get("Motivo_Modelo", pd.Series("", index=base.index)).fillna(""),
    })
    eval_tmp = _evaluate_confidence_frame(eval_tmp)
    needs_fb = ~eval_tmp["recomendable_elasticidad"].fillna(False)
    if not needs_fb.any() or base_sku.empty:
        return base

    merge_on = [c for c in ["prod_nbr", "periodo", "fecha_inicio", "fecha_fin"] if c in base_sku.columns]
    if not merge_on:
        return base

    fb_cols_src = ["Elasticidad", "R2", "P_Value", "Observaciones_Modelo",
                   "Precios_Distintos_Modelo", "Fuente_Elasticidad", "Motivo_Modelo"]
    fb_cols_present = [c for c in fb_cols_src if c in base_sku.columns]
    fb = base_sku[merge_on + fb_cols_present].copy()
    fb = fb.rename(columns={c: f"{c}_fb" for c in fb_cols_present})

    left = base.loc[needs_fb, [NSE_COL, *merge_on]].copy()
    left["_idx"] = left.index
    candidate = left.merge(fb, on=merge_on, how="left")

    if candidate.empty or "Elasticidad_fb" not in candidate.columns:
        return base

    usable = candidate["Elasticidad_fb"].notna()
    if not usable.any():
        return base

    usable_cands = candidate.loc[usable].drop_duplicates("_idx", keep="first")
    target_idx = usable_cands["_idx"].astype(int).to_numpy()

    for src_col in fb_cols_src:
        fb_col = f"{src_col}_fb"
        if fb_col in usable_cands.columns and src_col in base.columns:
            base.loc[target_idx, src_col] = usable_cands[fb_col].values

    base.loc[target_idx, "Motivo_Modelo"] = (
        "Fallback NSE→SKU completo: datos insuficientes en segmento NSE"
    )
    return base


def _assign_period_columns(df: pd.DataFrame, periodo_tipo: str) -> pd.DataFrame:
    """Agrega columnas de periodo a una copia para cálculos vectorizados."""
    out = df.copy()
    if periodo_tipo == "mensual":
        out["periodo"] = out["mes"].astype(str)
        out["fecha_inicio"] = out["mes"].dt.to_timestamp(how="start").dt.date
        out["fecha_fin"] = out["mes"].dt.to_timestamp(how="end").dt.date
    elif periodo_tipo == "trimestral":
        mapa = {}
        for bloque in build_three_month_blocks(out):
            for mes in bloque["meses"]:
                mapa[mes] = {
                    "periodo": bloque["trimestre"],
                    "fecha_inicio": bloque["mes_inicio"].to_timestamp(how="start").date(),
                    "fecha_fin": bloque["mes_fin"].to_timestamp(how="end").date(),
                    "periodo_3m": bloque["periodo_3m"],
                    "trimestre": bloque["trimestre"],
                    "mes_inicio": str(bloque["mes_inicio"]),
                    "mes_fin": str(bloque["mes_fin"]),
                }
        meta = out["mes"].map(mapa)
        out = out[meta.notna()].copy()
        if out.empty:
            return out
        meta_df = pd.DataFrame(out["mes"].map(mapa).tolist(), index=out.index)
        for col in meta_df.columns:
            out[col] = meta_df[col].values
        return out
    elif periodo_tipo == "semestral":
        meses_ordenados = sorted(out["mes"].dropna().unique())
        mapa = {}
        for i in range(0, len(meses_ordenados), 6):
            meses_bloque = meses_ordenados[i : i + 6]
            if len(meses_bloque) < 6:
                continue
            fecha_inicio = meses_bloque[0].to_timestamp(how="start").date()
            fecha_fin = meses_bloque[-1].to_timestamp(how="end").date()
            periodo = _period_label(periodo_tipo, fecha_inicio, fecha_fin)
            for mes in meses_bloque:
                mapa[mes] = {"periodo": periodo, "fecha_inicio": fecha_inicio, "fecha_fin": fecha_fin}
        meta = out["mes"].map(mapa)
        out = out[meta.notna()].copy()
        if out.empty:
            return out
        meta_df = pd.DataFrame(out["mes"].map(mapa).tolist(), index=out.index)
        for col in meta_df.columns:
            out[col] = meta_df[col].values
        return out
    elif periodo_tipo == "anual":
        out["periodo"] = out["mes"].dt.year.astype(str)
        anio_inicio = out.groupby("periodo", observed=True)["mes"].transform("min")
        anio_fin = out.groupby("periodo", observed=True)["mes"].transform("max")
        out["fecha_inicio"] = anio_inicio.dt.to_timestamp(how="start").dt.date
        out["fecha_fin"] = anio_fin.dt.to_timestamp(how="end").dt.date
    else:
        meses = sorted(out["mes"].dropna().unique())
        if not meses:
            return out.iloc[0:0].copy()
        out["periodo"] = periodo_tipo
        out["fecha_inicio"] = meses[0].to_timestamp(how="start").date()
        out["fecha_fin"] = meses[-1].to_timestamp(how="end").date()
    return out


def _build_model_rows(df_periodo: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Replica preparar_df_modelo por grupo, pero en una sola agregación vectorizada."""
    cols = group_cols + ["fecha_dia", "precio_modelo", "qty", "net_sale"]
    if df_periodo.empty:
        return pd.DataFrame(columns=group_cols + ["qty_modelo", "precio_modelo"])
    tmp = df_periodo[cols].dropna(subset=["fecha_dia", "precio_modelo", "qty", "net_sale"]).copy()
    tmp = tmp[(tmp["qty"] > 0) & (tmp["net_sale"] > 0) & (tmp["precio_modelo"] > 0)]
    if tmp.empty:
        return pd.DataFrame(columns=group_cols + ["qty_modelo", "precio_modelo"])
    model = (
        tmp.groupby(group_cols + ["fecha_dia", "precio_modelo"], observed=True, sort=False, as_index=False)
        .agg(qty_modelo=("qty", "sum"), venta_modelo=("net_sale", "sum"))
    )
    model["precio_modelo"] = model["venta_modelo"] / model["qty_modelo"]
    model = model.replace([np.inf, -np.inf], np.nan).dropna(subset=["qty_modelo", "precio_modelo"])
    model = model[(model["qty_modelo"] > 0) & (model["precio_modelo"] > 0)].copy()
    return model


def _estimate_loglog_grouped(model: pd.DataFrame, group_cols: list[str], fuente: str) -> pd.DataFrame:
    """Estima OLS log-log para todos los grupos con fórmulas cerradas."""
    result_cols = group_cols + [
        "Beta",
        "Elasticidad",
        "Alfa",
        "R2",
        "P_Value",
        "Observaciones_Modelo",
        "Precios_Distintos_Modelo",
        "Fuente_Elasticidad",
        "Motivo_Modelo",
    ]
    if model.empty:
        return pd.DataFrame(columns=result_cols)

    m = model[group_cols + ["qty_modelo", "precio_modelo"]].copy()
    m["log_qty"] = np.log(m["qty_modelo"])
    m["log_precio"] = np.log(m["precio_modelo"])
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_qty", "log_precio"])
    if m.empty:
        return pd.DataFrame(columns=result_cols)

    m["x2"] = m["log_precio"] ** 2
    m["y2"] = m["log_qty"] ** 2
    m["xy"] = m["log_precio"] * m["log_qty"]
    agg = (
        m.groupby(group_cols, observed=True, sort=False)
        .agg(
            Observaciones_Modelo=("log_qty", "size"),
            Precios_Distintos_Modelo=("precio_modelo", "nunique"),
            Qty_Distintas_Modelo=("qty_modelo", "nunique"),
            sum_x=("log_precio", "sum"),
            sum_y=("log_qty", "sum"),
            sum_x2=("x2", "sum"),
            sum_y2=("y2", "sum"),
            sum_xy=("xy", "sum"),
        )
        .reset_index()
    )

    n = agg["Observaciones_Modelo"].astype(float)
    sxx = agg["sum_x2"] - (agg["sum_x"] ** 2) / n
    syy = agg["sum_y2"] - (agg["sum_y"] ** 2) / n
    sxy = agg["sum_xy"] - (agg["sum_x"] * agg["sum_y"]) / n

    valid = (agg["Observaciones_Modelo"] >= MIN_OBSERVACIONES) & (agg["Precios_Distintos_Modelo"] >= MIN_PRECIOS_DISTINTOS) & (sxx > 0)
    constant_qty = valid & (agg["Qty_Distintas_Modelo"] < 2)
    model_ok = valid & ~constant_qty

    agg["Beta"] = np.nan
    agg["Elasticidad"] = np.nan
    agg["Alfa"] = np.nan
    agg["R2"] = np.nan
    agg["P_Value"] = np.nan
    agg["Motivo_Modelo"] = "Modelo no estimable"

    agg.loc[agg["Observaciones_Modelo"] < MIN_OBSERVACIONES, "Motivo_Modelo"] = f"Menos de {MIN_OBSERVACIONES} observaciones agregadas"
    agg.loc[(agg["Observaciones_Modelo"] >= MIN_OBSERVACIONES) & (agg["Precios_Distintos_Modelo"] < MIN_PRECIOS_DISTINTOS), "Motivo_Modelo"] = "Sin variación suficiente de precio"

    agg.loc[constant_qty, "Beta"] = 0.0
    agg.loc[constant_qty, "Elasticidad"] = 0.0
    agg.loc[constant_qty, "Alfa"] = agg.loc[constant_qty, "sum_y"] / n[constant_qty]
    agg.loc[constant_qty, "R2"] = 0.0
    agg.loc[constant_qty, "P_Value"] = 1.0
    agg.loc[constant_qty, "Motivo_Modelo"] = "Cantidad agregada constante; elasticidad aproximada a 0"

    beta = sxy[model_ok] / sxx[model_ok]
    alfa = (agg.loc[model_ok, "sum_y"] / n[model_ok]) - beta * (agg.loc[model_ok, "sum_x"] / n[model_ok])
    r2 = (sxy[model_ok] ** 2) / (sxx[model_ok] * syy[model_ok])
    r2 = r2.where(np.isfinite(r2), np.nan).clip(lower=0, upper=1)
    sse = (syy[model_ok] - beta * sxy[model_ok]).clip(lower=0)
    df_resid = n[model_ok] - 2
    se_beta = np.sqrt((sse / df_resid) / sxx[model_ok])
    t_stat = beta / se_beta
    p_values = pd.Series(2 * stats.t.sf(np.abs(t_stat), df_resid), index=beta.index)
    p_values = p_values.where(np.isfinite(p_values), np.nan)

    agg.loc[model_ok, "Beta"] = beta
    agg.loc[model_ok, "Elasticidad"] = beta
    agg.loc[model_ok, "Alfa"] = alfa
    agg.loc[model_ok, "R2"] = r2
    agg.loc[model_ok, "P_Value"] = p_values
    agg.loc[model_ok, "Motivo_Modelo"] = "Modelo estimado correctamente"
    agg["Fuente_Elasticidad"] = fuente

    return agg[result_cols].replace([np.inf, -np.inf], np.nan)


def _raw_metrics_grouped(df_periodo: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df_periodo.empty:
        return pd.DataFrame(columns=group_cols)
    tmp = df_periodo.copy()
    if "margen_total" not in tmp.columns:
        tmp["margen_total"] = np.nan
    return (
        tmp.groupby(group_cols, observed=True, sort=False)
        .agg(
            num_observaciones=("prod_nbr", "size"),
            num_precios_distintos=("precio_unitario", lambda s: s.round(2).nunique()),
            precio_promedio=("precio_unitario", "mean"),
            unidades_promedio=("qty", "mean"),
            ingreso_promedio=("net_sale", "mean"),
            margen_promedio=("margen_total", "mean"),
        )
        .reset_index()
    )


def _evaluate_confidence_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    elasticidad = pd.to_numeric(out.get("elasticidad"), errors="coerce")
    r2 = pd.to_numeric(out.get("r2"), errors="coerce")
    p_value = pd.to_numeric(out.get("p_value"), errors="coerce")
    n_obs = pd.to_numeric(out.get("num_observaciones"), errors="coerce").fillna(0)
    n_modelo = pd.to_numeric(out.get("Observaciones_Modelo"), errors="coerce").fillna(0)
    precios = pd.to_numeric(out.get("num_precios_distintos"), errors="coerce").fillna(0)
    motivo = out.get("Motivo_Modelo", pd.Series("", index=out.index)).fillna("").astype(str)

    no_usable = elasticidad.isna() | ~np.isfinite(elasticidad) | (elasticidad >= 0) | (n_obs < MIN_OBSERVACIONES) | (n_modelo < MIN_OBSERVACIONES) | (precios < 3)
    baja = (~no_usable) & ((n_obs < 8) | (n_modelo < 5) | r2.isna() | (r2 < 0.15) | (p_value > 0.20) | (elasticidad < -10))
    alta = (~no_usable) & (~baja) & (n_obs >= 15) & (n_modelo >= 10) & (precios >= 4) & (r2 >= 0.50) & (elasticidad >= -5) & (elasticidad < 0)

    out["confianza_elasticidad"] = np.select([no_usable, baja, alta], ["No usable", "Baja", "Alta"], default="Media")
    out["recomendable_elasticidad"] = out["confianza_elasticidad"].isin(["Media", "Alta"])
    out["razon_no_recomendable"] = ""
    out.loc[elasticidad.isna() | ~np.isfinite(elasticidad), "razon_no_recomendable"] = motivo.where(motivo.ne(""), "elasticidad NaN o infinita")
    out.loc[elasticidad > 0, "razon_no_recomendable"] = "elasticidad positiva"
    out.loc[elasticidad == 0, "razon_no_recomendable"] = "elasticidad cero sospechosa"
    out.loc[(out["razon_no_recomendable"] == "") & ((n_obs < MIN_OBSERVACIONES) | (n_modelo < MIN_OBSERVACIONES)), "razon_no_recomendable"] = "datos insuficientes"
    out.loc[(out["razon_no_recomendable"] == "") & (precios < 3), "razon_no_recomendable"] = "menos de 3 precios distintos"
    out.loc[(out["razon_no_recomendable"] == "") & baja & ((n_obs < 8) | (n_modelo < 5)), "razon_no_recomendable"] = "pocos datos"
    out.loc[(out["razon_no_recomendable"] == "") & baja & (r2.isna() | (r2 < 0.15)), "razon_no_recomendable"] = "bajo R2"
    out.loc[(out["razon_no_recomendable"] == "") & baja & (p_value > 0.20), "razon_no_recomendable"] = "p-value alto"
    out.loc[(out["razon_no_recomendable"] == "") & baja & (elasticidad < -10), "razon_no_recomendable"] = "comportamiento inestable"
    return out


def _period_estimates_fast(ventas: pd.DataFrame, periodo_tipo: str, entity_cols: list[str], fuente: str) -> pd.DataFrame:
    df_periodo = _assign_period_columns(ventas, periodo_tipo)
    group_cols = entity_cols + ["periodo", "fecha_inicio", "fecha_fin"]
    if df_periodo.empty:
        return pd.DataFrame(columns=group_cols)
    model = _build_model_rows(df_periodo, group_cols)
    estimates = _estimate_loglog_grouped(model, group_cols, fuente)
    metrics = _raw_metrics_grouped(df_periodo, group_cols)
    out = metrics.merge(estimates, on=group_cols, how="left")
    return out.replace([np.inf, -np.inf], np.nan)

def _row_elasticidad_periodo(
    sku: str,
    df_sku: pd.DataFrame,
    bloque: dict,
    estimacion: dict,
    periodo_tipo: str,
    categoria: str | None = None,
    departamento: str | None = None,
) -> dict:
    metricas = _metricas_periodo(df_sku)
    confianza = evaluar_confianza_elasticidad(estimacion, metricas)
    return {
        "SKU": str(sku),
        "categoria": categoria if categoria is not None else _moda_segura(df_sku, "subdept_nm"),
        "departamento": departamento if departamento is not None else _moda_segura(df_sku, "dept_nm"),
        "periodo_tipo": periodo_tipo,
        "periodo": bloque["periodo"],
        "fecha_inicio": bloque["fecha_inicio"],
        "fecha_fin": bloque["fecha_fin"],
        "elasticidad": _finite_or_none(estimacion.get("Elasticidad")),
        "r2": _finite_or_none(estimacion.get("R2")),
        "p_value": _finite_or_none(estimacion.get("P_Value")),
        **metricas,
        **confianza,
    }


def _fallback_categoria_departamento(
    df_sku: pd.DataFrame,
    df_periodo: pd.DataFrame,
    periodo_tipo: str,
) -> dict | None:
    """Intenta fallback confiable por categoría y departamento para un SKU-periodo."""
    categoria = _moda_segura(df_sku, "subdept_nm")
    departamento = _moda_segura(df_sku, "dept_nm")

    candidatos: list[tuple[str, pd.DataFrame]] = []
    if pd.notna(categoria) and "subdept_nm" in df_periodo.columns:
        candidatos.append(("Categoría/departamento-periodo", df_periodo[df_periodo["subdept_nm"] == categoria].copy()))
    if pd.notna(departamento) and "dept_nm" in df_periodo.columns:
        candidatos.append(("Departamento-periodo", df_periodo[df_periodo["dept_nm"] == departamento].copy()))

    for fuente, df_fallback in candidatos:
        est = estimar_elasticidad_loglog(df_fallback, fuente=f"{fuente}-{periodo_tipo}")
        metricas = _metricas_periodo(df_fallback)
        confianza = evaluar_confianza_elasticidad(est, metricas)
        if confianza["recomendable_elasticidad"]:
            est["Motivo_Modelo"] = f"Fallback confiable: {fuente}"
            return est
    return None


def _calculate_elasticity_period_prepared(ventas: pd.DataFrame, periodo_tipo: str) -> pd.DataFrame:
    """Calcula elasticidades_periodo con agregaciones vectorizadas."""
    if periodo_tipo not in PERIODOS_ELASTICIDAD:
        raise ValueError(f"periodo_tipo inválido: {periodo_tipo}. Opciones: {', '.join(PERIODOS_ELASTICIDAD)}")
    if ventas.empty:
        return _empty_elasticidades_periodo()

    if periodo_tipo == "categoria_departamento":
        group_cols = [col for col in ["subdept_nm", "dept_nm"] if col in ventas.columns]
        if not group_cols:
            return _empty_elasticidades_periodo()
        base = _period_estimates_fast(ventas, periodo_tipo, group_cols, "Categoría/departamento-global")
        if base.empty:
            return _empty_elasticidades_periodo()
        if "subdept_nm" not in base.columns:
            base["subdept_nm"] = np.nan
        if "dept_nm" not in base.columns:
            base["dept_nm"] = np.nan
        out = base.rename(
            columns={
                "subdept_nm": "categoria",
                "dept_nm": "departamento",
                "Elasticidad": "elasticidad",
                "R2": "r2",
                "P_Value": "p_value",
            }
        )
        out["SKU"] = out["departamento"].astype(str) + "|" + out["categoria"].astype(str)
        out["periodo_tipo"] = periodo_tipo
        out[NSE_COL] = np.nan  # No se estratifica por NSE a nivel categoría/departamento
        out = _evaluate_confidence_frame(out)
        out = out.replace([np.inf, -np.inf], np.nan)
        return out[ELASTICIDADES_PERIODO_COLUMNS]

    # ── SKU-level base (siempre se calcula; sirve como fallback NSE) ───────────
    base_sku = _period_estimates_fast(ventas, periodo_tipo, ["prod_nbr"], f"SKU-{periodo_tipo}")
    if base_sku.empty:
        return _empty_elasticidades_periodo()

    period_df = _assign_period_columns(ventas, periodo_tipo)

    # ── Intento NSE-estratificado ──────────────────────────────────────────────
    # Si existe la columna NSE con valores válidos, calcula la elasticidad por
    # SKU × NSE × periodo. Las filas sin suficientes datos en el segmento NSE
    # hacen fallback a la elasticidad SKU × periodo (sin estratificación).
    nse_available = (
        NSE_COL in ventas.columns
        and ventas[NSE_COL].notna().any()
        and ventas[NSE_COL].astype(str).str.strip().ne("").any()
        and ventas[NSE_COL].astype(str).str.strip().ne("nan").any()
    )

    if nse_available:
        ventas_nse_ok = ventas[
            ventas[NSE_COL].notna()
            & ventas[NSE_COL].astype(str).str.strip().ne("")
            & ventas[NSE_COL].astype(str).str.strip().ne("nan")
            & ventas[NSE_COL].astype(str).str.lower().ne("nse_no_asignado")
        ].copy()

        base_nse_raw = (
            _period_estimates_fast(
                ventas_nse_ok, periodo_tipo,
                ["prod_nbr", NSE_COL],
                f"SKU-NSE-{periodo_tipo}",
            )
            if not ventas_nse_ok.empty
            else pd.DataFrame()
        )

        if not base_nse_raw.empty:
            # Fallback por segmento: si un segmento NSE no tiene suficientes datos,
            # sustituye con la elasticidad del SKU completo (todos los NSE juntos).
            base_nse_raw = _apply_nse_fallback_vectorized(base_nse_raw, base_sku)
            base = base_nse_raw
            entity_group_cols = ["prod_nbr", NSE_COL, "periodo", "fecha_inicio", "fecha_fin"]
        else:
            # Sin datos NSE válidos para ningún segmento → SKU-nivel
            base = base_sku
            entity_group_cols = ["prod_nbr", "periodo", "fecha_inicio", "fecha_fin"]
    else:
        base = base_sku
        entity_group_cols = ["prod_nbr", "periodo", "fecha_inicio", "fecha_fin"]

    # ── Adjuntar descriptores (categoria, departamento) ────────────────────────
    descriptoras = ["subdept_nm", "dept_nm"]
    attach_group_cols = [c for c in entity_group_cols if c in base.columns and c in period_df.columns]
    base = _attach_fast_modes(base, period_df, attach_group_cols, descriptoras)

    # ── Renombrar columnas al esquema canónico ─────────────────────────────────
    base = base.rename(
        columns={
            "prod_nbr": "SKU",
            "subdept_nm": "categoria",
            "dept_nm": "departamento",
            "Elasticidad": "elasticidad",
            "R2": "r2",
            "P_Value": "p_value",
        }
    )

    # Asegurar que NSE_COL esté presente (NaN si no hubo estratificación)
    if NSE_COL not in base.columns:
        base[NSE_COL] = np.nan

    base["periodo_tipo"] = periodo_tipo
    base = _evaluate_confidence_frame(base)

    # Fallback vectorizado: para filas no recomendables, intenta primero categoría y luego departamento
    # del mismo periodo. Evita estimar un modelo por SKU-periodo, que era el principal cuello de botella.
    needs_fallback = ~base["recomendable_elasticidad"]
    if needs_fallback.any():
        fallback_sources = []
        if "subdept_nm" in period_df.columns:
            cat = _period_estimates_fast(ventas, periodo_tipo, ["subdept_nm"], f"Categoría/departamento-periodo-{periodo_tipo}")
            if not cat.empty:
                cat = cat.rename(
                    columns={
                        "subdept_nm": "categoria",
                        "Elasticidad": "elasticidad_fb",
                        "R2": "r2_fb",
                        "P_Value": "p_value_fb",
                        "Observaciones_Modelo": "Observaciones_Modelo_fb",
                        "Precios_Distintos_Modelo": "Precios_Distintos_Modelo_fb",
                        "Motivo_Modelo": "Motivo_Modelo_fb",
                    }
                )
                cat["_fallback_nivel"] = "categoría/departamento"
                fallback_sources.append((cat, ["categoria", "periodo", "fecha_inicio", "fecha_fin"]))
        if "dept_nm" in period_df.columns:
            dept = _period_estimates_fast(ventas, periodo_tipo, ["dept_nm"], f"Departamento-periodo-{periodo_tipo}")
            if not dept.empty:
                dept = dept.rename(
                    columns={
                        "dept_nm": "departamento",
                        "Elasticidad": "elasticidad_fb",
                        "R2": "r2_fb",
                        "P_Value": "p_value_fb",
                        "Observaciones_Modelo": "Observaciones_Modelo_fb",
                        "Precios_Distintos_Modelo": "Precios_Distintos_Modelo_fb",
                        "Motivo_Modelo": "Motivo_Modelo_fb",
                    }
                )
                dept["_fallback_nivel"] = "departamento"
                fallback_sources.append((dept, ["departamento", "periodo", "fecha_inicio", "fecha_fin"]))

        for fallback_df, merge_cols in fallback_sources:
            left = base.loc[needs_fallback, ["SKU", *merge_cols]].copy()
            left["_base_index"] = left.index
            candidate = left.merge(fallback_df, on=merge_cols, how="left")
            if candidate.empty:
                continue
            candidate_eval = pd.DataFrame(
                {
                    "SKU": candidate["SKU"],
                    "elasticidad": candidate["elasticidad_fb"],
                    "r2": candidate["r2_fb"],
                    "p_value": candidate["p_value_fb"],
                    "num_observaciones": candidate["num_observaciones"],
                    "num_precios_distintos": candidate["num_precios_distintos"],
                    "Observaciones_Modelo": candidate["Observaciones_Modelo_fb"],
                    "Motivo_Modelo": candidate["Motivo_Modelo_fb"],
                }
            )
            candidate_eval = _evaluate_confidence_frame(candidate_eval)
            usable = candidate_eval["recomendable_elasticidad"].fillna(False)
            if not usable.any():
                continue
            usable_candidates = candidate.loc[usable.values].copy().drop_duplicates("_base_index", keep="first")
            target_idx = usable_candidates["_base_index"].astype(int).to_numpy()
            base.loc[target_idx, "elasticidad"] = usable_candidates["elasticidad_fb"].values
            base.loc[target_idx, "r2"] = usable_candidates["r2_fb"].values
            base.loc[target_idx, "p_value"] = usable_candidates["p_value_fb"].values
            base.loc[target_idx, "Observaciones_Modelo"] = usable_candidates["Observaciones_Modelo_fb"].values
            base.loc[target_idx, "Precios_Distintos_Modelo"] = usable_candidates["Precios_Distintos_Modelo_fb"].values
            base.loc[target_idx, "Motivo_Modelo"] = "Fallback confiable: " + usable_candidates["_fallback_nivel"].astype(str).values
            base = _evaluate_confidence_frame(base)
            needs_fallback = ~base["recomendable_elasticidad"]
            if not needs_fallback.any():
                break

    base = base.replace([np.inf, -np.inf], np.nan).sort_values(
        by=["SKU", "periodo_tipo", "fecha_inicio"], kind="stable"
    ).reset_index(drop=True)
    return base[ELASTICIDADES_PERIODO_COLUMNS]




def _apply_legacy_fallbacks(base: pd.DataFrame, ventas: pd.DataFrame, period_df: pd.DataFrame) -> pd.DataFrame:
    """Aplica fallback legacy solo cuando la elasticidad SKU-trimestre es NaN."""
    out = base.copy()
    needs = out["Elasticidad"].isna()
    if not needs.any():
        return out

    global_sku = _period_estimates_fast(ventas, "global_sku", ["prod_nbr"], "SKU-global")
    if not global_sku.empty:
        global_sku = global_sku.drop_duplicates("prod_nbr").rename(
            columns={c: f"{c}_fb" for c in ["Beta", "Elasticidad", "Alfa", "R2", "P_Value", "Observaciones_Modelo", "Precios_Distintos_Modelo", "Fuente_Elasticidad", "Motivo_Modelo"]}
        )
        candidate = out.loc[needs, ["prod_nbr"]].merge(global_sku, on="prod_nbr", how="left")
        usable = candidate["Elasticidad_fb"].notna()
        if usable.any():
            idx = out.index[needs][usable.values]
            for col in ["Beta", "Elasticidad", "Alfa", "R2", "P_Value", "Observaciones_Modelo", "Precios_Distintos_Modelo", "Fuente_Elasticidad", "Motivo_Modelo"]:
                out.loc[idx, col] = candidate.loc[usable, f"{col}_fb"].values
            needs = out["Elasticidad"].isna()
            if not needs.any():
                return out

    fallback_specs = []
    if "subdept_nm" in period_df.columns:
        fallback_specs.append((["subdept_nm"], "Subdepartamento-trimestre"))
    if "dept_nm" in period_df.columns:
        fallback_specs.append((["dept_nm"], "Departamento-trimestre"))

    for entity_cols, fuente in fallback_specs:
        fallback = _period_estimates_fast(ventas, "trimestral", entity_cols, fuente)
        if fallback.empty:
            continue
        rename_cols = {c: f"{c}_fb" for c in ["Beta", "Elasticidad", "Alfa", "R2", "P_Value", "Observaciones_Modelo", "Precios_Distintos_Modelo", "Fuente_Elasticidad", "Motivo_Modelo"]}
        fallback = fallback.rename(columns=rename_cols)
        merge_cols = entity_cols + ["periodo", "fecha_inicio", "fecha_fin"]
        candidate = out.loc[needs, ["prod_nbr", *merge_cols]].merge(fallback, on=merge_cols, how="left")
        usable = candidate["Elasticidad_fb"].notna()
        if usable.any():
            idx = out.index[needs][usable.values]
            for col in ["Beta", "Elasticidad", "Alfa", "R2", "P_Value", "Observaciones_Modelo", "Precios_Distintos_Modelo", "Fuente_Elasticidad", "Motivo_Modelo"]:
                out.loc[idx, col] = candidate.loc[usable, f"{col}_fb"].values
            needs = out["Elasticidad"].isna()
            if not needs.any():
                return out

    return out


def _calculate_legacy_quarterly_fast(ventas: pd.DataFrame) -> pd.DataFrame:
    """Construye la salida trimestral legacy sin ejecutar statsmodels por SKU-periodo."""
    period_df = _assign_period_columns(ventas, "trimestral")
    if period_df.empty:
        return pd.DataFrame()

    base = _period_estimates_fast(ventas, "trimestral", ["prod_nbr"], "SKU-trimestre")
    if base.empty:
        return pd.DataFrame()

    meta_cols = ["prod_nbr", "periodo", "fecha_inicio", "fecha_fin", "periodo_3m", "trimestre", "mes_inicio", "mes_fin"]
    meta = period_df[meta_cols].drop_duplicates()
    base = base.merge(meta, on=["prod_nbr", "periodo", "fecha_inicio", "fecha_fin"], how="left")

    descriptoras = ["dept_nm", "subdept_nm", "marca", "tipo_marca", "categoria_est_socio", "estado"]
    base = _attach_fast_modes(base, period_df, ["prod_nbr", "periodo", "fecha_inicio", "fecha_fin"], descriptoras)
    base = _apply_legacy_fallbacks(base, ventas, period_df)

    promo = (
        period_df.groupby(["prod_nbr", "periodo", "fecha_inicio", "fecha_fin"], observed=True, sort=False)
        .agg(Tiene_Promocion=("tiene_promocion", "max"), Num_Promociones=("num_promociones", "sum"))
        .reset_index()
    )
    base = base.merge(promo, on=["prod_nbr", "periodo", "fecha_inicio", "fecha_fin"], how="left")

    out = pd.DataFrame(
        {
            "prod_nbr": base["prod_nbr"].astype(str),
            "SKU": base["prod_nbr"].astype(str),
            "periodo_3m": base["periodo_3m"],
            "trimestre": base["trimestre"],
            "mes_inicio": base["mes_inicio"],
            "mes_fin": base["mes_fin"],
            "Beta": base["Beta"],
            "Elasticidad": base["Elasticidad"],
            "Alfa": base["Alfa"],
            "R2": base["R2"],
            "P_Value": base["P_Value"],
            "Observaciones": base["num_observaciones"],
            "Precios_Distintos": base["num_precios_distintos"],
            "Observaciones_Modelo": base["Observaciones_Modelo"],
            "Precios_Distintos_Modelo": base["Precios_Distintos_Modelo"],
            "Fuente_Elasticidad": base["Fuente_Elasticidad"],
            "Motivo_Modelo": base["Motivo_Modelo"],
            "Tiene_Promocion": base["Tiene_Promocion"].fillna(0),
            "Num_Promociones": base["Num_Promociones"].fillna(0),
            "Diagnostico": base["Elasticidad"].map(diagnosticar_elasticidad),
        }
    )
    for col in descriptoras:
        out[col] = base[col] if col in base.columns else np.nan
    return out.replace([np.inf, -np.inf], np.nan).sort_values(by=["prod_nbr", "mes_inicio"]).reset_index(drop=True)

def calculate_elasticity_by_period(
    ventas_nse: pd.DataFrame,
    promociones: pd.DataFrame | None = None,
    periodo_tipo: str = "trimestral",
) -> pd.DataFrame:
    """Función general de elasticidad para mensual/trimestral/semestral/anual/global/fallback."""
    ventas, _ = _preparar_ventas_elasticidad(ventas_nse, promociones)
    return _calculate_elasticity_period_prepared(ventas, periodo_tipo)


def calculate_elasticidades_periodo(
    ventas_nse: pd.DataFrame,
    promociones: pd.DataFrame | None = None,
    periodo_tipos: list[str] | None = None,
) -> pd.DataFrame:
    """Construye la tabla interna elasticidades_periodo para todos los niveles solicitados."""
    periodo_tipos = periodo_tipos or PERIODOS_ELASTICIDAD
    ventas, _ = _preparar_ventas_elasticidad(ventas_nse, promociones)
    tablas = [_calculate_elasticity_period_prepared(ventas, periodo_tipo) for periodo_tipo in periodo_tipos]
    tablas = [tabla for tabla in tablas if tabla is not None and not tabla.empty]
    if not tablas:
        return _empty_elasticidades_periodo()
    elasticidades_periodo = pd.concat(tablas, ignore_index=True)
    elasticidades_periodo = elasticidades_periodo.replace([np.inf, -np.inf], np.nan)
    return elasticidades_periodo[ELASTICIDADES_PERIODO_COLUMNS]

def calculate_elasticity(
    ventas_nse: pd.DataFrame,
    promociones: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Calcula elasticidad trimestral legacy y tabla multi-periodo con motor vectorizado."""
    ventas, bloques = _preparar_ventas_elasticidad(ventas_nse, promociones)
    tablas_periodo = [
        _calculate_elasticity_period_prepared(ventas, periodo_tipo)
        for periodo_tipo in PERIODOS_ELASTICIDAD
    ]
    tablas_periodo = [tabla for tabla in tablas_periodo if tabla is not None and not tabla.empty]
    elasticidades_periodo = (
        pd.concat(tablas_periodo, ignore_index=True)[ELASTICIDADES_PERIODO_COLUMNS]
        if tablas_periodo
        else _empty_elasticidades_periodo()
    )

    elasticidad = _calculate_legacy_quarterly_fast(ventas)
    elasticidad.attrs["elasticidades_periodo"] = elasticidades_periodo
    ventas.attrs["elasticidades_periodo"] = elasticidades_periodo

    elasticidad.attrs["elasticidades_periodo"] = elasticidades_periodo
    ventas.attrs["elasticidades_periodo"] = elasticidades_periodo

    return elasticidad, ventas, bloques


def build_elasticity_download(elasticidad_df: pd.DataFrame) -> pd.DataFrame:
    """Construye un CSV limpio de elasticidades, priorizando elasticidades_periodo.

    Si la entrada contiene ``periodo_tipo`` se asume que ya es la tabla consolidada
    multi-periodo. Para compatibilidad, si recibe la tabla legacy trimestral se
    transforma al esquema más cercano posible y se marca como ``trimestral``.
    """
    if elasticidad_df is None or elasticidad_df.empty:
        return pd.DataFrame(columns=ELASTICIDADES_PERIODO_COLUMNS)

    out = elasticidad_df.copy().replace([np.inf, -np.inf], np.nan)

    if "periodo_tipo" not in out.columns:
        legacy_rename = {
            "dept_nm": "departamento",
            "subdept_nm": "categoria",
            "trimestre": "periodo",
            "mes_inicio": "fecha_inicio",
            "mes_fin": "fecha_fin",
            "Elasticidad": "elasticidad",
            "R2": "r2",
            "P_Value": "p_value",
            "Observaciones": "num_observaciones",
            "Precios_Distintos": "num_precios_distintos",
        }
        out = out.rename(columns=legacy_rename)
        out["periodo_tipo"] = "trimestral"

    for col in ELASTICIDADES_PERIODO_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out[ELASTICIDADES_PERIODO_COLUMNS].copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    # Evita objetos complejos y NaN problemáticos en el CSV descargable.
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or str(out[col].dtype).startswith("category"):
            out[col] = out[col].map(
                lambda value: ""
                if value is None or (isinstance(value, float) and pd.isna(value))
                else str(value)
                if isinstance(value, (list, tuple, dict, set))
                else value
            )
    out = out.where(pd.notna(out), "")
    return out


def build_dynamic_explanation_elasticity(
    df_filtered: pd.DataFrame,
    filtros: dict,
) -> str:
    """Explicación dinámica para el dashboard de elasticidad."""
    if df_filtered is None or df_filtered.empty:
        return "No hay datos suficientes para explicar esta selección."

    elasticidad_prom = df_filtered["Elasticidad"].mean()
    r2_prom = df_filtered["R2"].mean()
    p_prom = df_filtered["P_Value"].mean()
    obs_total = df_filtered["Observaciones"].sum()
    diagnostico = (
        df_filtered["Diagnostico"].dropna().mode().iloc[0]
        if "Diagnostico" in df_filtered.columns and not df_filtered["Diagnostico"].dropna().mode().empty
        else "Sin diagnóstico"
    )

    riesgos = []
    if pd.notna(r2_prom) and r2_prom < 0.30:
        riesgos.append("R² bajo")
    if pd.notna(p_prom) and p_prom > 0.10:
        riesgos.append("p-value alto")
    if obs_total < 30:
        riesgos.append("pocas observaciones")
    if pd.notna(elasticidad_prom) and elasticidad_prom >= 0:
        riesgos.append("elasticidad positiva o sospechosa")

    filtros_txt = ", ".join([f"{k}: {v}" for k, v in filtros.items() if v not in [None, "Todos", "Todas", []]])
    filtros_txt = filtros_txt or "sin filtros específicos"

    riesgo_txt = " Riesgos detectados: " + ", ".join(riesgos) + "." if riesgos else " No se detectan alertas críticas inmediatas."

    return (
        f"Con {filtros_txt}, la elasticidad promedio es {elasticidad_prom:.3f} "
        f"y el diagnóstico dominante es '{diagnostico}'. "
        f"Un valor entre 0 y -1 sugiere demanda inelástica; menor a -1 sugiere demanda elástica; "
        f"un valor positivo debe revisarse porque puede indicar ruido o relación precio-demanda no esperada."
        f"{riesgo_txt}"
    )
