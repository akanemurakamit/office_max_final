"""Simulación de pricing dinámico y selección de mejores escenarios optimizada.

La vista de pricing depende de:
1. Base limpia + cruzada con NSE.
2. Elasticidad SKU × trimestre ya calculada en la vista 2.

Esta versión evita `groupby.apply()` para seleccionar mejor escenario, que suele ser
uno de los cuellos de botella más grandes en Streamlit con muchos SKUs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ELASTICIDAD_CAP_MAX,
    ELASTICIDAD_CAP_MIN,
    ESCENARIOS_PRICING,
    LIMITE_NUMERICO_RF,
    MIN_OBSERVACIONES,
    MIN_PRECIOS_DISTINTOS,
    USE_RANDOM_FOREST_CLASSIFIER,
)
from .promotions import DEMANDA_BASE_BAJA_MINIMA
from .utils import format_money, format_num, format_pct, normalizar_categoria_est_socio, parse_transaction_dates


# =========================================================
# Utilidades
# =========================================================

def _moda_no_vacia(serie: pd.Series):
    s = serie.replace(["", " ", "nan", "NaN", "None", "none", "null", "Null"], np.nan).dropna()
    if s.empty:
        return "Sin dato"
    moda = s.mode(dropna=True)
    return moda.iloc[0] if not moda.empty else s.iloc[0]


def _fast_mode_by_group(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """Moda por grupo sin groupby.apply, más rápida para muchos SKUs."""
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



def _ensure_sku_alias(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura compatibilidad entre el nombre operativo del notebook (`prod_nbr`) y la app (`SKU`).

    En la base original el SKU suele llamarse `prod_nbr`, pero las vistas y descargas
    de la app usan `SKU`. Esta función evita KeyError: 'SKU' cuando una etapa trae
    solo uno de los dos nombres.
    """
    if df is None:
        return df
    if df.empty:
        # Aun en DataFrames vacíos, crea columnas si existe la contraparte para evitar KeyError.
        if "SKU" not in df.columns and "prod_nbr" in df.columns:
            df = df.copy()
            df["SKU"] = df["prod_nbr"]
        elif "prod_nbr" not in df.columns and "SKU" in df.columns:
            df = df.copy()
            df["prod_nbr"] = df["SKU"]
        return df

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


def _attach_descriptor_modes(base: pd.DataFrame, ventas: pd.DataFrame, group_cols: list[str], descriptoras: list[str]) -> pd.DataFrame:
    out = base.copy()
    for col in descriptoras:
        if col in ventas.columns:
            modes = _fast_mode_by_group(ventas, group_cols, col)
            out = out.merge(modes, on=group_cols, how="left")
        elif col not in out.columns:
            out[col] = np.nan
    return out


def categorize_sku(row: pd.Series) -> str:
    """Clasificación por reglas del notebook base."""
    elasticidad = row.get("Elasticidad", np.nan)
    n_obs = row.get("Observaciones", np.nan)
    n_precios = row.get("Precios_Distintos", np.nan)

    datos_insuficientes = (
        pd.isna(elasticidad)
        or (pd.notna(n_obs) and n_obs < MIN_OBSERVACIONES)
        or (pd.notna(n_precios) and n_precios < MIN_PRECIOS_DISTINTOS)
    )

    if datos_insuficientes:
        return "No recomendar"
    if elasticidad >= 0:
        return "Mantener precio"
    if -1 <= elasticidad < 0:
        return "Subir precio"
    if elasticidad < -1:
        return "Bajar precio / promociones"
    return "No recomendar"


def _classify_vectorized(base_pricing: pd.DataFrame) -> pd.DataFrame:
    """Clasificación vectorizada. Mantiene RF desactivable por rendimiento."""
    df = base_pricing.copy()
    elasticidad = pd.to_numeric(df.get("Elasticidad", np.nan), errors="coerce")
    n_obs = pd.to_numeric(df.get("Observaciones", np.nan), errors="coerce")
    n_precios = pd.to_numeric(df.get("Precios_Distintos", np.nan), errors="coerce")

    datos_insuficientes = (
        elasticidad.isna()
        | (n_obs.notna() & (n_obs < MIN_OBSERVACIONES))
        | (n_precios.notna() & (n_precios < MIN_PRECIOS_DISTINTOS))
    )

    df["Categoria_Regla"] = np.select(
        [
            datos_insuficientes,
            elasticidad >= 0,
            (elasticidad >= -1) & (elasticidad < 0),
            elasticidad < -1,
        ],
        [
            "No recomendar",
            "Mantener precio",
            "Subir precio",
            "Bajar precio / promociones",
        ],
        default="No recomendar",
    )

    if not USE_RANDOM_FOREST_CLASSIFIER:
        df["Categoria_RF"] = df["Categoria_Regla"]
        df["Categoria_RF_Original"] = df["Categoria_Regla"]
        df["Probabilidad_RF_Max"] = 1.0
        return df

    # RF opcional. Si necesitas máxima velocidad, deja USE_RANDOM_FOREST_CLASSIFIER=False.
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        features_numericas = [
            "Elasticidad", "R2", "P_Value", "Tiene_Promocion", "Num_Promociones",
            "Precio_Base", "Costo_Unitario_Base", "Margen_Unitario_Base", "Unidades_Base",
            "Ingreso_Base", "Margen_Base", "Ticket_Promedio_Linea", "Precio_Promedio_Linea",
        ]
        features_categoricas = [
            c for c in ["dept_nm", "subdept_nm", "marca", "tipo_marca", "categoria_est_socio"] if c in df.columns
        ]
        features_numericas = [c for c in features_numericas if c in df.columns]

        for col in features_categoricas:
            df[col] = df[col].astype("string").str.strip().replace("", pd.NA).fillna("Sin dato")
        for col in features_numericas:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].clip(lower=-LIMITE_NUMERICO_RF, upper=LIMITE_NUMERICO_RF)

        X = df[features_numericas + features_categoricas].copy()
        y = df["Categoria_Regla"].copy()
        if len(y.dropna().unique()) < 2 or len(df) < 30:
            raise ValueError("No hay suficientes clases para entrenar RF.")

        transformers = []
        if features_numericas:
            transformers.append(("num", SimpleImputer(strategy="median"), features_numericas))
        if features_categoricas:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    features_categoricas,
                )
            )

        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=3,
            max_leaf_nodes=6,
            min_samples_leaf=max(5, int(np.ceil(len(df) * 0.015))),
            min_samples_split=max(12, int(np.ceil(len(df) * 0.03))),
            max_features="sqrt",
            bootstrap=True,
            max_samples=0.70,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        pipe = Pipeline(steps=[("preprocesador", ColumnTransformer(transformers=transformers)), ("modelo", model)])
        pipe.fit(X, y)
        df["Categoria_RF"] = pipe.predict(X)
        proba = pipe.predict_proba(X)
        df["Probabilidad_RF_Max"] = proba.max(axis=1)
        df["Categoria_RF_Original"] = df["Categoria_RF"]
        df.loc[df["Probabilidad_RF_Max"] < 0.45, "Categoria_RF"] = "No recomendar"
    except Exception:
        df["Categoria_RF"] = df["Categoria_Regla"]
        df["Categoria_RF_Original"] = df["Categoria_Regla"]
        df["Probabilidad_RF_Max"] = 1.0

    return df


# Alias compatible con versiones previas.
_classify_with_conservative_rf = _classify_vectorized


# =========================================================
# Base financiera
# =========================================================

def _prepare_financial_base(ventas_base: pd.DataFrame, elasticidad_df: pd.DataFrame, bloques: list[dict]) -> pd.DataFrame:
    """Crea base financiera SKU-trimestre desde la base limpia + NSE."""
    if ventas_base is None or ventas_base.empty:
        return pd.DataFrame()

    ventas_base = _ensure_sku_alias(ventas_base)
    elasticidad_df = _ensure_sku_alias(elasticidad_df)

    rows = []
    for bloque in bloques:
        for mes in bloque["meses"]:
            rows.append(
                {
                    "mes": mes,
                    "periodo_3m": bloque["periodo_3m"],
                    "trimestre": bloque["trimestre"],
                    "mes_inicio": str(bloque["mes_inicio"]),
                    "mes_fin": str(bloque["mes_fin"]),
                }
            )
    mapa = pd.DataFrame(rows)
    if mapa.empty:
        return pd.DataFrame()

    ventas = ventas_base.copy()
    ventas.columns = ventas.columns.astype(str).str.strip()
    ventas = _ensure_sku_alias(ventas)
    if "prod_nbr" not in ventas.columns:
        raise ValueError("No se encontró columna de SKU. La base debe tener `prod_nbr` o `SKU`.")
    ventas["prod_nbr"] = ventas["prod_nbr"].astype("string").str.strip().astype(str)
    ventas["qty"] = pd.to_numeric(ventas["qty"], errors="coerce")
    ventas["net_sale"] = pd.to_numeric(ventas["net_sale"], errors="coerce")

    if "mes" not in ventas.columns:
        ventas["tran_date"] = parse_transaction_dates(ventas["tran_date"])
        ventas["mes"] = ventas["tran_date"].dt.to_period("M")

    ventas = ventas.merge(mapa, on="mes", how="inner")
    if ventas.empty:
        return pd.DataFrame()

    if "precio_unitario" not in ventas.columns:
        ventas["precio_unitario"] = ventas["net_sale"] / ventas["qty"]
    else:
        ventas["precio_unitario"] = pd.to_numeric(ventas["precio_unitario"], errors="coerce")
        ventas["precio_unitario"] = ventas["precio_unitario"].fillna(ventas["net_sale"] / ventas["qty"])

    if "costo_unitario" not in ventas.columns:
        if "costo2" not in ventas.columns:
            raise ValueError("No existe costo_unitario ni costo2 para calcular margen.")
        ventas["costo_unitario"] = pd.to_numeric(ventas["costo2"], errors="coerce")
    else:
        ventas["costo_unitario"] = pd.to_numeric(ventas["costo_unitario"], errors="coerce")

    ventas = ventas.replace([np.inf, -np.inf], np.nan)
    ventas = ventas.dropna(subset=["prod_nbr", "qty", "net_sale", "precio_unitario", "costo_unitario"])
    ventas = ventas[(ventas["qty"] > 0) & (ventas["net_sale"] > 0) & (ventas["precio_unitario"] > 0)]

    if ventas.empty:
        return pd.DataFrame()

    ventas["costo_total_linea"] = ventas["costo_unitario"] * ventas["qty"]
    ventas["margen_unitario"] = ventas["precio_unitario"] - ventas["costo_unitario"]
    ventas["margen_total"] = ventas["margen_unitario"] * ventas["qty"]

    if "categoria_est_socio" in ventas.columns:
        ventas["categoria_est_socio"] = ventas["categoria_est_socio"].apply(normalizar_categoria_est_socio)

    group_cols = ["prod_nbr", "periodo_3m", "trimestre", "mes_inicio", "mes_fin"]
    base_financiera = (
        ventas.groupby(group_cols, observed=True, sort=False)
        .agg(
            Venta_Neta_Total=("net_sale", "sum"),
            Unidades_Base=("qty", "sum"),
            Costo_Total=("costo_total_linea", "sum"),
            Margen_Base=("margen_total", "sum"),
            Ticket_Promedio_Linea=("net_sale", "mean"),
            Precio_Promedio_Linea=("precio_unitario", "mean"),
        )
        .reset_index()
    )

    descriptoras = ["dept_nm", "subdept_nm", "marca", "tipo_marca", "categoria_est_socio", "estado"]
    base_financiera = _attach_descriptor_modes(base_financiera, ventas, group_cols, descriptoras)

    base_financiera["Precio_Base"] = base_financiera["Venta_Neta_Total"] / base_financiera["Unidades_Base"]
    base_financiera["Costo_Unitario_Base"] = base_financiera["Costo_Total"] / base_financiera["Unidades_Base"]
    base_financiera["Margen_Unitario_Base"] = base_financiera["Precio_Base"] - base_financiera["Costo_Unitario_Base"]
    base_financiera["Ingreso_Base"] = base_financiera["Precio_Base"] * base_financiera["Unidades_Base"]
    base_financiera = base_financiera.replace([np.inf, -np.inf], np.nan)
    base_financiera = base_financiera[(base_financiera["Unidades_Base"] > 0) & (base_financiera["Precio_Base"] > 0)].copy()

    elasticidad = _ensure_sku_alias(elasticidad_df.copy())
    if "prod_nbr" not in elasticidad.columns:
        # Sin elasticidad compatible no se rompe la app; se conserva base financiera
        # y la clasificación terminará como No recomendar por datos insuficientes.
        elasticidad = pd.DataFrame(columns=["prod_nbr", "periodo_3m", "trimestre", "mes_inicio", "mes_fin"])
    elasticidad["prod_nbr"] = elasticidad["prod_nbr"].astype("string").str.strip().astype(str)
    base_financiera = _ensure_sku_alias(base_financiera)
    base_financiera["prod_nbr"] = base_financiera["prod_nbr"].astype(str)

    base_pricing = base_financiera.merge(
        elasticidad,
        on=["prod_nbr", "periodo_3m", "trimestre", "mes_inicio", "mes_fin"],
        how="left",
        suffixes=("", "_elasticidad"),
    )
    base_pricing = _ensure_sku_alias(base_pricing)

    for col in descriptoras:
        col_el = f"{col}_elasticidad"
        if col in base_pricing.columns and col_el in base_pricing.columns:
            base_pricing[col] = base_pricing[col].replace(["Sin dato", ""], np.nan).fillna(base_pricing[col_el])
        elif col_el in base_pricing.columns:
            base_pricing[col] = base_pricing[col_el]

    return base_pricing


# =========================================================
# Simulación y mejor escenario vectorizado
# =========================================================

def _select_best_scenario_vectorized(simulacion: pd.DataFrame) -> pd.DataFrame:
    """Selecciona mejor escenario por SKU-periodo sin groupby.apply."""
    if simulacion is None or simulacion.empty:
        return pd.DataFrame()

    sim = _ensure_sku_alias(simulacion)
    if "SKU" not in sim.columns:
        raise ValueError("No se pudo identificar la columna SKU para seleccionar el mejor escenario.")
    group_cols = ["SKU", "periodo_3m"]
    sim["_row_id"] = np.arange(len(sim))

    # Fila base: escenario 0% o el cambio más cercano a 0.
    tmp_base = sim.copy()
    tmp_base["_abs_change"] = pd.to_numeric(tmp_base["Cambio_Precio"], errors="coerce").abs()
    base_rows = tmp_base.sort_values(group_cols + ["_abs_change"], kind="stable").drop_duplicates(group_cols)

    valid = sim.dropna(subset=["Unidades_Simuladas", "Ingreso_Simulado", "Margen_Simulado"]).copy()
    if "Riesgo_Promocion" in valid.columns:
        valid = valid[~(valid["Tipo_Escenario"].eq("Promoción") & valid["Riesgo_Promocion"].eq("Alto"))].copy()
    if valid.empty:
        resumen = base_rows.copy()
        resumen["Criterio_Escenario_Ideal"] = "Sin recomendación por datos insuficientes o baja confianza"
    else:
        # Candidatos por categoría.
        no_rec = sim[sim["Categoria_RF"].eq("No recomendar")].copy()
        mant = sim[sim["Categoria_RF"].eq("Mantener precio")].copy()

        subir = valid[
            sim.loc[valid.index, "Categoria_RF"].eq("Subir precio")
            & (sim.loc[valid.index, "Cambio_Precio"] >= 0)
            & (sim.loc[valid.index, "Cambio_Margen"] >= 0)
        ].copy()
        # Fallback de subir precio si no hay candidatos positivos: cualquier escenario válido de ese grupo.
        subir_groups = set(map(tuple, subir[group_cols].drop_duplicates().to_numpy())) if not subir.empty else set()
        subir_all = valid[valid["Categoria_RF"].eq("Subir precio")].copy()
        if not subir_all.empty:
            mask_missing = ~subir_all[group_cols].apply(tuple, axis=1).isin(subir_groups)
            subir = pd.concat([subir, subir_all[mask_missing]], ignore_index=False)

        bajar = valid[
            sim.loc[valid.index, "Categoria_RF"].eq("Bajar precio / promociones")
            & ((sim.loc[valid.index, "Cambio_Precio"] < 0) | (sim.loc[valid.index, "Tipo_Escenario"].eq("Promoción")))
        ].copy()
        bajar_groups = set(map(tuple, bajar[group_cols].drop_duplicates().to_numpy())) if not bajar.empty else set()
        bajar_all = valid[valid["Categoria_RF"].eq("Bajar precio / promociones")].copy()
        if not bajar_all.empty:
            mask_missing = ~bajar_all[group_cols].apply(tuple, axis=1).isin(bajar_groups)
            bajar = pd.concat([bajar, bajar_all[mask_missing]], ignore_index=False)

        otros = valid[~valid["Categoria_RF"].isin(["No recomendar", "Mantener precio", "Subir precio", "Bajar precio / promociones"])].copy()

        winners = []
        criterios = []

        if not no_rec.empty:
            w = base_rows.merge(no_rec[group_cols].drop_duplicates(), on=group_cols, how="inner")
            w["Criterio_Escenario_Ideal"] = "Sin recomendación por datos insuficientes o baja confianza"
            winners.append(w)
        if not mant.empty:
            w = base_rows.merge(mant[group_cols].drop_duplicates(), on=group_cols, how="inner")
            w["Criterio_Escenario_Ideal"] = "Mantener precio por elasticidad positiva o sin señal clara"
            winners.append(w)
        if not subir.empty:
            w = subir.sort_values(group_cols + ["Margen_Simulado", "Ingreso_Simulado", "Unidades_Simuladas"], ascending=[True, True, False, False, False], kind="stable").drop_duplicates(group_cols)
            w["Criterio_Escenario_Ideal"] = "Maximiza margen simulado con escenario de subida o neutro"
            winners.append(w)
        if not bajar.empty:
            w = bajar.sort_values(group_cols + ["Unidades_Simuladas", "Ingreso_Simulado", "Margen_Simulado"], ascending=[True, True, False, False, False], kind="stable").drop_duplicates(group_cols)
            w["Criterio_Escenario_Ideal"] = "Maximiza unidades simuladas con descuento/promoción"
            winners.append(w)
        if not otros.empty:
            w = otros.sort_values(group_cols + ["Margen_Simulado", "Ingreso_Simulado", "Unidades_Simuladas"], ascending=[True, True, False, False, False], kind="stable").drop_duplicates(group_cols)
            w["Criterio_Escenario_Ideal"] = "Maximiza margen, ingreso y unidades simuladas"
            winners.append(w)

        resumen = pd.concat(winners, ignore_index=True) if winners else base_rows.copy()

    # Métricas máximas por grupo, también vectorizadas.
    valid = sim.dropna(subset=["Unidades_Simuladas", "Ingreso_Simulado", "Margen_Simulado"]).copy()
    if "Riesgo_Promocion" in valid.columns:
        valid = valid[~(valid["Tipo_Escenario"].eq("Promoción") & valid["Riesgo_Promocion"].eq("Alto"))].copy()
    if valid.empty:
        max_unidades = base_rows[group_cols + ["Nombre_Escenario", "Unidades_Simuladas"]].rename(columns={"Nombre_Escenario": "Escenario_Max_Unidades", "Unidades_Simuladas": "Unidades_Max"})
        max_ingreso = base_rows[group_cols + ["Nombre_Escenario", "Ingreso_Simulado"]].rename(columns={"Nombre_Escenario": "Escenario_Max_Ingreso", "Ingreso_Simulado": "Ingreso_Max"})
        max_margen = base_rows[group_cols + ["Nombre_Escenario", "Margen_Simulado"]].rename(columns={"Nombre_Escenario": "Escenario_Max_Margen", "Margen_Simulado": "Margen_Max"})
    else:
        max_unidades = valid.sort_values(group_cols + ["Unidades_Simuladas"], ascending=[True, True, False], kind="stable").drop_duplicates(group_cols)[group_cols + ["Nombre_Escenario", "Unidades_Simuladas"]].rename(columns={"Nombre_Escenario": "Escenario_Max_Unidades", "Unidades_Simuladas": "Unidades_Max"})
        max_ingreso = valid.sort_values(group_cols + ["Ingreso_Simulado"], ascending=[True, True, False], kind="stable").drop_duplicates(group_cols)[group_cols + ["Nombre_Escenario", "Ingreso_Simulado"]].rename(columns={"Nombre_Escenario": "Escenario_Max_Ingreso", "Ingreso_Simulado": "Ingreso_Max"})
        max_margen = valid.sort_values(group_cols + ["Margen_Simulado"], ascending=[True, True, False], kind="stable").drop_duplicates(group_cols)[group_cols + ["Nombre_Escenario", "Margen_Simulado"]].rename(columns={"Nombre_Escenario": "Escenario_Max_Margen", "Margen_Simulado": "Margen_Max"})

    resumen = resumen.merge(max_unidades, on=group_cols, how="left")
    resumen = resumen.merge(max_ingreso, on=group_cols, how="left")
    resumen = resumen.merge(max_margen, on=group_cols, how="left")

    resumen = resumen.rename(
        columns={
            "Nombre_Escenario": "Escenario_Ideal",
            "Escenario_ID": "Escenario_ID_Ideal",
            "Tipo_Escenario": "Tipo_Escenario_Ideal",
            "Mecanica_Promocion": "Mecanica_Promocion_Ideal",
            "Cambio_Precio_%": "Cambio_Precio_Ideal_%",
            "Precio_Nuevo": "Precio_Nuevo_Ideal",
            "Unidades_Simuladas": "Unidades_Simuladas_Ideal",
            "Ingreso_Simulado": "Ingreso_Simulado_Ideal",
            "Margen_Simulado": "Margen_Simulado_Ideal",
            "Cambio_Unidades_%": "Cambio_Unidades_Ideal_%",
            "Cambio_Ingreso_%": "Cambio_Ingreso_Ideal_%",
            "Cambio_Margen_%": "Cambio_Margen_Ideal_%",
        }
    )

    cols = [
        "SKU", "prod_nbr", "periodo_3m", "trimestre", "mes_inicio", "mes_fin",
        "Categoria_Regla", "Categoria_RF", "Probabilidad_RF_Max", "Elasticidad", "Beta",
        "R2", "P_Value", "Observaciones", "Precios_Distintos", "Tiene_Promocion",
        "Num_Promociones", "Precio_Base", "Costo_Unitario_Base", "Unidades_Base",
        "Ingreso_Base", "Margen_Base", "Escenario_Ideal", "Escenario_ID_Ideal",
        "Tipo_Escenario_Ideal", "Mecanica_Promocion_Ideal", "Cambio_Precio_Ideal_%",
        "Precio_Nuevo_Ideal", "Unidades_Simuladas_Ideal", "Ingreso_Simulado_Ideal",
        "Margen_Simulado_Ideal", "Cambio_Unidades_Ideal_%", "Cambio_Ingreso_Ideal_%",
        "Cambio_Margen_Ideal_%", "Criterio_Escenario_Ideal", "Escenario_Max_Unidades",
        "Unidades_Max", "Escenario_Max_Ingreso", "Ingreso_Max", "Escenario_Max_Margen",
        "Margen_Max", "dept_nm", "subdept_nm", "marca", "tipo_marca", "categoria_est_socio", "estado",
    ]
    for col in cols:
        if col not in resumen.columns:
            resumen[col] = np.nan
    return resumen[cols].reset_index(drop=True)


def simulate_pricing_scenarios(
    ventas_base: pd.DataFrame,
    elasticidad_df: pd.DataFrame,
    bloques: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simula todos los escenarios por SKU-trimestre y elige mejor escenario."""
    base_pricing = _prepare_financial_base(ventas_base, elasticidad_df, bloques)
    base_pricing = _ensure_sku_alias(base_pricing)
    base_pricing = _classify_vectorized(base_pricing)
    base_pricing = _ensure_sku_alias(base_pricing)

    if base_pricing.empty:
        return base_pricing, pd.DataFrame(), pd.DataFrame()

    base = base_pricing.copy()
    esc = ESCENARIOS_PRICING.copy()
    base["_join_key"] = 1
    esc["_join_key"] = 1
    simulacion = base.merge(esc, on="_join_key", how="left", sort=False).drop(columns="_join_key")
    simulacion = _ensure_sku_alias(simulacion)

    cambio = pd.to_numeric(simulacion["Cambio_Efectivo"], errors="coerce")
    elasticidad_original = pd.to_numeric(simulacion["Elasticidad"], errors="coerce")
    precio_base = pd.to_numeric(simulacion["Precio_Base"], errors="coerce")
    costo_unitario = pd.to_numeric(simulacion["Costo_Unitario_Base"], errors="coerce")
    unidades_base = pd.to_numeric(simulacion["Unidades_Base"], errors="coerce")
    ingreso_base = pd.to_numeric(simulacion["Ingreso_Base"], errors="coerce")
    margen_base = pd.to_numeric(simulacion["Margen_Base"], errors="coerce")

    valid = (
        elasticidad_original.notna()
        & precio_base.notna()
        & costo_unitario.notna()
        & unidades_base.notna()
        & (precio_base > 0)
        & (unidades_base > 0)
        & (1 + cambio > 0)
    )

    simulacion["Precio_Nuevo"] = np.where(valid, precio_base * (1 + cambio), np.nan)
    elasticidad_usada = elasticidad_original.clip(ELASTICIDAD_CAP_MIN, ELASTICIDAD_CAP_MAX)
    simulacion["Elasticidad_Usada"] = np.where(valid, elasticidad_usada, np.nan)
    simulacion["Elasticidad_Capada"] = np.where(valid, elasticidad_usada != elasticidad_original, False)

    promo_mask = simulacion["Tipo_Escenario"].eq("Promoción")
    cambio_unidades_pct = elasticidad_usada * cambio
    simulacion["Unidades_Simuladas"] = np.where(
        valid & promo_mask,
        unidades_base * (1 + cambio_unidades_pct),
        np.where(valid, unidades_base * np.exp(elasticidad_usada * np.log1p(cambio)), np.nan),
    )
    simulacion["Ingreso_Simulado"] = simulacion["Precio_Nuevo"] * simulacion["Unidades_Simuladas"]
    simulacion["Margen_Simulado"] = (simulacion["Precio_Nuevo"] - costo_unitario) * simulacion["Unidades_Simuladas"]
    simulacion["Riesgo_Promocion"] = "No aplica"
    promo_alto = promo_mask & (
        elasticidad_original.isna()
        | ~np.isfinite(elasticidad_original)
        | elasticidad_original.ge(0)
        | unidades_base.lt(DEMANDA_BASE_BAJA_MINIMA)
        | costo_unitario.ge(simulacion["Precio_Nuevo"])
        | simulacion["Margen_Simulado"].lt(0)
    )
    simulacion.loc[promo_mask & ~promo_alto, "Riesgo_Promocion"] = "Bajo"
    simulacion.loc[promo_alto, "Riesgo_Promocion"] = "Alto"

    for col in ["Unidades_Simuladas", "Ingreso_Simulado", "Margen_Simulado"]:
        simulacion[col] = pd.to_numeric(simulacion[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    simulacion["Cambio_Unidades"] = simulacion["Unidades_Simuladas"] - unidades_base
    simulacion["Cambio_Ingreso"] = simulacion["Ingreso_Simulado"] - ingreso_base
    simulacion["Cambio_Margen"] = simulacion["Margen_Simulado"] - margen_base

    simulacion["Cambio_Unidades_%"] = np.where(unidades_base != 0, ((simulacion["Unidades_Simuladas"] / unidades_base) - 1) * 100, np.nan)
    simulacion["Cambio_Ingreso_%"] = np.where(ingreso_base != 0, ((simulacion["Ingreso_Simulado"] / ingreso_base) - 1) * 100, np.nan)
    simulacion["Cambio_Margen_%"] = np.where(margen_base != 0, ((simulacion["Margen_Simulado"] / margen_base) - 1) * 100, np.nan)

    simulacion = simulacion.rename(columns={"Cambio_Efectivo": "Cambio_Precio"})
    simulacion["Cambio_Precio_%"] = simulacion["Cambio_Precio"] * 100
    simulacion["Descuento_Equivalente_%"] = np.where(simulacion["Cambio_Precio"] < 0, simulacion["Cambio_Precio"].abs() * 100, 0)
    # Especificación Fase 6: tipo_escenario es "simple" o "promocional".
    es_promo = simulacion["Mecanica_Promocion"].isin(["2x1", "3x2", "2do al 50%"])
    simulacion["tipo_escenario"] = np.where(es_promo, "promocional", "simple")
    simulacion["nombre_escenario"] = simulacion["Nombre_Escenario"]
    simulacion["precio_lista"] = simulacion["Precio_Base"]
    simulacion["precio_efectivo"] = simulacion["Precio_Nuevo"]
    simulacion["descuento_efectivo"] = simulacion["Descuento_Equivalente_%"]
    simulacion["cambio_precio_pct"] = simulacion["Cambio_Precio_%"]
    simulacion["riesgo_promocion"] = simulacion["Riesgo_Promocion"]

    if "Tiene_Promocion" not in simulacion.columns:
        simulacion["Tiene_Promocion"] = simulacion.get("tiene_promocion", np.nan)
    if "Num_Promociones" not in simulacion.columns:
        simulacion["Num_Promociones"] = simulacion.get("num_promociones", np.nan)

    defaults = {
        "Beta": np.nan, "R2": np.nan, "P_Value": np.nan, "Observaciones": np.nan,
        "Precios_Distintos": np.nan, "Tiene_Promocion": np.nan, "Num_Promociones": np.nan,
        "Probabilidad_RF_Max": np.nan, "Categoria_Regla": np.nan, "Categoria_RF": np.nan,
        "dept_nm": np.nan, "subdept_nm": np.nan, "marca": np.nan, "tipo_marca": np.nan,
        "categoria_est_socio": np.nan, "estado": np.nan,
    }
    for col, value in defaults.items():
        if col not in simulacion.columns:
            simulacion[col] = value

    front_cols = [
        "SKU", "prod_nbr", "periodo_3m", "trimestre", "mes_inicio", "mes_fin",
        "Categoria_Regla", "Categoria_RF", "Probabilidad_RF_Max", "Elasticidad",
        "Beta", "R2", "P_Value", "Observaciones", "Precios_Distintos",
        "Tiene_Promocion", "Num_Promociones", "Escenario_ID", "Nombre_Escenario",
        "Nombre_Corto", "Tipo_Escenario", "Mecanica_Promocion", "Riesgo_Promocion", "Cambio_Precio",
        "Cambio_Precio_%", "Descuento_Equivalente_%", "Precio_Base", "Precio_Nuevo",
        "Costo_Unitario_Base", "Unidades_Base", "Unidades_Simuladas",
        "Cambio_Unidades", "Cambio_Unidades_%", "Ingreso_Base", "Ingreso_Simulado",
        "Cambio_Ingreso", "Cambio_Ingreso_%", "Margen_Base", "Margen_Simulado",
        "Cambio_Margen", "Cambio_Margen_%", "Elasticidad_Usada", "Elasticidad_Capada",
        "dept_nm", "subdept_nm", "marca", "tipo_marca", "categoria_est_socio", "estado",
    ]
    rest_cols = [c for c in simulacion.columns if c not in front_cols]
    simulacion = simulacion[[c for c in front_cols if c in simulacion.columns] + rest_cols]
    simulacion = _ensure_sku_alias(simulacion)

    resumen = _select_best_scenario_vectorized(simulacion)

    resumen = _ensure_sku_alias(resumen)
    cols_ideal = [
        "SKU", "periodo_3m", "Escenario_Ideal", "Escenario_ID_Ideal",
        "Tipo_Escenario_Ideal", "Mecanica_Promocion_Ideal",
        "Cambio_Precio_Ideal_%", "Criterio_Escenario_Ideal",
    ]
    for col in cols_ideal:
        if col not in resumen.columns:
            resumen[col] = np.nan
    simulacion = simulacion.merge(resumen[cols_ideal], on=["SKU", "periodo_3m"], how="left")

    return base_pricing, simulacion, resumen


# =========================================================
# Compatibilidad: función anterior, ya no se usa en la ruta rápida
# =========================================================

def choose_best_scenario(g: pd.DataFrame) -> pd.Series:
    """Wrapper compatible; la ruta rápida usa _select_best_scenario_vectorized."""
    return _select_best_scenario_vectorized(g).iloc[0]


# =========================================================
# Descargas y explicación
# =========================================================

def _series_or_na(df: pd.DataFrame, col: str) -> pd.Series:
    """Devuelve una columna existente o una serie NA del mismo largo.

    Evita que las descargas se rompan cuando alguna columna opcional no existe
    por diferencias entre bases o notebooks.
    """
    if df is None:
        return pd.Series(dtype="object")
    if col in df.columns:
        return df[col].reset_index(drop=True)
    return pd.Series([pd.NA] * len(df), dtype="object")


def build_pricing_downloads(simulacion: pd.DataFrame, resumen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye los archivos descargables obligatorios de pricing.

    Esta versión está blindada para bases grandes y para diferencias entre
    `prod_nbr` y `SKU`. También evita DataFrames corruptos cuando falta una
    columna opcional.
    """
    if simulacion is None or simulacion.empty:
        return pd.DataFrame(), pd.DataFrame()

    exp = _ensure_sku_alias(simulacion.copy())
    exp = exp.reset_index(drop=True)

    exp_out = pd.DataFrame(
        {
            "SKU": _series_or_na(exp, "SKU"),
            "dept_nm": _series_or_na(exp, "dept_nm"),
            "marca": _series_or_na(exp, "marca"),
            "tipo_marca": _series_or_na(exp, "tipo_marca"),
            "categoria_est_socio": _series_or_na(exp, "categoria_est_socio"),
            "trimestre": _series_or_na(exp, "trimestre"),
            "escenario aplicado": _series_or_na(exp, "Nombre_Escenario"),
            "unidades simuladas": _series_or_na(exp, "Unidades_Simuladas"),
            "ingreso simulado": _series_or_na(exp, "Ingreso_Simulado"),
            "margen simulado": _series_or_na(exp, "Margen_Simulado"),
            "tipo escenario": _series_or_na(exp, "Tipo_Escenario"),
            "mecánica promoción": _series_or_na(exp, "Mecanica_Promocion"),
            "riesgo promoción": _series_or_na(exp, "Riesgo_Promocion"),
            "tipo_escenario": _series_or_na(exp, "tipo_escenario"),
            "precio lista": _series_or_na(exp, "precio_lista"),
            "precio efectivo": _series_or_na(exp, "Precio_Nuevo"),
            "descuento efectivo %": _series_or_na(exp, "Descuento_Equivalente_%"),
            "mejor escenario": _series_or_na(exp, "Escenario_Ideal"),
            "categoría de SKU": _series_or_na(exp, "Categoria_RF"),
        }
    )

    if resumen is None or resumen.empty:
        return exp_out, pd.DataFrame()

    best = _ensure_sku_alias(resumen.copy()).reset_index(drop=True)
    best_out = pd.DataFrame(
        {
            "SKU": _series_or_na(best, "SKU"),
            "trimestre": _series_or_na(best, "trimestre"),
            "categoría de SKU": _series_or_na(best, "Categoria_RF"),
            "dept_nm": _series_or_na(best, "dept_nm"),
            "marca": _series_or_na(best, "marca"),
            "tipo_marca": _series_or_na(best, "tipo_marca"),
            "categoria_est_socio": _series_or_na(best, "categoria_est_socio"),
            "elasticidad": _series_or_na(best, "Elasticidad"),
            "unidades simuladas": _series_or_na(best, "Unidades_Simuladas_Ideal"),
            "ingreso simulado": _series_or_na(best, "Ingreso_Simulado_Ideal"),
            "margen simulado": _series_or_na(best, "Margen_Simulado_Ideal"),
            "mejor escenario": _series_or_na(best, "Escenario_Ideal"),
            "tipo escenario": _series_or_na(best, "Tipo_Escenario_Ideal"),
            "mecánica promoción": _series_or_na(best, "Mecanica_Promocion_Ideal"),
            "precio efectivo": _series_or_na(best, "Precio_Nuevo_Ideal"),
        }
    )

    return exp_out, best_out


def build_dynamic_explanation_pricing(df_selected: pd.DataFrame, escenario: str, sku: str | None = None) -> str:
    """Explicación dinámica para pricing."""
    if df_selected is not None:
        df_selected = _ensure_sku_alias(df_selected)
    if df_selected is None or df_selected.empty:
        return "No hay datos suficientes para explicar esta selección."

    unidades_base = df_selected["Unidades_Base"].sum()
    unidades_sim = df_selected["Unidades_Simuladas"].sum()
    ingreso_base = df_selected["Ingreso_Base"].sum()
    ingreso_sim = df_selected["Ingreso_Simulado"].sum()
    margen_base = df_selected["Margen_Base"].sum()
    margen_sim = df_selected["Margen_Simulado"].sum()
    elasticidad_prom = df_selected["Elasticidad"].mean()
    categoria = (
        df_selected["Categoria_RF"].dropna().mode().iloc[0]
        if "Categoria_RF" in df_selected.columns and not df_selected["Categoria_RF"].dropna().mode().empty
        else "Sin categoría"
    )

    cambio_u = ((unidades_sim / unidades_base) - 1) * 100 if unidades_base else np.nan
    cambio_i = ((ingreso_sim / ingreso_base) - 1) * 100 if ingreso_base else np.nan
    cambio_m = ((margen_sim / margen_base) - 1) * 100 if margen_base else np.nan

    riesgos = []
    if pd.notna(elasticidad_prom) and elasticidad_prom >= 0:
        riesgos.append("elasticidad positiva")
    if "R2" in df_selected.columns and pd.notna(df_selected["R2"].mean()) and df_selected["R2"].mean() < 0.30:
        riesgos.append("R² bajo")
    if "P_Value" in df_selected.columns and pd.notna(df_selected["P_Value"].mean()) and df_selected["P_Value"].mean() > 0.10:
        riesgos.append("p-value alto")
    if "Observaciones" in df_selected.columns and df_selected["Observaciones"].sum() < 30:
        riesgos.append("pocas observaciones")
    if "Costo_Unitario_Base" in df_selected.columns and "Precio_Base" in df_selected.columns:
        if (df_selected["Costo_Unitario_Base"] >= df_selected["Precio_Base"]).any():
            riesgos.append("costo mayor o igual a precio")

    sujeto = f"el SKU {sku}" if sku else "el grupo filtrado"
    mejora_ingreso = "mejora" if pd.notna(cambio_i) and cambio_i > 0 else "no mejora"
    mejora_margen = "mejora" if pd.notna(cambio_m) and cambio_m > 0 else "no mejora"
    direccion_unidades = "suben" if pd.notna(cambio_u) and cambio_u > 0 else "bajan o se mantienen"

    riesgo_txt = " Riesgos: " + ", ".join(riesgos) + "." if riesgos else " No se observan riesgos críticos inmediatos."

    return (
        f"Para {sujeto}, el escenario '{escenario}' deja una categoría dominante '{categoria}' "
        f"con elasticidad promedio {format_num(elasticidad_prom, 3)}. "
        f"El ingreso {mejora_ingreso} ({format_pct(cambio_i)}), el margen {mejora_margen} ({format_pct(cambio_m)}) "
        f"y las unidades {direccion_unidades} ({format_pct(cambio_u)}). "
        f"Unidades simuladas: {format_num(unidades_sim, 0)}, ingreso simulado: {format_money(ingreso_sim)}, "
        f"margen simulado: {format_money(margen_sim)}.{riesgo_txt}"
    )
