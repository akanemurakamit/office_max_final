"""Modelos ML para entender comportamiento histórico de ventas antes del pronóstico."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import parse_transaction_dates


def _safe_mode(series: pd.Series):
    clean = series.replace(
        ["", " ", "nan", "NaN", "None", "none", "null", "Null"], np.nan
    ).dropna()
    if clean.empty:
        return np.nan
    mode = clean.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else clean.iloc[0]


def _safe_roc_auc(y_true: pd.Series, probability: np.ndarray) -> float:
    try:
        if len(pd.Series(y_true).dropna().unique()) < 2:
            return np.nan
        return float(roc_auc_score(y_true, probability))
    except Exception:
        return np.nan


def _metric_row(
    model_name: str, y_true: pd.Series, predictions: np.ndarray, probability: np.ndarray
) -> dict:
    return {
        "modelo": model_name,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": _safe_roc_auc(y_true, probability),
    }


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names: list[str] = []
        for name, _, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            names.extend([str(col) for col in cols])
        return names


def _feature_importance_frame(
    model_name: str, feature_names: list[str], values: np.ndarray, top_n: int = 12
) -> pd.DataFrame:
    if len(feature_names) != len(values):
        feature_names = [f"feature_{i}" for i in range(len(values))]
    out = pd.DataFrame(
        {"modelo": model_name, "variable": feature_names, "importancia": np.abs(values)}
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["importancia"])
    return (
        out.sort_values("importancia", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _prepare_monthly_sales(ventas: pd.DataFrame) -> pd.DataFrame:
    ventas = ventas.copy()
    ventas.columns = ventas.columns.astype(str).str.strip()
    ventas["tran_date"] = parse_transaction_dates(ventas["tran_date"])
    ventas["qty"] = pd.to_numeric(ventas["qty"], errors="coerce")
    ventas["net_sale"] = pd.to_numeric(ventas["net_sale"], errors="coerce")
    ventas["prod_nbr"] = ventas["prod_nbr"].astype(str)
    if "precio_unitario" in ventas.columns:
        ventas["precio_unitario"] = pd.to_numeric(
            ventas["precio_unitario"], errors="coerce"
        )
    else:
        ventas["precio_unitario"] = ventas["net_sale"] / ventas["qty"]
    ventas = ventas.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["tran_date", "qty", "net_sale", "prod_nbr", "precio_unitario"]
    )
    ventas = ventas[
        (ventas["qty"] > 0) & (ventas["net_sale"] > 0) & (ventas["precio_unitario"] > 0)
    ].copy()
    if ventas.empty:
        return pd.DataFrame()

    ventas["mes"] = ventas["tran_date"].dt.to_period("M")
    ventas["mes_num"] = ventas["tran_date"].dt.month
    ventas["trimestre_num"] = ventas["tran_date"].dt.quarter

    group_cols = ["prod_nbr", "mes"]
    aggregations = {
        "unidades_mes": ("qty", "sum"),
        "ingreso_mes": ("net_sale", "sum"),
        "precio_promedio": ("precio_unitario", "mean"),
        "transacciones": ("qty", "size"),
        "mes_num": ("mes_num", "first"),
        "trimestre_num": ("trimestre_num", "first"),
    }
    for col in [
        "dept_nm",
        "subdept_nm",
        "marca",
        "tipo_marca",
        "categoria_est_socio",
        "estado",
        "state",
    ]:
        if col in ventas.columns:
            aggregations[col] = (col, _safe_mode)
    for col in ["tiene_promocion", "num_promociones"]:
        if col in ventas.columns:
            aggregations[col] = (col, "max" if col == "tiene_promocion" else "sum")

    monthly = (
        ventas.groupby(group_cols, observed=True).agg(**aggregations).reset_index()
    )
    monthly = monthly.replace([np.inf, -np.inf], np.nan)
    if monthly.empty:
        return monthly

    monthly["venta_alta"] = (
        monthly["unidades_mes"] > monthly["unidades_mes"].median()
    ).astype(int)
    monthly["periodo"] = monthly["mes"].astype(str)
    return monthly


def build_historical_sales_ml_summary(ventas: pd.DataFrame) -> dict:
    """Entrena regresión logística y random forest para explicar ventas históricas.

    El objetivo es diagnóstico: clasifica meses SKU con venta alta vs baja para
    entender drivers históricos antes de alimentar pronósticos o simulaciones.
    """
    if ventas is None or ventas.empty:
        return {
            "status": "empty",
            "message": "No hay ventas procesadas para entrenar modelos históricos.",
        }

    monthly = _prepare_monthly_sales(ventas)
    if monthly.empty or len(monthly) < 30 or monthly["venta_alta"].nunique() < 2:
        return {
            "status": "insufficient_data",
            "message": "No hay suficientes observaciones mensuales o clases para entrenar regresión logística y random forest.",
            "dataset": monthly,
        }

    numeric_features = [
        col
        for col in [
            "precio_promedio",
            "mes_num",
            "trimestre_num",
            "tiene_promocion",
            "num_promociones",
        ]
        if col in monthly.columns
    ]
    categorical_features = [
        col
        for col in [
            "dept_nm",
            "subdept_nm",
            "marca",
            "tipo_marca",
            "categoria_est_socio",
            "estado",
            "state",
        ]
        if col in monthly.columns
    ]
    if not numeric_features and not categorical_features:
        return {
            "status": "insufficient_features",
            "message": "No hay variables explicativas suficientes para entrenar modelos ML históricos.",
            "dataset": monthly,
        }

    X = monthly[numeric_features + categorical_features].copy()
    y = monthly["venta_alta"].astype(int)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # Optimización de rendimiento (sin alterar resultados): el ColumnTransformer se
    # ajusta UNA sola vez y las matrices transformadas se reutilizan para ambos
    # modelos, en lugar de re-ajustar y re-transformar el preprocesador dentro de dos
    # Pipelines. Las transformaciones (imputación, one-hot, escalado) son
    # deterministas, por lo que las predicciones y métricas son idénticas.
    preprocessor.fit(X_train, y_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    X_all_t = preprocessor.transform(X)
    feature_names = _feature_names(preprocessor)

    models = [
        (
            "Regresión logística",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=120,
                max_depth=6,
                min_samples_leaf=max(2, int(np.ceil(len(monthly) * 0.01))),
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
        ),
    ]
    metrics = []
    importances = []
    scored = monthly.copy()

    for model_name, model in models:
        model.fit(X_train_t, y_train)
        predictions = model.predict(X_test_t)
        probability = model.predict_proba(X_test_t)[:, 1]
        metrics.append(_metric_row(model_name, y_test, predictions, probability))
        scored[
            f"prob_venta_alta_{model_name.lower().replace(' ', '_').replace('ó', 'o')}"
        ] = model.predict_proba(X_all_t)[:, 1]

        if hasattr(model, "coef_"):
            values = model.coef_[0]
        else:
            values = model.feature_importances_
        importances.append(_feature_importance_frame(model_name, feature_names, values))

    prob_cols = [col for col in scored.columns if col.startswith("prob_venta_alta_")]
    segment_cols = [
        col
        for col in ["dept_nm", "subdept_nm", "estado", "state"]
        if col in scored.columns
    ]
    if segment_cols:
        segment_col = segment_cols[0]
        segments = (
            scored.groupby(segment_col, observed=True)
            .agg(
                registros=("prod_nbr", "size"),
                skus=("prod_nbr", "nunique"),
                unidades_promedio=("unidades_mes", "mean"),
                probabilidad_venta_alta=(prob_cols[-1], "mean"),
            )
            .reset_index()
            .rename(columns={segment_col: "segmento"})
            .sort_values("probabilidad_venta_alta", ascending=False)
            .head(15)
        )
    else:
        segments = pd.DataFrame()

    return {
        "status": "ok",
        "message": "Modelos entrenados para clasificar ventas históricas altas vs bajas por SKU-mes.",
        "metrics": pd.DataFrame(metrics),
        "feature_importance": (
            pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
        ),
        "segments": segments,
        "dataset_summary": pd.DataFrame(
            [
                {
                    "observaciones_sku_mes": len(monthly),
                    "skus": monthly["prod_nbr"].nunique(),
                    "periodos": monthly["periodo"].nunique(),
                    "venta_alta": int(monthly["venta_alta"].sum()),
                    "venta_baja": int((1 - monthly["venta_alta"]).sum()),
                }
            ]
        ),
    }
