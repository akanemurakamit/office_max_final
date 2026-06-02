"""Utilidades compartidas para escenarios promocionales de pricing."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEMANDA_BASE_BAJA_MINIMA = 5

# Especificación Fase 6: `tipo_escenario` debe ser "simple" o "promocional".
# La identidad concreta de la promoción (2x1/3x2/2do 50%) se conserva en
# `escenario_id` y `nombre_escenario`, no en `tipo_escenario`.
TIPO_ESCENARIO_SIMPLE = "simple"
TIPO_ESCENARIO_PROMOCIONAL = "promocional"

PROMOCIONES_PRICING = pd.DataFrame(
    [
        {
            "escenario_id": "promocion_2x1",
            "nombre_escenario": "promoción 2x1",
            "tipo_escenario": TIPO_ESCENARIO_PROMOCIONAL,
            "factor_precio": 0.50,
            "descuento_efectivo": 0.50,
            "cambio_precio_pct": -0.50,
        },
        {
            "escenario_id": "promocion_3x2",
            "nombre_escenario": "promoción 3x2",
            "tipo_escenario": TIPO_ESCENARIO_PROMOCIONAL,
            "factor_precio": 2 / 3,
            "descuento_efectivo": 0.3333,
            "cambio_precio_pct": -0.3333,
        },
        {
            "escenario_id": "promocion_segundo_50",
            "nombre_escenario": "promoción segundo producto al 50%",
            "tipo_escenario": TIPO_ESCENARIO_PROMOCIONAL,
            "factor_precio": 0.75,
            "descuento_efectivo": 0.25,
            "cambio_precio_pct": -0.25,
        },
    ]
)

# Valores reconocidos como promoción. Incluye el valor canónico "promocional" y,
# por compatibilidad, los identificadores granulares legacy que aún emite
# modules/pricing.py.
PROMOTION_TYPES = {
    TIPO_ESCENARIO_PROMOCIONAL,
    "promocion_2x1",
    "promocion_3x2",
    "promocion_segundo_50",
}


def escenarios_con_promociones(escenarios_base: pd.DataFrame) -> pd.DataFrame:
    """Agrega promociones retail estándar a una tabla de escenarios simples."""
    base = escenarios_base.copy()
    if "factor_precio" not in base.columns:
        base["factor_precio"] = 1 + pd.to_numeric(base["cambio_precio_pct"], errors="coerce")
    if "descuento_efectivo" not in base.columns:
        base["descuento_efectivo"] = np.nan
    return pd.concat([base, PROMOCIONES_PRICING.copy()], ignore_index=True, sort=False)


def es_promocion(tipo_escenario) -> pd.Series | bool:
    """Indica si el tipo de escenario corresponde a una promoción."""
    if isinstance(tipo_escenario, pd.Series):
        return tipo_escenario.isin(PROMOTION_TYPES)
    return tipo_escenario in PROMOTION_TYPES


def evaluar_riesgo_promocion(
    tipo_escenario,
    elasticidad,
    demanda_base,
    costo_unitario,
    precio_efectivo,
    margen_simulado,
    confianza_demanda=None,
    confianza_elasticidad=None,
) -> pd.Series:
    """Devuelve Bajo/Medio/Alto para promociones según guardrails de negocio."""
    idx = getattr(tipo_escenario, "index", None)
    promo = es_promocion(tipo_escenario)

    # Escala de riesgo promocional según especificación Fase 6:
    # "Alto" | "Medio" | "Bajo" | "No evaluar". Las filas que NO son promoción
    # se marcan "No evaluar" porque el riesgo promocional no aplica a ellas.
    _BAJA = {"baja", "no usable", "no recomendable"}

    def _serie(value, index):
        if isinstance(value, pd.Series):
            return value.reindex(index)
        return pd.Series(value, index=index)

    if idx is None:
        if not promo:
            return "No evaluar"
        elasticidad_ok = pd.notna(elasticidad) and np.isfinite(elasticidad) and elasticidad < 0
        costo_ok = pd.isna(costo_unitario) or (pd.notna(precio_efectivo) and costo_unitario < precio_efectivo)
        demanda_ok = pd.notna(demanda_base) and demanda_base >= DEMANDA_BASE_BAJA_MINIMA
        margen_ok = pd.isna(margen_simulado) or margen_simulado >= 0
        conf_d = str(confianza_demanda).strip().lower()
        conf_e = str(confianza_elasticidad).strip().lower()
        if not elasticidad_ok or not costo_ok or not demanda_ok or not margen_ok:
            return "Alto"
        if conf_d in _BAJA or conf_e in _BAJA:
            return "Alto"
        if conf_d == "media" or conf_e == "media":
            return "Medio"
        return "Bajo"

    riesgo = pd.Series("No evaluar", index=idx)
    elasticidad_s = pd.to_numeric(_serie(elasticidad, idx), errors="coerce")
    demanda_s = pd.to_numeric(_serie(demanda_base, idx), errors="coerce")
    costo_s = pd.to_numeric(_serie(costo_unitario, idx), errors="coerce")
    precio_s = pd.to_numeric(_serie(precio_efectivo, idx), errors="coerce")
    margen_s = pd.to_numeric(_serie(margen_simulado, idx), errors="coerce")
    conf_dem = _serie(confianza_demanda, idx).astype(str).str.strip().str.lower()
    conf_ela = _serie(confianza_elasticidad, idx).astype(str).str.strip().str.lower()

    hard_fail = (
        elasticidad_s.isna()
        | ~np.isfinite(elasticidad_s)
        | elasticidad_s.ge(0)
        | demanda_s.lt(DEMANDA_BASE_BAJA_MINIMA)
        | demanda_s.isna()
        | precio_s.isna()
        | (costo_s.notna() & costo_s.ge(precio_s))
        | margen_s.lt(0)
    )
    low_conf = conf_dem.isin(_BAJA) | conf_ela.isin(_BAJA)
    medium_conf = conf_dem.eq("media") | conf_ela.eq("media")

    alto = promo & (hard_fail | low_conf)
    medio = promo & ~alto & medium_conf
    bajo = promo & ~alto & ~medio
    riesgo.loc[alto] = "Alto"
    riesgo.loc[medio] = "Medio"
    riesgo.loc[bajo] = "Bajo"
    return riesgo


def razon_promocion_no_margen(costo_unitario) -> pd.Series | bool:
    """Detecta ausencia de costo para explicar que se optimiza por ingreso."""
    if isinstance(costo_unitario, pd.Series):
        return pd.to_numeric(costo_unitario, errors="coerce").isna()
    return pd.isna(costo_unitario)


# Nombres canónicos (ya normalizados: minúsculas, sin acentos, espacios->_) para
# detectar columnas de la base de promociones opcional, sin importar mayúsculas,
# acentos ni espacios en el archivo del usuario.
_PROMO_SKU_CANON = {
    "prod_nbr", "sku", "producto", "product_id", "item_id", "id_producto",
    "codigo", "codigo_producto", "clave", "clave_producto",
}
_PROMO_INICIO_CANON = {
    "fecha_inicio", "inicio_promocion", "fecha_ini", "start_date", "tran_date",
    "fecha", "fecha_de_inicio", "inicio", "fecha_inicio_promocion",
    "desde", "fecha_desde",
}
_PROMO_FIN_CANON = {
    "fecha_fin", "fin_promocion", "fecha_final", "end_date", "fecha_termino",
    "fecha_de_fin", "fin", "fecha_fin_promocion", "hasta", "fecha_hasta",
}
_PROMO_MECANICA_CANON = {
    "mecanica", "tipo_promo", "tipo_promocion", "promo", "promocion",
    "descripcion", "mecanica_promocion", "2x1", "3x2",
}


def _normalizar_nombre_columna(nombre: str) -> str:
    """minúsculas, sin acentos y espacios/guiones -> guion bajo."""
    import unicodedata

    texto = str(nombre).strip().lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )
    for ch in (" ", "-", ".", "/"):
        texto = texto.replace(ch, "_")
    while "__" in texto:
        texto = texto.replace("__", "_")
    return texto.strip("_")


def _detectar_columna(promos: pd.DataFrame, candidatos: set) -> str | None:
    """Devuelve el nombre ORIGINAL de la primera columna cuyo nombre normalizado
    coincida (exacto o por inclusión) con alguno de los candidatos."""
    mapa = {col: _normalizar_nombre_columna(col) for col in promos.columns}
    # 1) coincidencia exacta normalizada
    for original, norm in mapa.items():
        if norm in candidatos:
            return original
    # 2) coincidencia por inclusión (p. ej. "fecha_inicio_promo" contiene "fecha_inicio")
    for original, norm in mapa.items():
        if any(cand in norm for cand in candidatos):
            return original
    return None


def normalizar_ventanas_promocion(promociones: "pd.DataFrame | None") -> pd.DataFrame:
    """Normaliza la base de promociones a ventanas (SKU, fecha_inicio, fecha_fin).

    Devuelve un DataFrame con columnas ``SKU``, ``fecha_inicio``, ``fecha_fin`` y,
    si existe, ``mecanica``. Detecta los nombres de columna de forma flexible
    (insensible a mayúsculas, acentos y espacios). Si no reconoce SKU o fecha de
    inicio, devuelve un DataFrame vacío (nunca crashea).
    """
    from .utils import parse_transaction_dates

    columnas = ["SKU", "fecha_inicio", "fecha_fin", "mecanica"]
    if promociones is None or promociones.empty:
        return pd.DataFrame(columns=columnas)

    promos = promociones.copy()
    promos.columns = promos.columns.astype(str).str.strip()

    col_sku = _detectar_columna(promos, _PROMO_SKU_CANON)
    col_inicio = _detectar_columna(promos, _PROMO_INICIO_CANON)
    if col_sku is None or col_inicio is None:
        return pd.DataFrame(columns=columnas)

    col_fin = _detectar_columna(promos, _PROMO_FIN_CANON)
    col_mecanica = _detectar_columna(promos, _PROMO_MECANICA_CANON)

    out = pd.DataFrame()
    out["SKU"] = promos[col_sku].astype("string").str.strip().astype(str)
    out["fecha_inicio"] = parse_transaction_dates(promos[col_inicio])
    if col_fin is not None:
        out["fecha_fin"] = parse_transaction_dates(promos[col_fin])
    else:
        out["fecha_fin"] = out["fecha_inicio"]
    out["fecha_fin"] = out["fecha_fin"].fillna(out["fecha_inicio"])
    out["mecanica"] = promos[col_mecanica].astype(str) if col_mecanica is not None else "Promoción"

    out = out.dropna(subset=["SKU", "fecha_inicio"])
    out = out[out["SKU"].ne("") & out["SKU"].ne("nan")]
    return out[columnas].reset_index(drop=True)
