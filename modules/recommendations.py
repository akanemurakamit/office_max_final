"""Recommendation Engine (Fase 7): reglas de negocio + simulación financiera.

Motor híbrido que reemplaza la lógica de recomendación basada únicamente en
Random Forest. La decisión se toma con reglas de negocio sobre los escenarios ya
simulados en ``pricing_futuro_escenarios`` (que ya incorpora demanda base futura,
elasticidades existentes y promociones). El Random Forest, si se usa, solo aporta
``probabilidad_exito`` y ajusta ``riesgo``; nunca decide ``categoria_recomendacion``.

Orden lógico:
  1. Reglas de exclusión -> "No recomendar".
  2. Selección del mejor escenario (margen si hay costo, si no ingreso).
  3. Clasificación en dos niveles (categoria_recomendacion + estrategia_especifica).
  4. Explicación textual en español.
  5. Integración opcional de RF (desactivada por defecto).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .promotions import es_promocion

# Si la mejor opción no supera al escenario "mantener precio" en al menos este
# margen relativo, se recomienda mantener el precio.
MEJORA_MINIMA_RELATIVA = 0.01
# Colapso extremo de unidades: se descartan escenarios con caída > 80% vs demanda base.
COLAPSO_UNIDADES_MAX = -0.80

RECOMENDACIONES_SKU_COLUMNS = [
    "SKU",
    "categoria",
    "departamento",
    "horizonte",
    "metodo_proyeccion",
    "precio_actual",
    "costo_unitario",
    "elasticidad_usada",
    "demanda_base",
    "mejor_escenario_precio",
    "precio_recomendado",
    "precio_efectivo",
    "descuento_efectivo",
    "unidades_esperadas",
    "ingreso_esperado",
    "margen_esperado",
    "categoria_recomendacion",
    "estrategia_especifica",
    "confianza_final",
    "riesgo",
    "razon_recomendacion",
    "modelo_apoyo_usado",
    "probabilidad_exito",
]

# Mapeo de nombre_escenario promocional -> estrategia comercial legible.
_PROMO_ESTRATEGIA = {
    "promoción 2x1": "2x1",
    "promoción 3x2": "3x2",
    "promoción segundo producto al 50%": "Segundo producto al 50%",
}


def empty_recomendaciones_sku() -> pd.DataFrame:
    """Devuelve la estructura interna ``recomendaciones_sku`` sin filas."""
    return pd.DataFrame(columns=RECOMENDACIONES_SKU_COLUMNS)


def _confidence_rank(value: str) -> int:
    return {"alta": 3, "media": 2, "baja": 1}.get(str(value).strip().lower(), 0)


def _ensure_sku(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SKU" not in out.columns and "prod_nbr" in out.columns:
        out["SKU"] = out["prod_nbr"]
    if "SKU" in out.columns:
        out["SKU"] = out["SKU"].astype("string").str.strip().astype(str)
    return out


def _costos_lookup(costos) -> dict:
    """Construye un diccionario SKU -> costo_unitario desde varias formas de entrada."""
    if costos is None:
        return {}
    if isinstance(costos, dict):
        return {str(k).strip(): float(v) for k, v in costos.items() if pd.notna(v)}
    if isinstance(costos, pd.Series):
        return {str(k).strip(): float(v) for k, v in costos.dropna().items()}
    if isinstance(costos, pd.DataFrame):
        df = _ensure_sku(costos)
        cost_col = next((c for c in ["costo_unitario", "costo2", "unit_cost", "costo"] if c in df.columns), None)
        if "SKU" not in df.columns or cost_col is None:
            return {}
        serie = pd.to_numeric(df[cost_col], errors="coerce")
        return {str(sku).strip(): float(c) for sku, c in zip(df["SKU"], serie) if pd.notna(c)}
    return {}


def _derive_costo_unitario(grupo: pd.DataFrame, costos_map: dict) -> float:
    """Obtiene el costo unitario del SKU: primero del mapa externo, si no, lo deriva."""
    sku = str(grupo["SKU"].iloc[0]).strip()
    if sku in costos_map:
        return costos_map[sku]
    # Derivación: margen_simulado = (precio_efectivo - costo) * unidades.
    derivable = grupo[
        grupo["margen_simulado"].notna()
        & grupo["unidades_simuladas"].gt(0)
        & grupo["precio_efectivo"].gt(0)
    ]
    if not derivable.empty:
        fila = derivable.iloc[0]
        costo = fila["precio_efectivo"] - (fila["margen_simulado"] / fila["unidades_simuladas"])
        if np.isfinite(costo):
            return float(costo)
    return np.nan


def _is_excluded(grupo: pd.DataFrame) -> str | None:
    """Aplica las reglas de exclusión. Devuelve la razón si se excluye, si no None."""
    elasticidad = pd.to_numeric(grupo["elasticidad_usada"], errors="coerce").iloc[0]
    demanda = pd.to_numeric(grupo["demanda_base"], errors="coerce").iloc[0]
    precio = pd.to_numeric(grupo["precio_actual"], errors="coerce").iloc[0]
    conf_ela = str(grupo["confianza_elasticidad"].iloc[0]).strip().lower()
    conf_dem = str(grupo["confianza_demanda"].iloc[0]).strip().lower()

    if pd.isna(elasticidad) or not np.isfinite(elasticidad):
        return "No se recomienda acción porque la elasticidad no es utilizable (vacía o no finita)."
    if elasticidad >= 0:
        return "No se recomienda acción porque la elasticidad es positiva o atípica."
    if abs(elasticidad) < 1e-9:
        return "No se recomienda acción porque la elasticidad es cero o sospechosa."
    if conf_ela in {"no usable", "no recomendable"}:
        return "No se recomienda acción porque la confianza de elasticidad es no usable."
    if conf_dem in {"no usable", "no recomendable"}:
        return "No se recomienda acción porque la confianza de demanda es no usable."
    if pd.isna(precio) or precio <= 0:
        return "No se recomienda acción porque no hay un precio actual válido."
    if pd.isna(demanda) or demanda <= 0:
        return "No se recomienda acción porque la demanda base proyectada es nula o insuficiente."
    return None


def _eligible_scenarios(grupo: pd.DataFrame, costo_disponible: bool) -> pd.DataFrame:
    """Filtra escenarios candidatos según guardrails de negocio."""
    elegibles = grupo[
        grupo["unidades_simuladas"].gt(0)
        & grupo["precio_efectivo"].gt(0)
        & grupo["ingreso_simulado"].notna()
        # Evita colapsos extremos de volumen (> 80% de caída vs demanda base).
        & (pd.to_numeric(grupo["variacion_unidades"], errors="coerce") > COLAPSO_UNIDADES_MAX)
        # No considerar promociones marcadas de alto riesgo.
        & ~(es_promocion(grupo["tipo_escenario"]) & grupo["riesgo_promocion"].eq("Alto"))
    ].copy()
    if costo_disponible:
        # Nunca elegir un escenario con margen negativo cuando hay costo.
        elegibles = elegibles[elegibles["margen_simulado"].notna() & elegibles["margen_simulado"].ge(0)]
    return elegibles


def _objective(fila: pd.Series, costo_disponible: bool) -> float:
    if costo_disponible and pd.notna(fila.get("margen_simulado")):
        return float(fila["margen_simulado"])
    return float(fila["ingreso_simulado"]) if pd.notna(fila.get("ingreso_simulado")) else -np.inf


def _estrategia_simple(cambio_pct: float) -> str:
    magnitud = int(round(abs(cambio_pct)))
    if cambio_pct > 0:
        return f"Subir precio {magnitud}%"
    return f"Bajar precio {magnitud}%"


def _build_row_excluded(grupo: pd.DataFrame, costo_unitario: float, razon: str) -> dict:
    base = grupo.iloc[0]
    return {
        "SKU": base["SKU"],
        "categoria": base.get("categoria", "Sin dato"),
        "departamento": base.get("departamento", "Sin dato"),
        "horizonte": base["horizonte"],
        "metodo_proyeccion": base.get("metodo_proyeccion", "Sin dato"),
        "precio_actual": pd.to_numeric(base.get("precio_actual"), errors="coerce"),
        "costo_unitario": costo_unitario,
        "elasticidad_usada": pd.to_numeric(base.get("elasticidad_usada"), errors="coerce"),
        "demanda_base": pd.to_numeric(base.get("demanda_base"), errors="coerce"),
        "mejor_escenario_precio": "No recomendar",
        "precio_recomendado": np.nan,
        "precio_efectivo": np.nan,
        "descuento_efectivo": np.nan,
        "unidades_esperadas": np.nan,
        "ingreso_esperado": np.nan,
        "margen_esperado": np.nan,
        "categoria_recomendacion": "No recomendar",
        "estrategia_especifica": "No recomendar",
        "confianza_final": base.get("confianza_final", "No usable"),
        "riesgo": "Alto",
        "razon_recomendacion": razon,
        "modelo_apoyo_usado": False,
        "probabilidad_exito": None,
    }


def _classify_and_explain(mejor: pd.Series, mantener: pd.Series | None, costo_disponible: bool) -> tuple[str, str, str]:
    """Devuelve (categoria_recomendacion, estrategia_especifica, razon)."""
    cambio = float(pd.to_numeric(mejor.get("cambio_precio_pct"), errors="coerce"))
    promo = bool(es_promocion(mejor.get("tipo_escenario")))
    elasticidad = float(pd.to_numeric(mejor.get("elasticidad_usada"), errors="coerce"))

    # ¿La mejor opción supera claramente al escenario "mantener precio"?
    supera_mantener = True
    if mantener is not None:
        obj_mejor = _objective(mejor, costo_disponible)
        obj_mantener = _objective(mantener, costo_disponible)
        umbral = abs(obj_mantener) * MEJORA_MINIMA_RELATIVA
        supera_mantener = obj_mejor > (obj_mantener + umbral)

    if (not supera_mantener) or (not promo and abs(cambio) < 1e-9):
        return (
            "Mantener precio",
            "Mantener precio",
            "Se recomienda mantener el precio porque ningún escenario supera claramente al escenario base.",
        )

    # Variación de margen frente a la base, para la explicación.
    var_margen = pd.to_numeric(mejor.get("variacion_margen"), errors="coerce")
    var_ingreso = pd.to_numeric(mejor.get("variacion_ingreso"), errors="coerce")
    base_margen = pd.to_numeric(mejor.get("margen_base"), errors="coerce")
    pct_margen = float(var_margen / base_margen * 100) if pd.notna(var_margen) and pd.notna(base_margen) and base_margen != 0 else np.nan

    if promo:
        estrategia = _PROMO_ESTRATEGIA.get(str(mejor.get("nombre_escenario")), str(mejor.get("nombre_escenario")))
        razon = (
            f"Se recomienda {estrategia} porque el aumento estimado de unidades compensa el descuento "
            f"efectivo y mejora el resultado total frente al escenario base."
        )
        return "Bajar precio / promover", estrategia, razon

    if cambio > 0:
        estrategia = _estrategia_simple(cambio)
        detalle_margen = f" y el margen esperado mejora {pct_margen:.1f}%" if pd.notna(pct_margen) else ""
        razon = (
            f"Se recomienda subir precio porque la elasticidad es inelástica (e={elasticidad:.2f}), "
            f"la demanda proyectada es estable{detalle_margen}."
        )
        return "Subir precio", estrategia, razon

    estrategia = _estrategia_simple(cambio)
    razon = (
        f"Se recomienda bajar precio porque la elasticidad es elástica (e={elasticidad:.2f}) y el aumento "
        f"estimado de unidades mejora el resultado total frente al escenario base."
    )
    return "Bajar precio / promover", estrategia, razon


def generar_recomendaciones(
    pricing_futuro_escenarios: pd.DataFrame,
    elasticidades_periodo: pd.DataFrame | None = None,
    demanda_base_futura: pd.DataFrame | None = None,
    costos=None,
    usar_random_forest: bool = False,
) -> pd.DataFrame:
    """Genera ``recomendaciones_sku`` (una fila por SKU x horizonte).

    Motor híbrido de reglas + simulación. ``elasticidades_periodo`` y
    ``demanda_base_futura`` se aceptan por compatibilidad con la firma de la fase;
    los valores económicos se consumen desde ``pricing_futuro_escenarios``, que ya
    los integra. ``usar_random_forest`` está desactivado por defecto: el RF solo
    aporta ``probabilidad_exito`` y ajusta ``riesgo``, nunca la decisión.
    """
    if pricing_futuro_escenarios is None or pricing_futuro_escenarios.empty:
        return empty_recomendaciones_sku()

    df = _ensure_sku(pricing_futuro_escenarios).copy()
    requeridas = {"SKU", "horizonte", "cambio_precio_pct", "ingreso_simulado", "unidades_simuladas"}
    if not requeridas.issubset(df.columns):
        return empty_recomendaciones_sku()

    for col in ["precio_efectivo", "ingreso_simulado", "unidades_simuladas", "margen_simulado",
                "margen_base", "variacion_unidades", "variacion_ingreso", "variacion_margen",
                "cambio_precio_pct", "demanda_base", "precio_actual", "elasticidad_usada", "descuento_efectivo"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    costos_map = _costos_lookup(costos)

    filas: list[dict] = []
    for (_sku, _horizonte), grupo in df.groupby(["SKU", "horizonte"], observed=True, sort=False):
        costo_unitario = _derive_costo_unitario(grupo, costos_map)
        costo_disponible = pd.notna(costo_unitario)

        razon_exclusion = _is_excluded(grupo)
        if razon_exclusion is not None:
            filas.append(_build_row_excluded(grupo, costo_unitario, razon_exclusion))
            continue

        elegibles = _eligible_scenarios(grupo, costo_disponible)
        if elegibles.empty:
            filas.append(_build_row_excluded(
                grupo, costo_unitario,
                "No se recomienda acción porque ningún escenario cumple los guardrails de margen, volumen o riesgo.",
            ))
            continue

        elegibles = elegibles.assign(_obj=elegibles.apply(lambda r: _objective(r, costo_disponible), axis=1))
        mejor = elegibles.sort_values(
            ["_obj", "ingreso_simulado", "unidades_simuladas"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]

        es_mantener = grupo["cambio_precio_pct"].abs() < 1e-9
        if "tipo_escenario" in grupo.columns:
            es_mantener = es_mantener & grupo["tipo_escenario"].eq("simple")
        mantener_rows = grupo[es_mantener]
        mantener = mantener_rows.iloc[0] if not mantener_rows.empty else None

        categoria_reco, estrategia, razon = _classify_and_explain(mejor, mantener, costo_disponible)

        base = grupo.iloc[0]
        margen_esperado = mejor.get("margen_simulado")
        filas.append({
            "SKU": base["SKU"],
            "categoria": base.get("categoria", "Sin dato"),
            "departamento": base.get("departamento", "Sin dato"),
            "horizonte": base["horizonte"],
            "metodo_proyeccion": base.get("metodo_proyeccion", "Sin dato"),
            "precio_actual": pd.to_numeric(base.get("precio_actual"), errors="coerce"),
            "costo_unitario": costo_unitario,
            "elasticidad_usada": pd.to_numeric(base.get("elasticidad_usada"), errors="coerce"),
            "demanda_base": pd.to_numeric(base.get("demanda_base"), errors="coerce"),
            "mejor_escenario_precio": mejor.get("nombre_escenario"),
            "precio_recomendado": mejor.get("precio_efectivo"),
            "precio_efectivo": mejor.get("precio_efectivo"),
            "descuento_efectivo": mejor.get("descuento_efectivo"),
            "unidades_esperadas": mejor.get("unidades_simuladas"),
            "ingreso_esperado": mejor.get("ingreso_simulado"),
            "margen_esperado": margen_esperado if pd.notna(margen_esperado) else np.nan,
            "categoria_recomendacion": categoria_reco,
            "estrategia_especifica": estrategia,
            "confianza_final": base.get("confianza_final", "No usable"),
            "riesgo": mejor.get("riesgo", "Bajo"),
            "razon_recomendacion": razon,
            "modelo_apoyo_usado": False,
            "probabilidad_exito": None,
        })

    if not filas:
        return empty_recomendaciones_sku()

    out = pd.DataFrame(filas, columns=RECOMENDACIONES_SKU_COLUMNS)

    # Paso 5 — Integración opcional de RF (solo soporte; nunca decide).
    if usar_random_forest:
        out = _integrar_random_forest(out, pricing_futuro_escenarios)

    return out.sort_values(["horizonte", "SKU"]).reset_index(drop=True)


def _integrar_random_forest(recomendaciones: pd.DataFrame, pricing_futuro_escenarios: pd.DataFrame) -> pd.DataFrame:
    """Integra un modelo de apoyo opcional para poblar probabilidad_exito y ajustar riesgo.

    Solo se activa explícitamente. Valida que haya suficientes datos; de lo
    contrario deja ``modelo_apoyo_usado=False`` y procede solo con reglas. El RF
    nunca sobreescribe ``categoria_recomendacion``.
    """
    MIN_FILAS_RF = 30
    recomendables = recomendaciones[recomendaciones["categoria_recomendacion"].ne("No recomendar")]
    if len(recomendables) < MIN_FILAS_RF:
        # Datos insuficientes para entrenar de forma fiable: reglas solamente.
        return recomendaciones
    # Hook de integración: en este repo el RF disponible es un diagnóstico
    # histórico con fuga de objetivo para este uso, por lo que no se emplea como
    # apoyo automático. Se mantiene la salida basada en reglas.
    return recomendaciones
