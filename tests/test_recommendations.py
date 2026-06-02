import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.demand_forecast import build_demanda_base_futura
from modules.future_pricing import build_pricing_futuro_escenarios
from modules.recommendations import (
    RECOMENDACIONES_SKU_COLUMNS,
    generar_recomendaciones,
)


def _demanda():
    return pd.DataFrame(
        [
            {
                "SKU": "SKU1",
                "categoria": "Cuadernos",
                "departamento": "Papelería",
                "horizonte": "1 mes",
                "metodo_proyeccion": "Automático recomendado",
                "fecha_inicio_proyeccion": "2025-01-01",
                "fecha_fin_proyeccion": "2025-01-31",
                "demanda_base": 100,
                "confianza_demanda": "Alta",
            },
            {
                "SKU": "SKU1",
                "categoria": "Cuadernos",
                "departamento": "Papelería",
                "horizonte": "3 meses",
                "metodo_proyeccion": "Automático recomendado",
                "fecha_inicio_proyeccion": "2025-01-01",
                "fecha_fin_proyeccion": "2025-03-31",
                "demanda_base": 300,
                "confianza_demanda": "Alta",
            },
        ]
    )


def _precios(costo=6):
    row = {"SKU": "SKU1", "precio_actual": 10, "precio_lista": 10}
    if costo is not None:
        row["costo_unitario"] = costo
    return pd.DataFrame([row])


def test_generar_recomendaciones_inelastic_recommends_price_increase():
    # Elasticidad inelástica (-0.5): subir precio mejora margen.
    elasticidades = pd.DataFrame(
        [{"SKU": "SKU1", "categoria": "Cuadernos", "departamento": "Papelería",
          "periodo_tipo": "global_sku", "periodo": "global_sku",
          "elasticidad": -0.5, "confianza_elasticidad": "Alta"}]
    )
    fut = build_pricing_futuro_escenarios(_demanda(), elasticidades, _precios())
    reco = generar_recomendaciones(fut, elasticidades, _demanda(), _precios())

    assert list(reco.columns) == RECOMENDACIONES_SKU_COLUMNS
    assert set(reco["horizonte"]) == {"1 mes", "3 meses"}
    assert (reco["razon_recomendacion"].str.len() > 0).all()
    assert reco["categoria_recomendacion"].notna().all()
    assert reco["estrategia_especifica"].notna().all()
    assert reco["modelo_apoyo_usado"].eq(False).all()

    fila = reco[reco["horizonte"].eq("1 mes")].iloc[0]
    assert fila["categoria_recomendacion"] == "Subir precio"
    assert fila["estrategia_especifica"].startswith("Subir precio")
    assert fila["margen_esperado"] >= 0


def test_generar_recomendaciones_excludes_positive_elasticity():
    elasticidades = pd.DataFrame(
        [{"SKU": "SKU1", "categoria": "Cuadernos", "departamento": "Papelería",
          "periodo_tipo": "global_sku", "periodo": "global_sku",
          "elasticidad": 0.8, "confianza_elasticidad": "Alta"}]
    )
    # build_pricing_futuro filtra elasticidad positiva, así que construimos un
    # escenario mínimo manual para forzar la regla de exclusión.
    fut = pd.DataFrame(
        [{"SKU": "SKU1", "categoria": "Cuadernos", "departamento": "Papelería",
          "horizonte": "1 mes", "metodo_proyeccion": "Automático recomendado",
          "precio_actual": 10, "precio_efectivo": 10, "descuento_efectivo": 0,
          "cambio_precio_pct": 0, "demanda_base": 100, "unidades_simuladas": 100,
          "ingreso_base": 1000, "ingreso_simulado": 1000, "margen_base": 400,
          "margen_simulado": 400, "variacion_unidades": 0, "variacion_ingreso": 0,
          "variacion_margen": 0, "elasticidad_usada": 0.8, "confianza_elasticidad": "Alta",
          "confianza_demanda": "Alta", "confianza_final": "Alta", "riesgo": "Bajo",
          "riesgo_promocion": "No evaluar", "tipo_escenario": "simple",
          "nombre_escenario": "mantener precio"}]
    )
    reco = generar_recomendaciones(fut, elasticidades, _demanda(), _precios())
    fila = reco.iloc[0]
    assert fila["categoria_recomendacion"] == "No recomendar"
    assert fila["estrategia_especifica"] == "No recomendar"
    assert "positiva" in fila["razon_recomendacion"].lower()


def test_generar_recomendaciones_without_cost_uses_revenue_and_flags():
    elasticidades = pd.DataFrame(
        [{"SKU": "SKU1", "categoria": "Cuadernos", "departamento": "Papelería",
          "periodo_tipo": "global_sku", "periodo": "global_sku",
          "elasticidad": -1.5, "confianza_elasticidad": "Alta"}]
    )
    fut = build_pricing_futuro_escenarios(_demanda(), elasticidades, _precios(costo=None))
    reco = generar_recomendaciones(fut, elasticidades, _demanda(), costos=None)

    assert not reco.empty
    assert set(reco["horizonte"]) == {"1 mes", "3 meses"}
    # Sin costo, la recomendación se basa en ingreso y no debe romperse.
    assert reco["razon_recomendacion"].notna().all()


def test_generar_recomendaciones_empty_input_returns_empty_schema():
    out = generar_recomendaciones(pd.DataFrame())
    assert list(out.columns) == RECOMENDACIONES_SKU_COLUMNS
    assert out.empty
