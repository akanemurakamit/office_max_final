import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.future_pricing import (
    PRICING_FUTURO_ESCENARIOS_COLUMNS,
    build_pricing_futuro_escenarios,
)


def _demanda_base_futura():
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


def test_build_pricing_futuro_escenarios_uses_future_demand_and_existing_elasticity():
    ventas = pd.DataFrame(
        [
            {
                "tran_date": "2024-12-01",
                "prod_nbr": "SKU1",
                "qty": 10,
                "net_sale": 100,
                "costo2": 6,
                "dept_nm": "Papelería",
                "subdept_nm": "Cuadernos",
            }
        ]
    )
    elasticidades = pd.DataFrame(
        [
            {
                "SKU": "SKU1",
                "categoria": "Cuadernos",
                "departamento": "Papelería",
                "periodo_tipo": "global_sku",
                "periodo": "global_sku",
                "elasticidad": -1.0,
                "confianza_elasticidad": "Alta",
            }
        ]
    )

    out = build_pricing_futuro_escenarios(_demanda_base_futura(), elasticidades, ventas)

    assert list(out.columns) == PRICING_FUTURO_ESCENARIOS_COLUMNS
    assert set(out["horizonte"]) == {"1 mes", "3 meses"}
    assert len(out) == 24
    assert not np.isinf(out.select_dtypes(include="number")).any().any()
    assert not out.isna().any().any()

    up_10 = out[(out["horizonte"].eq("1 mes")) & (out["cambio_precio_pct"].eq(10))].iloc[0]
    assert up_10["precio_actual"] == 10
    assert up_10["precio_efectivo"] == 11
    # Fórmula requerida: 100 * (1 + -1.0 * 0.10)
    assert up_10["unidades_simuladas"] == 90
    assert up_10["ingreso_base"] == 1000
    assert up_10["ingreso_simulado"] == 990
    assert up_10["margen_base"] == 400
    assert up_10["margen_simulado"] == 450
    assert up_10["tipo_elasticidad_usada"] == "elasticidad_sku_global"


def test_build_pricing_futuro_escenarios_marks_low_confidence_suspicious_scenarios():
    demanda = _demanda_base_futura()
    demanda["confianza_demanda"] = "Baja"
    elasticidades = pd.DataFrame(
        [
            {
                "SKU": "SKU1",
                "categoria": "Cuadernos",
                "departamento": "Papelería",
                "periodo_tipo": "global_sku",
                "periodo": "global_sku",
                "elasticidad": 0.5,
                "confianza_elasticidad": "Baja",
            }
        ]
    )
    precios = pd.DataFrame([{"SKU": "SKU1", "precio_actual": 10, "precio_lista": 10, "costo_unitario": 6}])

    out = build_pricing_futuro_escenarios(demanda, elasticidades, precios)

    assert not out.empty
    assert set(out["confianza_final"]) == {"Baja"}
    changed = out[out["cambio_precio_pct"].ne(0)]
    assert changed["riesgo"].eq("Alto").all()
    assert changed["recomendacion"].eq("No recomendar").all()


def test_build_pricing_futuro_escenarios_calculates_promotions_and_blocks_dangerous_margin():
    demanda = pd.DataFrame([{"SKU": "SKU1", "categoria": "Cuadernos", "departamento": "Papelería", "horizonte": "1 mes", "metodo_proyeccion": "Automático recomendado", "fecha_inicio_proyeccion": "2025-01-01", "fecha_fin_proyeccion": "2025-01-31", "demanda_base": 100, "confianza_demanda": "Alta"}])
    elasticidades = pd.DataFrame([{"SKU": "SKU1", "categoria": "Cuadernos", "departamento": "Papelería", "periodo_tipo": "global_sku", "periodo": "global_sku", "elasticidad": -1.0, "confianza_elasticidad": "Alta"}])
    precios = pd.DataFrame([{"SKU": "SKU1", "precio_actual": 10, "precio_lista": 10, "costo_unitario": 6}])

    out = build_pricing_futuro_escenarios(demanda, elasticidades, precios)

    # Fase 6: tipo_escenario es "simple"|"promocional"; la identidad concreta
    # de la promoción vive en nombre_escenario.
    promo_rows = out[out["tipo_escenario"].eq("promocional")]
    assert set(promo_rows["nombre_escenario"]) == {
        "promoción 2x1",
        "promoción 3x2",
        "promoción segundo producto al 50%",
    }
    assert set(out.loc[~out["tipo_escenario"].eq("promocional"), "tipo_escenario"]) == {"simple"}

    promo_2x1 = out[out["nombre_escenario"].eq("promoción 2x1")].iloc[0]
    assert promo_2x1["tipo_escenario"] == "promocional"
    assert promo_2x1["precio_efectivo"] == 5
    assert promo_2x1["descuento_efectivo"] == 50
    assert promo_2x1["cambio_precio_pct"] == -50
    assert promo_2x1["unidades_simuladas"] == 150
    assert promo_2x1["ingreso_simulado"] == 750
    assert promo_2x1["margen_simulado"] == -150
    assert promo_2x1["riesgo_promocion"] == "Alto"
    assert promo_2x1["recomendacion"] == "No recomendar"

    promo_3x2 = out[out["nombre_escenario"].eq("promoción 3x2")].iloc[0]
    assert round(promo_3x2["precio_efectivo"], 4) == 6.6667
    assert round(promo_3x2["descuento_efectivo"], 2) == 33.33
    assert round(promo_3x2["unidades_simuladas"], 2) == 133.33

    promo_segundo_50 = out[out["nombre_escenario"].eq("promoción segundo producto al 50%")].iloc[0]
    assert promo_segundo_50["precio_efectivo"] == 7.5
    assert promo_segundo_50["descuento_efectivo"] == 25
    assert promo_segundo_50["unidades_simuladas"] == 125
