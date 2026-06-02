import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.historical_pricing import (
    PRICING_HISTORICO_ESCENARIOS_COLUMNS,
    build_pricing_historico_escenarios,
)


def test_build_pricing_historico_escenarios_uses_real_period_and_existing_elasticity():
    ventas = pd.DataFrame(
        [
            {
                "tran_date": "2024-01-05",
                "prod_nbr": "SKU1",
                "qty": 10,
                "net_sale": 100,
                "costo2": 6,
                "dept_nm": "Papelería",
                "subdept_nm": "Cuadernos",
            },
            {
                "tran_date": "2024-01-20",
                "prod_nbr": "SKU1",
                "qty": 10,
                "net_sale": 100,
                "costo2": 6,
                "dept_nm": "Papelería",
                "subdept_nm": "Cuadernos",
            },
        ]
    )
    elasticidades = pd.DataFrame(
        [
            {
                "SKU": "SKU1",
                "categoria": "Cuadernos",
                "departamento": "Papelería",
                "periodo_tipo": "mensual",
                "periodo": "2024-01",
                "fecha_inicio": "2024-01-01",
                "fecha_fin": "2024-01-31",
                "elasticidad": -1.0,
                "r2": 0.8,
                "p_value": 0.02,
                "num_observaciones": 10,
                "num_precios_distintos": 3,
                "precio_promedio": 10,
                "unidades_promedio": 20,
                "ingreso_promedio": 200,
                "margen_promedio": 80,
                "confianza_elasticidad": "Alta",
                "recomendable_elasticidad": True,
                "razon_no_recomendable": "",
            }
        ]
    )

    out = build_pricing_historico_escenarios(ventas, elasticidades, periodo_tipos=["mensual"])

    assert list(out.columns) == PRICING_HISTORICO_ESCENARIOS_COLUMNS
    assert len(out) == 12
    assert set(out["nombre_escenario"]) == {
        "bajar precio 20%",
        "bajar precio 15%",
        "bajar precio 10%",
        "bajar precio 5%",
        "mantener precio",
        "subir precio 5%",
        "subir precio 10%",
        "subir precio 15%",
        "subir precio 20%",
        "promoción 2x1",
        "promoción 3x2",
        "promoción segundo producto al 50%",
    }
    keep = out.loc[out["nombre_escenario"].eq("mantener precio")].iloc[0]
    assert keep["precio_real"] == 10
    assert keep["precio_efectivo"] == 10
    assert keep["unidades_reales"] == 20
    assert keep["unidades_simuladas"] == 20
    assert keep["ingreso_real"] == 200
    assert keep["margen_real"] == 80
    assert keep["tipo_elasticidad_usada"] == "elasticidad_sku_periodo"


def test_build_pricing_historico_escenarios_calculates_promotions_and_flags_risk():
    ventas = pd.DataFrame([{"tran_date": "2024-01-05", "prod_nbr": "SKU1", "qty": 20, "net_sale": 200, "costo2": 6}])
    elasticidades = pd.DataFrame([{"SKU": "SKU1", "periodo_tipo": "mensual", "periodo": "2024-01", "elasticidad": -1.0, "r2": 0.8, "p_value": 0.02, "confianza_elasticidad": "Alta"}])

    out = build_pricing_historico_escenarios(ventas, elasticidades, periodo_tipos=["mensual"])

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
    assert promo_2x1["unidades_simuladas"] == 30
    assert promo_2x1["ingreso_simulado"] == 150
    assert promo_2x1["margen_simulado"] == -30
    assert promo_2x1["riesgo_promocion"] == "Alto"
    assert promo_2x1["recomendacion_historica"] == "No recomendar"

    promo_3x2 = out[out["nombre_escenario"].eq("promoción 3x2")].iloc[0]
    assert round(promo_3x2["precio_efectivo"], 4) == 6.6667
    assert round(promo_3x2["descuento_efectivo"], 2) == 33.33
    assert round(promo_3x2["unidades_simuladas"], 4) == 26.666

    promo_segundo_50 = out[out["nombre_escenario"].eq("promoción segundo producto al 50%")].iloc[0]
    assert promo_segundo_50["precio_efectivo"] == 7.5
    assert promo_segundo_50["descuento_efectivo"] == 25
    assert promo_segundo_50["unidades_simuladas"] == 25
