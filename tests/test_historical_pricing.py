import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.historical_pricing import (
    HISTORICAL_PRICE_SCENARIOS,
    PRICING_HISTORICO_ESCENARIOS_COLUMNS,
    simulate_historical_pricing_scenarios,
)


def test_historical_pricing_uses_real_periods_and_existing_elasticity():
    ventas = pd.DataFrame(
        [
            {
                "tran_date": "2024-01-05",
                "qty": 10,
                "net_sale": 100,
                "prod_nbr": "SKU1",
                "costo2": 6,
                "dept_nm": "Dept A",
                "subdept_nm": "Cat A",
            },
            {
                "tran_date": "2024-01-20",
                "qty": 15,
                "net_sale": 150,
                "prod_nbr": "SKU1",
                "costo2": 6,
                "dept_nm": "Dept A",
                "subdept_nm": "Cat A",
            },
        ]
    )
    elasticidades_periodo = pd.DataFrame(
        [
            {
                "SKU": "SKU1",
                "categoria": "Cat A",
                "departamento": "Dept A",
                "periodo_tipo": "mensual",
                "periodo": "2024-01",
                "fecha_inicio": "2024-01-01",
                "fecha_fin": "2024-01-31",
                "elasticidad": -1.5,
                "r2": 0.8,
                "p_value": 0.02,
                "confianza_elasticidad": "Alta",
                "recomendable_elasticidad": True,
                "razon_no_recomendable": "",
            }
        ]
    )

    result = simulate_historical_pricing_scenarios(ventas, elasticidades_periodo)

    assert list(result.columns) == PRICING_HISTORICO_ESCENARIOS_COLUMNS
    assert len(result) == len(HISTORICAL_PRICE_SCENARIOS)
    assert set(result["nombre_escenario"]) == {name for _, name in HISTORICAL_PRICE_SCENARIOS}
    assert result["periodo"].eq("2024-01").all()
    assert result["unidades_reales"].eq(25).all()
    assert result["ingreso_real"].eq(250).all()
    mantener = result.loc[result["nombre_escenario"].eq("mantener precio")].iloc[0]
    assert mantener["unidades_simuladas"] == mantener["unidades_reales"]
    assert mantener["ingreso_simulado"] == mantener["ingreso_real"]
    assert result["tipo_elasticidad_usada"].eq("SKU-mensual").all()
