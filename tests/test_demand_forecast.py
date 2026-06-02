import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.demand_forecast import DEMANDA_BASE_FUTURA_COLUMNS, build_demanda_base_futura


def _sales_for_sku(sku="SKU1"):
    rows = []
    # 24 meses completos, enero 2023 a diciembre 2024.
    for month in pd.period_range("2023-01", "2024-12", freq="M"):
        rows.append(
            {
                "tran_date": month.to_timestamp(how="start"),
                "prod_nbr": sku,
                "qty": month.month * 10,
                "net_sale": month.month * 100,
                "dept_nm": "Papelería",
                "subdept_nm": "Cuadernos",
            }
        )
    return pd.DataFrame(rows)


def test_build_demanda_base_futura_calculates_1m_and_3m_without_elasticity():
    ventas = _sales_for_sku()

    out = build_demanda_base_futura(ventas, metodos=["Automático recomendado"])

    assert list(out.columns) == DEMANDA_BASE_FUTURA_COLUMNS
    assert set(out["horizonte"]) == {"1 mes", "3 meses"}
    assert "elasticidad" not in out.columns
    assert out["confianza_demanda"].notna().all()

    one_month = out[out["horizonte"].eq("1 mes")].iloc[0]
    # Próximo mes: enero 2025.
    assert str(one_month["fecha_inicio_proyeccion"]) == "2025-01-01"
    assert str(one_month["fecha_fin_proyeccion"]) == "2025-01-31"
    expected_1m = 0.50 * 110 + 0.30 * 65 + 0.20 * 10
    assert one_month["demanda_base"] == expected_1m
    # pesos_usados se almacena como JSON string (no dict), según la especificación.
    assert isinstance(one_month["pesos_usados"], str)
    assert json.loads(one_month["pesos_usados"]) == {
        "ultimos_3_meses": 0.50,
        "ultimos_12_meses": 0.30,
        "mismo_mes_historico": 0.20,
    }

    three_month = out[out["horizonte"].eq("3 meses")].iloc[0]
    assert str(three_month["fecha_inicio_proyeccion"]) == "2025-01-01"
    assert str(three_month["fecha_fin_proyeccion"]) == "2025-03-31"
    expected_3m = 0.40 * (95 * 3) + 0.30 * (65 * 3) + 0.30 * (20 * 3)
    assert three_month["demanda_base"] == expected_3m


def test_build_demanda_base_futura_redistributes_missing_windows_and_keeps_rows():
    ventas = pd.DataFrame(
        [
            {"tran_date": "2024-10-01", "SKU": "SKU2", "qty": 10, "categoria": "Papel", "departamento": "Oficina"},
            {"tran_date": "2024-11-01", "SKU": "SKU2", "qty": 20, "categoria": "Papel", "departamento": "Oficina"},
            {"tran_date": "2024-12-01", "SKU": "SKU2", "qty": 30, "categoria": "Papel", "departamento": "Oficina"},
        ]
    )

    out = build_demanda_base_futura(ventas, metodos=["Automático recomendado", "Estacional"])

    auto_1m = out[(out["horizonte"].eq("1 mes")) & (out["metodo_proyeccion"].eq("Automático recomendado"))].iloc[0]
    assert auto_1m["demanda_base"] == 20
    assert isinstance(auto_1m["pesos_usados"], str)
    assert json.loads(auto_1m["pesos_usados"]) == {"ultimos_3_meses": 1.0}
    assert auto_1m["confianza_demanda"] in {"Media", "Baja"}
    assert "Ventanas sin datos suficientes" in auto_1m["razon_confianza_demanda"]

    seasonal_3m = out[(out["horizonte"].eq("3 meses")) & (out["metodo_proyeccion"].eq("Estacional"))].iloc[0]
    assert pd.isna(seasonal_3m["demanda_base"])
    assert seasonal_3m["confianza_demanda"] == "No usable"
