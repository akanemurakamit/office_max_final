"""Tests de la cascada de filtros de la Vista 4 (Pricing futuro).

Verifica la lógica pura de filtrado dependiente que alimenta los selectbox de
la vista, en particular el nuevo filtro de Departamento que antecede al de SKU.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


def _sim():
    """DataFrame mínimo tipo pricing_futuro_escenarios con 2 departamentos."""
    return pd.DataFrame(
        [
            {"horizonte": "1 mes", "metodo_proyeccion": "Automático recomendado",
             "departamento": "Papelería", "SKU": "SKU1", "nombre_escenario": "Cambio de precio +0%"},
            {"horizonte": "1 mes", "metodo_proyeccion": "Automático recomendado",
             "departamento": "Papelería", "SKU": "SKU2", "nombre_escenario": "Cambio de precio +0%"},
            {"horizonte": "1 mes", "metodo_proyeccion": "Automático recomendado",
             "departamento": "Tecnología", "SKU": "SKU3", "nombre_escenario": "Cambio de precio +0%"},
        ]
    )


def test_departamento_options_present():
    sim = _sim()
    opciones = app._safe_sorted_options(sim, "departamento")
    assert opciones == ["Papelería", "Tecnología"]


def test_sku_options_narrow_to_selected_departamento():
    """Al elegir un departamento, el SKU debe restringirse a ese departamento."""
    sim = _sim()
    df_dep = app._filter_fast(sim, "departamento", "Papelería")
    skus = app._safe_sorted_options(df_dep, "SKU")
    assert skus == ["SKU1", "SKU2"]
    assert "SKU3" not in skus


def test_full_cascade_horizonte_metodo_departamento_sku():
    """La cascada completa filtra correctamente hasta un único registro."""
    sim = _sim()
    df_h = app._filter_fast(sim, "horizonte", "1 mes")
    df_m = app._filter_fast(df_h, "metodo_proyeccion", "Automático recomendado")
    df_d = app._filter_fast(df_m, "departamento", "Tecnología")
    df_s = app._filter_fast(df_d, "SKU", "SKU3")
    assert len(df_s) == 1
    assert df_s.iloc[0]["SKU"] == "SKU3"


def test_departamento_todos_keeps_all_rows():
    """El valor 'Todos' no debe filtrar nada (cascada transparente)."""
    sim = _sim()
    df = app._filter_fast(sim, "departamento", "Todos")
    assert len(df) == len(sim)
