"""App principal de Streamlit para pricing dinámico, elasticidad y proyección de ventas.

Versión optimizada:
- Solo se renderiza una vista a la vez.
- La vista 1 solo limpia/cruza/calcula calidad.
- La elasticidad se calcula únicamente desde la vista 2.
- Pricing dinámico se calcula únicamente desde la vista 3.
- La base cruzada con NSE es la base maestra para elasticidad y pricing.
- Pricing depende explícitamente de la elasticidad SKU × trimestre calculada en la vista 2.
- Los cálculos pesados se guardan en caché de sesión y las lecturas/gráficas usan st.cache_data para evitar recálculos innecesarios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from modules.config import (
    COLUMNAS_LECTURA_NSE,
    COLUMNAS_LECTURA_PROMOCIONES,
    COLUMNAS_LECTURA_VENTAS,
    COLUMNAS_MINIMAS_VENTAS,
    ESCENARIOS_PRICING,
    LEER_SOLO_COLUMNAS_NECESARIAS,
    MAX_ROWS_PREVIEW,
    MAX_SKUS_CURVA_ELASTICIDAD,
)
from modules.utils import (
    build_default_nse,
    clean_sales_data,
    convert_df_to_csv,
    filter_dataframe_dependently,
    format_money,
    format_num,
    get_default_nse_path,
    get_uploaded_file_info,
    get_uploaded_file_signature,
    merge_sales_with_nse,
    normalize_column_names,
    read_uploaded_file,
    render_kpi_card,
    validate_columns,
    validate_custom_nse,
)


st.set_page_config(
    page_title="Pricing dinámico retail",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .main .block-container {padding-top: 1.4rem;}
    .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px 18px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        min-height: 112px;
    }
    .kpi-title {
        color: #4b5563;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: #111827;
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.15;
    }
    .kpi-subtitle {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 8px;
    }
    .section-card {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        background: #f9fafb;
        margin-bottom: 16px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# Caché de cálculos pesados
# =========================================================

def process_quality_cached(
    sales_df: pd.DataFrame,
    nse_df: pd.DataFrame | None,
    fuente_nse: str = "default",
    estado_validacion_nse: str = "default_precargada",
    advertencias_nse: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Limpia ventas, cruza NSE y calcula semáforo de calidad.

    Esta función NO calcula elasticidad ni pricing. Así la app responde rápido
    después de cargar ventas y solo calcula la vista activa.
    """
    from modules.quality import build_quality_diagnostics, calculate_quality_diagnosis

    ventas_limpias, resumen_limpieza, summary = clean_sales_data(sales_df)
    ventas_nse, nse_info = merge_sales_with_nse(
        ventas_limpias,
        nse_df,
        fuente_nse=fuente_nse,
        estado_validacion_nse=estado_validacion_nse,
        advertencias_nse=advertencias_nse,
    )
    semaforo, calidad_varianza = calculate_quality_diagnosis(ventas_nse, resumen_limpieza, summary)
    diagnostico_calidad = build_quality_diagnostics(
        ventas_nse,
        resumen_limpieza,
        summary,
        semaforo,
        calidad_varianza,
        nse_info,
    )
    return ventas_nse, resumen_limpieza, summary, semaforo, calidad_varianza, diagnostico_calidad, nse_info


def calculate_elasticity_cached(
    ventas_nse: pd.DataFrame,
    promo_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Calcula elasticidad con caché.

    Se ejecuta solo desde la vista 2 o cuando la vista 3 necesita elasticidad.
    """
    from modules.elasticity import calculate_elasticity

    return calculate_elasticity(ventas_nse, promo_df)


def simulate_historical_pricing_cached(
    ventas_historicas: pd.DataFrame,
    elasticidades_periodo: pd.DataFrame,
) -> pd.DataFrame:
    """Simula pricing histórico con caché sin calcular demanda futura."""
    from modules.historical_pricing import build_pricing_historico_escenarios

    return build_pricing_historico_escenarios(ventas_historicas, elasticidades_periodo)


def simulate_pricing_cached(
    ventas_base_elasticidad: pd.DataFrame,
    elasticidad: pd.DataFrame,
    bloques: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simula escenarios de pricing legacy; la vista activa usa pricing histórico."""
    from modules.pricing import simulate_pricing_scenarios

    return simulate_pricing_scenarios(ventas_base_elasticidad, elasticidad, bloques)


def build_demand_forecast_cached(
    ventas_nse: pd.DataFrame,
    metodos: tuple[str, ...] | None = None,
    horizontes: tuple[str, ...] | None = None,
    pesos_manual: tuple[tuple[str, float], ...] | None = None,
) -> pd.DataFrame:
    """Calcula demanda_base_futura sin recalcular elasticidad ni aplicar promociones.

    Los parámetros de método/horizonte/ventanas se pasan como tuplas para que el
    caché de Streamlit pueda hashearlos. ``pesos_manual`` permite a la vista 4
    sobreescribir las ventanas del método "Manual avanzado" sin tocar el motor.
    """
    from modules.demand_forecast import build_demanda_base_futura

    pesos_config = None
    if pesos_manual:
        pesos_dict = {k: v for k, v in pesos_manual}
        if pesos_dict:
            pesos_config = {
                "1 mes": {"Manual avanzado": pesos_dict},
                "3 meses": {"Manual avanzado": pesos_dict},
            }
    return build_demanda_base_futura(
        ventas_nse,
        horizontes=list(horizontes) if horizontes else None,
        metodos=list(metodos) if metodos else None,
        pesos_config=pesos_config,
    )


@st.cache_data(show_spinner=False, max_entries=5)
def build_future_pricing_cached(
    demanda_base_futura: pd.DataFrame,
    elasticidades_periodo: pd.DataFrame,
    ventas_nse: pd.DataFrame,
) -> pd.DataFrame:
    """Simula pricing futuro usando demanda_base_futura y elasticidades ya calculadas."""
    from modules.future_pricing import build_pricing_futuro_escenarios

    return build_pricing_futuro_escenarios(demanda_base_futura, elasticidades_periodo, ventas_nse)


@st.cache_data(show_spinner=False, max_entries=5)
def build_recommendations_cached(
    pricing_futuro_escenarios: pd.DataFrame,
    elasticidades_periodo: pd.DataFrame,
    demanda_base_futura: pd.DataFrame,
    ventas_nse: pd.DataFrame,
) -> pd.DataFrame:
    """Fase 7: motor de recomendaciones híbrido (reglas + simulación; RF opcional desactivado)."""
    from modules.recommendations import generar_recomendaciones

    return generar_recomendaciones(
        pricing_futuro_escenarios,
        elasticidades_periodo,
        demanda_base_futura,
        ventas_nse,
    )


@st.cache_data(show_spinner=False, max_entries=5)
def build_historical_sales_ml_cached(ventas_nse: pd.DataFrame) -> dict:
    """Entrena modelos ML ligeros para entender ventas históricas antes del pronóstico."""
    from modules.historical_ml import build_historical_sales_ml_summary

    return build_historical_sales_ml_summary(ventas_nse)


@st.cache_data(show_spinner=False, max_entries=10)
def build_elasticity_curve_data(
    curva_df: pd.DataFrame,
    max_skus: int = 8,
) -> pd.DataFrame:
    """Construye la curva precio-demanda log-log por SKU desde elasticidades_periodo.

    Usa la forma constante de elasticidad: Q(P) = unidades_promedio * (P/precio_promedio)^elasticidad.
    Genera un rango de precios alrededor del precio promedio observado de cada SKU.
    Devuelve columnas: SKU, periodo, Precio, "Demanda estimada".
    """
    import numpy as np

    if curva_df is None or curva_df.empty:
        return pd.DataFrame()

    requeridas = {"SKU", "elasticidad", "precio_promedio", "unidades_promedio"}
    if not requeridas.issubset(curva_df.columns):
        return pd.DataFrame()

    base = curva_df.copy()
    base["elasticidad"] = pd.to_numeric(base["elasticidad"], errors="coerce")
    base["precio_promedio"] = pd.to_numeric(base["precio_promedio"], errors="coerce")
    base["unidades_promedio"] = pd.to_numeric(base["unidades_promedio"], errors="coerce")
    base = base.dropna(subset=["elasticidad", "precio_promedio", "unidades_promedio"])
    base = base[(base["precio_promedio"] > 0) & (base["unidades_promedio"] > 0)]
    if base.empty:
        return pd.DataFrame()

    curva_rows = []
    for _, row in base.head(max_skus).iterrows():
        beta = float(row["elasticidad"])
        p0 = float(row["precio_promedio"])
        q0 = float(row["unidades_promedio"])
        sku = row.get("SKU")
        periodo = row.get("periodo", "")
        precios = np.linspace(p0 * 0.6, p0 * 1.4, 50)
        for precio in precios:
            demanda = q0 * (precio / p0) ** beta
            if np.isfinite(demanda) and demanda >= 0:
                curva_rows.append(
                    {
                        "SKU": str(sku),
                        "periodo": str(periodo),
                        "Precio": precio,
                        "Demanda estimada": demanda,
                    }
                )
    return pd.DataFrame(curva_rows)


@st.cache_data(show_spinner=False, max_entries=10)
def aggregate_weekly_demand(ventas_f: pd.DataFrame) -> pd.DataFrame:
    """Agrega demanda semanal con caché para la vista de elasticidad."""
    if ventas_f is None or ventas_f.empty:
        return pd.DataFrame()

    df = ventas_f.copy()
    # tran_date puede llegar como string (dd/mm/YYYY); pd.Grouper(freq="W") exige
    # datetime, de lo contrario el gráfico sale vacío o roto. Se normaliza primero.
    if "tran_date" not in df.columns or "qty" not in df.columns:
        return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(df["tran_date"]):
        from modules.utils import parse_transaction_dates

        df["tran_date"] = parse_transaction_dates(df["tran_date"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df.dropna(subset=["tran_date", "qty"])
    if df.empty:
        return pd.DataFrame()

    if "tiene_promocion" in df.columns and pd.to_numeric(df["tiene_promocion"], errors="coerce").fillna(0).sum() > 0:
        serie = (
            df.groupby([pd.Grouper(key="tran_date", freq="W"), "tiene_promocion"])
            .agg(Demanda=("qty", "sum"))
            .reset_index()
        )
        serie["Promoción"] = serie["tiene_promocion"].map({1: "Con promoción", 0: "Sin promoción"}).fillna("Sin promoción")
        return serie

    # reset_index explícito: con un único Grouper, as_index=False no siempre
    # devuelve tran_date como columna, dejando la serie sin eje X.
    return (
        df.groupby(pd.Grouper(key="tran_date", freq="W"))
        .agg(Demanda=("qty", "sum"))
        .reset_index()
    )


@st.cache_data(show_spinner=False, max_entries=10)
def aggregate_pricing_chart_data(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Agrega tablas para las tres gráficas de pricing con caché."""
    if selected is None or selected.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    money_group = (
        selected.groupby("trimestre", as_index=False)
        .agg(Ventas_normales=("Ingreso_Base", "sum"), Ventas_simuladas=("Ingreso_Simulado", "sum"))
    )
    money_long = money_group.melt(
        id_vars="trimestre",
        value_vars=["Ventas_normales", "Ventas_simuladas"],
        var_name="Serie",
        value_name="Ventas",
    )

    qty_group = (
        selected.groupby("trimestre", as_index=False)
        .agg(Cantidad_normal=("Unidades_Base", "sum"), Cantidad_simulada=("Unidades_Simuladas", "sum"))
    )
    qty_long = qty_group.melt(
        id_vars="trimestre",
        value_vars=["Cantidad_normal", "Cantidad_simulada"],
        var_name="Serie",
        value_name="Unidades",
    )

    im_group = (
        selected.groupby("trimestre", as_index=False)
        .agg(Ingreso_simulado=("Ingreso_Simulado", "sum"), Margen_simulado=("Margen_Simulado", "sum"))
    )
    im_long = im_group.melt(
        id_vars="trimestre",
        value_vars=["Ingreso_simulado", "Margen_simulado"],
        var_name="Métrica",
        value_name="Monto",
    )

    return money_long, qty_long, im_long


# Orden lógico de escenarios de precio: primero los decrementos (de mayor a menor
# descuento), luego mantener, luego los incrementos, y al final las promociones.
_ORDEN_ESCENARIOS = [
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
]
_ORDEN_ESCENARIOS_INDEX = {nombre: i for i, nombre in enumerate(_ORDEN_ESCENARIOS)}


def _sort_escenarios(valores: list[str]) -> list[str]:
    """Ordena nombres de escenario por lógica de precio (decremento -> incremento)."""
    conocidos = [v for v in valores if v.lower() in _ORDEN_ESCENARIOS_INDEX]
    otros = sorted(v for v in valores if v.lower() not in _ORDEN_ESCENARIOS_INDEX)
    conocidos.sort(key=lambda v: _ORDEN_ESCENARIOS_INDEX[v.lower()])
    return conocidos + otros


def _safe_sorted_options(df: pd.DataFrame, col: str | None) -> list[str]:
    """Devuelve opciones limpias y ordenadas para filtros dependientes."""
    if df is None or df.empty or col is None or col not in df.columns:
        return []
    values = (
        df[col]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    values = values[values != ""]
    unicos = values.unique().tolist()
    # Para columnas de escenario se usa el orden lógico de precio, no el alfabético.
    if col in {"nombre_escenario", "escenario"}:
        return _sort_escenarios(unicos)
    return sorted(unicos)


def _filter_fast(df: pd.DataFrame, col: str | None, value: object) -> pd.DataFrame:
    """Filtro ligero para cascadas de Streamlit sin copiar todo el DataFrame si no hace falta."""
    if df is None or df.empty or col is None or col not in df.columns or value in [None, "Todos", "Todas"]:
        return df
    return df.loc[df[col].astype(str) == str(value)]


def _dependent_selectbox(
    label: str,
    options: list[str],
    key: str,
    default: str,
    container,
) -> str:
    """Selectbox que se resetea si una selección previa ya no existe por filtros anteriores."""
    if not options:
        options = [default]
    if default not in options:
        options = [default] + options
    if st.session_state.get(key) not in options:
        st.session_state[key] = default
    with container:
        return st.selectbox(label, options, key=key)



ELASTICITY_EXPECTED_COLUMNS = [
    "SKU",
    "categoria",
    "departamento",
    "periodo_tipo",
    "periodo",
    "fecha_inicio",
    "fecha_fin",
    "elasticidad",
    "r2",
    "p_value",
    "num_observaciones",
    "num_precios_distintos",
    "precio_promedio",
    "unidades_promedio",
    "ingreso_promedio",
    "margen_promedio",
    "confianza_elasticidad",
    "recomendable_elasticidad",
    "razon_no_recomendable",
]

ELASTICITY_NUMERIC_COLUMNS = [
    "elasticidad",
    "r2",
    "p_value",
    "num_observaciones",
    "num_precios_distintos",
    "precio_promedio",
    "unidades_promedio",
    "ingreso_promedio",
    "margen_promedio",
    "descuento_efectivo",
    "cambio_precio_pct",
]

ELASTICITY_TEXT_COLUMNS = [
    "SKU",
    "categoria",
    "departamento",
    "periodo_tipo",
    "periodo",
    "confianza_elasticidad",
    "recomendable_elasticidad",
    "razon_no_recomendable",
    "fuente_nse",
    "categoria_est_socio",
    "nse_match_status",
]


def _stringify_complex_value(value):
    """Convierte objetos complejos a texto y conserva nulos para casteos posteriores."""
    if isinstance(value, (list, tuple, dict, set)):
        return str(value)
    return value


def prepare_dataframe_for_streamlit(
    df: pd.DataFrame | None,
    numeric_columns: list[str] | None = None,
    text_columns: list[str] | None = None,
    force_text_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Normaliza tipos para evitar errores PyArrow/Streamlit al mostrar o exportar tablas."""
    if df is None:
        return pd.DataFrame()

    clean = df.copy().replace([np.inf, -np.inf], np.nan)
    numeric_columns = numeric_columns or []
    text_columns = text_columns or []
    force_text_columns = force_text_columns or []

    for col in clean.columns:
        if clean[col].dtype == "object" or str(clean[col].dtype).startswith("category"):
            clean[col] = clean[col].map(_stringify_complex_value)

    for col in numeric_columns:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    for col in text_columns:
        if col in clean.columns and col not in numeric_columns:
            clean[col] = clean[col].fillna("").astype(str)

    for col in force_text_columns:
        if col in clean.columns:
            clean[col] = clean[col].fillna("").astype(str)

    for col in clean.columns:
        if clean[col].dtype == "object" or str(clean[col].dtype).startswith("category"):
            non_null = clean[col].dropna()
            if not non_null.empty:
                clean[col] = clean[col].fillna("").astype(str)

    return clean


def prepare_elasticity_dataframe_for_display(df: pd.DataFrame | None) -> pd.DataFrame:
    """Normaliza elasticidades manteniendo métricas numéricas y textos homogéneos."""
    return prepare_dataframe_for_streamlit(
        df,
        numeric_columns=ELASTICITY_NUMERIC_COLUMNS,
        text_columns=ELASTICITY_TEXT_COLUMNS,
    )


def _empty_elasticity_periodo_frame() -> pd.DataFrame:
    """Devuelve una tabla vacía segura con el esquema mínimo esperado."""
    return pd.DataFrame(columns=ELASTICITY_EXPECTED_COLUMNS)

def _df_to_excel_friendly_csv_bytes(df: pd.DataFrame, sep: str = ",") -> bytes:
    """CSV estándar separado por comas, compatible con Excel y Google Sheets.

    Usa coma como separador (estándar CSV universal) y UTF-8 con BOM para que los
    acentos se vean bien. Así cada campo cae en su propia columna al abrirlo.
    """
    if df is None or df.empty:
        return b""
    clean = df.copy()
    return clean.to_csv(index=False, sep=sep, encoding="utf-8-sig", lineterminator="\n").encode("utf-8-sig")


def _dataframes_to_zip_csv_bytes(files: dict[str, pd.DataFrame], sep: str = ",") -> bytes:
    """Empaqueta uno o varios DataFrames como CSV dentro de un ZIP en memoria."""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, df in files.items():
            csv_bytes = _df_to_excel_friendly_csv_bytes(df, sep=sep)
            zf.writestr(filename, csv_bytes)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# Estado de la app
# =========================================================

def init_state() -> None:
    """Inicializa session_state."""
    defaults = {
        "active_nse_df": build_default_nse(),
        "nse_mode": "Usar base NSE default",
        "nse_source": "Base NSE default",
        "nse_validation_status": "default_precargada",
        "nse_warnings": [],
        "processed": False,
        "elasticity_ready": False,
        "pricing_ready": False,
        "historical_pricing_ready": False,
        "demand_forecast_ready": False,
        "future_pricing_ready": False,
        "recommendations_ready": False,
        "recomendaciones_sku": pd.DataFrame(),
        "future_metodo": "Automático recomendado",
        "future_horizonte_sel": "Ambos",
        "future_manual_ventanas": [],
        "ventas_limpias": pd.DataFrame(),
        "ventas_nse": pd.DataFrame(),
        "promo_df": None,
        "elasticidad": pd.DataFrame(),
        "elasticidades_periodo": pd.DataFrame(),
        "ventas_base_elasticidad": pd.DataFrame(),
        "ventas_base_pricing": pd.DataFrame(),
        "bloques": [],
        "base_pricing": pd.DataFrame(),
        "simulacion": pd.DataFrame(),
        "resumen_pricing": pd.DataFrame(),
        "pricing_historico_escenarios": pd.DataFrame(),
        "demanda_base_futura": pd.DataFrame(),
        "pricing_futuro_escenarios": pd.DataFrame(),
        "semaforo": pd.DataFrame(),
        "calidad_varianza": pd.DataFrame(),
        "resumen_limpieza": pd.DataFrame(),
        "diagnostico_calidad": pd.DataFrame(),
        "nse_info": {},
        "sales_signature": None,
        "promo_signature": None,
        "nse_signature": "default_nse",
        "quality_cache_key": None,
        "elasticity_cache_key": None,
        "pricing_cache_key": None,
        "manual_cache": {"quality": {}, "elasticity": {}, "pricing": {}, "pricing_historico": {}, "demand_forecast": {}, "pricing_futuro": {}, "recomendaciones": {}},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_model_results() -> None:
    """Limpia resultados derivados cuando cambia la base de ventas o NSE."""
    st.session_state.elasticity_ready = False
    st.session_state.pricing_ready = False
    st.session_state.historical_pricing_ready = False
    st.session_state.demand_forecast_ready = False
    st.session_state.future_pricing_ready = False
    st.session_state.recommendations_ready = False
    st.session_state.recomendaciones_sku = pd.DataFrame()
    st.session_state.elasticidad = pd.DataFrame()
    st.session_state.elasticidades_periodo = pd.DataFrame()
    st.session_state.ventas_base_elasticidad = pd.DataFrame()
    st.session_state.bloques = []
    st.session_state.base_pricing = pd.DataFrame()
    st.session_state.simulacion = pd.DataFrame()
    st.session_state.resumen_pricing = pd.DataFrame()
    st.session_state.pricing_historico_escenarios = pd.DataFrame()
    st.session_state.demanda_base_futura = pd.DataFrame()
    st.session_state.pricing_futuro_escenarios = pd.DataFrame()
    st.session_state.recomendaciones_sku = pd.DataFrame()


def render_sidebar() -> str:
    """Renderiza sidebar. La lectura real ocurre solo con botones explícitos."""
    st.sidebar.title("📊 Pricing dinámico")
    st.sidebar.caption("Carga tus bases y navega entre vistas. Solo se ejecuta la vista activa.")

    vista = st.sidebar.radio(
        "Vista",
        [
            "1. Carga y diagnóstico de datos",
            "2. Elasticidad",
            "3. Pricing histórico (backtesting)",
            "4. Pricing futuro (simulador)",
            "5. Recomendaciones ejecutivas",
            "6. Exportables",
        ],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Archivos")

    with st.sidebar.expander("A. Base de ventas obligatoria", expanded=True):
        st.info(
            "Sube CSV, Excel o Parquet con ventas. Columnas mínimas: "
            "`tran_date`, `qty`, `net_sale`, `prod_nbr`, `costo2`."
        )
        sales_file = st.file_uploader(
            "Base de ventas",
            type=["csv", "xlsx", "xls", "parquet"],
            key="sales_file",
        )

    with st.sidebar.expander("B. Base de promociones opcional", expanded=False):
        st.info(
            "Opcional. Si no se carga, la app funciona sin promociones. "
            "Se lee solo al presionar `Procesar / actualizar datos`."
        )
        promo_file = st.file_uploader(
            "Base de promociones",
            type=["csv", "xlsx", "xls", "parquet"],
            key="promo_file",
        )

    with st.sidebar.expander("C. Configuración de nivel socioeconómico", expanded=False):
        st.info(
            "La opción predeterminada usa la base NSE default precargada. "
            "Si subes una base personalizada, se validará al procesar ventas; si falla, se usará default como fallback."
        )

        default_nse = build_default_nse()
        st.caption(f"Base default: `{get_default_nse_path()}`")
        st.download_button(
            "Descargar base NSE default",
            data=convert_df_to_csv(default_nse),
            file_name="base_nse_default.csv",
            mime="text/csv",
        )

        nse_mode = st.radio(
            "Configuración de nivel socioeconómico",
            ["Usar base NSE default", "Subir base NSE personalizada"],
            index=0 if st.session_state.get("nse_mode", "Usar base NSE default") == "Usar base NSE default" else 1,
            key="nse_mode_selector",
        )
        st.session_state.nse_mode = nse_mode

        nse_file = None
        if nse_mode == "Subir base NSE personalizada":
            nse_file = st.file_uploader(
                "Subir base NSE personalizada",
                type=["csv", "xlsx", "xls", "parquet"],
                key="nse_file",
            )
            st.caption("Se validan columnas NSE, claves de cruce, nulos, duplicados conflictivos, valores válidos y compatibilidad con ventas.")
        else:
            st.session_state.active_nse_df = default_nse
            st.session_state.nse_signature = "default_nse"
            st.session_state.nse_source = "Base NSE default"
            st.session_state.nse_validation_status = "default_precargada"
            st.session_state.nse_warnings = []

        st.caption(f"Modo NSE seleccionado: {nse_mode}")

    if sales_file is not None:
        st.sidebar.success(f"Ventas listas: {get_uploaded_file_info(sales_file)}")
    if promo_file is not None:
        st.sidebar.success(f"Promociones listas: {get_uploaded_file_info(promo_file)}")

    process = st.sidebar.button("Procesar / actualizar datos", type="primary", use_container_width=True)
    if st.sidebar.button("Limpiar caché de esta sesión", use_container_width=True):
        st.cache_data.clear()
        st.session_state.manual_cache = {"quality": {}, "elasticity": {}, "pricing": {}, "pricing_historico": {}, "demand_forecast": {}}
        st.session_state.processed = False
        reset_model_results()
        st.sidebar.success("Caché limpiado. Vuelve a procesar la base si lo necesitas.")

    if process:
        if sales_file is None:
            st.sidebar.error("Primero sube la base de ventas.")
        else:
            try:
                columnas_ventas = COLUMNAS_LECTURA_VENTAS if LEER_SOLO_COLUMNAS_NECESARIAS else None
                columnas_promos = COLUMNAS_LECTURA_PROMOCIONES if LEER_SOLO_COLUMNAS_NECESARIAS else None
                columnas_nse = COLUMNAS_LECTURA_NSE if LEER_SOLO_COLUMNAS_NECESARIAS else None

                with st.spinner("Leyendo archivo de ventas y preparando vista de calidad..."):
                    sales_signature = get_uploaded_file_signature(sales_file)
                    promo_signature = get_uploaded_file_signature(promo_file) if promo_file is not None else "sin_promociones"
                    sales_df = read_uploaded_file(sales_file, usecols=columnas_ventas)
                    promo_df = read_uploaded_file(promo_file, usecols=columnas_promos) if promo_file is not None else None

                    nse_df = build_default_nse()
                    fuente_nse = "default"
                    estado_validacion_nse = "default_precargada"
                    advertencias_nse: list[str] = []
                    nse_signature = "default_nse"

                    if st.session_state.get("nse_mode") == "Subir base NSE personalizada":
                        if nse_file is None:
                            advertencias_nse = ["Se seleccionó NSE personalizada, pero no se subió archivo. Se usa NSE default como fallback."]
                            estado_validacion_nse = "usada_default_por_fallback"
                            st.sidebar.warning(advertencias_nse[0])
                        else:
                            custom_signature = get_uploaded_file_signature(nse_file)
                            custom_nse_df = read_uploaded_file(nse_file, usecols=columnas_nse)
                            is_valid, advertencias_nse, validation_info = validate_custom_nse(custom_nse_df, sales_df)
                            if is_valid:
                                nse_df = custom_nse_df
                                fuente_nse = "personalizada"
                                estado_validacion_nse = validation_info.get("estado_validacion_nse", "valida")
                                nse_signature = custom_signature
                                st.sidebar.success("Base NSE personalizada válida. Se usará para el cruce.")
                            else:
                                fuente_nse = "default"
                                estado_validacion_nse = "usada_default_por_fallback"
                                nse_signature = f"default_fallback_{custom_signature}"
                                st.sidebar.warning("La NSE personalizada no es válida; se usará la base default como fallback.")
                                for warning in advertencias_nse[:5]:
                                    st.sidebar.warning(warning)

                st.session_state.sales_signature = sales_signature
                st.session_state.promo_signature = promo_signature
                st.session_state.nse_signature = nse_signature
                st.session_state.active_nse_df = nse_df
                st.session_state.nse_source = "Base NSE personalizada" if fuente_nse == "personalizada" else "Base NSE default"
                st.session_state.nse_validation_status = estado_validacion_nse
                st.session_state.nse_warnings = advertencias_nse
                process_quality_pipeline(
                    sales_df,
                    promo_df,
                    nse_df,
                    fuente_nse=fuente_nse,
                    estado_validacion_nse=estado_validacion_nse,
                    advertencias_nse=advertencias_nse,
                    cache_key=(sales_signature, nse_signature, estado_validacion_nse),
                )
            except Exception as exc:
                st.session_state.processed = False
                st.sidebar.error(str(exc))

    return vista


def process_quality_pipeline(
    sales_df: pd.DataFrame,
    promo_df: pd.DataFrame | None,
    nse_df: pd.DataFrame | None,
    fuente_nse: str = "default",
    estado_validacion_nse: str = "default_precargada",
    advertencias_nse: list[str] | None = None,
    cache_key: tuple | None = None,
) -> None:
    """Ejecuta solo limpieza, cruce NSE y semáforo."""
    if sales_df is None or sales_df.empty:
        st.sidebar.error("La base de ventas está vacía o no se pudo leer.")
        return

    sales_df = normalize_column_names(sales_df)
    missing = validate_columns(sales_df, COLUMNAS_MINIMAS_VENTAS)
    if missing:
        st.sidebar.error("Faltan columnas obligatorias: " + ", ".join(missing))
        st.session_state.processed = False
        return

    try:
        cache_key = cache_key or (st.session_state.get("sales_signature"), st.session_state.get("nse_signature"))
        cache = st.session_state.manual_cache.setdefault("quality", {})

        if cache_key in cache:
            ventas_nse, resumen_limpieza, summary, semaforo, calidad_varianza, diagnostico_calidad, nse_info = cache[cache_key]
        else:
            with st.spinner("Limpiando ventas, cruzando NSE y calculando calidad..."):
                ventas_nse, resumen_limpieza, summary, semaforo, calidad_varianza, diagnostico_calidad, nse_info = process_quality_cached(
                    sales_df,
                    nse_df,
                    fuente_nse=fuente_nse,
                    estado_validacion_nse=estado_validacion_nse,
                    advertencias_nse=advertencias_nse,
                )
            cache.clear()
            cache[cache_key] = (ventas_nse, resumen_limpieza, summary, semaforo, calidad_varianza, diagnostico_calidad, nse_info)

        st.session_state.quality_cache_key = cache_key

        # Base maestra del análisis:
        # ventas_nse = ventas limpias + cruce NSE.
        # Esta misma base se usa tanto para elasticidad como para pricing.
        st.session_state.ventas_nse = ventas_nse
        st.session_state.ventas_limpias = ventas_nse  # alias para compatibilidad visual
        st.session_state.ventas_base_elasticidad = ventas_nse
        st.session_state.ventas_base_pricing = ventas_nse

        st.session_state.promo_df = promo_df
        st.session_state.resumen_limpieza = resumen_limpieza
        st.session_state.semaforo = semaforo
        st.session_state.calidad_varianza = calidad_varianza
        st.session_state.diagnostico_calidad = diagnostico_calidad
        st.session_state.nse_info = nse_info
        st.session_state.processed = True
        reset_model_results()
        # reset_model_results limpia derivados; se restaura la base maestra.
        st.session_state.ventas_base_elasticidad = ventas_nse
        st.session_state.ventas_base_pricing = ventas_nse
        st.sidebar.success("Base limpia y cruzada con NSE. Elasticidad y pricing se calcularán solo en sus vistas.")

    except Exception as exc:
        st.session_state.processed = False
        st.sidebar.error(f"No se pudo procesar la base: {exc}")


def ensure_elasticity_ready(show_button: bool = True) -> bool:
    """Calcula elasticidad SKU × trimestre usando la base limpia y cruzada con NSE."""
    if not st.session_state.processed:
        return False

    if st.session_state.get("ventas_nse") is None or st.session_state.ventas_nse.empty:
        st.warning(
            "Primero procesa la base en la vista 1. La elasticidad necesita la base limpia y cruzada con NSE."
        )
        return False

    button_clicked = False
    if show_button:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            button_clicked = st.button(
                "Calcular / actualizar elasticidad",
                type="primary" if not st.session_state.elasticity_ready else "secondary",
                use_container_width=True,
            )
        with col_b:
            st.caption(
                "Este cálculo se hace una sola vez por base gracias al caché de sesión. "
                "Cambiar filtros no recalcula el modelo."
            )

    if st.session_state.elasticity_ready and not button_clicked:
        return True

    if show_button and not button_clicked and not st.session_state.elasticity_ready:
        st.info("Presiona **Calcular / actualizar elasticidad** para ejecutar esta vista.")
        return False

    try:
        cache_key = (
            st.session_state.get("sales_signature"),
            st.session_state.get("nse_signature"),
            st.session_state.get("promo_signature"),
            "elasticidades_periodo_v2_all_periodos",
        )
        cache = st.session_state.manual_cache.setdefault("elasticity", {})
        if cache_key in cache:
            elasticidad, ventas_base_elasticidad, bloques = cache[cache_key]
        else:
            with st.spinner("Calculando elasticidades multi-periodo usando base cruzada con NSE..."):
                elasticidad, ventas_base_elasticidad, bloques = calculate_elasticity_cached(
                    st.session_state.ventas_nse,
                    st.session_state.promo_df,
                )
            cache.clear()
            cache[cache_key] = (elasticidad, ventas_base_elasticidad, bloques)

        st.session_state.elasticity_cache_key = cache_key
        st.session_state.elasticidad = elasticidad
        elasticidades_periodo = elasticidad.attrs.get(
            "elasticidades_periodo",
            ventas_base_elasticidad.attrs.get("elasticidades_periodo", pd.DataFrame()),
        )
        if (elasticidades_periodo is None or elasticidades_periodo.empty or "periodo_tipo" not in elasticidades_periodo.columns):
            from modules.elasticity import calculate_elasticidades_periodo

            elasticidades_periodo = calculate_elasticidades_periodo(
                st.session_state.ventas_nse,
                st.session_state.promo_df,
            )
        st.session_state.elasticidades_periodo = elasticidades_periodo
        st.session_state.ventas_base_elasticidad = ventas_base_elasticidad
        st.session_state.bloques = bloques
        st.session_state.elasticity_ready = True
        st.session_state.pricing_ready = False
        st.session_state.historical_pricing_ready = False
        st.session_state.demand_forecast_ready = False
        st.session_state.future_pricing_ready = False
        st.session_state.pricing_historico_escenarios = pd.DataFrame()
        st.session_state.demanda_base_futura = pd.DataFrame()
        st.session_state.pricing_futuro_escenarios = pd.DataFrame()
        st.success("Elasticidad calculada correctamente. Cambiar filtros no volverá a calcularla.")
        return True
    except Exception as exc:
        st.session_state.elasticity_ready = False
        st.error(f"No se pudo calcular elasticidad: {exc}")
        return False


def ensure_pricing_ready() -> bool:
    """Calcula pricing solo si ya existe elasticidad SKU × trimestre de la misma base NSE.

    Dependencias obligatorias:
    1. Vista 1 debe generar ventas_nse = ventas limpias + cruce NSE.
    2. Vista 2 debe calcular elasticidad SKU × trimestre sobre ventas_nse.
    3. Vista 3 usa ventas_nse + elasticidad; no recalcula elasticidad automáticamente.
    """
    if not st.session_state.processed:
        return False

    if st.session_state.get("ventas_nse") is None or st.session_state.ventas_nse.empty:
        st.warning(
            "Primero procesa la base en la vista **1. Carga y diagnóstico de datos**. "
            "Pricing necesita la base limpia y cruzada con NSE."
        )
        return False

    required_cols_pricing = ["prod_nbr", "tran_date", "qty", "net_sale", "categoria_est_socio"]
    missing_cols = [col for col in required_cols_pricing if col not in st.session_state.ventas_nse.columns]
    if missing_cols:
        st.error(
            "La base limpia y cruzada con NSE no tiene las columnas necesarias para pricing: "
            + ", ".join(missing_cols)
        )
        return False

    analysis_key = (
        st.session_state.get("sales_signature"),
        st.session_state.get("nse_signature"),
        st.session_state.get("promo_signature"),
    )

    if (
        not st.session_state.get("elasticity_ready", False)
        or st.session_state.elasticidad is None
        or st.session_state.elasticidad.empty
    ):
        st.warning(
            "Primero calcula la elasticidad en la vista **2. Elasticidad**. "
            "La vista de pricing depende de la elasticidad por SKU y trimestre."
        )
        return False

    if st.session_state.get("elasticity_cache_key") != analysis_key:
        st.warning(
            "La elasticidad guardada no corresponde a la base actual de ventas + NSE + promociones. "
            "Vuelve a calcular elasticidad en la vista **2. Elasticidad** antes de calcular pricing."
        )
        st.session_state.pricing_ready = False
        return False

    col_a, col_b = st.columns([1, 2])
    with col_a:
        button_clicked = st.button(
            "Calcular / actualizar pricing",
            type="primary" if not st.session_state.pricing_ready else "secondary",
            use_container_width=True,
        )
    with col_b:
        st.caption(
            "Pricing usa la base limpia + NSE y la elasticidad SKU × trimestre ya calculada. "
            "Cambiar filtros no recalcula elasticidad ni vuelve a simular todo."
        )

    if st.session_state.pricing_ready and not button_clicked:
        return True

    if not button_clicked and not st.session_state.pricing_ready:
        st.info("Presiona **Calcular / actualizar pricing** para ejecutar simulaciones y proyecciones.")
        return False

    try:
        pricing_key = (analysis_key, "pricing_depende_ventas_nse_y_elasticidad_sku_trimestre")
        cache_pr = st.session_state.manual_cache.setdefault("pricing", {})
        if pricing_key in cache_pr:
            base_pricing, simulacion, resumen_pricing = cache_pr[pricing_key]
        else:
            with st.spinner("Simulando escenarios de pricing con base NSE + elasticidad SKU-trimestre..."):
                base_pricing, simulacion, resumen_pricing = simulate_pricing_cached(
                    st.session_state.ventas_nse,
                    st.session_state.elasticidad,
                    st.session_state.bloques,
                )
            cache_pr.clear()
            cache_pr[pricing_key] = (base_pricing, simulacion, resumen_pricing)

        st.session_state.pricing_cache_key = pricing_key
        st.session_state.base_pricing = base_pricing
        st.session_state.simulacion = simulacion
        st.session_state.resumen_pricing = resumen_pricing
        st.session_state.pricing_ready = True
        st.session_state.ventas_base_pricing = st.session_state.ventas_nse
        st.success("Pricing calculado correctamente usando base NSE + elasticidad SKU-trimestre.")
        return True
    except Exception as exc:
        st.session_state.pricing_ready = False
        st.error(f"No se pudo calcular pricing dinámico: {exc}")
        return False


def ensure_historical_pricing_ready() -> bool:
    """Calcula pricing histórico solo con ventas reales + elasticidades_periodo ya calculadas."""
    if not st.session_state.processed:
        return False

    if st.session_state.get("ventas_nse") is None or st.session_state.ventas_nse.empty:
        st.warning("Primero procesa la base en la vista **1. Carga y diagnóstico de datos**.")
        return False

    elasticidades_periodo = st.session_state.get("elasticidades_periodo", pd.DataFrame())
    if (
        not st.session_state.get("elasticity_ready", False)
        or elasticidades_periodo is None
        or elasticidades_periodo.empty
    ):
        st.warning(
            "Primero calcula la elasticidad en la vista **2. Elasticidad**. "
            "Pricing histórico usa exclusivamente elasticidades ya calculadas desde `elasticidades_periodo`."
        )
        return False

    analysis_key = (
        st.session_state.get("sales_signature"),
        st.session_state.get("nse_signature"),
        st.session_state.get("promo_signature"),
    )
    elasticity_key = st.session_state.get("elasticity_cache_key")
    if isinstance(elasticity_key, tuple) and elasticity_key[:3] != analysis_key:
        st.warning(
            "La elasticidad guardada no corresponde a la base actual de ventas + NSE + promociones. "
            "Vuelve a calcular elasticidad en la vista **2. Elasticidad**."
        )
        st.session_state.historical_pricing_ready = False
        return False

    col_a, col_b = st.columns([1, 2])
    with col_a:
        button_clicked = st.button(
            "Calcular / actualizar pricing histórico",
            type="primary" if not st.session_state.historical_pricing_ready else "secondary",
            use_container_width=True,
        )
    with col_b:
        st.caption(
            "Se calcula una sola vez sobre la base histórica y `elasticidades_periodo`; "
            "los filtros posteriores no recalculan toda la app."
        )

    if st.session_state.historical_pricing_ready and not button_clicked:
        return True

    if not button_clicked and not st.session_state.historical_pricing_ready:
        st.info("Presiona **Calcular / actualizar pricing histórico** para ejecutar el backtesting.")
        return False

    try:
        pricing_key = (analysis_key, "pricing_historico_escenarios_v1")
        cache_pr = st.session_state.manual_cache.setdefault("pricing_historico", {})
        if pricing_key in cache_pr:
            pricing_historico_escenarios = cache_pr[pricing_key]
        else:
            with st.spinner("Simulando escenarios históricos con ventas reales y elasticidades_periodo..."):
                pricing_historico_escenarios = simulate_historical_pricing_cached(
                    st.session_state.ventas_nse,
                    elasticidades_periodo,
                )
            cache_pr.clear()
            cache_pr[pricing_key] = pricing_historico_escenarios

        st.session_state.pricing_cache_key = pricing_key
        st.session_state.pricing_historico_escenarios = pricing_historico_escenarios
        st.session_state.historical_pricing_ready = True
        st.session_state.pricing_ready = False
        st.success("Pricing histórico calculado correctamente como backtesting; no se calculó demanda futura.")
        return True
    except Exception as exc:
        st.session_state.historical_pricing_ready = False
        st.error(f"No se pudo calcular pricing histórico: {exc}")
        return False



def ensure_future_pricing_ready() -> bool:
    """Calcula pricing futuro con demanda_base_futura y elasticidades ya calculadas."""
    if not st.session_state.processed:
        return False

    if st.session_state.get("ventas_nse") is None or st.session_state.ventas_nse.empty:
        st.warning("Primero procesa la base en la vista **1. Carga y diagnóstico de datos**.")
        return False

    elasticidades_periodo = st.session_state.get("elasticidades_periodo", pd.DataFrame())
    if (
        not st.session_state.get("elasticity_ready", False)
        or elasticidades_periodo is None
        or elasticidades_periodo.empty
    ):
        st.warning(
            "Primero calcula la elasticidad en la vista **2. Elasticidad**. "
            "El simulador futuro usa exclusivamente elasticidades ya calculadas."
        )
        return False

    analysis_key = (
        st.session_state.get("sales_signature"),
        st.session_state.get("nse_signature"),
        st.session_state.get("promo_signature"),
    )
    elasticity_key = st.session_state.get("elasticity_cache_key")
    if isinstance(elasticity_key, tuple) and elasticity_key[:3] != analysis_key:
        st.warning(
            "La elasticidad guardada no corresponde a la base actual de ventas + NSE + promociones. "
            "Vuelve a calcular elasticidad en la vista **2. Elasticidad**."
        )
        st.session_state.future_pricing_ready = False
        return False

    col_a, col_b = st.columns([1, 2])
    with col_a:
        button_clicked = st.button(
            "Calcular / actualizar pricing futuro",
            type="primary" if not st.session_state.future_pricing_ready else "secondary",
            use_container_width=True,
        )
    with col_b:
        st.caption(
            "Fase 5: primero construye `demanda_base_futura` para horizontes de 1 y 3 meses; "
            "luego aplica elasticidades existentes a escenarios de -20% a +20%."
        )

    if st.session_state.future_pricing_ready and not button_clicked:
        return True

    if not button_clicked and not st.session_state.future_pricing_ready:
        st.info("Presiona **Calcular / actualizar pricing futuro** para ejecutar la Fase 5.")
        return False

    try:
        # La selección de método/horizonte forma parte de la firma del caché para
        # recomputar solo cuando cambia la configuración (no en cada filtro visual).
        metodo_sel = st.session_state.get("future_metodo", "Automático recomendado")
        horizonte_sel = st.session_state.get("future_horizonte_sel", "Ambos")
        horizontes = ("1 mes", "3 meses") if horizonte_sel == "Ambos" else (horizonte_sel,)
        pesos_manual = None
        if metodo_sel == "Manual avanzado":
            ventanas = tuple(st.session_state.get("future_manual_ventanas") or ())
            if ventanas:
                peso = 1.0 / len(ventanas)
                pesos_manual = tuple((v, peso) for v in ventanas)
        config_sig = (metodo_sel, horizontes, pesos_manual)

        pricing_key = (analysis_key, config_sig, "pricing_futuro_escenarios_v2")
        cache_pr = st.session_state.manual_cache.setdefault("pricing_futuro", {})
        if pricing_key in cache_pr:
            demanda_base_futura, pricing_futuro_escenarios = cache_pr[pricing_key]
        else:
            with st.spinner("Calculando demanda_base_futura y simulando escenarios futuros..."):
                demanda_base_futura = build_demand_forecast_cached(
                    st.session_state.ventas_nse,
                    metodos=(metodo_sel,),
                    horizontes=horizontes,
                    pesos_manual=pesos_manual,
                )
                pricing_futuro_escenarios = build_future_pricing_cached(
                    demanda_base_futura,
                    elasticidades_periodo,
                    st.session_state.ventas_nse,
                )
            cache_pr.clear()
            cache_pr[pricing_key] = (demanda_base_futura, pricing_futuro_escenarios)

        st.session_state.demanda_base_futura = demanda_base_futura
        st.session_state.pricing_futuro_escenarios = pricing_futuro_escenarios
        st.session_state.demand_forecast_ready = True
        st.session_state.future_pricing_ready = True
        st.session_state.historical_pricing_ready = False
        st.session_state.pricing_ready = False
        st.success("Pricing futuro calculado correctamente usando demanda_base_futura y elasticidades existentes.")
        return True
    except Exception as exc:
        st.session_state.future_pricing_ready = False
        st.error(f"No se pudo calcular pricing futuro: {exc}")
        return False


DEMANDA_VENTANAS_MANUAL = [
    ("ultimos_3_meses", "Últimos 3 meses"),
    ("ultimos_6_meses", "Últimos 6 meses"),
    ("ultimos_12_meses", "Últimos 12 meses"),
    ("ultimos_24_meses", "Últimos 24 meses"),
    ("mismo_mes_historico", "Mismo mes histórico"),
    ("mismo_trimestre_historico", "Mismo trimestre histórico"),
]


def _ranking_escenarios_futuro(selected: pd.DataFrame) -> pd.DataFrame:
    """Agrega los escenarios filtrados y los ordena por atractivo (margen, luego ingreso).

    Cuando hay costo, el mejor escenario es el de mayor margen simulado; si no hay
    costo, se usa el ingreso simulado. Solo considera escenarios viables (unidades
    > 0, precio efectivo > 0 y, si es promoción, sin riesgo Alto).
    """
    if selected is None or selected.empty:
        return pd.DataFrame()

    viables = selected.copy()
    viables["unidades_simuladas"] = pd.to_numeric(viables["unidades_simuladas"], errors="coerce")
    viables["precio_efectivo"] = pd.to_numeric(viables["precio_efectivo"], errors="coerce")
    viables = viables[viables["unidades_simuladas"].gt(0) & viables["precio_efectivo"].gt(0)]
    if "riesgo_promocion" in viables.columns and "tipo_escenario" in viables.columns:
        es_promo = viables["tipo_escenario"].eq("promocional")
        viables = viables[~(es_promo & viables["riesgo_promocion"].eq("Alto"))]
    if viables.empty:
        return pd.DataFrame()

    ranking = (
        viables.groupby("nombre_escenario", observed=True, sort=False)
        .agg(
            unidades_simuladas=("unidades_simuladas", "sum"),
            ingreso_simulado=("ingreso_simulado", "sum"),
            margen_simulado=("margen_simulado", "sum"),
            cambio_precio_pct=("cambio_precio_pct", "mean"),
        )
        .reset_index()
    )
    hay_margen = ranking["margen_simulado"].notna().any()
    ranking["_score"] = ranking["margen_simulado"] if hay_margen else ranking["ingreso_simulado"]
    ranking = ranking.sort_values(
        ["_score", "ingreso_simulado"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    return ranking


def _mejor_escenario_futuro(selected: pd.DataFrame) -> tuple[str, str]:
    """Devuelve (nombre_mejor_escenario, subtítulo) para la tarjeta de indicador."""
    ranking = _ranking_escenarios_futuro(selected)
    if ranking.empty:
        return "Sin escenario viable", "Revisa filtros"
    fila = ranking.iloc[0]
    if pd.notna(fila.get("margen_simulado")) and ranking["margen_simulado"].notna().any():
        sub = f"Margen {format_money(fila['margen_simulado'])}"
    else:
        sub = f"Ingreso {format_money(fila['ingreso_simulado'])}"
    return str(fila["nombre_escenario"]), sub


def _render_mejor_escenario_detalle(selected: pd.DataFrame) -> None:
    """Muestra un resumen del mejor escenario y el ranking completo."""
    ranking = _ranking_escenarios_futuro(selected)
    if ranking.empty:
        st.info("No hay un escenario viable para la selección actual (sin unidades o precios válidos).")
        return

    mejor = ranking.iloc[0]
    skus_sel = selected["SKU"].nunique()
    alcance = "el SKU seleccionado" if skus_sel == 1 else f"los {skus_sel} SKUs filtrados"
    if pd.notna(mejor.get("margen_simulado")) and ranking["margen_simulado"].notna().any():
        criterio = f"mayor **margen simulado** ({format_money(mejor['margen_simulado'])})"
    else:
        criterio = f"mayor **ingreso simulado** ({format_money(mejor['ingreso_simulado'])})"
    st.success(
        f"Para {alcance}, el mejor escenario es **{mejor['nombre_escenario']}** "
        f"(cambio de precio promedio {mejor['cambio_precio_pct']:+.1f}%), por {criterio}."
    )

    ranking_display = ranking.drop(columns=["_score"]).rename(
        columns={
            "nombre_escenario": "Escenario",
            "unidades_simuladas": "Unidades simuladas",
            "ingreso_simulado": "Ingreso simulado",
            "margen_simulado": "Margen simulado",
            "cambio_precio_pct": "Cambio precio %",
        }
    )
    st.dataframe(ranking_display, use_container_width=True)


def _render_graficas_futuro(selected: pd.DataFrame) -> None:
    """Gráficas base vs proyectado por escenario: margen, ingreso y unidades.

    En el pricing futuro no existe un valor "real"; la referencia es la base
    (demanda al precio actual, sin cambio). Cada gráfica compara ese valor base
    contra el proyectado por cada escenario de precio.
    """
    import plotly.express as px

    if selected is None or selected.empty:
        st.info("No hay datos suficientes para graficar escenarios.")
        return

    df = selected.copy()
    for col in ["margen_base", "margen_simulado", "ingreso_base", "ingreso_simulado",
                "demanda_base", "unidades_simuladas"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg = (
        df.groupby("nombre_escenario", observed=True, sort=False)
        .agg(
            margen_base=("margen_base", "sum"),
            margen_simulado=("margen_simulado", "sum"),
            ingreso_base=("ingreso_base", "sum"),
            ingreso_simulado=("ingreso_simulado", "sum"),
            demanda_base=("demanda_base", "sum"),
            unidades_simuladas=("unidades_simuladas", "sum"),
        )
        .reset_index()
    )
    if agg.empty:
        st.info("No hay datos suficientes para graficar escenarios.")
        return

    def _grafica_comparativa(base_col: str, proj_col: str, base_lbl: str, proj_lbl: str, titulo: str, eje_y: str):
        if not (agg[base_col].notna().any() or agg[proj_col].notna().any()):
            st.info(f"No hay datos para {titulo.lower()}.")
            return
        long = agg.melt(
            id_vars="nombre_escenario",
            value_vars=[base_col, proj_col],
            var_name="tipo",
            value_name="valor",
        )
        long["tipo"] = long["tipo"].map({base_col: base_lbl, proj_col: proj_lbl})
        fig = px.bar(
            long, x="nombre_escenario", y="valor", color="tipo", barmode="group",
            category_orders={"tipo": [base_lbl, proj_lbl]}, title=titulo,
        )
        fig.update_layout(xaxis_title="Escenario", yaxis_title=eje_y, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        _grafica_comparativa(
            "margen_base", "margen_simulado", "Margen base", "Margen proyectado",
            "Margen: base vs proyectado por escenario", "Margen",
        )
    with g2:
        _grafica_comparativa(
            "ingreso_base", "ingreso_simulado", "Ingreso base", "Ingreso proyectado",
            "Ingreso: base vs proyectado por escenario", "Ingreso",
        )
    _grafica_comparativa(
        "demanda_base", "unidades_simuladas", "Unidades base", "Unidades proyectadas",
        "Unidades: base vs proyectado por escenario", "Unidades",
    )


def render_future_pricing_controls() -> None:
    """Controles de horizonte y método de proyección (Vista 4, Fase 8)."""
    st.subheader("Configuración de proyección de demanda")

    col_h, col_m = st.columns([1, 2])
    with col_h:
        st.radio(
            "Horizonte",
            ["1 mes", "3 meses", "Ambos"],
            index=["1 mes", "3 meses", "Ambos"].index(st.session_state.get("future_horizonte_sel", "Ambos")),
            key="future_horizonte_sel",
            horizontal=True,
        )
    with col_m:
        from modules.config import DEMANDA_FUTURA_METODOS

        st.selectbox(
            "Método de proyección",
            DEMANDA_FUTURA_METODOS,
            index=DEMANDA_FUTURA_METODOS.index(st.session_state.get("future_metodo", "Automático recomendado")),
            key="future_metodo",
        )

    if st.session_state.get("future_metodo") == "Manual avanzado":
        st.caption("Selecciona las ventanas históricas a combinar (se reparten con peso uniforme):")
        cols = st.columns(3)
        seleccionadas: list[str] = []
        previas = set(st.session_state.get("future_manual_ventanas") or ["ultimos_3_meses", "ultimos_12_meses", "mismo_mes_historico"])
        for i, (clave, etiqueta) in enumerate(DEMANDA_VENTANAS_MANUAL):
            with cols[i % 3]:
                if st.checkbox(etiqueta, value=clave in previas, key=f"manual_ventana_{clave}"):
                    seleccionadas.append(clave)
        st.session_state.future_manual_ventanas = seleccionadas

    st.info(
        "El sistema calculará la demanda base usando las ventanas históricas seleccionadas y "
        "después aplicará la elasticidad estimada para simular escenarios de precio."
    )


def render_future_pricing_view() -> None:
    """Vista 4: Future Pricing Simulator (Fase 5)."""
    st.title("4. Pricing futuro (simulador)")
    st.caption("Fase 5: escenarios futuros de pricing para horizontes de 1 mes y 3 meses.")
    st.info(
        "Este módulo **no recalcula elasticidad**. Usa `demanda_base_futura` y "
        "`elasticidades_periodo` ya calculadas para simular escenarios simples de cambio de precio."
    )

    if not require_processed():
        return

    render_future_pricing_controls()

    if not ensure_future_pricing_ready():
        return

    sim = st.session_state.pricing_futuro_escenarios
    demanda = st.session_state.demanda_base_futura
    if sim is None or sim.empty:
        st.warning("No hay escenarios futuros. Revisa demanda_base_futura, precios actuales y elasticidades disponibles.")
        if demanda is not None and not demanda.empty:
            st.subheader("Tabla interna: demanda_base_futura")
            st.dataframe(demanda, use_container_width=True)
        return

    st.subheader("Filtros")
    f1, f2, f3, f4 = st.columns(4)
    horizonte = _dependent_selectbox("Horizonte", ["Todos"] + _safe_sorted_options(sim, "horizonte"), "future_pricing_horizonte", "Todos", f1)
    df_h = _filter_fast(sim, "horizonte", horizonte)
    metodo = _dependent_selectbox("Método", ["Todos"] + _safe_sorted_options(df_h, "metodo_proyeccion"), "future_pricing_metodo", "Todos", f2)
    df_m = _filter_fast(df_h, "metodo_proyeccion", metodo)
    sku = _dependent_selectbox("SKU", ["Todos"] + _safe_sorted_options(df_m, "SKU"), "future_pricing_sku", "Todos", f3)
    df_s = _filter_fast(df_m, "SKU", sku)
    escenario = _dependent_selectbox("Escenario", ["Todos"] + _safe_sorted_options(df_s, "nombre_escenario"), "future_pricing_escenario", "Todos", f4)
    selected = _filter_fast(df_s, "nombre_escenario", escenario)

    if selected.empty:
        st.warning("No hay resultados para la combinación de filtros seleccionada.")
        return

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Unidades simuladas", format_num(selected["unidades_simuladas"].sum(), 0), "Futuro")
    with k2:
        render_kpi_card("Ingreso simulado", format_money(selected["ingreso_simulado"].sum()), "Futuro")
    with k3:
        render_kpi_card("Margen simulado", format_money(selected["margen_simulado"].sum()), "Futuro")
    with k4:
        pct_reco = (selected["recomendacion"].eq("Recomendar").mean() * 100) if "recomendacion" in selected.columns else 0
        render_kpi_card("Escenarios recomendados", f"{pct_reco:.1f}%", "Dentro del filtro")
    with k5:
        mejor_nombre, mejor_sub = _mejor_escenario_futuro(selected)
        render_kpi_card("Mejor escenario", mejor_nombre, mejor_sub)

    st.subheader("Mejor escenario según filtros")
    _render_mejor_escenario_detalle(selected)

    st.subheader("Gráficas de escenarios")
    _render_graficas_futuro(selected)

    st.subheader("Confianza y riesgo")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(selected["confianza_final"].value_counts(dropna=False).rename_axis("confianza_final").reset_index(name="Escenarios"), use_container_width=True)
    with c2:
        st.dataframe(selected["riesgo"].value_counts(dropna=False).rename_axis("riesgo").reset_index(name="Escenarios"), use_container_width=True)

    st.subheader("Tabla interna: pricing_futuro_escenarios")
    table_cols = [
        "SKU", "categoria", "departamento", "horizonte", "metodo_proyeccion",
        "tipo_elasticidad_usada", "tipo_escenario", "nombre_escenario", "precio_actual", "precio_lista",
        "precio_efectivo", "descuento_efectivo", "cambio_precio_pct", "riesgo_promocion", "demanda_base",
        "unidades_simuladas", "ingreso_base", "ingreso_simulado", "margen_base",
        "margen_simulado", "variacion_unidades", "variacion_ingreso", "variacion_margen",
        "elasticidad_usada", "confianza_elasticidad", "confianza_demanda", "confianza_final",
        "riesgo", "recomendacion", "razon_recomendacion",
    ]
    st.dataframe(selected[[col for col in table_cols if col in selected.columns]], use_container_width=True)

    with st.expander("Ver demanda_base_futura usada", expanded=False):
        st.dataframe(demanda, use_container_width=True)

    st.subheader("Descarga")
    st.download_button(
        "Descargar pricing_futuro_escenarios filtrado",
        data=_df_to_excel_friendly_csv_bytes(selected, sep=","),
        file_name="pricing_futuro_escenarios.csv",
        mime="text/csv; charset=utf-8",
        use_container_width=True,
    )

def ensure_recommendations_ready() -> bool:
    """Genera recomendaciones_sku (Fase 7) a partir de pricing_futuro_escenarios."""
    if not st.session_state.get("future_pricing_ready", False):
        st.warning(
            "Primero calcula el **pricing futuro** en la vista **4. Pricing futuro (simulador)**. "
            "Las recomendaciones se construyen sobre los escenarios futuros simulados."
        )
        return False

    sim = st.session_state.get("pricing_futuro_escenarios", pd.DataFrame())
    if sim is None or sim.empty:
        st.warning("No hay escenarios futuros disponibles para generar recomendaciones.")
        return False

    col_a, col_b = st.columns([1, 2])
    with col_a:
        button_clicked = st.button(
            "Calcular / actualizar recomendaciones",
            type="primary" if not st.session_state.recommendations_ready else "secondary",
            use_container_width=True,
        )
    with col_b:
        st.caption(
            "Fase 7: motor híbrido (reglas de negocio + simulación financiera). "
            "El Random Forest queda como apoyo opcional y nunca decide la recomendación."
        )

    if st.session_state.recommendations_ready and not button_clicked:
        return True
    if not button_clicked and not st.session_state.recommendations_ready:
        st.info("Presiona **Calcular / actualizar recomendaciones** para ejecutar la Fase 7.")
        return False

    try:
        analysis_key = (
            st.session_state.get("sales_signature"),
            st.session_state.get("nse_signature"),
            st.session_state.get("promo_signature"),
        )
        reco_key = (analysis_key, "recomendaciones_sku_v1")
        cache_re = st.session_state.manual_cache.setdefault("recomendaciones", {})
        if reco_key in cache_re:
            recomendaciones = cache_re[reco_key]
        else:
            with st.spinner("Generando recomendaciones ejecutivas..."):
                recomendaciones = build_recommendations_cached(
                    sim,
                    st.session_state.get("elasticidades_periodo", pd.DataFrame()),
                    st.session_state.get("demanda_base_futura", pd.DataFrame()),
                    st.session_state.ventas_nse,
                )
            cache_re.clear()
            cache_re[reco_key] = recomendaciones

        st.session_state.recomendaciones_sku = recomendaciones
        st.session_state.recommendations_ready = True
        st.success("Recomendaciones generadas correctamente.")
        return True
    except Exception as exc:
        st.session_state.recommendations_ready = False
        st.error(f"No se pudieron generar las recomendaciones: {exc}")
        return False


def render_recommendations_view() -> None:
    """Vista 5: Recomendaciones ejecutivas (Fase 7)."""
    st.title("5. Recomendaciones ejecutivas")
    st.caption("Ranking accionable por SKU: qué hacer con el precio y por qué.")
    st.info(
        "Cada SKU recibe una **categoría de recomendación** (subir / bajar-promover / mantener / no recomendar) "
        "y una **estrategia específica**, con su razón en español. Decisión por reglas; ML solo de apoyo."
    )

    if not require_processed():
        return
    if not ensure_recommendations_ready():
        return

    reco = st.session_state.get("recomendaciones_sku", pd.DataFrame())
    if reco is None or reco.empty:
        st.warning("No hay recomendaciones disponibles para la base actual.")
        return

    st.subheader("Filtros")
    f1, f2, f3, f4, f5 = st.columns(5)
    categoria = _dependent_selectbox("Categoría", ["Todos"] + _safe_sorted_options(reco, "categoria"), "reco_categoria", "Todos", f1)
    df_c = _filter_fast(reco, "categoria", categoria)
    departamento = _dependent_selectbox("Departamento", ["Todos"] + _safe_sorted_options(df_c, "departamento"), "reco_departamento", "Todos", f2)
    df_d = _filter_fast(df_c, "departamento", departamento)
    # El horizonte se muestra explícitamente como 1 mes / 3 meses / Ambos.
    # "Ambos" equivale a no filtrar por horizonte.
    horizontes_disponibles = [h for h in ["1 mes", "3 meses"] if h in _safe_sorted_options(df_d, "horizonte")]
    horizonte = _dependent_selectbox("Horizonte", ["Ambos"] + horizontes_disponibles, "reco_horizonte", "Ambos", f3)
    df_h = df_d if horizonte == "Ambos" else _filter_fast(df_d, "horizonte", horizonte)
    cat_reco = _dependent_selectbox("Recomendación", ["Todos"] + _safe_sorted_options(df_h, "categoria_recomendacion"), "reco_cat_reco", "Todos", f4)
    df_r = _filter_fast(df_h, "categoria_recomendacion", cat_reco)
    confianza = _dependent_selectbox("Confianza", ["Todos"] + _safe_sorted_options(df_r, "confianza_final"), "reco_confianza", "Todos", f5)
    selected = _filter_fast(df_r, "confianza_final", confianza)

    if selected.empty:
        st.warning("No hay datos para esta selección.")
        return

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("SKUs en ranking", format_num(selected["SKU"].nunique(), 0), "Dentro del filtro")
    with k2:
        render_kpi_card("Ingreso esperado", format_money(pd.to_numeric(selected["ingreso_esperado"], errors="coerce").sum()), "Futuro")
    with k3:
        render_kpi_card("Margen esperado", format_money(pd.to_numeric(selected["margen_esperado"], errors="coerce").sum()), "Futuro")
    with k4:
        recomendables = selected["categoria_recomendacion"].ne("No recomendar").mean() * 100
        render_kpi_card("SKUs con acción", f"{recomendables:.1f}%", "Recomendados")

    st.subheader("Distribución de recomendaciones")
    st.dataframe(
        selected["categoria_recomendacion"].value_counts(dropna=False).rename_axis("categoria_recomendacion").reset_index(name="SKUs"),
        use_container_width=True,
    )

    st.subheader("Ranking de SKUs")
    ranking_cols = [
        "SKU", "categoria", "departamento", "horizonte",
        "categoria_recomendacion", "estrategia_especifica",
        "precio_recomendado", "ingreso_esperado", "margen_esperado",
        "confianza_final", "riesgo", "razon_recomendacion",
    ]
    ranking = selected[[c for c in ranking_cols if c in selected.columns]].copy()
    ranking = ranking.sort_values("ingreso_esperado", ascending=False, na_position="last")
    st.dataframe(ranking, use_container_width=True)
    st.caption("La columna `razon_recomendacion` explica en español el porqué de cada recomendación.")

    st.subheader("Descarga")
    st.download_button(
        "Descargar recomendaciones_sku filtrado",
        data=_df_to_excel_friendly_csv_bytes(selected, sep=","),
        file_name="recomendaciones_sku.csv",
        mime="text/csv; charset=utf-8",
        use_container_width=True,
    )


def render_exportables_view() -> None:
    """Vista 6: Exportables (placeholder de UI; se cablea por completo en Fase 10)."""
    st.title("6. Exportables")
    st.caption("Descarga las tablas internas calculadas en formato CSV.")
    st.info(
        "Cada botón exporta una tabla ya calculada. Si una tabla aún no se ha generado, "
        "ve a su vista correspondiente y ejecútala primero."
    )

    if not require_processed():
        return

    exportables = [
        ("diagnostico_calidad.csv", "diagnostico_calidad", "Diagnóstico de calidad"),
        ("ventas_limpias.csv", "ventas_limpias", "Ventas limpias"),
        ("elasticidades_periodo.csv", "elasticidades_periodo", "Elasticidades por periodo"),
        ("pricing_historico_escenarios.csv", "pricing_historico_escenarios", "Pricing histórico (escenarios)"),
        ("demanda_base_futura.csv", "demanda_base_futura", "Demanda base futura"),
        ("pricing_futuro_escenarios.csv", "pricing_futuro_escenarios", "Pricing futuro (escenarios)"),
        ("recomendaciones_sku.csv", "recomendaciones_sku", "Recomendaciones por SKU"),
    ]

    for file_name, state_key, label in exportables:
        df = st.session_state.get(state_key, pd.DataFrame())
        col_a, col_b = st.columns([3, 2])
        with col_a:
            disponible = isinstance(df, pd.DataFrame) and not df.empty
            estado = f"✅ {len(df)} filas" if disponible else "⏳ aún no generada"
            st.markdown(f"**{label}** — `{file_name}` · {estado}")
        with col_b:
            st.download_button(
                f"Descargar {file_name}",
                data=_df_to_excel_friendly_csv_bytes(df if isinstance(df, pd.DataFrame) else pd.DataFrame(), sep=","),
                file_name=file_name,
                mime="text/csv; charset=utf-8",
                use_container_width=True,
                disabled=not (isinstance(df, pd.DataFrame) and not df.empty),
                key=f"export_{state_key}",
            )


def _ventas_fase_promocion(ventas_nse: pd.DataFrame, ventanas: pd.DataFrame, sku: str) -> pd.DataFrame:
    """Etiqueta las ventas semanales de un SKU como Antes / Durante / Después de promoción.

    Toma la primera ventana de promoción del SKU y compara una ventana simétrica de
    semanas antes y después. Devuelve un DataFrame con columnas `semana`, `unidades`,
    `fase`. Si no hay datos suficientes, devuelve vacío (no crashea).
    """
    from modules.utils import parse_transaction_dates

    if ventas_nse is None or ventas_nse.empty or ventanas is None or ventanas.empty:
        return pd.DataFrame()

    vent_sku = ventanas[ventanas["SKU"].astype(str) == str(sku)].dropna(subset=["fecha_inicio"])
    if vent_sku.empty:
        return pd.DataFrame()
    inicio = pd.to_datetime(vent_sku["fecha_inicio"].min())
    fin = pd.to_datetime(vent_sku["fecha_fin"].max())
    if pd.isna(fin) or fin < inicio:
        fin = inicio

    ventas = ventas_nse.copy()
    sku_col = "SKU" if "SKU" in ventas.columns else ("prod_nbr" if "prod_nbr" in ventas.columns else None)
    if sku_col is None or "qty" not in ventas.columns or "tran_date" not in ventas.columns:
        return pd.DataFrame()
    ventas = ventas[ventas[sku_col].astype(str) == str(sku)].copy()
    if ventas.empty:
        return pd.DataFrame()

    ventas["tran_date"] = parse_transaction_dates(ventas["tran_date"])
    ventas["qty"] = pd.to_numeric(ventas["qty"], errors="coerce")
    ventas = ventas.dropna(subset=["tran_date", "qty"])
    if ventas.empty:
        return pd.DataFrame()

    duracion = max((fin - inicio).days, 7)
    margen = pd.Timedelta(days=duracion)
    desde = inicio - margen
    hasta = fin + margen
    ventana = ventas[(ventas["tran_date"] >= desde) & (ventas["tran_date"] <= hasta)].copy()
    if ventana.empty:
        return pd.DataFrame()

    def _fase(fecha):
        if fecha < inicio:
            return "Antes"
        if fecha <= fin:
            return "Durante"
        return "Después"

    ventana["fase"] = ventana["tran_date"].map(_fase)
    serie = (
        ventana.groupby([pd.Grouper(key="tran_date", freq="W"), "fase"], observed=True)["qty"]
        .sum()
        .reset_index()
        .rename(columns={"tran_date": "semana", "qty": "unidades"})
    )
    return serie


def render_promociones_historico(selected: pd.DataFrame) -> None:
    """Sección de promociones dentro del pricing histórico (antes/durante/después)."""
    from modules.promotions import normalizar_ventanas_promocion

    st.subheader("Impacto de promociones en ventas reales")

    promo_df = st.session_state.get("promo_df")
    ventanas = normalizar_ventanas_promocion(promo_df)

    if promo_df is None or (isinstance(promo_df, pd.DataFrame) and promo_df.empty):
        st.info(
            "No se subió una base de promociones. Es **opcional**: cárgala en la barra lateral "
            "(sección B) para ver el efecto de las promociones en las ventas reales. "
            "Las promociones detectadas sí se incorporan al cálculo de elasticidad cuando se proveen."
        )
        return

    if ventanas.empty:
        st.warning(
            "La base de promociones se cargó pero no se reconocieron columnas de SKU/fecha. "
            "Asegúrate de incluir un identificador de SKU (`prod_nbr`/`SKU`) y una fecha de inicio "
            "(`fecha_inicio`/`start_date`)."
        )
        return

    skus_con_promo = sorted(ventanas["SKU"].astype(str).unique())
    st.success(
        f"Base de promociones considerada: {len(ventanas)} ventanas de promoción sobre "
        f"{len(skus_con_promo)} SKU(s). También se usa al calcular elasticidad."
    )

    # SKUs presentes en el filtro actual que además tienen promoción.
    skus_filtro = set(selected["SKU"].astype(str).unique()) if "SKU" in selected.columns else set()
    opciones = [s for s in skus_con_promo if not skus_filtro or s in skus_filtro] or skus_con_promo
    sku_promo = st.selectbox(
        "SKU para ver ventas antes / durante / después de la promoción",
        opciones,
        key="hist_promo_sku",
    )

    serie = _ventas_fase_promocion(st.session_state.get("ventas_nse", pd.DataFrame()), ventanas, sku_promo)
    if serie.empty:
        st.info("No hay suficientes ventas alrededor de la promoción de este SKU para graficar.")
        return

    import plotly.express as px

    orden = {"Antes": 0, "Durante": 1, "Después": 2}
    serie = serie.sort_values("semana")
    fig = px.bar(
        serie, x="semana", y="unidades", color="fase",
        category_orders={"fase": ["Antes", "Durante", "Después"]},
        title=f"Ventas semanales del SKU {sku_promo}: antes, durante y después de la promoción",
    )
    fig.update_layout(xaxis_title="Semana", yaxis_title="Unidades vendidas", legend_title="Fase")
    st.plotly_chart(fig, use_container_width=True)

    resumen = (
        serie.groupby("fase", observed=True)["unidades"]
        .agg(["sum", "mean"])
        .reindex(["Antes", "Durante", "Después"])
        .rename(columns={"sum": "Unidades totales", "mean": "Promedio semanal"})
        .reset_index()
        .rename(columns={"fase": "Fase"})
    )
    st.dataframe(resumen, use_container_width=True)
    antes = resumen.loc[resumen["Fase"].eq("Antes"), "Promedio semanal"].fillna(0).sum()
    durante = resumen.loc[resumen["Fase"].eq("Durante"), "Promedio semanal"].fillna(0).sum()
    if antes > 0:
        cambio = (durante - antes) / antes * 100
        st.caption(f"Durante la promoción, el promedio semanal de unidades cambió {cambio:+.1f}% frente al periodo previo.")


def render_historical_pricing_view() -> None:
    """Vista 3: Historical Pricing Simulator separado de pricing futuro."""
    st.title("3. Pricing histórico (backtesting)")
    st.caption("Simulación histórica de escenarios de precio con ventas reales y elasticidades ya calculadas.")
    st.info(
        "Este módulo **no predice futuro** y **no calcula demanda base futura**. "
        "Responde: ¿qué habría pasado en un periodo pasado si hubiera cambiado el precio?"
    )

    if not require_processed():
        return

    if not ensure_historical_pricing_ready():
        return

    sim = st.session_state.get("pricing_historico_escenarios", pd.DataFrame())
    if sim is None or sim.empty:
        st.warning("No hay escenarios históricos. Revisa que existan ventas reales, costos y elasticidades_periodo compatibles.")
        return

    required_cols = [
        "categoria", "departamento", "periodo_tipo", "periodo", "SKU",
        "tipo_elasticidad_usada", "nombre_escenario",
    ]
    missing = [col for col in required_cols if col not in sim.columns]
    if missing:
        st.error("La tabla pricing_historico_escenarios no tiene columnas obligatorias: " + ", ".join(missing))
        return

    st.subheader("Filtros")
    st.caption(
        "Filtros dependientes: categoría → departamento → periodo_tipo → periodo → SKU → "
        "tipo de elasticidad usada → escenario de precio. Cambiarlos solo filtra `pricing_historico_escenarios`."
    )

    f1, f2, f3, f4 = st.columns(4)
    categoria = _dependent_selectbox("Categoría", ["Todas"] + _safe_sorted_options(sim, "categoria"), "hist_pricing_categoria", "Todas", f1)
    df_cat = _filter_fast(sim, "categoria", categoria)

    departamento = _dependent_selectbox("Departamento", ["Todos"] + _safe_sorted_options(df_cat, "departamento"), "hist_pricing_departamento", "Todos", f2)
    df_dept = _filter_fast(df_cat, "departamento", departamento)

    periodo_tipo = _dependent_selectbox("periodo_tipo", ["Todos"] + _safe_sorted_options(df_dept, "periodo_tipo"), "hist_pricing_periodo_tipo", "Todos", f3)
    df_tipo = _filter_fast(df_dept, "periodo_tipo", periodo_tipo)

    periodo = _dependent_selectbox("Periodo", ["Todos"] + _safe_sorted_options(df_tipo, "periodo"), "hist_pricing_periodo", "Todos", f4)
    df_periodo = _filter_fast(df_tipo, "periodo", periodo)

    f5, f6, f7 = st.columns(3)
    sku = _dependent_selectbox("SKU", ["Todos"] + _safe_sorted_options(df_periodo, "SKU"), "hist_pricing_sku", "Todos", f5)
    df_sku = _filter_fast(df_periodo, "SKU", sku)

    tipo_elasticidad = _dependent_selectbox(
        "Tipo de elasticidad usada",
        ["Todos"] + _safe_sorted_options(df_sku, "tipo_elasticidad_usada"),
        "hist_pricing_tipo_elasticidad",
        "Todos",
        f6,
    )
    df_elasticidad = _filter_fast(df_sku, "tipo_elasticidad_usada", tipo_elasticidad)

    escenario = _dependent_selectbox(
        "Escenario de precio",
        ["Todos"] + _safe_sorted_options(df_elasticidad, "nombre_escenario"),
        "hist_pricing_escenario",
        "Todos",
        f7,
    )
    selected = _filter_fast(df_elasticidad, "nombre_escenario", escenario)

    if selected.empty:
        st.warning("No hay resultados para la combinación de filtros seleccionada.")
        return

    st.subheader("KPIs del backtesting histórico")
    unidades_reales = selected["unidades_reales"].sum()
    unidades_sim = selected["unidades_simuladas"].sum()
    ingreso_real = selected["ingreso_real"].sum()
    ingreso_sim = selected["ingreso_simulado"].sum()
    margen_real = selected["margen_real"].sum()
    margen_sim = selected["margen_simulado"].sum()

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card("Unidades simuladas", format_num(unidades_sim, 0), f"Real: {format_num(unidades_reales, 0)}")
    with k2:
        render_kpi_card("Ingreso simulado", format_money(ingreso_sim), f"Real: {format_money(ingreso_real)}")
    with k3:
        render_kpi_card("Margen simulado", format_money(margen_sim), f"Real: {format_money(margen_real)}")

    best = selected[selected["mejor_escenario_historico"].fillna(False).astype(bool)]
    if not best.empty:
        st.success("Mejores escenarios históricos dentro del filtro: " + ", ".join(best["nombre_escenario"].dropna().astype(str).unique()[:5]))

    import plotly.express as px

    chart = (
        selected.groupby("nombre_escenario", observed=True, sort=False)
        .agg(
            ingreso_real=("ingreso_real", "sum"),
            ingreso_simulado=("ingreso_simulado", "sum"),
            margen_real=("margen_real", "sum"),
            margen_simulado=("margen_simulado", "sum"),
        )
        .reset_index()
    )
    if not chart.empty:
        chart_long = chart.melt(id_vars="nombre_escenario", var_name="métrica", value_name="monto")
        fig = px.bar(chart_long, x="nombre_escenario", y="monto", color="métrica", barmode="group", title="Real vs simulado por escenario histórico")
        st.plotly_chart(fig, use_container_width=True)

    render_promociones_historico(selected)

    st.subheader("Tabla interna: pricing_historico_escenarios")
    table_cols = [
        "SKU", "categoria", "departamento", "periodo_tipo", "periodo",
        "tipo_elasticidad_usada", "tipo_escenario", "nombre_escenario", "precio_real", "precio_lista",
        "precio_efectivo", "descuento_efectivo", "cambio_precio_pct", "riesgo_promocion",
        "unidades_reales", "unidades_simuladas", "ingreso_real", "ingreso_simulado",
        "margen_real", "margen_simulado", "variacion_unidades", "variacion_ingreso",
        "variacion_margen", "recomendacion_historica", "confianza", "razon_recomendacion",
    ]
    st.dataframe(selected[[col for col in table_cols if col in selected.columns]], use_container_width=True)

    st.subheader("Descarga")
    st.download_button(
        "Descargar pricing_historico_escenarios filtrado",
        data=_df_to_excel_friendly_csv_bytes(selected, sep=","),
        file_name="pricing_historico_escenarios.csv",
        mime="text/csv; charset=utf-8",
        use_container_width=True,
    )

def require_processed() -> bool:
    """Valida que haya datos procesados."""
    if not st.session_state.processed:
        st.warning(
            "Carga una base de ventas y presiona **Procesar / actualizar datos** en el sidebar. "
            "Después, cada vista calcula únicamente lo que necesita."
        )
        return False
    return True


# =========================================================
# Vistas
# =========================================================

def render_quality_view() -> None:
    """Vista 1: carga y diagnóstico."""
    st.title("1. Carga y diagnóstico de datos")
    st.caption("Validación, limpieza, cruce NSE y semáforo de calidad.")

    st.markdown(
        """
        Esta vista solo ejecuta limpieza, cruce NSE y diagnóstico de calidad.  
        **No calcula elasticidad ni pricing**, para que la app cargue más rápido.
        """
    )


    st.subheader("Configuración de nivel socioeconómico")
    st.caption(
        "La app usa la base NSE default precargada si no se sube una personalizada válida. "
        "Las bases personalizadas se validan antes del cruce y, si fallan, no rompen la app."
    )
    st.write(f"**Opción seleccionada:** {st.session_state.get('nse_mode', 'Usar base NSE default')}")
    st.write(f"**Ubicación de bases NSE default:** `{get_default_nse_path()}`")

    if not require_processed():
        return

    ventas = st.session_state.ventas_nse
    semaforo = st.session_state.semaforo
    resumen_limpieza = st.session_state.resumen_limpieza
    calidad_varianza = st.session_state.calidad_varianza
    nse_info = st.session_state.nse_info
    diagnostico_calidad = st.session_state.diagnostico_calidad

    if not semaforo.empty:
        row = semaforo.iloc[0]
        color = "#dc2626" if "Rojo" in row["Semaforo"] else "#f59e0b" if "Amarillo" in row["Semaforo"] else "#16a34a"
        st.markdown(
            f"""
            <div style="border:2px solid {color}; border-radius:16px; padding:18px; background:#ffffff;">
                <h3 style="margin-top:0;">Semáforo de calidad: {row['Semaforo']}</h3>
                <p style="margin-bottom:0;">{row['Interpretacion']}</p>
                <small>{row['Motivos']}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("KPIs de calidad")
    row = semaforo.iloc[0] if not semaforo.empty else {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Registros originales", f"{int(row.get('Filas_Originales', 0)):,}", "Antes de limpieza")
    with c2:
        render_kpi_card("Registros limpios", f"{int(row.get('Filas_Limpias', 0)):,}", "Después de limpieza")
    with c3:
        render_kpi_card("Registros eliminados", f"{int(row.get('Registros_Removidos', 0)):,}", f"{row.get('%_Registros_Removidos', 0):.1f}% removido")
    with c4:
        render_kpi_card("Datos faltantes", f"{row.get('Porcentaje_Datos_Faltantes_Original', 0):.1f}%", "Promedio original")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_kpi_card("Duplicados", f"{int(row.get('Duplicados_Originales', 0)):,}", "Detectados originalmente")
    with c6:
        render_kpi_card("Valores infinitos", f"{int(row.get('Valores_Infinitos_Detectados', 0)):,}", "Antes de limpieza")
    with c7:
        render_kpi_card("Precio inválido", f"{int(row.get('Registros_Precio_Invalido', 0)):,}", "Después de crear precio")
    with c8:
        render_kpi_card("Cantidad inválida", f"{int(row.get('Registros_Cantidad_Invalida', 0)):,}", "qty <= 0")

    st.subheader("Cruce NSE")
    st.info(nse_info.get("mensaje", "NSE no aplicado."))
    nse_c1, nse_c2, nse_c3, nse_c4 = st.columns(4)
    with nse_c1:
        render_kpi_card("Fuente NSE usada", nse_info.get("fuente_nse_usada", "default"), nse_info.get("estado_validacion_nse", ""))
    with nse_c2:
        render_kpi_card("Registros con NSE", f"{nse_info.get('porcentaje_match_nse', 0):.1f}%", "NSE asignado")
    with nse_c3:
        render_kpi_card("Sin NSE asignado", f"{int(nse_info.get('registros_sin_match_nse', 0)):,}", "Marcados como NSE_no_asignado")
    with nse_c4:
        advertencias = nse_info.get("advertencias_nse", []) or []
        render_kpi_card("Advertencias NSE", f"{len(advertencias):,}", "Validación y cruce")
    if nse_info.get("advertencias_nse"):
        with st.expander("Advertencias del cruce NSE", expanded=True):
            for warning in nse_info.get("advertencias_nse", []):
                st.warning(warning)
    if "categoria_est_socio" in ventas.columns:
        st.dataframe(
            ventas["categoria_est_socio"]
            .fillna("Sin dato")
            .value_counts(dropna=False)
            .rename_axis("categoria_est_socio")
            .reset_index(name="Registros"),
            use_container_width=True,
        )

    with st.expander("Resumen de limpieza"):
        st.dataframe(resumen_limpieza, use_container_width=True)

    with st.expander("Métricas de varianza"):
        st.dataframe(calidad_varianza, use_container_width=True)

    with st.expander("Diagnóstico de calidad consolidado"):
        st.dataframe(
            prepare_dataframe_for_streamlit(diagnostico_calidad, force_text_columns=["valor"]),
            use_container_width=True,
        )

    st.subheader("Comportamiento histórico de ventas con Machine Learning")
    st.markdown(
        """
        Antes de generar un pronóstico de ventas, la herramienta resume el comportamiento histórico
        con dos modelos supervisados: **regresión logística** y **Random Forest**. Ambos modelos
        clasifican meses SKU con venta alta vs. baja para identificar señales históricas asociadas
        a precio, temporalidad, categoría y geografía.
        """
    )
    if st.button("Analizar ventas históricas con regresión logística y Random Forest", use_container_width=True):
        with st.spinner("Entrenando modelos ML sobre ventas históricas..."):
            ml_summary = build_historical_sales_ml_cached(ventas)

        if ml_summary.get("status") != "ok":
            st.warning(ml_summary.get("message", "No se pudo entrenar el análisis histórico con ML."))
        else:
            st.success(ml_summary.get("message", "Modelos históricos entrenados correctamente."))
            summary_df = ml_summary.get("dataset_summary", pd.DataFrame())
            if summary_df is not None and not summary_df.empty:
                st.caption("Base de entrenamiento SKU-mes usada antes de cualquier pronóstico.")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            metrics_df = ml_summary.get("metrics", pd.DataFrame())
            if metrics_df is not None and not metrics_df.empty:
                st.caption("Desempeño comparativo de los modelos históricos.")
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            importance_df = ml_summary.get("feature_importance", pd.DataFrame())
            if importance_df is not None and not importance_df.empty:
                st.caption("Variables que más explican el comportamiento histórico según cada modelo.")
                st.dataframe(importance_df, use_container_width=True, hide_index=True)

            segments_df = ml_summary.get("segments", pd.DataFrame())
            if segments_df is not None and not segments_df.empty:
                st.caption("Segmentos con mayor probabilidad histórica de venta alta.")
                st.dataframe(segments_df, use_container_width=True, hide_index=True)

    st.subheader("Vista previa de ventas_limpias")
    st.dataframe(ventas.head(MAX_ROWS_PREVIEW), use_container_width=True)
    st.success("La base está lista. Pasa a Elasticidad o Pricing cuando quieras calcular esas vistas.")


def render_elasticity_view() -> None:
    """Vista 2: elasticidad multi-periodo."""
    st.title("2. Elasticidad")
    st.caption("Elasticidad log-log multi-periodo calculada desde elasticidades_periodo.")

    if not require_processed():
        return

    if not ensure_elasticity_ready(show_button=True):
        return

    import plotly.express as px

    from modules.elasticity import PERIODOS_ELASTICIDAD, build_elasticity_download
    from modules.utils import add_state_coordinates

    elasticidades_periodo = st.session_state.get("elasticidades_periodo", pd.DataFrame())
    legacy_df = st.session_state.get("elasticidad", pd.DataFrame())
    ventas = st.session_state.ventas_base_elasticidad

    if (
        (elasticidades_periodo is None or elasticidades_periodo.empty)
        and legacy_df is not None
        and not legacy_df.empty
    ):
        elasticidades_periodo = legacy_df.attrs.get(
            "elasticidades_periodo",
            st.session_state.ventas_base_elasticidad.attrs.get("elasticidades_periodo", pd.DataFrame()),
        )

    if elasticidades_periodo is None or elasticidades_periodo.empty:
        st.warning("No se generaron resultados en elasticidades_periodo. Revisa fechas, SKUs y variación de precios.")
        df_periodo = _empty_elasticity_periodo_frame()
    else:
        df_periodo = prepare_elasticity_dataframe_for_display(elasticidades_periodo)

    if "periodo_tipo" not in df_periodo.columns:
        st.warning("La tabla elasticidades_periodo no tiene la columna obligatoria periodo_tipo. Recalcula elasticidades.")
        df_periodo = _empty_elasticity_periodo_frame()

    for col in ELASTICITY_EXPECTED_COLUMNS:
        if col not in df_periodo.columns:
            df_periodo[col] = np.nan

    df_periodo = prepare_elasticity_dataframe_for_display(df_periodo)
    if "periodo_tipo" in df_periodo.columns:
        df_periodo["periodo_tipo"] = df_periodo["periodo_tipo"].astype(str).str.strip()
        df_periodo = df_periodo[df_periodo["periodo_tipo"] != ""].copy()

    if df_periodo.empty:
        st.warning("No hay elasticidades_periodo válidas para mostrar o descargar.")
        return

    if "elasticidad" not in df_periodo.columns:
        st.warning("La tabla elasticidades_periodo no tiene la columna elasticidad; se mostrará sin métricas numéricas.")
        df_periodo["elasticidad"] = np.nan
    else:
        df_periodo["elasticidad"] = pd.to_numeric(df_periodo["elasticidad"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if df_periodo["elasticidad"].isna().all():
            st.warning("La columna elasticidad no contiene valores numéricos válidos con los filtros actuales.")

    tipo_labels = {
        "Todos": None,
        "Mensual": "mensual",
        "Trimestral": "trimestral",
        "Semestral": "semestral",
        "Anual": "anual",
        "Global SKU": "global_sku",
        "Categoría/Departamento": "categoria_departamento",
    }


    label_by_periodo = {v: k for k, v in tipo_labels.items() if v is not None}

    missing_periods = [
        p
        for p in PERIODOS_ELASTICIDAD
        if p not in set(df_periodo["periodo_tipo"].dropna().astype(str))
    ]

    for periodo_tipo in missing_periods:
        st.warning(
            f"No hay suficientes datos para calcular elasticidad {label_by_periodo.get(periodo_tipo, periodo_tipo)} "
            "con los filtros seleccionados."
        )

    with st.expander("Cómo interpretar este dashboard", expanded=True):
        st.markdown(
            """
            La elasticidad mide qué tanto cambia la demanda ante un cambio de precio.
            Una elasticidad entre **0 y -1** indica demanda inelástica: puede tolerar incrementos.
            Una elasticidad **menor a -1** indica demanda elástica: conviene tener cuidado con subidas y evaluar promociones.
            Una elasticidad **positiva** es sospechosa o requiere revisión, porque sugiere que precio y demanda suben juntos.
            La columna **periodo_tipo** identifica si el registro es mensual, trimestral, semestral, anual, global SKU o categoría/departamento.
            """
        )

    st.subheader("Resumen de disponibilidad")

    resumen = (
        df_periodo.groupby("periodo_tipo", dropna=False)
        .agg(
            registros=("periodo_tipo", "size"),
            skus_unicos=("SKU", "nunique"),
            alta=("confianza_elasticidad", lambda s: (s == "Alta").sum()),
            media=("confianza_elasticidad", lambda s: (s == "Media").sum()),
            baja=("confianza_elasticidad", lambda s: (s == "Baja").sum()),
            no_usable=("confianza_elasticidad", lambda s: (s == "No usable").sum()),
        )
        .reset_index()
        .rename(
            columns={
                "periodo_tipo": "periodo_tipo",
                "registros": "número de registros",
                "skus_unicos": "número de SKUs únicos",
                "alta": "confianza Alta",
                "media": "confianza Media",
                "baja": "confianza Baja",
                "no_usable": "No usable",
            }
        )
    )

    st.dataframe(
        prepare_dataframe_for_streamlit(resumen),
        use_container_width=True,
        hide_index=True,
    )

    recomendables_raw = df_periodo.get("recomendable_elasticidad", pd.Series(False, index=df_periodo.index))
    if recomendables_raw.dtype == "object" or str(recomendables_raw.dtype).startswith("string"):
        recomendables = recomendables_raw.fillna("").astype(str).str.strip().str.lower().isin(["true", "1", "sí", "si", "yes"])
    else:
        recomendables = recomendables_raw.fillna(False).astype(bool)

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        render_kpi_card("Total registros", f"{len(df_periodo):,}", "elasticidades_periodo")

    with m2:
        render_kpi_card(
            "Tipos incluidos",
            f"{df_periodo['periodo_tipo'].nunique():,}",
            ", ".join(sorted(df_periodo["periodo_tipo"].unique())),
        )

    with m3:
        render_kpi_card("SKUs únicos", f"{df_periodo['SKU'].nunique():,}", "Incluye grupos categoría/depto")

    with m4:
        render_kpi_card("Recomendables", f"{int(recomendables.sum()):,}", "confianza Media/Alta")

    with m5:
        render_kpi_card("No recomendables", f"{int((~recomendables).sum()):,}", "Baja o No usable")

    # =====================================================
    # Filtros únicos de elasticidad
    # =====================================================
    st.subheader("Filtros")

    c0, c1, c2, c3 = st.columns(4)

    with c0:
        tipo_label = st.selectbox(
            "Tipo de elasticidad",
            list(tipo_labels.keys()),
            index=list(tipo_labels.keys()).index("Trimestral"),
            key="elasticity_tipo_selectbox",
        )

    selected_periodo_tipo = tipo_labels[tipo_label]
    filtered_base = df_periodo.copy()

    if selected_periodo_tipo is not None:
        filtered_base = filtered_base[
            filtered_base["periodo_tipo"] == selected_periodo_tipo
        ].copy()

    if filtered_base.empty:
        st.warning(
            f"No hay suficientes datos para calcular elasticidad {tipo_label} "
            "con los filtros seleccionados."
        )
        filtered = filtered_base.copy()
        dept = "Todos"
        periodo = "Todos"
        skus = []

    else:
        dept_options = ["Todos"] + sorted(
            filtered_base
            .get("departamento", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        with c1:
            dept = st.selectbox(
                "Departamento",
                dept_options,
                key="elasticity_departamento_selectbox",
            )

        filtered_dept = filtered_base.copy()

        if dept != "Todos" and "departamento" in filtered_dept.columns:
            filtered_dept = filtered_dept[
                filtered_dept["departamento"].astype(str) == str(dept)
            ]

        periodo_options = ["Todos"] + sorted(
            filtered_dept
            .get("periodo", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        with c2:
            periodo = st.selectbox(
                "Periodo",
                periodo_options,
                key="elasticity_periodo_selectbox",
            )

        filtered_periodo = filtered_dept.copy()

        if periodo != "Todos" and "periodo" in filtered_periodo.columns:
            filtered_periodo = filtered_periodo[
                filtered_periodo["periodo"].astype(str) == str(periodo)
            ]

        sku_options = sorted(
            filtered_periodo
            .get("SKU", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        with c3:
            skus = st.multiselect(
                "SKU",
                sku_options,
                default=sku_options[: min(5, len(sku_options))],
                key="elasticity_sku_multiselect",
            )

        filtered = filtered_periodo.copy()

        if skus:
            filtered = filtered[
                filtered["SKU"].astype(str).isin(skus)
            ]

        if filtered.empty:
            st.warning(
                f"No hay suficientes datos para calcular elasticidad {tipo_label} "
                "con los filtros seleccionados."
            )

    if not filtered.empty:
        st.subheader("KPIs de elasticidades filtradas")

        elasticidad_prom = pd.to_numeric(
            filtered.get("elasticidad"),
            errors="coerce",
        ).mean()

        r2_prom = pd.to_numeric(
            filtered.get("r2"),
            errors="coerce",
        ).mean()

        confianza_dom = (
            filtered["confianza_elasticidad"].dropna().mode().iloc[0]
            if (
                "confianza_elasticidad" in filtered.columns
                and not filtered["confianza_elasticidad"].dropna().mode().empty
            )
            else "Sin confianza"
        )

        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            render_kpi_card(
                "Elasticidad promedio",
                format_num(elasticidad_prom, 3),
                "Promedio filtrado",
            )

        with k2:
            render_kpi_card(
                "R² promedio",
                format_num(r2_prom, 3),
                "Ajuste promedio",
            )

        with k3:
            render_kpi_card(
                "SKUs analizados",
                f"{filtered['SKU'].nunique():,}",
                "Únicos",
            )

        with k4:
            render_kpi_card(
                "Registros",
                f"{len(filtered):,}",
                tipo_label,
            )

        with k5:
            render_kpi_card(
                "Confianza dominante",
                confianza_dom,
                "Moda",
            )

        st.subheader("Tabla de elasticidades")

        filtered_display = prepare_elasticity_dataframe_for_display(build_elasticity_download(filtered))

        st.dataframe(
            filtered_display,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Serie de tiempo de demanda")

        ventas_f = ventas.copy() if ventas is not None else pd.DataFrame()

        if ventas_f.empty:
            st.warning("No hay ventas para la serie de tiempo con estos filtros.")

        else:
            if (
                dept != "Todos"
                and "dept_nm" in ventas_f.columns
            ):
                ventas_f = ventas_f[
                    ventas_f["dept_nm"].astype(str) == str(dept)
                ]

            if (
                skus
                and selected_periodo_tipo != "categoria_departamento"
                and "prod_nbr" in ventas_f.columns
            ):
                ventas_f = ventas_f[
                    ventas_f["prod_nbr"].astype(str).isin(skus)
                ]

            if ventas_f.empty:
                st.warning("No hay ventas para la serie de tiempo con estos filtros.")

            else:
                serie = aggregate_weekly_demand(ventas_f)

                if "Promoción" in serie.columns:
                    fig = px.line(
                        serie,
                        x="tran_date",
                        y="Demanda",
                        color="Promoción",
                        markers=True,
                        title="Demanda semanal con/sin promoción",
                    )
                else:
                    fig = px.line(
                        serie,
                        x="tran_date",
                        y="Demanda",
                        markers=True,
                        title="Demanda semanal",
                    )
                fig.update_layout(xaxis_title="Semana", yaxis_title="Unidades")

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        st.subheader("Curva de elasticidad (precio vs demanda estimada)")
        curva = build_elasticity_curve_data(filtered, max_skus=MAX_SKUS_CURVA_ELASTICIDAD)
        if curva.empty:
            st.info(
                "No hay datos suficientes para la curva de elasticidad con los filtros actuales. "
                "Se requiere elasticidad, precio promedio y unidades promedio válidos por SKU."
            )
        else:
            fig_curva = px.line(
                curva,
                x="Precio",
                y="Demanda estimada",
                color="SKU",
                title="Curva precio-demanda log-log por SKU (Q = Q₀·(P/P₀)^elasticidad)",
            )
            fig_curva.update_layout(xaxis_title="Precio", yaxis_title="Demanda estimada")
            st.plotly_chart(fig_curva, use_container_width=True)
            st.caption(
                "Cada curva muestra cómo cambiaría la demanda estimada al variar el precio, "
                "según la elasticidad calculada del SKU. Pendiente más pronunciada = más elástico."
            )

        st.subheader("Mapa geográfico de México")

        geo = pd.DataFrame()
        estado_col = "estado"

        filtered_estado_col = next(
            (col for col in ["estado", "state"] if col in filtered.columns),
            None,
        )

        ventas_estado_col = (
            next((col for col in ["estado", "state"] if col in ventas.columns), None)
            if ventas is not None and not ventas.empty
            else None
        )

        if filtered_estado_col is not None and filtered[filtered_estado_col].dropna().any():
            geo = (
                filtered
                .dropna(subset=[filtered_estado_col])
                .groupby(filtered_estado_col, as_index=False)
                .agg(
                    elasticidad=("elasticidad", "mean"),
                    SKUs=("SKU", "nunique"),
                )
                .rename(columns={filtered_estado_col: estado_col})
            )

        elif ventas_estado_col is not None:
            ventas_geo = ventas.copy()

            if ventas_estado_col != estado_col:
                ventas_geo[estado_col] = ventas_geo[ventas_estado_col]

            if (
                dept != "Todos"
                and "dept_nm" in ventas_geo.columns
            ):
                ventas_geo = ventas_geo[
                    ventas_geo["dept_nm"].astype(str) == str(dept)
                ]

            if selected_periodo_tipo == "categoria_departamento":
                merge_cols_left = [
                    col for col in ["dept_nm", "subdept_nm"]
                    if col in ventas_geo.columns
                ]

                rename_for_merge = {
                    "departamento": "dept_nm",
                    "categoria": "subdept_nm",
                }

                elasticidad_geo = filtered.rename(
                    columns=rename_for_merge
                ).copy()

                merge_cols = [
                    col for col in merge_cols_left
                    if col in elasticidad_geo.columns
                ]

                if merge_cols:
                    elasticidad_geo = (
                        elasticidad_geo
                        .groupby(merge_cols, as_index=False)
                        .agg(elasticidad=("elasticidad", "mean"))
                    )

                    ventas_geo = ventas_geo.merge(
                        elasticidad_geo,
                        on=merge_cols,
                        how="inner",
                    )

                else:
                    ventas_geo = ventas_geo.iloc[0:0].copy()

            else:
                if (
                    skus
                    and "prod_nbr" in ventas_geo.columns
                ):
                    ventas_geo = ventas_geo[
                        ventas_geo["prod_nbr"].astype(str).isin(skus)
                    ]

                elasticidad_geo = (
                    filtered
                    .groupby("SKU", as_index=False)
                    .agg(elasticidad=("elasticidad", "mean"))
                    .rename(columns={"SKU": "prod_nbr"})
                )

                if "prod_nbr" in ventas_geo.columns:
                    ventas_geo["prod_nbr"] = ventas_geo["prod_nbr"].astype(str)
                    elasticidad_geo["prod_nbr"] = elasticidad_geo["prod_nbr"].astype(str)

                    ventas_geo = ventas_geo.merge(
                        elasticidad_geo,
                        on="prod_nbr",
                        how="inner",
                    )

                else:
                    ventas_geo = ventas_geo.iloc[0:0].copy()

            if not ventas_geo.empty:
                geo = (
                    ventas_geo
                    .dropna(subset=[estado_col])
                    .groupby(estado_col, as_index=False)
                    .agg(
                        elasticidad=("elasticidad", "mean"),
                        SKUs=("prod_nbr", "nunique"),
                    )
                )

        if geo.empty:
            st.info(
                "No hay estados/state disponibles en la base de ventas "
                "para construir el mapa con los filtros seleccionados."
            )

        else:
            geo["Elasticidad absoluta"] = pd.to_numeric(
                geo["elasticidad"],
                errors="coerce",
            ).abs()

            geo = add_state_coordinates(
                geo,
                estado_col=estado_col,
            )
            geo["lat"] = pd.to_numeric(geo.get("lat"), errors="coerce")
            geo["lon"] = pd.to_numeric(geo.get("lon"), errors="coerce")
            geo["marker_size"] = pd.to_numeric(geo["Elasticidad absoluta"], errors="coerce")
            geo["marker_size"] = geo["marker_size"].replace([np.inf, -np.inf], np.nan)
            geo["marker_size"] = geo["marker_size"].fillna(1).clip(lower=1)
            geo = geo.dropna(subset=["lat", "lon", "marker_size"])
            geo = geo[geo["marker_size"] >= 0].copy()

            if geo.empty:
                st.warning(
                    "No hay datos geográficos válidos para mostrar el mapa."
                )

            else:
                fig = px.scatter_geo(
                    geo,
                    lat="lat",
                    lon="lon",
                    color="Elasticidad absoluta",
                    size="marker_size",
                    hover_name=estado_col,
                    hover_data={
                        "elasticidad": ":.3f",
                        "SKUs": True,
                        "lat": False,
                        "lon": False,
                    },
                    scope="north america",
                    title="Intensidad de elasticidad absoluta por estado",
                )

                fig.update_geos(
                    fitbounds="locations",
                    visible=True,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

    st.subheader("Descarga")

    all_csv = prepare_elasticity_dataframe_for_display(build_elasticity_download(df_periodo))

    if all_csv.empty:
        st.warning("No hay elasticidades disponibles para descargar.")
    else:
        st.download_button(
            "Descargar todas las elasticidades",
            data=_df_to_excel_friendly_csv_bytes(all_csv, sep=","),
            file_name="elasticidades_periodo.csv",
            mime="text/csv; charset=utf-8",
            key="elasticity_download_all_button",
        )


def render_pricing_view() -> None:
    """Vista 3: pricing dinámico y proyección.

    Iteraciones aplicadas:
    - Se quitaron los filtros de estado y nivel socioeconómico de la interfaz.
    - La dependencia de filtros queda en este orden:
      1) Categoría de SKU -> 2) Departamento -> 3) Trimestre -> 4) SKU.
    - Cada filtro delimita automáticamente las opciones de los siguientes.
    - La descarga completa se prepara bajo demanda y se entrega como ZIP con CSV,
      para evitar archivos corruptos o demasiado pesados en Streamlit/Excel.
    """
    st.title("3. Pricing dinámico + proyección de ventas")
    st.caption("Simulación de escenarios, KPIs, proyección y recomendación del mejor escenario.")
    st.info(
        "Esta vista depende de dos insumos: **base limpia + NSE** y **elasticidad SKU × trimestre**. "
        "Los filtros visibles son categoría, departamento, trimestre y SKU. "
        "El NSE sigue dentro de la base para el análisis, pero ya no aparece como filtro en esta vista."
    )

    if not require_processed():
        return

    if not ensure_pricing_ready():
        return

    import plotly.express as px

    from modules.pricing import build_dynamic_explanation_pricing, build_pricing_downloads

    sim = st.session_state.simulacion
    resumen = st.session_state.resumen_pricing

    if sim is None or sim.empty:
        st.warning("No hay simulaciones de pricing. Revisa elasticidad, costos y bloques trimestrales.")
        return

    # Validaciones mínimas para evitar errores por columnas faltantes.
    required_filter_cols = ["Categoria_RF", "trimestre", "SKU", "Nombre_Escenario"]
    missing = [c for c in required_filter_cols if c not in sim.columns]
    if missing:
        st.error(
            "La tabla de simulaciones no tiene las columnas necesarias para la vista de pricing: "
            + ", ".join(missing)
        )
        return

    st.subheader("Filtros")
    st.caption(
        "Orden de dependencia: **Categoría de SKU → Departamento → Trimestre → SKU**. "
        "Al cambiar un filtro, los siguientes muestran únicamente opciones válidas."
    )

    f1, f2, f3, f4 = st.columns(4)

    # 1) Categoría de SKU
    cat_options = ["Todas"] + _safe_sorted_options(sim, "Categoria_RF")
    categoria = _dependent_selectbox(
        label="1. Categoría de SKU",
        options=cat_options,
        key="pricing_filter_categoria",
        default="Todas",
        container=f1,
    )
    df_cat = _filter_fast(sim, "Categoria_RF", categoria)

    # 2) Departamento, delimitado por categoría.
    dept_col = "dept_nm" if "dept_nm" in df_cat.columns else None
    dept_options = ["Todos"] + (_safe_sorted_options(df_cat, dept_col) if dept_col else [])
    dept = _dependent_selectbox(
        label="2. Departamento",
        options=dept_options,
        key="pricing_filter_departamento",
        default="Todos",
        container=f2,
    )
    df_dept = _filter_fast(df_cat, dept_col, dept) if dept_col else df_cat

    # 3) Trimestre, delimitado por categoría + departamento.
    tri_options = ["Todos"] + _safe_sorted_options(df_dept, "trimestre")
    trimestre = _dependent_selectbox(
        label="3. Trimestre",
        options=tri_options,
        key="pricing_filter_trimestre",
        default="Todos",
        container=f3,
    )
    df_tri = _filter_fast(df_dept, "trimestre", trimestre)

    # 4) SKU, delimitado por categoría + departamento + trimestre.
    sku_options = ["Todos"] + _safe_sorted_options(df_tri, "SKU")
    sku = _dependent_selectbox(
        label="4. SKU",
        options=sku_options,
        key="pricing_filter_sku",
        default="Todos",
        container=f4,
    )
    df_sku = _filter_fast(df_tri, "SKU", sku)

    if df_sku.empty:
        st.warning("No hay resultados para la combinación de filtros seleccionada.")
        return

    # Escenario: se filtra después de la cascada principal.
    escenario_options = _safe_sorted_options(df_sku, "Nombre_Escenario")
    if not escenario_options:
        escenario_options = ESCENARIOS_PRICING["Nombre_Escenario"].astype(str).tolist()

    escenario = st.selectbox(
        "Escenario de pricing",
        escenario_options,
        key="pricing_filter_escenario",
    )

    selected = _filter_fast(df_sku, "Nombre_Escenario", escenario)

    if selected.empty:
        st.warning("No hay resultados para la combinación de filtros y escenario seleccionado.")
        return

    card1, card2 = st.columns(2)
    with card1:
        cat_sel = (
            selected["Categoria_RF"].dropna().mode().iloc[0]
            if "Categoria_RF" in selected.columns and not selected["Categoria_RF"].dropna().mode().empty
            else "Sin categoría"
        )
        render_kpi_card("Categoría del SKU/grupo", cat_sel, "Según reglas de elasticidad y rentabilidad")
    with card2:
        if sku != "Todos":
            best = resumen.copy() if resumen is not None else pd.DataFrame()
            if not best.empty and "SKU" in best.columns:
                best = best[best["SKU"].astype(str) == str(sku)]
                if trimestre != "Todos" and "trimestre" in best.columns:
                    best = best[best["trimestre"].astype(str) == str(trimestre)]
                best_scen = best["Escenario_Ideal"].iloc[0] if "Escenario_Ideal" in best.columns and not best.empty else "Sin dato"
            else:
                best_scen = "Sin dato"
            render_kpi_card("Mejor escenario", best_scen, "Para el SKU seleccionado")
        else:
            render_kpi_card("Mejor escenario", "Selecciona un SKU", "Disponible por SKU")

    st.subheader("KPIs proyectados")
    unidades = selected["Unidades_Simuladas"].sum() if "Unidades_Simuladas" in selected.columns else 0
    ingreso = selected["Ingreso_Simulado"].sum() if "Ingreso_Simulado" in selected.columns else 0
    margen = selected["Margen_Simulado"].sum() if "Margen_Simulado" in selected.columns else 0

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card("Unidades simuladas", format_num(unidades, 0), escenario)
    with k2:
        render_kpi_card("Ingreso predicho", format_money(ingreso), escenario)
    with k3:
        render_kpi_card("Margen predicho", format_money(margen), escenario)

    money_long, qty_long, im_long = aggregate_pricing_chart_data(selected)

    st.subheader("Ventas en dinero")
    if not money_long.empty:
        fig = px.line(
            money_long,
            x="trimestre",
            y="Ventas",
            color="Serie",
            markers=True,
            title="Ventas normales vs ventas simuladas",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos suficientes para graficar ventas en dinero.")
    st.caption(build_dynamic_explanation_pricing(selected, escenario, None if sku == "Todos" else sku))

    st.subheader("Ventas en cantidad")
    if not qty_long.empty:
        fig = px.line(
            qty_long,
            x="trimestre",
            y="Unidades",
            color="Serie",
            markers=True,
            title="Cantidad normal vs cantidad simulada",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos suficientes para graficar ventas en cantidad.")
    st.caption(build_dynamic_explanation_pricing(selected, escenario, None if sku == "Todos" else sku))

    st.subheader("Ingreso vs margen")
    if not im_long.empty:
        fig = px.bar(
            im_long,
            x="trimestre",
            y="Monto",
            color="Métrica",
            barmode="group",
            title="Ingreso simulado vs margen simulado",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos suficientes para graficar ingreso vs margen.")
    st.caption(build_dynamic_explanation_pricing(selected, escenario, None if sku == "Todos" else sku))

    st.subheader("Conclusión personalizada")
    st.info(build_dynamic_explanation_pricing(selected, escenario, None if sku == "Todos" else sku))

    with st.expander("Tabla de resultados filtrados"):
        cols = [
            "SKU",
            "trimestre",
            "Nombre_Escenario",
            "Categoria_RF",
            "dept_nm",
            "Elasticidad",
            "R2",
            "P_Value",
            "Unidades_Base",
            "Unidades_Simuladas",
            "Ingreso_Base",
            "Ingreso_Simulado",
            "Margen_Base",
            "Margen_Simulado",
            "Escenario_Ideal",
        ]
        st.dataframe(selected[[c for c in cols if c in selected.columns]], use_container_width=True)

    st.subheader("Descargas")
    st.caption(
        "Para evitar archivos corruptos o muy pesados, los archivos se preparan solo cuando presionas el botón. "
        "El archivo completo de todos los escenarios se descarga como ZIP con un CSV adentro."
    )

    download_key = st.session_state.get("pricing_cache_key")
    prepared_key = st.session_state.get("pricing_download_key")
    downloads_ready = prepared_key == download_key and st.session_state.get("pricing_full_zip_bytes") is not None

    if st.button("Preparar archivos de descarga", use_container_width=True):
        try:
            with st.spinner("Preparando archivos de descarga..."):
                exp_csv, best_csv = build_pricing_downloads(sim, resumen)

                if exp_csv is None or exp_csv.empty:
                    st.warning("No se pudo construir el archivo completo porque la tabla de simulaciones está vacía.")
                    st.session_state.pricing_download_key = None
                    st.session_state.pricing_full_zip_bytes = None
                    st.session_state.pricing_best_csv_bytes = None
                else:
                    st.session_state.pricing_full_zip_bytes = _dataframes_to_zip_csv_bytes(
                        {
                            "pricing_todos_los_escenarios.csv": exp_csv,
                        },
                        sep=",",
                    )
                    st.session_state.pricing_best_csv_bytes = _df_to_excel_friendly_csv_bytes(best_csv, sep=",")
                    st.session_state.pricing_download_rows = len(exp_csv)
                    st.session_state.pricing_best_rows = len(best_csv) if best_csv is not None else 0
                    st.session_state.pricing_download_key = download_key
                    st.success("Archivos preparados correctamente.")
        except Exception as exc:
            st.error(f"No se pudieron preparar las descargas: {exc}")
            st.session_state.pricing_download_key = None
            st.session_state.pricing_full_zip_bytes = None
            st.session_state.pricing_best_csv_bytes = None

    prepared_key = st.session_state.get("pricing_download_key")
    downloads_ready = prepared_key == download_key and st.session_state.get("pricing_full_zip_bytes") is not None

    if downloads_ready:
        rows_full = st.session_state.get("pricing_download_rows", 0)
        if rows_full > 1_048_576:
            st.warning(
                f"El archivo completo tiene {rows_full:,} filas. Excel tiene un límite aproximado de 1,048,576 filas por hoja. "
                "El CSV está completo dentro del ZIP, pero para bases muy grandes conviene abrirlo en Power BI, Python, Tableau o dividirlo por filtros."
            )

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Descargar ZIP con CSV completo de todos los escenarios",
                data=st.session_state.pricing_full_zip_bytes,
                file_name="pricing_todos_los_escenarios.zip",
                mime="application/zip",
                use_container_width=True,
            )
        with d2:
            best_bytes = st.session_state.get("pricing_best_csv_bytes") or b""
            st.download_button(
                "Descargar CSV con mejor escenario",
                data=best_bytes,
                file_name="pricing_mejor_escenario.csv",
                mime="text/csv; charset=utf-8",
                use_container_width=True,
                disabled=not bool(best_bytes),
            )
    else:
        st.info("Presiona **Preparar archivos de descarga** para generar los archivos.")


# =========================================================
# Router principal: solo una vista por rerun
# =========================================================

def main() -> None:
    """Punto de entrada de la app."""
    init_state()
    vista = render_sidebar()

    # Router explícito: solo se ejecuta una rama por rerun.
    if vista.startswith("1."):
        render_quality_view()
        return

    if vista.startswith("2."):
        render_elasticity_view()
        return

    if vista.startswith("3."):
        render_historical_pricing_view()
        return

    if vista.startswith("4."):
        render_future_pricing_view()
        return

    if vista.startswith("5."):
        render_recommendations_view()
        return

    render_exportables_view()


if __name__ == "__main__":
    main()
