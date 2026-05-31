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


def simulate_pricing_cached(
    ventas_base_elasticidad: pd.DataFrame,
    elasticidad: pd.DataFrame,
    bloques: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simula escenarios de pricing con caché.

    Se ejecuta solo desde la vista 3.
    """
    from modules.pricing import simulate_pricing_scenarios

    return simulate_pricing_scenarios(ventas_base_elasticidad, elasticidad, bloques)


@st.cache_data(show_spinner=False, max_entries=5)
def build_historical_sales_ml_cached(ventas_nse: pd.DataFrame) -> dict:
    """Entrena modelos ML ligeros para entender ventas históricas antes del pronóstico."""
    from modules.historical_ml import build_historical_sales_ml_summary

    return build_historical_sales_ml_summary(ventas_nse)


@st.cache_data(show_spinner=False, max_entries=10)
def build_elasticity_curve_data(
    curva_df: pd.DataFrame,
    min_price: float,
    max_price: float,
    max_skus: int,
) -> pd.DataFrame:
    """Construye datos para la curva de elasticidad usando caché."""
    import numpy as np

    if curva_df is None or curva_df.empty:
        return pd.DataFrame()

    precios = np.linspace(max(0.01, float(min_price)), max(0.02, float(max_price)), 60)
    curva_rows = []
    for _, row in curva_df.head(max_skus).iterrows():
        alfa = row.get("Alfa")
        beta = row.get("Elasticidad")
        sku = row.get("SKU")
        trimestre = row.get("trimestre")
        if pd.isna(alfa) or pd.isna(beta):
            continue
        for precio in precios:
            demanda = np.exp(alfa + beta * np.log(precio))
            if np.isfinite(demanda):
                curva_rows.append(
                    {
                        "SKU": sku,
                        "Precio": precio,
                        "Demanda estimada": demanda,
                        "trimestre": trimestre,
                    }
                )
    return pd.DataFrame(curva_rows)


@st.cache_data(show_spinner=False, max_entries=10)
def aggregate_weekly_demand(ventas_f: pd.DataFrame) -> pd.DataFrame:
    """Agrega demanda semanal con caché para la vista de elasticidad."""
    if ventas_f is None or ventas_f.empty:
        return pd.DataFrame()

    if "tiene_promocion" in ventas_f.columns and ventas_f["tiene_promocion"].sum() > 0:
        serie = (
            ventas_f.groupby([pd.Grouper(key="tran_date", freq="W"), "tiene_promocion"], as_index=False)
            .agg(Demanda=("qty", "sum"))
        )
        serie["Promoción"] = serie["tiene_promocion"].map({1: "Con promoción", 0: "Sin promoción"}).fillna("Sin promoción")
        return serie

    return ventas_f.groupby(pd.Grouper(key="tran_date", freq="W"), as_index=False).agg(Demanda=("qty", "sum"))


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
    return sorted(values.unique().tolist())


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


def _df_to_excel_friendly_csv_bytes(df: pd.DataFrame, sep: str = ";") -> bytes:
    """CSV compatible con Excel en configuración regional de México/España.

    Usa UTF-8 con BOM y separador `;` para que Excel abra columnas correctamente.
    """
    if df is None or df.empty:
        return b""
    clean = df.copy()
    return clean.to_csv(index=False, sep=sep, encoding="utf-8-sig", lineterminator="\n").encode("utf-8-sig")


def _dataframes_to_zip_csv_bytes(files: dict[str, pd.DataFrame], sep: str = ";") -> bytes:
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
        "manual_cache": {"quality": {}, "elasticity": {}, "pricing": {}},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_model_results() -> None:
    """Limpia resultados derivados cuando cambia la base de ventas o NSE."""
    st.session_state.elasticity_ready = False
    st.session_state.pricing_ready = False
    st.session_state.elasticidad = pd.DataFrame()
    st.session_state.elasticidades_periodo = pd.DataFrame()
    st.session_state.ventas_base_elasticidad = pd.DataFrame()
    st.session_state.bloques = []
    st.session_state.base_pricing = pd.DataFrame()
    st.session_state.simulacion = pd.DataFrame()
    st.session_state.resumen_pricing = pd.DataFrame()


def render_sidebar() -> str:
    """Renderiza sidebar. La lectura real ocurre solo con botones explícitos."""
    st.sidebar.title("📊 Pricing dinámico")
    st.sidebar.caption("Carga tus bases y navega entre vistas. Solo se ejecuta la vista activa.")

    vista = st.sidebar.radio(
        "Vista",
        [
            "1. Carga y diagnóstico de datos",
            "2. Elasticidad",
            "3. Pricing dinámico + proyección de ventas",
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
        st.session_state.manual_cache = {"quality": {}, "elasticity": {}, "pricing": {}}
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
        st.dataframe(diagnostico_calidad, use_container_width=True)

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
        return

    if "periodo_tipo" not in elasticidades_periodo.columns:
        st.warning("La tabla elasticidades_periodo no tiene la columna obligatoria periodo_tipo. Recalcula elasticidades.")
        return

    df_periodo = elasticidades_periodo.copy().replace([np.inf, -np.inf], np.nan)
    df_periodo["periodo_tipo"] = df_periodo["periodo_tipo"].astype(str)

    tipo_labels = {
        "Todos": None,
        "Mensual": "mensual",
        "Trimestral": "trimestral",
        "Semestral": "semestral",
        "Anual": "anual",
        "Global SKU": "global_sku",
        "Categoría/Departamento": "categoria_departamento",
    }

    filename_by_tipo = {
        "Todos": "elasticidades_filtradas.csv",
        "Mensual": "elasticidades_mensual.csv",
        "Trimestral": "elasticidades_trimestral.csv",
        "Semestral": "elasticidades_semestral.csv",
        "Anual": "elasticidades_anual.csv",
        "Global SKU": "elasticidades_global_sku.csv",
        "Categoría/Departamento": "elasticidades_categoria_departamento.csv",
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

    st.dataframe(resumen, use_container_width=True, hide_index=True)

    recomendables = (
        df_periodo
        .get("recomendable_elasticidad", pd.Series(False, index=df_periodo.index))
        .fillna(False)
        .astype(bool)
    )

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
            key="elasticidad_tipo_label",
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
                key="elasticidad_departamento",
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
                key="elasticidad_periodo",
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
                key="elasticidad_skus",
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

        filtered_display = build_elasticity_download(filtered)

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
                        title="Demanda semanal con/sin promoción",
                    )
                else:
                    fig = px.line(
                        serie,
                        x="tran_date",
                        y="Demanda",
                        title="Demanda semanal",
                    )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
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
            ).dropna(subset=["lat", "lon"])

            if geo.empty:
                st.warning(
                    "No se pudieron homologar los estados a coordenadas de México."
                )

            else:
                fig = px.scatter_geo(
                    geo,
                    lat="lat",
                    lon="lon",
                    color="Elasticidad absoluta",
                    size="Elasticidad absoluta",
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

    all_csv = build_elasticity_download(df_periodo)
    filtered_csv = build_elasticity_download(filtered)

    if all_csv.empty:
        st.warning("No hay elasticidades disponibles para descargar.")

    else:
        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "Descargar todas las elasticidades",
                data=convert_df_to_csv(all_csv),
                file_name="elasticidades_periodo.csv",
                mime="text/csv",
                key="download_elasticidades_todas",
            )

        with d2:
            if filtered_csv.empty:
                st.warning("No hay elasticidades filtradas disponibles para descargar.")

            else:
                st.download_button(
                    "Descargar elasticidades filtradas",
                    data=convert_df_to_csv(filtered_csv),
                    file_name=filename_by_tipo[tipo_label],
                    mime="text/csv",
                    key="download_elasticidades_filtradas",
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
                        sep=";",
                    )
                    st.session_state.pricing_best_csv_bytes = _df_to_excel_friendly_csv_bytes(best_csv, sep=";")
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

    render_pricing_view()


if __name__ == "__main__":
    main()
