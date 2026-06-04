"""Funciones utilitarias de lectura, validación, limpieza y UI."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
import unicodedata
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from .config import (
    DEFAULT_NSE_DIR, DEFAULT_NSE_FILENAME, NSE_CATEGORIAS_VALIDAS, STATE_COORDINATES,
    INEGI_DIR, INEGI_GEO_CATALOG_FILENAME, INEGI_HOGARES_FILENAME,
)


def normalize_text(value) -> str:
    """Normaliza texto para comparaciones robustas."""
    if pd.isna(value):
        return ""
    txt = str(value).strip().lower()
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("utf-8")
    txt = re.sub(r"[_\-]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt




# Alias de columnas frecuentes para estabilizar la carga inicial sin exigir un
# layout único al usuario. Los nombres canónicos conservan compatibilidad con
# elasticidad/pricing, pero la Fase 1 también crea aliases de negocio legibles.
COLUMN_ALIASES = {
    "tran_date": ["tran_date", "fecha", "date", "transaction_date", "fecha_transaccion", "fecha_venta"],
    "qty": ["qty", "unidades", "cantidad", "quantity", "unit_sales", "units"],
    "net_sale": ["net_sale", "ingreso", "venta", "venta_neta", "net_sales", "sales", "importe"],
    "prod_nbr": ["prod_nbr", "sku", "SKU", "producto", "product_id", "item_id", "prod_id"],
    "costo2": ["costo2", "costo", "cost", "costo_unitario", "unit_cost"],
    "precio_base": ["precio_base", "precio", "price", "precio_unitario", "unit_price"],
    "store_nm": ["store_nm", "tienda", "store", "sucursal", "nombre_tienda"],
    "dept_nm": ["dept_nm", "departamento", "department", "depto"],
    "subdept_nm": ["subdept_nm", "categoria", "categoría", "category", "subcategoria", "sub_category"],
    "estado": ["estado", "state", "entidad"],
    "municipio": ["municipio", "city", "ciudad", "localidad"],
    "categoria_est_socio": ["categoria_est_socio", "nivel_socioeconomico", "nivel socioeconómico", "nse", "categoria_nse"],
}


def _canonical_column_token(value: str) -> str:
    """Convierte nombres de columnas a tokens comparables sin perder canónicos."""
    txt = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("utf-8")
    txt = txt.strip().lower()
    txt = re.sub(r"[^0-9a-zA-Z]+", "_", txt)
    return re.sub(r"_+", "_", txt).strip("_")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza encabezados y renombra aliases conocidos a columnas canónicas."""
    if df is None or df.empty:
        return df

    out = df.copy()
    normalized_originals = {}
    renamed = []
    for col in out.columns:
        base = re.sub(r"\s+", " ", str(col)).strip()
        if base not in normalized_originals:
            normalized_originals[base] = col
        renamed.append(base)
    out.columns = renamed

    token_to_col = {_canonical_column_token(c): c for c in out.columns}
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            found = token_to_col.get(_canonical_column_token(alias))
            if found is not None and found not in rename_map:
                rename_map[found] = canonical
                break

    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Quita espacios dobles y espacios finales/iniciales de todas las columnas de texto."""
    if df is None or df.empty:
        return df
    out = df.copy()
    text_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    priority_cols = [c for c in ["store_nm", "tienda"] if c in out.columns]
    for col in dict.fromkeys(priority_cols + text_cols):
        out[col] = out[col].where(
            out[col].isna(),
            out[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip(),
        )
    return out


def add_business_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva columnas clave canónicas y agrega aliases de negocio para diagnóstico."""
    out = df.copy()
    alias_sources = {
        "SKU": "prod_nbr",
        "fecha": "tran_date",
        "precio": "precio_unitario",
        "unidades": "qty",
        "ingreso": "net_sale",
        "costo": "costo_unitario",
        "margen": "margen_total",
        "categoría": "subdept_nm",
        "departamento": "dept_nm",
        "tienda": "store_nm",
        "nivel socioeconómico": "categoria_est_socio",
    }
    for alias, source in alias_sources.items():
        if alias not in out.columns and source in out.columns:
            out[alias] = out[source]
    return out


def add_period_variables(df: pd.DataFrame, date_col: str = "tran_date") -> pd.DataFrame:
    """Crea mes/año/trimestre/semestre y periodos en varios niveles."""
    out = df.copy()
    if date_col not in out.columns:
        return out
    fechas = parse_transaction_dates(out[date_col])
    out[date_col] = fechas
    out["mes"] = fechas.dt.month.astype("Int64")
    out["año"] = fechas.dt.year.astype("Int64")
    out["trimestre"] = fechas.dt.quarter.astype("Int64")
    out["semestre"] = np.where(fechas.dt.month.le(6), 1, 2)
    out["semestre"] = pd.Series(out["semestre"], index=out.index).where(fechas.notna()).astype("Int64")
    mensual = fechas.dt.to_period("M")
    trimestral = fechas.dt.to_period("Q")
    anual = fechas.dt.to_period("Y")
    out["periodo_mensual"] = mensual.astype("string")
    out["periodo_trimestral"] = trimestral.astype("string")
    out["periodo_semestral"] = np.where(
        fechas.notna(),
        fechas.dt.year.astype("Int64").astype("string") + "-S" + out["semestre"].astype("string"),
        pd.NA,
    )
    out["periodo_anual"] = anual.astype("string")
    return out

def _normalizar_lista_columnas(usecols: Optional[Sequence[str]]) -> tuple[str, ...] | None:
    """Normaliza usecols para poder cachear lecturas con columnas seleccionadas."""
    if not usecols:
        return None
    return tuple(dict.fromkeys(str(c).strip() for c in usecols if str(c).strip()))


def _usecols_callable(columnas_permitidas: tuple[str, ...] | None):
    """Crea un filtro flexible de columnas para pandas."""
    if not columnas_permitidas:
        return None
    permitidas = {str(c).strip() for c in columnas_permitidas}
    permitidas_tokens = {_canonical_column_token(c) for c in permitidas}
    return lambda col: str(col).strip() in permitidas or _canonical_column_token(col) in permitidas_tokens


def _dtype_map_for_columns(columnas: tuple[str, ...] | None) -> dict:
    """Tipos ligeros para columnas categóricas; las numéricas se limpian después."""
    if not columnas:
        columnas = tuple()
    text_cols = {
        "prod_nbr", "SKU", "store_nm", "dept_nm", "subdept_nm", "marca", "tipo_marca",
        "categoria_est_socio", "estado", "key", "id_municipio", "ubica_geo", "municipio",
        "est_socio", "sku", "promo_id", "id_promocion", "mecanica",
        "tienda", "departamento", "categoria", "categoría", "municipio",
        "nivel_socioeconomico", "nivel socioeconómico",
    }
    return {c: "string" for c in columnas if c in text_cols}


def _fast_bytes_hash(data: bytes) -> str:
    """Hash rápido para caché de archivos grandes.

    Evita que Streamlit tarde demasiado hasheando bases completas en cada rerun.
    Usa tamaño + inicio + final del archivo. Para trabajo académico/prototipo es
    suficientemente robusto y acelera mucho la lectura en deploy/local.
    """
    if data is None:
        return "none"
    size = len(data)
    chunk = 1024 * 1024
    h = hashlib.blake2b(digest_size=16)
    h.update(str(size).encode("utf-8"))
    h.update(data[:chunk])
    if size > chunk:
        h.update(data[-chunk:])
    return h.hexdigest()


def get_uploaded_file_signature(uploaded_file) -> str:
    """Firma liviana para detectar si cambió un archivo subido."""
    if uploaded_file is None:
        return "sin_archivo"
    file_bytes = uploaded_file.getvalue()
    return f"{uploaded_file.name}|{len(file_bytes)}|{_fast_bytes_hash(file_bytes)}"


def _detect_available_columns_csv(file_bytes: bytes, encoding: str) -> list[str]:
    """Lee solo encabezados de CSV para elegir usecols existentes."""
    try:
        header = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, nrows=0)
        return header.columns.astype(str).str.strip().tolist()
    except Exception:
        return []


def _select_existing_usecols(file_bytes: bytes, encoding: str, usecols_tuple: tuple[str, ...] | None) -> list[str] | None:
    """Selecciona solo columnas requeridas que realmente existan en el CSV."""
    if not usecols_tuple:
        return None
    available = _detect_available_columns_csv(file_bytes, encoding)
    if not available:
        return list(usecols_tuple)
    requested = {str(c).strip() for c in usecols_tuple}
    requested_tokens = {_canonical_column_token(c) for c in requested}
    selected = [c for c in available if c in requested or _canonical_column_token(c) in requested_tokens]
    # Si no hay coincidencias exactas/flexibles, no forzamos usecols para evitar
    # errores de pandas y permitir que la normalización posterior diagnostique.
    return selected if selected else None



@st.cache_data(show_spinner=False, max_entries=4, hash_funcs={bytes: _fast_bytes_hash})
def _read_uploaded_file_cached(
    file_name: str,
    file_size: int,
    file_bytes: bytes,
    usecols_tuple: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Lee archivos desde bytes con caché.

    Optimización importante:
    - Si usecols_tuple viene definido, pandas solo carga esas columnas.
    - Esto conserva la calidad del modelo porque las columnas seleccionadas son las que usa la app.
    - Parquet se lee mucho más rápido que CSV/Excel cuando está disponible.
    """
    del file_size  # ayuda a invalidar cache si cambia el archivo, aunque no se usa directamente.
    name = file_name.lower()
    usecols = _usecols_callable(usecols_tuple)
    dtype_map = _dtype_map_for_columns(usecols_tuple)

    if name.endswith(".parquet"):
        if not usecols_tuple:
            return pd.read_parquet(io.BytesIO(file_bytes))
        try:
            return pd.read_parquet(io.BytesIO(file_bytes), columns=list(usecols_tuple))
        except Exception:
            df = pd.read_parquet(io.BytesIO(file_bytes))
            selector = _usecols_callable(usecols_tuple)
            selected = [col for col in df.columns if selector(col)] if selector else df.columns.tolist()
            return df[selected] if selected else df

    if name.endswith(".csv"):
        # Ruta rápida: intenta engine=pyarrow con columnas existentes.
        # Si falla por encoding/separador, cae al engine C de pandas.
        # Se prueban varios encodings en orden; UnicodeDecodeError (y cualquier
        # excepción de decodificación) simplemente pasa al siguiente encoding.
        # NOTA: UnicodeDecodeError es subclase de ValueError, por lo que se
        # captura de forma genérica para no re-lanzar antes de agotar encodings.
        last_error = None
        for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
            selected_usecols = _select_existing_usecols(file_bytes, encoding, usecols_tuple)
            # Intento 1: pyarrow (rápido pero solo soporta UTF-8 realmente)
            try:
                return pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding,
                    usecols=selected_usecols,
                    engine="pyarrow",
                )
            except Exception as exc:
                last_error = exc

            # Intento 2: engine C de pandas (robusto con todos los encodings)
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding,
                    usecols=_usecols_callable(tuple(selected_usecols)) if selected_usecols else None,
                    dtype=dtype_map if dtype_map else None,
                    low_memory=False,
                )
                # Si parece que el CSV venía separado por ;, reintenta con ;.
                if df.shape[1] == 1 and ";" in str(df.columns[0]):
                    df = pd.read_csv(
                        io.BytesIO(file_bytes),
                        encoding=encoding,
                        sep=";",
                        usecols=_usecols_callable(usecols_tuple),
                        dtype=dtype_map if dtype_map else None,
                        low_memory=False,
                    )
                return df
            except (UnicodeDecodeError, UnicodeError):
                # Error de encoding: prueba el siguiente
                last_error = exc
                continue
            except ValueError as exc2:
                # ValueError que NO sea de encoding (e.g. columnas inválidas):
                # si aún quedan encodings por probar, continúa; de lo contrario lanza.
                last_error = exc2
                continue

        # Último recurso: latin1 con errors='replace' para no bloquear la carga
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="latin1",
                encoding_errors="replace",
                usecols=_usecols_callable(usecols_tuple),
                dtype=dtype_map if dtype_map else None,
                low_memory=False,
            )
            return df
        except Exception as exc:
            last_error = exc

        raise ValueError(f"No se pudo leer el CSV. Último detalle: {last_error}")

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(
            io.BytesIO(file_bytes),
            usecols=usecols,
            dtype=dtype_map if dtype_map else None,
        )

    raise ValueError("Formato no compatible. Sube un archivo CSV, XLSX, XLS o PARQUET.")


def read_uploaded_file(uploaded_file, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Lee CSV, Excel o Parquet cargado desde Streamlit.

    Nota de rendimiento:
    Esta función debe llamarse solo al presionar el botón de procesar, no en cada rerun.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    try:
        file_bytes = uploaded_file.getvalue()
        usecols_tuple = _normalizar_lista_columnas(usecols)
        return _read_uploaded_file_cached(
            uploaded_file.name,
            getattr(uploaded_file, "size", len(file_bytes)),
            file_bytes,
            usecols_tuple,
        )
    except Exception as exc:
        raise ValueError(f"No se pudo leer el archivo '{uploaded_file.name}': {exc}") from exc


def get_uploaded_file_info(uploaded_file) -> str:
    """Devuelve nombre y tamaño sin leer todo el archivo como DataFrame."""
    if uploaded_file is None:
        return "Sin archivo"
    size = getattr(uploaded_file, "size", None)
    if size is None:
        return uploaded_file.name
    mb = size / (1024 ** 2)
    return f"{uploaded_file.name} ({mb:.1f} MB)"

def validate_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> list[str]:
    """Regresa las columnas faltantes."""
    if df is None or df.empty:
        return list(required_columns)
    disponibles = set(df.columns.astype(str).str.strip())
    return [col for col in required_columns if col not in disponibles]


def clean_store_names(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia espacios dobles, iniciales y finales de store_nm y textos equivalentes."""
    return clean_text_columns(df)


def parse_transaction_dates(series: pd.Series) -> pd.Series:
    """Convierte fechas de transacción de forma explícita y rápida.

    Prioriza el formato mexicano dd/mm/YYYY para eliminar el warning de pandas y
    evitar interpretaciones ambiguas. Incluye fallbacks para fecha con hora,
    formato ISO y formatos mixtos con dayfirst=True.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    fechas = series.astype("string").str.strip()

    # Formatos frecuentes en bases mexicanas. Primero se prueban rutas rápidas.
    parsed = pd.to_datetime(fechas, format="%d/%m/%Y", errors="coerce", cache=True)

    pendientes = parsed.isna() & fechas.notna() & (fechas != "")
    if pendientes.any():
        parsed_hora = pd.to_datetime(
            fechas.loc[pendientes],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
            cache=True,
        )
        parsed.loc[pendientes] = parsed_hora

    pendientes = parsed.isna() & fechas.notna() & (fechas != "")
    if pendientes.any():
        parsed_iso = pd.to_datetime(
            fechas.loc[pendientes],
            format="%Y-%m-%d",
            errors="coerce",
            cache=True,
        )
        parsed.loc[pendientes] = parsed_iso

    pendientes = parsed.isna() & fechas.notna() & (fechas != "")
    if pendientes.any():
        parsed.loc[pendientes] = pd.to_datetime(
            fechas.loc[pendientes],
            errors="coerce",
            dayfirst=True,
            cache=True,
        )

    return parsed


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convierte texto monetario o numérico a número."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def clean_sales_data(
    sales_df: pd.DataFrame,
    costo2_es_unitario: bool = True,
    eliminar_costo_mayor_o_igual_precio: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Limpia la base de ventas siguiendo la lógica del notebook:
    duplicados, fechas, numéricos, qty/net_sale válidos, precio/costo/margen.
    """
    if sales_df is None or sales_df.empty:
        raise ValueError("La base de ventas está vacía.")

    ventas = normalize_column_names(sales_df)
    ventas = clean_text_columns(ventas)
    filas_originales = len(ventas)
    resumen = []

    def registrar(paso: str, antes: int, despues: int) -> None:
        resumen.append(
            {
                "Paso": paso,
                "Filas_Antes": int(antes),
                "Filas_Despues": int(despues),
                "Registros_Removidos": int(antes - despues),
                "%_Removido": float(((antes - despues) / antes * 100) if antes > 0 else 0),
            }
        )

    ventas = clean_store_names(ventas)

    antes = len(ventas)
    ventas = ventas.drop_duplicates().copy()
    registrar("Eliminar registros duplicados", antes, len(ventas))

    ventas["tran_date"] = parse_transaction_dates(ventas["tran_date"])

    for col in ["qty", "net_sale", "costo2"]:
        ventas[col] = _safe_numeric(ventas[col])
    ventas["prod_nbr"] = ventas["prod_nbr"].astype("string").str.strip()

    antes = len(ventas)
    ventas = ventas.dropna(subset=["tran_date", "qty", "net_sale", "prod_nbr", "costo2"]).copy()
    ventas = ventas[ventas["prod_nbr"].astype("string").str.len() > 0].copy()
    registrar("Eliminar nulos en columnas críticas", antes, len(ventas))

    antes = len(ventas)
    ventas = ventas[(ventas["qty"] > 0) & (ventas["net_sale"] > 0)].copy()
    registrar("Eliminar qty <= 0 y net_sale <= 0", antes, len(ventas))

    ventas["prod_nbr"] = ventas["prod_nbr"].astype("string").str.strip().astype(str)
    ventas["precio_unitario"] = ventas["net_sale"] / ventas["qty"]

    antes = len(ventas)
    ventas = ventas.replace([np.inf, -np.inf], np.nan)
    ventas = ventas.dropna(subset=["precio_unitario"]).copy()
    ventas = ventas[ventas["precio_unitario"] > 0].copy()
    registrar("Eliminar precio <= 0 o no calculable", antes, len(ventas))

    if costo2_es_unitario:
        ventas["costo_unitario"] = ventas["costo2"]
    else:
        ventas["costo_unitario"] = ventas["costo2"] / ventas["qty"]

    ventas = ventas.replace([np.inf, -np.inf], np.nan)

    antes = len(ventas)
    ventas = ventas.dropna(subset=["costo_unitario"]).copy()
    ventas = ventas[ventas["costo_unitario"] >= 0].copy()
    registrar("Eliminar costo negativo o no calculable", antes, len(ventas))

    ventas["Costo_Mayor_O_Igual_Precio_Linea"] = ventas["costo_unitario"] >= ventas["precio_unitario"]
    filas_costo_invalido = int(ventas["Costo_Mayor_O_Igual_Precio_Linea"].sum())

    if filas_costo_invalido > 0 and eliminar_costo_mayor_o_igual_precio:
        antes = len(ventas)
        ventas = ventas[~ventas["Costo_Mayor_O_Igual_Precio_Linea"]].copy()
        registrar("Eliminar costo unitario >= precio", antes, len(ventas))
    else:
        registrar("Verificar costo unitario < precio", len(ventas), len(ventas))

    ventas["precio_base"] = pd.to_numeric(
        ventas["precio_base"], errors="coerce"
    ) if "precio_base" in ventas.columns else ventas["precio_unitario"]
    ventas["precio_base"] = ventas["precio_base"].fillna(ventas["precio_unitario"])

    ventas["ingreso_base"] = pd.to_numeric(
        ventas["ingreso_base"], errors="coerce"
    ) if "ingreso_base" in ventas.columns else ventas["precio_base"] * ventas["qty"]
    ventas["ingreso_base"] = ventas["ingreso_base"].fillna(ventas["precio_base"] * ventas["qty"])

    ventas["margen_unitario"] = pd.to_numeric(
        ventas["margen_unitario"], errors="coerce"
    ) if "margen_unitario" in ventas.columns else ventas["precio_base"] - ventas["costo_unitario"]
    ventas["margen_unitario"] = ventas["margen_unitario"].fillna(
        ventas["precio_base"] - ventas["costo_unitario"]
    )

    ventas["margen_total"] = pd.to_numeric(
        ventas["margen_total"], errors="coerce"
    ) if "margen_total" in ventas.columns else ventas["margen_unitario"] * ventas["qty"]
    ventas["margen_total"] = ventas["margen_total"].fillna(ventas["margen_unitario"] * ventas["qty"])

    ventas = add_period_variables(ventas, "tran_date")
    ventas = add_business_alias_columns(ventas)

    summary = {
        "filas_originales": int(filas_originales),
        "filas_limpias": int(len(ventas)),
        "registros_removidos": int(filas_originales - len(ventas)),
        "porcentaje_removido": float(((filas_originales - len(ventas)) / filas_originales) if filas_originales else 1),
        "duplicados_originales": int(sales_df.duplicated().sum()),
        "duplicados_eliminados": int(resumen[0]["Registros_Removidos"] if resumen else 0),
        "nulos_por_columna_original": sales_df.isna().sum().astype(int).to_dict(),
        "nulos_por_columna_final": ventas.isna().sum().astype(int).to_dict(),
        "columnas_finales": ventas.columns.astype(str).tolist(),
        "faltantes_pct_original": float(sales_df.isna().mean().mean() * 100),
        "infinitos_detectados_original": int(
            np.isinf(sales_df.select_dtypes(include=[np.number]).to_numpy()).sum()
        )
        if len(sales_df.select_dtypes(include=[np.number]).columns) > 0
        else 0,
        "registros_precio_invalido": int((ventas["precio_unitario"] <= 0).sum()) if "precio_unitario" in ventas else 0,
        "registros_cantidad_invalida": int((ventas["qty"] <= 0).sum()) if "qty" in ventas else 0,
        "registros_costo_mayor_o_igual_precio": filas_costo_invalido,
    }

    return ventas.reset_index(drop=True), pd.DataFrame(resumen), summary


def normalizar_categoria_est_socio(valor):
    """Normaliza est_socio o categoria_est_socio a bajo, medio bajo, medio alto, alto."""
    if pd.isna(valor):
        return np.nan

    try:
        numero = float(str(valor).strip().replace(",", "."))
        if np.isfinite(numero):
            codigo = str(int(round(numero)))
            mapa_num = {"1": "bajo", "2": "medio bajo", "3": "medio alto", "4": "alto"}
            if codigo in mapa_num:
                return mapa_num[codigo]
    except Exception:
        pass

    txt = normalize_text(valor)
    mapa_txt = {
        "bajo": "bajo",
        "medio bajo": "medio bajo",
        "medio alto": "medio alto",
        "alto": "alto",
    }
    return mapa_txt.get(txt, np.nan)


def _limpiar_id_municipio(valor):
    """Homologa llaves geográficas numéricas."""
    if pd.isna(valor):
        return np.nan
    txt = str(valor).strip().replace(",", "")
    txt = re.sub(r"\.0$", "", txt)
    return txt


def apply_geo_catalog(ventas: pd.DataFrame, geo_catalog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 1 del cruce de dos bases: key → id_municipio.

    Replica el bloque 2B del notebook original que usaba df_inegi_id /
    catalogo_ubica_geo para generar la columna id_municipio a partir de
    la llave de texto 'key' presente en ventas.

    El catálogo debe tener al menos las columnas 'key' y 'ubica_geo'.
    La columna resultante en ventas se llama 'id_municipio'.
    Si ventas ya tiene 'id_municipio', se sobreescribe con los valores del catálogo
    y se conservan los existentes donde el catálogo no aporte dato.
    """
    if geo_catalog_df is None or geo_catalog_df.empty:
        return ventas

    cat = clean_text_columns(normalize_column_names(geo_catalog_df.copy()))
    cat.columns = cat.columns.astype(str).str.strip()

    if "key" not in cat.columns or "ubica_geo" not in cat.columns:
        return ventas

    cat["key"] = cat["key"].astype(str).str.strip().str.lower()
    cat_reducido = cat[["key", "ubica_geo"]].drop_duplicates("key").copy()
    cat_reducido = cat_reducido.rename(columns={"ubica_geo": "id_municipio_cat"})
    cat_reducido["id_municipio_cat"] = cat_reducido["id_municipio_cat"].apply(_limpiar_id_municipio)

    if "key" not in ventas.columns:
        return ventas

    ventas = ventas.copy()
    ventas["key"] = ventas["key"].astype(str).str.strip().str.lower()

    # Eliminamos columna previa de id_municipio para regenerarla limpia
    ventas = ventas.drop(columns=["id_municipio"], errors="ignore")

    ventas = ventas.merge(cat_reducido, on="key", how="left")
    ventas = ventas.rename(columns={"id_municipio_cat": "id_municipio"})
    return ventas


def _default_nse_hardcoded() -> pd.DataFrame:
    """Base NSE default embebida como respaldo si falta el CSV versionado."""
    rows = [
        ("Aguascalientes-Aguascalientes", 1001, "Aguascalientes", "Aguascalientes", 3),
        ("Mexicali-Baja California", 2002, "Baja California", "Mexicali", 3),
        ("La Paz-Baja California Sur", 3003, "Baja California Sur", "La Paz", 3),
        ("Campeche-Campeche", 4002, "Campeche", "Campeche", 2),
        ("Tuxtla Gutierrez-Chiapas", 7101, "Chiapas", "Tuxtla Gutierrez", 2),
        ("Chihuahua-Chihuahua", 8019, "Chihuahua", "Chihuahua", 3),
        ("Saltillo-Coahuila", 5030, "Coahuila", "Saltillo", 3),
        ("Colima-Colima", 6002, "Colima", "Colima", 3),
        ("Cuauhtemoc-Ciudad de Mexico", 9015, "Ciudad de Mexico", "Cuauhtemoc", 4),
        ("Durango-Durango", 10005, "Durango", "Durango", 3),
        ("Leon-Guanajuato", 11020, "Guanajuato", "Leon", 3),
        ("Acapulco de Juarez-Guerrero", 12001, "Guerrero", "Acapulco de Juarez", 2),
        ("Pachuca de Soto-Hidalgo", 13048, "Hidalgo", "Pachuca de Soto", 3),
        ("Guadalajara-Jalisco", 14039, "Jalisco", "Guadalajara", 4),
        ("Toluca-Mexico", 15106, "Estado de Mexico", "Toluca", 3),
        ("Morelia-Michoacan", 16053, "Michoacan", "Morelia", 3),
        ("Cuernavaca-Morelos", 17007, "Morelos", "Cuernavaca", 3),
        ("Tepic-Nayarit", 18017, "Nayarit", "Tepic", 3),
        ("Monterrey-Nuevo Leon", 19039, "Nuevo Leon", "Monterrey", 4),
        ("Oaxaca de Juarez-Oaxaca", 20067, "Oaxaca", "Oaxaca de Juarez", 2),
        ("Puebla-Puebla", 21114, "Puebla", "Puebla", 3),
        ("Queretaro-Queretaro", 22014, "Queretaro", "Queretaro", 4),
        ("Benito Juarez-Quintana Roo", 23005, "Quintana Roo", "Benito Juarez", 3),
        ("San Luis Potosi-San Luis Potosi", 24028, "San Luis Potosi", "San Luis Potosi", 3),
        ("Culiacan-Sinaloa", 25006, "Sinaloa", "Culiacan", 3),
        ("Hermosillo-Sonora", 26030, "Sonora", "Hermosillo", 3),
        ("Centro-Tabasco", 27004, "Tabasco", "Centro", 3),
        ("Tampico-Tamaulipas", 28038, "Tamaulipas", "Tampico", 3),
        ("Tlaxcala-Tlaxcala", 29033, "Tlaxcala", "Tlaxcala", 3),
        ("Veracruz-Veracruz", 30193, "Veracruz", "Veracruz", 3),
        ("Merida-Yucatan", 31050, "Yucatan", "Merida", 4),
        ("Zacatecas-Zacatecas", 32056, "Zacatecas", "Zacatecas", 3),
    ]
    df = pd.DataFrame(rows, columns=["key", "ubica_geo", "estado", "municipio", "est_socio"])
    df["categoria_est_socio"] = df["est_socio"].apply(normalizar_categoria_est_socio)
    return df


@st.cache_data(show_spinner=False)
def build_default_nse() -> pd.DataFrame:
    """
    Carga la base NSE default precargada.

    Prioridad:
    1. data/inegi/hogares_INEGI.csv  — base INEGI de hogares precargada
    2. data/default_nse/base_nse_default.csv — base simplificada legacy
    3. Respaldo embebido de 32 municipios (hardcoded)
    """
    for path in [
        Path(INEGI_DIR) / INEGI_HOGARES_FILENAME,
        Path(DEFAULT_NSE_DIR) / DEFAULT_NSE_FILENAME,
    ]:
        try:
            if path.exists():
                df = pd.read_csv(path, encoding="latin1", low_memory=False)
                return clean_text_columns(normalize_column_names(df))
        except Exception:
            pass
    return _default_nse_hardcoded()


@st.cache_data(show_spinner=False)
def build_default_geo_catalog() -> pd.DataFrame:
    """
    Carga el catálogo geográfico precargado (data/inegi/catalogo_ubica_geo.csv).

    Contiene la relación key → ubica_geo para todos los municipios de México.
    Es el equivalente de df_inegi_id del flujo original del notebook.
    Retorna DataFrame vacío si el archivo no existe.
    """
    path = Path(INEGI_DIR) / INEGI_GEO_CATALOG_FILENAME
    try:
        if path.exists():
            df = pd.read_csv(path, encoding="latin1", low_memory=False)
            return clean_text_columns(normalize_column_names(df))
    except Exception:
        pass
    return pd.DataFrame()


def get_default_nse_path() -> str:
    """Ruta relativa de la base NSE default usada por la app."""
    inegi_path = Path(INEGI_DIR) / INEGI_HOGARES_FILENAME
    if inegi_path.exists():
        return str(inegi_path)
    return str(Path(DEFAULT_NSE_DIR) / DEFAULT_NSE_FILENAME)


def _nse_value_column(df: pd.DataFrame) -> str | None:
    for col in ["categoria_est_socio", "est_socio", "nivel_socioeconomico", "nse", "categoria_nse"]:
        if col in df.columns:
            return col
    return None


def _nse_join_columns(nse: pd.DataFrame, sales: pd.DataFrame) -> list[str]:
    joins: list[str] = []
    if "key" in nse.columns and "key" in sales.columns:
        joins.append("key")
    nse_has_municipio = "id_municipio" in nse.columns or "ubica_geo" in nse.columns
    if nse_has_municipio and "id_municipio" in sales.columns:
        joins.append("id_municipio")
    return joins


def validate_custom_nse(nse_df: pd.DataFrame | None, sales_df: pd.DataFrame | None) -> tuple[bool, list[str], dict]:
    """Valida una base NSE personalizada antes de permitir su uso en el cruce."""
    warnings: list[str] = []
    info: dict = {
        "estado_validacion_nse": "pendiente",
        "join_columns": [],
        "advertencias_nse": warnings,
    }

    if nse_df is None or nse_df.empty:
        warnings.append("La base NSE personalizada está vacía.")
        info["estado_validacion_nse"] = "fallida_base_vacia"
        return False, warnings, info

    nse = clean_text_columns(normalize_column_names(nse_df))
    sales = clean_text_columns(normalize_column_names(sales_df)) if sales_df is not None else pd.DataFrame()
    nse.columns = nse.columns.astype(str).str.strip()
    sales.columns = sales.columns.astype(str).str.strip()

    value_col = _nse_value_column(nse)
    if value_col is None:
        warnings.append("La base NSE debe contener una columna NSE: categoria_est_socio, est_socio, nivel_socioeconomico, nse o categoria_nse.")
        info["estado_validacion_nse"] = "fallida_columnas_nse"
        return False, warnings, info

    if "ubica_geo" in nse.columns and "id_municipio" not in nse.columns:
        nse["id_municipio"] = nse["ubica_geo"]
    if "ubica_geo" in sales.columns and "id_municipio" not in sales.columns:
        sales["id_municipio"] = sales["ubica_geo"]

    join_cols = _nse_join_columns(nse, sales)
    info["join_columns"] = join_cols
    if not join_cols:
        warnings.append("No hay claves de cruce compatibles entre ventas y NSE personalizada. Usa key o id_municipio/ubica_geo en ambas bases.")
        info["estado_validacion_nse"] = "fallida_claves_incompatibles"
        return False, warnings, info

    normalized_nse = nse[value_col].apply(normalizar_categoria_est_socio)
    invalid_mask = normalized_nse.isna() & nse[value_col].notna()
    if invalid_mask.any():
        warnings.append(
            "La base NSE contiene valores NSE no válidos. Valores aceptados: "
            + ", ".join(NSE_CATEGORIAS_VALIDAS)
            + " o códigos 1-4."
        )
        info["estado_validacion_nse"] = "fallida_valores_nse_invalidos"
        return False, warnings, info

    if normalized_nse.isna().any():
        warnings.append("La base NSE contiene valores nulos en la columna NSE usada para asignación.")
        info["estado_validacion_nse"] = "fallida_nulos_nse"
        return False, warnings, info

    for join_col in join_cols:
        if join_col not in nse.columns:
            continue
        key_series = nse[join_col].apply(_limpiar_id_municipio) if join_col == "id_municipio" else nse[join_col].astype("string").str.strip().str.lower()
        if key_series.isna().any() or key_series.eq("").any():
            warnings.append(f"La base NSE contiene valores nulos o vacíos en la clave de cruce `{join_col}`.")
            info["estado_validacion_nse"] = "fallida_nulos_clave"
            return False, warnings, info

        check = pd.DataFrame({join_col: key_series, "categoria_est_socio": normalized_nse})
        conflictivos = check.groupby(join_col)["categoria_est_socio"].nunique(dropna=True)
        conflictivos = conflictivos[conflictivos > 1]
        if not conflictivos.empty:
            warnings.append(f"La clave `{join_col}` tiene duplicados con NSE distintos; corrige {len(conflictivos)} clave(s) conflictiva(s).")
            info["estado_validacion_nse"] = "fallida_duplicados_conflictivos"
            return False, warnings, info

        duplicados = int(key_series.duplicated().sum())
        if duplicados > 0:
            warnings.append(f"La clave `{join_col}` contiene {duplicados:,} duplicados no conflictivos; se usará una fila por clave.")

        sales_key = sales[join_col].apply(_limpiar_id_municipio) if join_col == "id_municipio" else sales[join_col].astype("string").str.strip().str.lower()
        overlap = set(sales_key.dropna().astype(str)) & set(key_series.dropna().astype(str))
        if len(overlap) == 0:
            warnings.append(f"La clave `{join_col}` no comparte valores entre ventas y NSE personalizada.")
            info["estado_validacion_nse"] = "fallida_sin_compatibilidad_ventas"
            return False, warnings, info

    info["estado_validacion_nse"] = "valida"
    return True, warnings, info


def merge_sales_with_nse(
    sales_df: pd.DataFrame,
    nse_df: Optional[pd.DataFrame],
    fuente_nse: str = "default",
    estado_validacion_nse: str = "default_precargada",
    advertencias_nse: Optional[list[str]] = None,
    geo_catalog_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Cruza ventas con NSE de forma flexible.
    Prioriza categoria_est_socio y usa est_socio solo para construirla.

    Si se proporciona geo_catalog_df (catálogo geográfico con columnas 'key' y
    'ubica_geo', equivalente a df_inegi_id del notebook), se ejecuta primero el
    paso key → id_municipio antes del cruce NSE, replicando el flujo original de
    dos bases del bloque 2B.
    """
    ventas = clean_text_columns(normalize_column_names(sales_df))

    # Paso 1 (opcional): key → id_municipio usando el catálogo geográfico.
    if geo_catalog_df is not None and not geo_catalog_df.empty:
        ventas = apply_geo_catalog(ventas, geo_catalog_df)
    advertencias_nse = advertencias_nse or []
    info = {
        "nse_usado": False,
        "fuente_nse_usada": fuente_nse,
        "estado_validacion_nse": estado_validacion_nse,
        "registros_con_nse": 0,
        "registros_sin_nse": len(ventas),
        "porcentaje_asignado": 0.0,
        "porcentaje_match_nse": 0.0,
        "registros_sin_match_nse": len(ventas),
        "advertencias_nse": advertencias_nse,
        "mensaje": "No se aplicó cruce NSE.",
    }

    if "categoria_est_socio" in ventas.columns:
        ventas["categoria_est_socio"] = ventas["categoria_est_socio"].apply(normalizar_categoria_est_socio)

    if nse_df is None or nse_df.empty:
        if "categoria_est_socio" not in ventas.columns:
            ventas["categoria_est_socio"] = np.nan
        ventas["fuente_nse"] = fuente_nse
        ventas["nse_asignado"] = ventas["categoria_est_socio"].fillna("NSE_no_asignado")
        ventas["categoria_est_socio"] = ventas["nse_asignado"]
        ventas["nse_match_status"] = "sin_match"
        ventas = add_business_alias_columns(ventas)
        return ventas, info

    nse = clean_text_columns(normalize_column_names(nse_df))
    ventas.columns = ventas.columns.astype(str).str.strip()
    nse.columns = nse.columns.astype(str).str.strip()

    # Construir categoria_est_socio en NSE si no viene explícita.
    if "categoria_est_socio" in nse.columns:
        nse["categoria_est_socio"] = nse["categoria_est_socio"].apply(normalizar_categoria_est_socio)
    elif "est_socio" in nse.columns:
        nse["categoria_est_socio"] = nse["est_socio"].apply(normalizar_categoria_est_socio)
    else:
        if "categoria_est_socio" not in ventas.columns:
            ventas["categoria_est_socio"] = np.nan
        info["mensaje"] = "La base NSE no contiene categoria_est_socio ni est_socio."
        ventas["fuente_nse"] = fuente_nse
        ventas["nse_asignado"] = ventas["categoria_est_socio"].fillna("NSE_no_asignado")
        ventas["categoria_est_socio"] = ventas["nse_asignado"]
        ventas["nse_match_status"] = "sin_match"
        ventas = add_business_alias_columns(ventas)
        return ventas, info

    # Si la base NSE es granular por hogares, calcular moda por municipio.
    if "ubica_geo" in ventas.columns and "id_municipio" not in ventas.columns:
        ventas["id_municipio"] = ventas["ubica_geo"].apply(_limpiar_id_municipio)
    if "ubica_geo" in nse.columns:
        nse["id_municipio"] = nse["ubica_geo"].apply(_limpiar_id_municipio)

    if "id_municipio" in nse.columns:
        nse["id_municipio"] = nse["id_municipio"].apply(_limpiar_id_municipio)
        nse_municipio = (
            nse.dropna(subset=["id_municipio", "categoria_est_socio"])
            .groupby("id_municipio", as_index=False)
            .agg(
                categoria_est_socio=("categoria_est_socio", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
                estado=("estado", lambda x: x.dropna().mode().iloc[0] if not x.dropna().mode().empty else np.nan) if "estado" in nse.columns else ("id_municipio", "first"),
            )
        )
    else:
        nse_municipio = pd.DataFrame()

    # Catálogo key -> ubica_geo/id_municipio si viene en NSE o en catálogo predeterminado.
    if "key" in ventas.columns:
        ventas["key"] = ventas["key"].astype(str).str.strip().str.lower()

    if "key" in nse.columns:
        nse["key"] = nse["key"].astype(str).str.strip().str.lower()

    if "key" in ventas.columns and "key" in nse.columns:
        cols_key = ["key"]
        if "ubica_geo" in nse.columns:
            cols_key.append("ubica_geo")
        if "id_municipio" in nse.columns:
            cols_key.append("id_municipio")
        if "estado" in nse.columns:
            cols_key.append("estado")
        if "categoria_est_socio" in nse.columns:
            cols_key.append("categoria_est_socio")

        nse_key = nse[cols_key].drop_duplicates("key").copy()
        nse_key = nse_key.rename(
            columns={
                "categoria_est_socio": "categoria_est_socio_nse",
                "estado": "estado_nse",
            }
        )
        ventas = ventas.drop(columns=["id_municipio"], errors="ignore") if "id_municipio" in ventas.columns else ventas
        ventas = ventas.merge(nse_key, on="key", how="left")
        if "ubica_geo" in ventas.columns and "id_municipio" not in ventas.columns:
            ventas = ventas.rename(columns={"ubica_geo": "id_municipio"})
        elif "ubica_geo" in ventas.columns and "id_municipio" in ventas.columns:
            ventas["id_municipio"] = ventas["id_municipio"].fillna(ventas["ubica_geo"])
            ventas = ventas.drop(columns=["ubica_geo"], errors="ignore")

        if "categoria_est_socio_nse" in ventas.columns:
            if "categoria_est_socio" not in ventas.columns:
                ventas["categoria_est_socio"] = ventas["categoria_est_socio_nse"]
            else:
                ventas["categoria_est_socio"] = ventas["categoria_est_socio"].fillna(ventas["categoria_est_socio_nse"])
            ventas = ventas.drop(columns=["categoria_est_socio_nse"], errors="ignore")
        if "estado_nse" in ventas.columns:
            if "estado" not in ventas.columns:
                ventas["estado"] = ventas["estado_nse"]
            else:
                ventas["estado"] = ventas["estado"].fillna(ventas["estado_nse"])
            ventas = ventas.drop(columns=["estado_nse"], errors="ignore")

    # Si hay id_municipio, cruzar por municipio.
    if "id_municipio" in ventas.columns and not nse_municipio.empty:
        ventas["id_municipio"] = ventas["id_municipio"].apply(_limpiar_id_municipio)
        ventas = ventas.merge(
            nse_municipio.rename(
                columns={
                    "categoria_est_socio": "categoria_est_socio_mpio",
                    "estado": "estado_mpio",
                }
            ),
            on="id_municipio",
            how="left",
        )
        if "categoria_est_socio_mpio" in ventas.columns:
            if "categoria_est_socio" not in ventas.columns:
                ventas["categoria_est_socio"] = ventas["categoria_est_socio_mpio"]
            else:
                ventas["categoria_est_socio"] = ventas["categoria_est_socio"].fillna(ventas["categoria_est_socio_mpio"])
            ventas = ventas.drop(columns=["categoria_est_socio_mpio"], errors="ignore")
        if "estado_mpio" in ventas.columns:
            if "estado" not in ventas.columns:
                ventas["estado"] = ventas["estado_mpio"]
            else:
                ventas["estado"] = ventas["estado"].fillna(ventas["estado_mpio"])
            ventas = ventas.drop(columns=["estado_mpio"], errors="ignore")

    if "categoria_est_socio" not in ventas.columns:
        ventas["categoria_est_socio"] = np.nan

    ventas["categoria_est_socio"] = ventas["categoria_est_socio"].apply(normalizar_categoria_est_socio)
    asignados = int(ventas["categoria_est_socio"].notna().sum())
    total = len(ventas)
    match_status_asignado = (
        "match_personalizado"
        if fuente_nse == "personalizada"
        else "fallback_default"
        if estado_validacion_nse == "usada_default_por_fallback"
        else "match_default"
    )
    ventas["fuente_nse"] = fuente_nse
    ventas["nse_asignado"] = ventas["categoria_est_socio"].fillna("NSE_no_asignado")
    ventas["nse_match_status"] = np.where(ventas["categoria_est_socio"].notna(), match_status_asignado, "sin_match")
    ventas["categoria_est_socio"] = ventas["nse_asignado"]

    info.update(
        {
            "nse_usado": True,
            "registros_con_nse": asignados,
            "registros_sin_nse": int(total - asignados),
            "porcentaje_asignado": float((asignados / total * 100) if total else 0),
            "porcentaje_match_nse": float((asignados / total * 100) if total else 0),
            "registros_sin_match_nse": int(total - asignados),
            "mensaje": f"NSE asignado a {asignados:,} de {total:,} registros usando fuente {fuente_nse}.",
        }
    )
    ventas = add_business_alias_columns(ventas)
    return ventas, info


def build_quarter_label(periodo_3m: str) -> str:
    """Convierte '2025-01 a 2025-03' en 'ene 2025 - mar 2025'."""
    if pd.isna(periodo_3m):
        return "Sin trimestre"
    txt = str(periodo_3m)
    meses = {
        1: "ene",
        2: "feb",
        3: "mar",
        4: "abr",
        5: "may",
        6: "jun",
        7: "jul",
        8: "ago",
        9: "sep",
        10: "oct",
        11: "nov",
        12: "dic",
    }
    try:
        partes = [p.strip() for p in txt.split(" a ")]
        if len(partes) != 2:
            return txt
        inicio = pd.Period(partes[0], freq="M")
        fin = pd.Period(partes[1], freq="M")
        return f"{meses[inicio.month]} {inicio.year} - {meses[fin.month]} {fin.year}"
    except Exception:
        return txt


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convierte DataFrame a CSV descargable."""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def render_kpi_card(title: str, value, subtitle: str = "") -> None:
    """Renderiza una tarjeta KPI simple."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_dataframe_dependently(
    df: pd.DataFrame,
    filters: dict[str, object],
) -> pd.DataFrame:
    """Filtra un DataFrame con filtros dependientes tipo 'Todos' o listas."""
    out = df.copy()
    for col, value in filters.items():
        if col not in out.columns or value in [None, "Todos", "Todas"]:
            continue
        if isinstance(value, (list, tuple, set)):
            selected = [v for v in value if v not in ["Todos", "Todas"]]
            if selected:
                out = out[out[col].astype(str).isin([str(v) for v in selected])]
        else:
            out = out[out[col].astype(str) == str(value)]
    return out


def add_state_coordinates(df: pd.DataFrame, estado_col: str = "estado") -> pd.DataFrame:
    """Agrega lat/lon aproximados para estados de México."""
    out = df.copy()
    if estado_col not in out.columns:
        return out
    keys = out[estado_col].apply(normalize_text)
    out["lat"] = keys.map(lambda k: STATE_COORDINATES.get(k, (np.nan, np.nan))[0])
    out["lon"] = keys.map(lambda k: STATE_COORDINATES.get(k, (np.nan, np.nan))[1])
    return out


def format_money(x) -> str:
    if pd.isna(x):
        return "N/A"
    return f"${x:,.2f}"


def format_num(x, dec: int = 2) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{dec}f}"


def format_pct(x, dec: int = 1) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:+,.{dec}f}%"
