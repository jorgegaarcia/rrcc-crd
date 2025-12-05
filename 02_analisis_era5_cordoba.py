"""
Análisis de temperatura media anual en Córdoba a partir de datos ERA5.

Este script:
- Carga todos los archivos NetCDF descargados con 01_descarga_era5_cordoba.py
- Calcula la temperatura media anual (promediando espacio -> tiempo)
- Compara los primeros 10 años con los últimos 10 (señal de calentamiento)
- Calcula medias por década
- Genera una gráfica con la serie anual y la tendencia lineal

Requisitos:
- Archivos cordoba_completo_YYYY.nc en data/cordoba
- Librerías: xarray, pandas, numpy, matplotlib, seaborn
"""

from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Directorios del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "cordoba"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def cargar_dataset_era5(data_dir: Path = DATA_DIR) -> xr.Dataset:
    """
    Carga y concatena todos los archivos NetCDF de ERA5 para Córdoba.

    Devuelve:
        ds (xarray.Dataset): Dataset concatenado con la variable t2m.
    """
    nc_files = sorted(data_dir.glob("cordoba_completo_*.nc"))

    if not nc_files:
        raise FileNotFoundError(
            f"No se encontraron archivos 'cordoba_completo_*.nc' en {data_dir}. "
            "Ejecuta antes el script 01_descarga_era5_cordoba.py."
        )

    print(f"Se han encontrado {len(nc_files)} archivos NetCDF en {data_dir}.")

    # open_mfdataset concatena automáticamente por la coordenada temporal
    ds = xr.open_mfdataset(nc_files, combine="by_coords")

    if "t2m" not in ds.data_vars:
        raise KeyError(
            "La variable 't2m' no está en el dataset. "
            "Revisa que los archivos provengan de ERA5 con '2m_temperature'."
        )

    return ds


def calcular_medias_anuales(ds: xr.Dataset) -> pd.DataFrame:
    """
    Calcula la temperatura media anual en °C, promediando sobre el área.

    Devuelve:
        df_anual (DataFrame): columnas ['year', 'temp_C_annual'].
    """
    # Promedio espacial y conversión a °C
    if not {"latitude", "longitude"}.issubset(ds.dims):
        raise KeyError("No se encontraron dimensiones 'latitude' y 'longitude' en el dataset.")

    t2m_k = ds["t2m"]
    t2m_c = (t2m_k - 273.15).mean(dim=["latitude", "longitude"]).rename("temp_C")

    df = t2m_c.to_dataframe().reset_index()

    # Nombre de la columna temporal (time o valid_time)
    time_col = "time" if "time" in df.columns else "valid_time" if "valid_time" in df.columns else None
    if time_col is None:
        raise KeyError("No se encontró columna temporal 'time' ni 'valid_time' en el dataset.")

    df["year"] = pd.to_datetime(df[time_col]).dt.year

    df_anual = (
        df.groupby("year", as_index=False)["temp_C"]
        .mean()
        .rename(columns={"temp_C": "temp_C_annual"})
    )

    return df_anual


def resumen_calentamiento(df_anual: pd.DataFrame) -> None:
    """
    Imprime en consola un resumen de calentamiento:
    primeros 10 años vs últimos 10, y medias por década.
    """
    year_min = int(df_anual["year"].min())
    year_max = int(df_anual["year"].max())
    n_years = year_max - year_min + 1

    print("\n=== RESUMEN DE AÑOS DISPONIBLES ===")
    print(f"Años analizados: {year_min}–{year_max} ({n_years} años)\n")

    if n_years < 20:
        print("No hay al menos 20 años de datos; no se puede comparar primeros y últimos 10 años.")
        return

    first_start = year_min
    first_end = year_min + 9
    last_end = year_max
    last_start = year_max - 9

    mask_first = (df_anual["year"] >= first_start) & (df_anual["year"] <= first_end)
    mask_last = (df_anual["year"] >= last_start) & (df_anual["year"] <= last_end)

    mean_first = df_anual.loc[mask_first, "temp_C_annual"].mean()
    mean_last = df_anual.loc[mask_last, "temp_C_annual"].mean()
    delta = mean_last - mean_first

    años_intervalo = last_end - first_start
    delta_por_decada = delta / (años_intervalo / 10)

    print("=== COMPARACIÓN PRIMEROS 10 AÑOS VS ÚLTIMOS 10 AÑOS ===")
    print(f"Media {first_start}-{first_end}: {mean_first:.2f} °C")
    print(f"Media {last_start}-{last_end}: {mean_last:.2f} °C")
    print(f"Aumento medio: +{delta:.2f} °C en {años_intervalo} años")
    print(f"Aproximadamente: +{delta_por_decada:.2f} °C por década\n")

    # Medias por década
    df_decadas = df_anual.copy()
    df_decadas["decada"] = (df_decadas["year"] // 10) * 10
    df_decadas = (
        df_decadas.groupby("decada", as_index=False)["temp_C_annual"]
        .mean()
        .sort_values("decada")
    )

    print("=== TEMPERATURA MEDIA POR DÉCADA (°C) ===")
    print(df_decadas.to_string(index=False))


def grafica_tendencia(df_anual: pd.DataFrame, output_dir: Path = FIG_DIR) -> None:
    """
    Genera una gráfica de la temperatura media anual y la tendencia lineal
    y la guarda en la carpeta de figuras.
    """
    year_min = int(df_anual["year"].min())
    year_max = int(df_anual["year"].max())

    # Cálculo rápido para el título
    first_start = year_min
    first_end = min(year_min + 9, year_max)
    last_end = year_max
    last_start = max(year_max - 9, year_min)

    mask_first = (df_anual["year"] >= first_start) & (df_anual["year"] <= first_end)
    mask_last = (df_anual["year"] >= last_start) & (df_anual["year"] <= last_end)

    mean_first = df_anual.loc[mask_first, "temp_C_annual"].mean()
    mean_last = df_anual.loc[mask_last, "temp_C_annual"].mean()
    delta = mean_last - mean_first

    plt.figure(figsize=(10, 5))

    # Puntos anuales
    sns.scatterplot(
        data=df_anual,
        x="year",
        y="temp_C_annual",
        alpha=0.7,
        label="Media anual",
    )

    # Tendencia lineal
    sns.regplot(
        data=df_anual,
        x="year",
        y="temp_C_annual",
        scatter=False,
        line_kws={"linewidth": 2},
        label="Tendencia lineal",
    )

    plt.xlabel("Año")
    plt.ylabel("Temperatura media anual (°C)")
    plt.title(
        f"Córdoba: evolución de la temperatura media anual ({year_min}-{year_max})\n"
        f"Aumento aproximado: +{delta:.2f} °C entre {first_start}-{first_end} y {last_start}-{last_end}"
    )
    plt.tight_layout()

    output_path = output_dir / "cordoba_temp_media_anual.png"
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"\nGráfico guardado en: {output_path}")


def main() -> None:
    print("Cargando datos ERA5 de Córdoba...")
    ds = cargar_dataset_era5()
    print("Calculando medias anuales...")
    df_anual = calcular_medias_anuales(ds)
    print("Generando resumen de calentamiento...")
    resumen_calentamiento(df_anual)
    print("\nGenerando gráfica de tendencia...")
    grafica_tendencia(df_anual)


if __name__ == "__main__":
    main()
