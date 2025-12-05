"""
Descarga datos ERA5 de temperatura a 2 m para Córdoba capital,
un archivo NetCDF por año.

Requisitos:
- Tener cuenta en Copernicus Climate Data Store (CDS)
- Tener configurado el archivo ~/.cdsapirc con tus credenciales, por ejemplo:
    url: https://cds.climate.copernicus.eu/api
    key: *********
"""

from pathlib import Path
import cdsapi

# Carpeta base del proyecto (este archivo .py está dentro del repo)
BASE_DIR = Path(__file__).resolve().parent

# Carpeta donde se guardarán los NetCDF
CARPETA_SALIDA = BASE_DIR / "data" / "cordoba"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

# Área de Córdoba (Norte, Oeste, Sur, Este)
AREA_CORDOBA = [38.08, -5.00, 37.65, -4.55]

# Rango de años a descargar (en este caso de 1960 a 2021, ambos inclusive)
YEAR_START = 1960
YEAR_END = 2021


def descargar_era5_cordoba(
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
    area=None,
    carpeta_salida: Path = CARPETA_SALIDA,
) -> None:
    """
    Descarga datos ERA5 de temperatura a 2 m (2m_temperature) para un rango de años
    y los guarda como archivos NetCDF en la carpeta indicada.

    Si un archivo de un año ya existe, se omite.
    """
    if area is None:
        area = AREA_CORDOBA

    client = cdsapi.Client()  # Usa ~/.cdsapirc

    print(f"Carpeta de salida: {carpeta_salida}")
    print(f"Años a descargar: {year_start}–{year_end}")
    print(f"Área (N, W, S, E): {area}\n")

    for year in range(year_start, year_end + 1):
        nombre_archivo = f"cordoba_completo_{year}.nc"
        ruta_completa = carpeta_salida / nombre_archivo

        # Si el archivo ya existe, se salta
        if ruta_completa.exists():
            print(f"--> Año {year}: ya descargado. Saltando...")
            continue

        print(f"Solicitando año {year}...")

        try:
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": ["2m_temperature"],  # Temperatura aire a 2 m
                    "year": str(year),
                    "month": [f"{m:02d}" for m in range(1, 13)],    # 01–12
                    "day":   [f"{d:02d}" for d in range(1, 32)],    # 01–31
                    "time":  [f"{h:02d}:00" for h in range(24)],    # 00:00–23:00
                    "area": area,
                },
                str(ruta_completa),
            )
            print(f"    Año {year}: guardado correctamente ({ruta_completa.name}).")
        except Exception as e:
            print(f"    Error bajando {year}: {e}")

    print("\n¡LISTO!")


# Punto de entrada (si se ejecuta como script)
if __name__ == "__main__":
    descargar_era5_cordoba()

Add ERA5 download script
