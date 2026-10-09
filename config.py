"""Path and frozen-constant configuration.

Override the data root with the YREPCMP_DATABANK environment variable so the
pipeline can run on any machine where the source datasets are staged.
Map shapefiles, the DEM clip, and schematic images are bundled under data/.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATABANK = os.environ.get("YREPCMP_DATABANK", "/data-02/home/20091008/databank")

OUT = os.environ.get("YREPCMP_OUT", str(ROOT / "out"))
OUT_LEGACY = os.environ.get("YREPCMP_OUT_LEGACY", str(ROOT / "out_legacy46x82"))
FIGDIR = os.environ.get("YREPCMP_FIGURES", str(ROOT / "figures"))

for _d in (OUT, OUT_LEGACY, FIGDIR):
    os.makedirs(_d, exist_ok=True)

DATA = ROOT / "data"

# Bundled small inputs (self-contained)
SHAPES = str(DATA / "china-shapefiles" / "shapefiles")
BASIN_SHP = str(DATA / "china-shapefiles" / "huanghe_zhongxia" / "huanghe_zhongxia.shp")
CNTRY_SHP = os.path.join(SHAPES, "china_country.shp")
NINEDASH_SHP = os.path.join(SHAPES, "china_nine_dotted_line.shp")
PROV_SHP = os.path.join(SHAPES, "china.shp")
DEM = str(DATA / "china-shapefiles" / "china-dem" / "wc2.1_2.5m_elev.tif")
SCHEMATICS = DATA / "schematics"

# External datasets staged under $YREPCMP_DATABANK (see data/*/download-url.txt)
CHM_DIR = os.path.join(DATABANK, "CHM_PRE_V2", "daily")
NCAR_DIR = os.path.join(DATABANK, "NCAR", "Data-NCAR", "data", "daily")
CN051_PATH = os.path.join(DATABANK, "CN05.1", "CN05.1_Pre_1961_2023_daily_025x025.nc")


def chm_path(year):
    return os.path.join(CHM_DIR, f"CHM_PRE_V2_daily_{year}.nc")


YEARS = list(range(1960, 2025))
EARLY = (1960, 1991)
LATE = (1992, 2024)
PRODUCT_OVERLAP = (1961, 2023)
