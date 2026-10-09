# Yellow River YR-EPCMP CNN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23260552.svg)](https://doi.org/10.5281/zenodo.23260552)

Analysis code for:

> **Interpretable machine learning reveals circulation–moisture patterns
> associated with regional precipitation extremes over the middle and lower
> reaches of the Yellow River basin**

The study identifies a small set of large-scale circulation–moisture patterns
(YR-EPCMP) that control flood-related regional extreme precipitation over the
Yellow River basin. A CNN classifier is trained on daily reanalysis fields
(Z500, SLP, IVT, VIMFC, PW) to detect YR-EPCMP days; LRP, Integrated
Gradients, and SmoothGrad provide channel-level interpretability. A
closed-form decomposition separates observed intensification into dynamical
(frequency) and thermodynamical (intensity) contributions.

## Quick start

```bash
pip install -r requirements.txt
export YREPCMP_DATABANK=/path/to/databank   # see Data sources below
python 00_make_all.py                       # full pipeline + all figures
```

Individual scripts can be run in order: scripts 01–02 prepare the raw data;
03–08 run the analysis; the `09a_fig*.py` scripts generate figures from cached
results. Execution order is listed in `00_make_all.py`.

## Configuration

All paths are centralized in `config.py`. Set the `YREPCMP_DATABANK`
environment variable to point at the root directory containing the large
source datasets (below). Output goes to `out/` and `out_legacy46x82/`;
figures to `figures/`. Map shapefiles, the DEM clip, and schematic images are
bundled under `data/` and require no download.

## Data sources

| Dataset | Resolution | Staged path | Source |
|---|---|---|---|
| CHM_PRE V2 daily precipitation | 0.1° | `$YREPCMP_DATABANK/CHM_PRE_V2/daily/` | National Tibetan Plateau Data Center, https://data.tpdc.ac.cn (pointer: `data/CHM_PRE_V2/download-url.txt`) |
| NCEP/NCAR R1 reanalysis | 2.5° → bilinear 1° | `$YREPCMP_DATABANK/NCAR/Data-NCAR/data/daily/` | NOAA PSL, https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html (pointer: `data/NCAR/download-url.txt`) |
| CN05.1 daily precipitation (validation) | 0.25° | `$YREPCMP_DATABANK/CN05.1/` | Zenodo, https://doi.org/10.5281/zenodo.22700402 (pointer: `data/CN05.1/download-url.txt`) |
| Basin / national boundary shapefiles | vector | bundled at `data/china-shapefiles/` | — |
| WorldClim DEM | 2.5′ | bundled at `data/china-shapefiles/china-dem/` | https://www.worldclim.org |
| Schematic artwork (Fig 2a, Fig 3a) | raster | bundled at `data/schematics/` | — |

## Dual-track convention

Two CNN input domains are used in this study:

- **Legacy track** (46×82 grid, 70–151°E / 15–60°N): all figures except Fig 4
  and Fig S7. Cached results in `out_legacy46x82/`.
- **New-domain track** (82×122 grid, 59–180°E / −11–70°N): Fig 4 and Fig S7.
  Cached results in `out/`.

Scripts reference `OUT` (new domain) or `OUT_LEGACY` (legacy) from `config.py`.

## License

MIT (see LICENSE).
