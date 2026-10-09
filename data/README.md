# Data layout

Bundled small inputs live here (no download needed):

- `china-shapefiles/` — basin boundary (`huanghe_zhongxia/`: middle + lower +
  combined), national/province boundaries (`shapefiles/`), WorldClim 2.5′ DEM
  clip (`china-dem/wc2.1_2.5m_elev.tif`)
- `schematics/` — Fig 2a flowchart and Fig 3a architecture artwork

Large datasets are staged externally under `$YREPCMP_DATABANK`; each
subdirectory here holds the download address:

- `CHM_PRE_V2/download-url.txt`
- `NCAR/download-url.txt`
- `CN05.1/download-url.txt`

Expected external layout:

```
$YREPCMP_DATABANK/
├── CHM_PRE_V2/daily/CHM_PRE_V2_daily_{1960..2024}.nc
├── NCAR/Data-NCAR/data/daily/{hgt,slp,uwnd,vwnd,shum,pr_wtr}.{YYYY}.nc
└── CN05.1/CN05.1_Pre_1961_2023_daily_025x025.nc
```

Scripts write cached analysis products to `../out/` (new-domain track) and
`../out_legacy46x82/` (legacy 46×82 track); figures to `../figures/`.
