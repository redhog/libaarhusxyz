# Workbench dat export fixture

`LPNNRD2018_SkyTEM304_line300901_MOD_dat_40soundings.xyz` — the first 40 soundings
(80 rows, two segments each) of the Aarhus Workbench `_MOD_dat` export for flight line
300901 of the ENWRA 2018 SkyTEM 304 survey, Nebraska, as published by the Lower Platte
North / Lower Platte South NRDs and redistributed in `YmerFlow/ymerflow-demo-data`. The
header is as exported except that the operator's user name and directory path are
scrubbed. Column layout, gate times, dummy value and values are untouched.

`SkyTEM304_xyz.gex` — the system description in its xyz variant (sensor offsets zeroed,
GateFactor 1.0), derived from the survey's published skb GEX; the same file the demo
repository ships as `system_skytem304_for_delivered_data.gex`.

Together they are the smallest real case of the two-rows-per-sounding layout
`transforms.split_workbench_moments` exists for: 51 master gate times, 23 LM and 30 HM
columns, two shared, none unmatched.
