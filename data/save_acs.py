import pandas as pd

p = pd.read_stata(
    "https://www2.census.gov/programs-surveys/supplemental-poverty-measure/datasets/spm/spm_2018_pu.dta",
    columns=[
        "serialno",
        "sporder",
        "wt",
        "age",
        "spm_id",
        "spm_povthreshold",
        "spm_resources",
        "st",
    ],
)

p.to_csv("data/acs_poverty.csv.gz", compression="gzip", index=False)
