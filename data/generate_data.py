import numpy as np
import pandas as pd
import microdf as mdf
import us
import pyreadr

# Load data.
cu = pd.read_csv("data/carbon_footprint_2018.csv")
cuid_sporder = pd.DataFrame(
    list(pyreadr.read_r("data/CUID-SPORDER.rds").values())[0]
)
p = pd.read_csv("data/acs_poverty.csv.gz")

# Per Kevin Ummel, serialno is first 13 characters of CUID
def cuid2serialno(cuid):
    return cuid.str[6:13].astype(int)


cuid_sporder["serialno"] = cuid2serialno(cuid_sporder.CUID)
cu["serialno"] = cuid2serialno(cu.CUID)
# Household dataset formed from the CU dataset.
h_co2 = cu.drop(columns="CUID").groupby("serialno").sum().reset_index()
# Columns to rename as they're aggregated to household.
h_co2_cols = h_co2.columns.drop("serialno")
h_co2.rename(
    columns=dict(zip(h_co2_cols, [i + "_hh" for i in h_co2_cols])),
    inplace=True,
)
# Assign emissions equally across household members.
p_per_h = (
    p.groupby("serialno")
    .size()
    .reset_index()
    .rename(columns={0: "people_in_hh"})
)
p = p.merge(p_per_h, on="serialno").merge(h_co2, on="serialno")
for i in h_co2_cols:
    p[i] = p[i + "_hh"] / p.people_in_hh

SPMU_COLS = ["id", "povthreshold", "resources"]
p["person"] = 1
s = (
    p.groupby(["spm_" + i for i in SPMU_COLS])[["person", "tco2"]]
    .sum()
    .reset_index()
)

total_co2 = mdf.weighted_sum(p, "tco2", "wt")
total_pop = p.wt.sum()
co2_pp = total_co2 / total_pop
p["age_group"] = np.where(p.age < 18, "Child", "Adult")


def carbon_dividend(price):
    def pov_metrics(p):
        poverty = mdf.poverty_rate(
            p, "spm_resources_new", "spm_povthreshold", "wt",
        )
        deep_poverty = mdf.deep_poverty_rate(
            p, "spm_resources_new", "spm_povthreshold", "wt",
        )
        return pd.Series(dict(poverty=poverty, deep_poverty=deep_poverty))

    dividend = co2_pp * price
    s["dividend"] = s.person * dividend
    s["co2_tax"] = s.tco2 * price
    s["spm_resources_new"] = s.spm_resources + s.dividend - s.co2_tax
    p2 = p.merge(s[["spm_id", "spm_resources_new"]], on="spm_id")
    pov_overall = pd.DataFrame(pov_metrics(p2)).T
    pov_overall["age_group"] = "All"
    pov_overall["st"] = 0
    pov_age = p2.groupby("age_group").apply(pov_metrics).reset_index()
    pov_age["st"] = 0
    pov_state = p2.groupby("st").apply(pov_metrics).reset_index()
    pov_state["age_group"] = "All"
    pov_age_state = (
        p2.groupby(["st", "age_group"]).apply(pov_metrics).reset_index()
    )
    res = pd.concat([pov_overall, pov_age, pov_state, pov_age_state], axis=0)
    res["price"] = price
    return res


carbon_price = pd.concat(
    [carbon_dividend(i) for i in np.arange(0, 405, 5)]
).reset_index(drop=True)
base_poverty = (
    carbon_price[carbon_price.price == 0]
    .rename(
        columns={
            "poverty": "base_poverty",
            "deep_poverty": "base_deep_poverty",
        }
    )
    .drop(columns="price")
)
carbon_price = carbon_price.merge(base_poverty, on=["age_group", "st"])
carbon_price["poverty_chg"] = (
    carbon_price.poverty / carbon_price.base_poverty - 1
)
carbon_price["deep_poverty_chg"] = (
    carbon_price.deep_poverty / carbon_price.base_deep_poverty - 1
)


def get_state_info(st):
    fips = str(int(st)).zfill(2)
    if st > 0:
        state = us.states.lookup(fips)
        abbr = state.abbr
        name = state.name
    else:
        abbr = "US"
        name = "US"
    return pd.Series(dict(state_abbr=abbr, state_name=name))


carbon_price[["state_abbr", "state_name"]] = carbon_price.st.apply(
    get_state_info
)
carbon_price.to_csv("data/carbon_price.csv", index=False)
