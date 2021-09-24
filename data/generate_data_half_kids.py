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

for d in [cu, cuid_sporder]:
    d.columns = d.columns.str.lower()

CU_METRICS = ["tco2", "tco2_direct", "invest_assets"]
cu.rename(
    columns=dict(zip(CU_METRICS, [i + "_cu" for i in CU_METRICS])),
    inplace=True,
)

# Per Kevin Ummel, serialno is first 13 characters of CUID
def cuid2serialno(cuid):
    return cuid.str[6:13].astype(int)


cuid_sporder["serialno"] = cuid2serialno(cuid_sporder.cuid)

p = p.merge(cuid_sporder, on=["serialno", "sporder"]).merge(cu, on="cuid")
# Give kids half a credit.
p["person"] = np.where(p.age < 18, 0.5, 1)
p_per_cu = (
    p.groupby("cuid")[["person"]]
    .sum()
    .reset_index()
    .rename(columns=dict(person="people_in_cu"))
)
p = p.merge(p_per_cu, on="cuid")
for i in CU_METRICS:
    p[i] = p[i + "_cu"] / p.people_in_cu

# Adjust population and CO2 to match administrative sources for 2021.
# Source: https://www.macrotrends.net/countries/USA/united-states/population
TRUE_POP = 332_915_073
# https://data.census.gov/cedsci/table?q=us%20population%20by%20age&tid=ACSST1Y2019.S0101
TRUE_KID_SHARE = 72_967_785 / 328_239_523
# Source: https://www.rff.org/publications/data-tools/carbon-pricing-calculator
TRUE_CO2 = 5.05e9
tmp_pop = p.wt.sum()
tmp_co2 = mdf.weighted_sum(p, "tco2", "wt")
p.tco2 *= TRUE_CO2 / tmp_co2
p.wt *= TRUE_POP / tmp_pop
total_kids = TRUE_POP * TRUE_KID_SHARE
total_adults = TRUE_POP - total_kids
total_shares = total_adults + total_kids * 0.5
co2_per_share = TRUE_CO2 / total_shares
p["age_group"] = np.where(p.age < 18, "Child", "Adult")


SPMU_COLS = ["id", "povthreshold", "resources"]
s = (
    p.groupby(["spm_" + i for i in SPMU_COLS])[["person", "tco2"]]
    .sum()
    .reset_index()
)


def carbon_dividend(price):
    def pov_metrics(p):
        poverty = mdf.poverty_rate(
            p,
            "spm_resources_new",
            "spm_povthreshold",
            "wt",
        )
        deep_poverty = mdf.deep_poverty_rate(
            p,
            "spm_resources_new",
            "spm_povthreshold",
            "wt",
        )
        winner_share = (
            p[p.spm_resources_new > p.spm_resources].wt.sum() / p.wt.sum()
        )
        return pd.Series(
            dict(
                poverty=poverty,
                deep_poverty=deep_poverty,
                winner_share=winner_share,
            )
        )

    dividend = co2_per_share * price
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
carbon_price.to_csv("data/carbon_price_half_kids.csv", index=False)
