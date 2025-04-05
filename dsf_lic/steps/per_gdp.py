import pandas as pd


COLUMNS = [
    "BX",
    "BM",
    "Trade_Deficit",
    "NCT",
    "NCTG",
    "NFDI",
]


def main(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    per_gdp_columns = (
        data.loc[:, COLUMNS]
        .div(data["NGDPD"], axis="index")
        .mul(100)
        .add_suffix("_per_GDP")
    )

    data = data.join(per_gdp_columns)

    return data
