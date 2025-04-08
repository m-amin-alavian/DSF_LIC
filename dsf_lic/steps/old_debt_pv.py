import pandas as pd

from ..utils.metadata import metadata


EXTERNAL_DEBT_LIST = [
    "OLD_IMF",
    "OLD_IDA1",
    "OLD_IDA2",
    "OLD_MULTI1",
    "OLD_MULTI2",
    "OLD_MULTI3",
    "OLD_OTH_MULTI1",
    "OLD_OTH_MULTI2",
    "OLD_OTH_MULTI3",
    "OLD_PC1",
    "OLD_PC2",
    "OLD_PC3",
    "OLD_PC4",
    "OLD_PC5",
    "OLD_NPC1",
    "OLD_NPC2",
    "OLD_NPC3",
    "OLD_NPC4",
    "OLD_NPC5",
    "OLD_COM1",
    "OLD_COM2",
    "OLD_COM3",
    "OLD_COM4",
    "OLD_COM5",
]


def add_pv(data: pd.DataFrame, debt_name: str) -> pd.DataFrame:
    pv_multiplier = (
        pd.Series(metadata.setting.discount_rate, index=data.index)
        .div(100).add(1).cumprod()
    )
    data[f"{debt_name}_PV"] = (
        data[[debt_name]]
        .shift(-1 , fill_value=0)
        .div(pv_multiplier, axis="index")
        .iloc[::-1].cumsum().iloc[::-1]
        .mul(pv_multiplier.shift(1, fill_value=1), axis="index")
        .loc[2016:]
    )
    return data


def main(data: pd.DataFrame) -> pd.DataFrame:
    for debt_name in EXTERNAL_DEBT_LIST:
        data = add_pv(data, debt_name)
    data["OLD_EXT_PV"] = (
        data[[f"{col}_PV" for col in EXTERNAL_DEBT_LIST]]
        .sum(axis="columns")
    )
    return data
