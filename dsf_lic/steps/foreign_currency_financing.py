import pandas as pd

from ..financing import foreign_currency
from ..utils.metadata import metadata


def create_table_for_aggregation(name: str, data: pd.DataFrame):
    external = not name.endswith("_fc")
    debt_info = foreign_currency.get_debt_info(name, external=external)
    table = (
        foreign_currency.create_debt_table(name, external=external, data=data)
        .rename(
            columns={
                "disbursement": "",
                "interest": "INT",
                "amortization": "DP",
                "pv": "PV",
                "stock_of_debt": "NOM",
            }
        )
        .loc[:, ["", "INT", "DP", "PV", "NOM"]]
    )
    table = pd.concat(
        [table],
        axis="columns",
        keys=[(debt_info.group, debt_info.sub_group, debt_info.prefix)],
        names=["group", "sub_group", "loan", "variable"],
    )
    return table


def filter_empty_groups(df: pd.DataFrame) -> list:
    return df.columns.get_level_values(0).to_series().ne("").to_list()

def main(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    loans = (
        list(metadata.external_loans) +
        list(
            loan for loan in metadata.internal_financing if
            loan.endswith("_fc")
        )
    )
    all_debts = (
        pd.concat(
            [create_table_for_aggregation(loan, data) for loan in loans],
            axis="columns"
        )
        .fillna(0)
    )

    loan_level = all_debts.droplevel([0, 1], 1)
    loan_level.columns = loan_level.columns.map(lambda x: '_'.join(map(str, x))).str.strip("_")

    sub_group_level = (
        all_debts
        .transpose()
        .groupby(["sub_group", "variable"]).sum()
        .transpose()
        .loc[:, filter_empty_groups]
    )
    sub_group_level.columns = sub_group_level.columns.map(lambda x: '_'.join(map(str, x))).str.strip("_")

    group_level = (
        all_debts
        .transpose()
        .groupby(["group", "variable"]).sum()
        .transpose()
        .loc[:, filter_empty_groups]
    )
    group_level.columns = group_level.columns.map(lambda x: '_'.join(map(str, x))).str.strip("_")
    loan_table = pd.concat([loan_level, sub_group_level, group_level], axis="columns")

    columns = {}
    for variable in ["", "_INT", "_DP", "_PV", "_NOM"]:
        columns[f"NEW_EXT{variable}"] = (
            loan_table.eval(
                f"NEW_MULTI{variable} + NEW_OB{variable} + NEW_COM{variable}"
            )
        )

    loan_table = loan_table.join(pd.DataFrame(columns))
    loan_table = loan_table.astype("Float64")

    columns = loan_table.columns.to_series().loc[lambda s: - s.isin(data.columns)].to_list()
    data = data.join(loan_table.loc[:, columns])

    return data
