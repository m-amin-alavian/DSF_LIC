import pandas as pd

from ..utils.metadata import metadata
from ..financing import foreign_currency, local_currency


def outstanding_from_old_debt(
    initial_value: str,
    payment: str,
    data:pd.DataFrame
) -> pd.Series:
    last_year = metadata.setting.projection_year-1
    last_year_value = pd.Series(
        data.eval(initial_value).loc[last_year], # type: ignore
        index=pd.RangeIndex(last_year, metadata.setting.last_year),
    )
    cumulative_payment = (
        pd.Series(
            data.eval(payment).loc[last_year+1:].fillna(0).cumsum(), # type: ignore
            index=pd.RangeIndex(last_year, metadata.setting.last_year),
        )
        .fillna(0)
    )
    outstanding = last_year_value - cumulative_payment
    return outstanding


def foreign_currency_debt(
        name: str,
        column: str,
        data: pd.DataFrame,
        external: bool = True
    ) -> pd.Series:
    table = foreign_currency.create_debt_table(name, data, external=external)
    result = table[column]
    return result


def local_currency_debt(name: str, column: str, data: pd.DataFrame) -> pd.Series:
    return local_currency.create_debt_table(name, data)[column]
