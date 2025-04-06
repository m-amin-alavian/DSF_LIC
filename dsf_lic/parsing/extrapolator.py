from typing import Optional, NamedTuple, Literal

import pandas as pd


class ExtrapolateOptions(NamedTuple):
    method: Literal["arithmetic", "geometric"] = "arithmetic"
    number_of_periods: int = 5


def extrapolate(
    column: pd.Series,
    parameters: dict | bool = True,
) -> pd.Series:
    if parameters is False:
        return column

    parameters = {} if parameters is True else parameters
    options = ExtrapolateOptions(**parameters)

    if options.method == "arithmetic":
        growth_average = (
            column
            .div(column.shift(1))
            .sub(1)
            .loc[lambda s: s.notna()]
            .iloc[-options.number_of_periods:]
            .mean()
        )
    else:
        raise NotImplementedError

    last_year = column.loc[lambda s: s.notna()].index[-1]
    last_value = column.loc[last_year]

    extrapolated_values = (
        pd.Series(
            growth_average,
            index=column.loc[last_year+1:].index # type: ignore
        )
        .add(1).cumprod().mul(last_value)
    )
    column = column.fillna(extrapolated_values)

    return column
