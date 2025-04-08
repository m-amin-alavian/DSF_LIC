from pathlib import Path
import importlib

import pandas as pd

from .. import steps
from ..utils.metadata import metadata
from .extrapolator import extrapolate


def open_data() -> pd.DataFrame:
    metadata.reload_metadata()
    data = load_input()
    data = apply_metadata(data, "inputs")
    data = steps.add_old_debt_pv(data)
    data = steps.add_foreign_currency_financing(data)
    data = steps.add_local_currency_financing(data)
    data = steps.add_mlt_debts(data)
    data = apply_metadata(data, "financing")
    data = apply_metadata(data, "macroeconomics")
    data = steps.add_per_gdp(data)
    data = data.round(6)
    return data


def load_input() -> pd.DataFrame:
    file_path = Path("data", "inputs.csv")
    return (
        pd.read_csv(file_path, index_col=0)
        .transpose()
        .pipe(lambda df: df.set_axis(df.index.astype(int))) # type: ignore
        .reindex(metadata.year_index)
    )


def apply_metadata(data: pd.DataFrame, file: str) -> pd.DataFrame:
    for column_name in metadata.variables[file]:
        data = apply_metadata_for_column(file=file, column_name=column_name, data=data)
    return data


def apply_metadata_for_column(
    file: str,
    column_name: str,
    data: pd.DataFrame
) -> pd.DataFrame:
    column_metadata = metadata.variables[file][column_name]

    if column_metadata["Source"] == "Input":
        assert column_name in data.columns
    elif column_metadata["Source"] == "Calculation" and "Formula" in column_metadata:
        formula = column_metadata["Formula"]
        if isinstance(formula, dict) and "Residency_Based" in formula:
            if metadata.setting.residency_based:
                formula = formula["Residency_Based"]
            else:
                formula = formula["Currency_Based"]

        projection_year = metadata.setting.projection_year
        local_dict = {
            "projection_year": projection_year
        }
        if isinstance(formula, str):
            data[column_name] = data.eval(formula, local_dict=local_dict)
        elif isinstance(formula, dict) and "Pre_Projection" in formula:
            data[column_name] = pd.concat([
                data.loc[:projection_year-1].eval(formula["Pre_Projection"], local_dict=local_dict), # type: ignore
                data.loc[projection_year:].eval(formula["Post_Projection"], local_dict=local_dict), # type: ignore
            ])
        else:
            print(column_metadata)
            raise ValueError
    elif column_metadata["Source"] == "Calculation" and "Function" in column_metadata:
        variable_functions = importlib.import_module("dsf_lic.metadata.variable_functions")
        function_info = column_metadata["Function"]
        if isinstance(function_info, str):
            function_name = function_info
            parameters = {}
        elif isinstance(function_info, dict):
            function_name, parameters = list(function_info.items())[0]
        parameters["data"] = data
        func = getattr(variable_functions, function_name)
        data[column_name] = func(**parameters)
    else:
        print(column_metadata)
        raise KeyError
        

    if "Extrapolate" in column_metadata:
        data[column_name] = extrapolate(
            data[column_name],
            column_metadata["Extrapolate"]
        )

    return data
