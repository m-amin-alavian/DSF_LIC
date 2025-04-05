import pandas as pd

from ..utils.metadata import metadata


def main(data: pd.DataFrame) -> pd.DataFrame:
    columns = {}
    for variable in ["", "_INT", "_DP", "_PV", "_NOM"]:
        columns[f"NEW_MLT{variable}"] = data[f"NEW_EXT{variable}"] + data[f"New_dom_nr_fc{variable}"]
        if metadata.setting.residency_based:
            columns[f"NEW_MLT{variable}"] += data[f"New_dom_nr_lc{variable}"]
        else:
            columns[f"NEW_MLT{variable}"] += data[f"New_dom_fc{variable}"]
    data = data.join(pd.DataFrame(columns).astype("Float64"))

    return data
