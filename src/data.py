import pandas as pd

def read_data(paths: dict[str, str]):
    '''
    Read data and return dataframes as a dict

    Params:
        path : {name: path} : file paths to read data
    Returns:
        dfs : {name: df} : a dict of dataframes
    '''

    dfs = {}
    for name, path in paths.items():
        dfs[name] = pd.read_csv(path)
    return dfs