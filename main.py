import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd 

    from matplotlib import pyplot as plt
    import seaborn as sns

    df = pd.read_csv("data.csv")
    return df, np, sns


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    string_columns = list(df.dtypes[df.dtypes == 'Object'].index)
    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, sns):
    sns.histplot(df.msrp, bins=40)
    return


@app.cell
def _(df, sns):
    sns.histplot(df.msrp[df.msrp < 100000], bins=40)
    return


@app.cell
def _(df, np):
    log_price = np.log1p(df.msrp)
    return (log_price,)


@app.cell
def _(log_price, sns):
    sns.histplot(log_price)
    return


if __name__ == "__main__":
    app.run()
