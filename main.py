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


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df):
    n = len(df)
    return (n,)


@app.cell
def _(n):
    n_val = int(n*0.2)
    n_test = int(n*0.2)
    n_train = n - (n_val + n_test)
    return n_train, n_val


@app.cell
def _(n, np):
    np.random.seed(2)
    idx = np.arange(n)
    np.random.shuffle(idx)
    return (idx,)


@app.cell
def _(df, idx):
    df_shuffled = df.iloc[idx]
    return (df_shuffled,)


@app.cell
def _(df_shuffled, n_train, n_val):
    df_train = df_shuffled.iloc[:n_train].copy()
    df_test = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_val = df_shuffled.iloc[n_train+n_val:].copy()
    return


if __name__ == "__main__":
    app.run()
