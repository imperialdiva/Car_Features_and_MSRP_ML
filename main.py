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
    return df, np, plt, sns


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
    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
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
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test = df_shuffled.iloc[n_train+n_val:].copy()
    return df_test, df_train, df_val


@app.cell
def _(df_train):
    # Посмотрите все доступные колонки
    df_train.columns.tolist()
    return


@app.cell
def _(df_test, df_train, df_val, np):
    y_train = np.log1p(df_train.msrp.values)
    y_val = np.log1p(df_val.msrp.values)
    y_test = np.log1p(df_test.msrp.values)
    return y_train, y_val


@app.cell
def _(df_test, df_train, df_val):
    del df_train['msrp']
    del df_val['msrp']
    del df_test['msrp']
    return


@app.cell
def _(np):
    def train_linear_regression(X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)

        return w[0], w[1:]

    return (train_linear_regression,)


@app.cell
def _():
    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
    return (base,)


@app.cell
def function_1(np):
    def rmse(y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)

    return (rmse,)


@app.cell
def function(base):
    def prepare_X(df):
        df = df.copy()
        features = base.copy()

        df['age'] = 2017 - df.year
        features.append('age')

        df_num = df[features]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

    return (prepare_X,)


@app.cell
def _(df_train, prepare_X, train_linear_regression, y_train):
    X_train = prepare_X(df_train)
    w_0, w = train_linear_regression(X_train, y_train)
    return w, w_0


@app.cell
def _(df_val, prepare_X, rmse, w, w_0, y_val):
    X_val = prepare_X(df_val)
    y_pred = w_0 + X_val.dot(w)
    print('validation:', rmse(y_val, y_pred))
    return (y_pred,)


@app.cell
def _(plt, sns, y_pred, y_val):
    sns.histplot(y_pred, label='prediction')
    sns.histplot(y_val, label='target')
    plt.legend()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
