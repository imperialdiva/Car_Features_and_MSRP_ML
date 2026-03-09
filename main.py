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
    return df, np, pd, plt, sns


@app.cell
def _(df):
    df
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
def _(df, np):
    log_price = np.log1p(df.msrp)
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

    return


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

    return


@app.cell
def function(base):
    def prepare_X(df):
        df = df.copy()
        features = base.copy()

        df['age'] = 2017 - df.year
        features.append('age')
        for v in [2, 3, 4]:
            feature = 'num_doors_%s' % v
            value = (df['number_of_doors'] == v).astype(int)
            df[feature] = value
            features.append(feature)

        for v in ['chevrolet', 'ford', 'volkswagen','toyota', 'dodge']:
            feature = 'is_make_%s' % v
            value = (df['make'] == v).astype(int)
            df[feature] = value
            features.append(feature)

        for v in ['regular_unleaded', 'premium_unleaded_(required)', 'premium_unleaded_(recommended)','flex-fuel_(unleaded/e85)']:
            feature = 'is_type_%s' % v
            value = (df['engine_fuel_type'] == v).astype(int)
            df[feature] = value
            features.append(feature)

        for v in ['automatic', 'manual', 'automated_manual']:
            feature = 'is_transmission_%s' % v
            value = (df['transmission_type'] == v).astype(int)
            df[feature] = value
            features.append(feature)

        for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive','four_wheel_drive']:
            feature = 'is_driven_wheels_%s' % v
            value = (df['driven_wheels'] == v).astype(int)
            df[feature] = value
            features.append(feature)

        for v in ['crossover', 'flex_fuel', 'luxury','luxury,performance', 'hatchback']:
            feature = 'is_mc_%s' % v
            value = (df['market_category'] == v).astype(int)
            df[feature] = value
            features.append(feature)

        for v in ['compact', 'midsize', 'large']:
            feature = 'is_size_%s' % v
            value = (df['vehicle_size'] == v).astype(int)
            df[feature] = value
            features.append(feature)

        for v in ['sedan', '4dr_suv', 'coupe','convertible', '4dr_hatchback']:
            feature = 'is_style_%s' % v
            value = (df['vehicle_style'] == v).astype(int)
            df[feature] = value
            features.append(feature)

        df_num = df[features]
        df_num = df_num.fillna(0)
        X = df_num.values

        return X

    return (prepare_X,)


@app.cell
def _(np):
    def train_linear_regression_reg(X, y, r=0.0):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        reg = r * np.eye(XTX.shape[0])
        XTX = XTX + reg

        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)

        return w[0], w[1:]

    return (train_linear_regression_reg,)


@app.cell
def _(df):
    df['vehicle_style'].value_counts().head(5)
    return


@app.cell
def _(X_train, train_linear_regression_reg, y_train):
    w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)
    return w, w_0


@app.cell
def _(plt, sns, y_pred, y_val):
    sns.histplot(y_pred, label='prediction')
    sns.histplot(y_val, label='target')
    plt.legend()
    return


@app.cell
def _():
    ad = {

        'city_mpg': 40,

        'driven_wheels': 'all_wheel_drive',

        'engine_cylinders': 4.0,

        'engine_fuel_type': 'regular_unleaded',

        'engine_hp': 222.0,

        'highway_mpg': 37,

        'make': 'toyota',

        'market_category': 'crossover,performance',

        'model': 'venza',

        'number_of_doors': 4.0,

        'popularity': 2031,

        'transmission_type': 'automatic',

        'vehicle_size': 'midsize',

        'vehicle_style': 'wagon',

        'year': 2025

    }
    return (ad,)


@app.cell
def _(ad, pd, prepare_X):
    df_test1 = pd.DataFrame([ad])
    X_test_1 = prepare_X(df_test1)
    return (X_test_1,)


@app.cell
def _(X_test_1, w, w_0):
    y_pred1 = w_0 + X_test_1.dot(w)
    return (y_pred1,)


@app.cell
def _(np, y_pred1):
    suggestion = np.expm1(y_pred1)

    return (suggestion,)


@app.cell
def _(suggestion):
    suggestion
    return


if __name__ == "__main__":
    app.run()
