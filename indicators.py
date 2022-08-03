import numpy as np
import pandas as pd
import pandas_ta as ta


def get_custom01_strategy(df):
    CustomStrategy = ta.Strategy(
        name="Momo and Volatility",
        description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
        ta=[
            {"kind": "sma", "length": 50},
            {"kind": "sma", "length": 200},
            {"kind": "bbands", "length": 20},
            {"kind": "rsi"},
            {"kind": "macd", "fast": 8, "slow": 21},
            {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
        ],
    )
    return CustomStrategy

def get_custom02_strategy():
    custom02_strategy = ta.Strategy(
        name="custom02_strategy",
        description="custom02_strategy",
        ta=[
            {"kind": "bbands", "length": 20},
            {"kind": "rsi"},
            {"kind": "macd", "fast": 8, "slow": 21},
            {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
            {"kind": "stoch"},
            {"kind": "stochrsi"},
            {"kind": "willr"},
        ],
    )
    return custom02_strategy

def get_momentum_strategy():
    momentum_strategy = ta.Strategy(
        name="momentum_strategy",
        description="momentum_strategy",
        ta=[
            {"kind": "ao"},
            {"kind": "apo"},
            {"kind": "bias"},
            {"kind": "bop"},
            {"kind": "brar"},
            {"kind": "cci"},
            {"kind": "cfo"},
            {"kind": "cg"},
            {"kind": "cmo"},
            {"kind": "coppock"},
            {"kind": "cti"},
            {"kind": "dm"},
            {"kind": "er"},
            {"kind": "eri"},
            {"kind": "fisher"},
            {"kind": "inertia"},
            {"kind": "kdj"},
            {"kind": "kst"},
            {"kind": "macd"},
            {"kind": "mom"},
            {"kind": "pgo"},
            {"kind": "ppo"},
            {"kind": "psl"},
            {"kind": "pvo"},
            # {"kind": "qqe"},
            {"kind": "roc"},
            {"kind": "rsi"},
            {"kind": "rsx"},
            {"kind": "rvgi"},
            {"kind": "stc"},
            {"kind": "slope"},
            {"kind": "smi"},
            {"kind": "squeeze"},
            {"kind": "squeeze_pro"},
            {"kind": "stoch"},
            {"kind": "stochrsi"},
            # {"kind": "td_seq"},
            {"kind": "trix"},
            {"kind": "tsi"},
            {"kind": "uo"},
            {"kind": "willr"},
        ],
    )
    return momentum_strategy

def get_3EMA_strategy(length1=5, length2=25, length3=100):
    EMA3_strategy = ta.Strategy(
        name="3EMA_strategy",
        description="3EMA_strategy",
        ta=[
            {"kind": "ema", "length": length1},
            {"kind": "ema", "length": length2},
            {"kind": "ema", "length": length3},
        ],
    )
    return EMA3_strategy

def get_all_strategy():
    return ta.AllStrategy

def add_all_strategy(df):
    df.ta.strategy(ta.AllStrategy)
    return df

def get_common_strategy():
    return ta.CommonStrategy

def add_common_strategy(df):
    df.ta.strategy(ta.CommonStrategy)
    return df



if __name__ == "__main__":
    print("Indicator Functions")
    # modified_BTCUSDT_df = add_all_strategy(BTCUSDT_df)
    # modified_BTCUSDT_df = add_common_strategy(BTCUSDT_df)
    # print(ta.CommonStrategy)
    # print(modified_BTCUSDT_df.head())
    # print(modified_BTCUSDT_df.columns)
