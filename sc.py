from quant23rg.implied_volatility import ImpliedVolatility
from quant23rg.riskManagement import RiskManagementOneAsset, RiskManagementPortfolio

if __name__=="__main__" : 
    print("----------------------------------------------------------------------")
    print("Welcome on the implied volatilities module ")
    print("----------------------------------------------------------------------")

    print("----------------------------------------------------------------------")
    print("A demo simulation will be launched")
    print("----------------------------------------------------------------------")

    import pandas as pd 

    spy_opt = pd.read_csv(
        "ressources/spy-options.csv",
    ).dropna()
    spy_opt.columns = [
        col + "_call" if ".1" not in col and col != "Strike" else col
        for col in spy_opt.columns
    ]
    spy_opt.columns = [col.replace(".1", "_put") for col in spy_opt.columns]
    spy_opt["IV_call"] = spy_opt["IV_call"].apply(lambda x: float(x.replace("%", "")) / 100)
    spy_opt["IV_put"] = spy_opt["IV_put"].apply(lambda x: float(x.replace("%", "")) / 100)
    spy_opt["Last_call"] = spy_opt["Last_call"].apply(lambda x: float(x))
    spy_opt["Last_put"] = spy_opt["Last_put"].apply(lambda x: float(x))
    call_example = spy_opt.sort_values("Volume_call", ascending=False).iloc[0]

    # Last_call              27.9
    # Bid_call              27.85
    # Ask_call              28.18
    # Change_call            27.9
    # Volume_call             962
    # Open Int_call         2,741
    # IV_call              0.1815
    # Last Trade_call    05/26/23
    # Strike                410.0
    # Last_put              12.45
    # Bid_put               12.24
    # Ask_put                12.4
    # Change_put            12.45
    # Volume_put            1,234
    # Open Int_put         20,292
    # IV_put               0.1831
    # Last Trade_put     05/26/23
    # Name: 47, dtype: object

    s0 = 420.02  # 29 may 2023
    dt = 146 / 252  # in years, expiration date : 20 october 2023

    ivs = ImpliedVolatility(
        s0=s0,
        strike=call_example.Strike,
        dt=dt,
        interest_rate=0.05 * dt,  # fed rate  multiplied by our period of time
        volatility=0.1326,
    )

    ivs.show_implied_vol_and_compare(
        marketPrices_call=spy_opt["Last_call"].tolist(),
        marketPrices_put=spy_opt["Last_put"].tolist(),
        strikes=spy_opt.Strike.tolist(),
        marketVols=spy_opt.IV_call.tolist(),
    )
    print("----------------------------------------------------------------------")
    print("End of the demo, bye.\n\n Author : Adrien Calas - Le23RayGorbella - le23rg@icloud.com")
    print("----------------------------------------------------------------------")


if __name__=="__main__":
    ### Datas for the test ###
    print("----------------------------------------------------------------------")
    print("Welcome on the risk management module ")
    print("----------------------------------------------------------------------")

    import yfinance as yf


    sp500 = yf.Ticker("^GSPC")
    hist_sp500 = sp500.history("1y")
    hist_sp500["returns"] = hist_sp500.Close / hist_sp500.Close.shift(1) - 1
    returns_serie = hist_sp500.dropna().returns
    
    ### One asset ### 

    rkManageHistoric = RiskManagementOneAsset(returns_serie, input_type="returns")
    rkManageNormal = RiskManagementOneAsset(
        returns_serie,
        mode="normal",
    )
    rkManageMonteCarlo = RiskManagementOneAsset(
        hist_sp500.Close, "Monte-Carlo", returns=returns_serie, input_type="value"
    )

    print(
        f"VaR Historic 1 day : {rkManageHistoric.value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )
    print(
        f"VaR Normal 1 day  : {rkManageNormal.value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )
    print(
        f"VaR Monte-Carlo 1 day : {rkManageMonteCarlo.value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )

    print(
        f"Cond-VaR Historic 1 day : {rkManageHistoric.conditional_value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )
    print(
        f"Cond-VaR Normal 1 day  : {rkManageNormal.conditional_value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )
    print(
        f"Cond-VaR Monte-Carlo 1 day : {rkManageMonteCarlo.conditional_value_at_risk(nb_past_days=len(returns_serie),confiance=.95, nb_days=1 )}"
    )

    ### Portfolio ### 

    import yfinance as yf
    import numpy as np

    ## Datas for the test
    sp500 = yf.Ticker("^GSPC")
    hist_sp500 = sp500.history("1y")
    aapl = yf.Ticker("AAPL")
    hist_aapl = aapl.history("1y")
    hist_sp500["returns"] = hist_sp500.Close / hist_sp500.Close.shift(1) - 1
    ###############################@


    pf_risk_Test = RiskManagementPortfolio.init_and_instantiate_RK_PF(
        portfolio_time_series=np.array(
            [
                hist_sp500.Close.tolist(),
                hist_aapl.Close.tolist(),
            ]
        ),
        interest_free_rate=0.05 / 252,
        mode="Monte-Carlo",
        nb_days=1,
        input_type="value",
    )
    print("Correlation matrix :")
    print(pf_risk_Test.correlation_matrix)
    print(
        "Value-at-risk 1 day Monte-Carlo",
        pf_risk_Test.value_at_risk(
            nb_past_days=1,  # not important here
            confiance=0.95,
        ),
    )

    pf_risk_Test = RiskManagementPortfolio.init_and_instantiate_RK_PF(
        portfolio_time_series=np.array(
            [
                hist_sp500.Close.tolist(),
                hist_aapl.Close.tolist(),
            ]
        ),
        interest_free_rate=0.05 / 252,
        mode="normal",
        nb_days=1,
        input_type="value",
    )
    print("Correlation matrix :")
    print(pf_risk_Test.correlation_matrix)
    print(
        "Value-at-risk 1 day normal method",
        pf_risk_Test.value_at_risk(
            nb_past_days=1,  # not important here
            confiance=0.95,
        ),
    )

    print("----------------------------------------------------------------------")
    print("End of the demo, bye.\n\n Author : Adrien Calas - Le23RayGorbella - le23rg@icloud.com")
    print("----------------------------------------------------------------------")