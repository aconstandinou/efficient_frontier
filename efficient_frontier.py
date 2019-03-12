# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 18:55:42 2019

@author: antonio constandinou

Purpose of this code is to build an efficient frontier from our stock data in a given year
"""

import common_methods as CM
import pandas as pd
import numpy as np
import psycopg2
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def efficient_frontier(merged_df, list_tickers, iterations = 25000):
    """
    return XXXXXX
    args:
        merged_df: a merged dataframe of multiple stocks where 
                    index of DF = date and columns = stock prices
        list_tickers (array of strings): each index in array is a ticker (string) ex: 'MMM'
        iterations (integer): number of random portfolios to compute, default 25000
    returns:
        XXXXXXXXXXXXXXX
    """
    
    port_returns = []
    port_volatility = []
    sharpe_ratio = []
    stock_weights = []
    
    num_assets = len(merged_df.columns)
    num_portfolios = iterations
    
    # calculate daily and annual returns of the stocks
    daily_rtns = merged_df.pct_change()
    annual_rtns = daily_rtns.mean() * 252
    
    # get daily and covariance of returns of the stock
    cov_daily = daily_rtns.cov()
    cov_annual = cov_daily * 252

    # populate the empty lists with each portfolios returns,risk and weights
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        # divides each weight by the initial sum of all weights to normalize it back to a total of 1.0
        # /= is equal to / and =
        weights /= np.sum(weights)
        # computes sum of weights * it's respective annual_rtn
        # np.dot keeps the order: weight[0] * annual_rtns[0] + weight[1] * annual_rtns[1]
        returns = np.dot(weights, annual_rtns)
        # computes portfolio std deviation
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        # sharpe ratio typically needs to include risk free rate: Return - Rf / Volatility
        sharpe = returns / volatility
        # store all data into our lists
        sharpe_ratio.append(sharpe)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)
    
    # a dictionary for Returns and Risk values of each portfolio
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility,
                 'Sharpe Ratio': sharpe_ratio}
    
    # extend original dictionary to accomodate each ticker and weight in the portfolio
    # enumerate of a list will provide 0 -> element0, 1 -> element1, 2 -> element2
    # this allows us to then iterate through each weight and extract the proper index for our symbol
    for counter, symbol in enumerate(list_tickers):
        # for each weight (np array) in stock_weights (list of np arrays), extract out the appropriate
        # weight for each stock
        portfolio[symbol+' weight'] = [weight[counter] for weight in stock_weights]
    
    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio)
    
    # get better labels for desired arrangement of columns
    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' weight' for stock in list_tickers]
    
    # reorder dataframe columns
    df = df[column_order]
    print(df.head())

    # scatterplot frontier, max sharpe & min volatility values
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns',
                    edgecolors='black', figsize=(12, 6), grid=True)
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    plt.savefig('ef.jpg')
    plt.show()
    
    # find min Volatility & max sharpe values in the dataframe (df)
    min_volatility = df['Volatility'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    
    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    min_variance_port = df.loc[df['Volatility'] == min_volatility]
    
    # plot frontier, max sharpe & min Volatility values with a scatterplot
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(12, 6), grid=True)
    plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
    plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    plt.savefig('ef2.jpg')
    plt.show()
    
def unique_list_of_tickers(conn):
    """
    return a list of stock tickers
    args:
        conn: a Postgres DB connection object
    returns:
        list of tickers
    """
    cur = conn.cursor()
    SQL =   """
            SELECT DISTINCT(ticker) FROM symbol
            """
    # execute the SQL query
    cur.execute(SQL)
    # fetch our data from the cursor object
    data = cur.fetchall()
    # initialize an empty list
    ticker_list = []
    # iterate through list of tuples to extract our ticker
    for ticker in data:
        ticker_list.append(ticker[0])

    return ticker_list

def main():
    db_credential_info_p = "\\" + "database_info.txt"
    db_host, db_user, db_password, db_name = CM.load_db_credential_info(db_credential_info_p)
    conn = psycopg2.connect(host=db_host,database=db_name, user=db_user, password=db_password)
    
    # find the last trading day for a given year
    curr_year = 2010
    last_tr_day_start = CM.fetch_last_day_mth(curr_year-1, conn)
    last_tr_day_end = CM.fetch_last_day_mth(curr_year+3, conn)
    # create datetime objects of the date range    
    start_date = datetime.date(curr_year-1,12,last_tr_day_start)
    end_date = datetime.date(curr_year,12,last_tr_day_end)
    # retrieve a unique list of our tickers
    stock_tuple = CM.load_db_tickers_start_date(start_date, conn)
    stock_list = []
    for row in stock_tuple:
        stock_list.append(row[0])
    # find SPY ETF index in our stock_list & remove it
    spy_idx = stock_list.index('SPY')
    stock_list.pop(spy_idx)

    """test - override"""
    #stock_list = ['SPY','IEF','GSG', 'AAPL', 'MSFT', 'MMM']
    stock_list = ['NOC', 'AAPL', 'MSFT', 'MMM']
    # load each ticker with the dates provided
    loaded_data = CM.load_df_stock_data_array_index_date(stock_list, start_date, end_date, conn)
    # merge our list of pd dataframes (each index is a DF of a stock) into one df
    merged_df = pd.concat(loaded_data, axis=1)
    daily_rtns = merged_df.pct_change()
    correlation_df = daily_rtns.corr()
    sns.heatmap(correlation_df, 
                annot = True, 
                xticklabels=correlation_df.columns.values,
                yticklabels=correlation_df.columns.values,
                cmap = "Greens")
    print(merged_df.head(5))
    
    """
    NEED TO WORK ON EFFICIENT FRONTIER: error with all stocks ~ 506 and also
         not producing the expected efficient frontier
         FIXED: error was loading all stocks for a given year, but some stocks had no data
                stock_tuple = CM.load_db_tickers_start_date(start_date, conn)
    """
    efficient_frontier(merged_df, stock_list, 50000)
    

    
if __name__ == "__main__":
    main()