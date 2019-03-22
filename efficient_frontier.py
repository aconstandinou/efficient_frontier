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
import csv

def efficient_frontier(merged_df, list_tickers, year, iterations = 25000):
    """
    args:
        merged_df: a merged dataframe of multiple stocks where 
                    index of DF = date and columns = stock prices
        list_tickers (array of strings): each index in array is a ticker (string) ex: 'MMM'
        iterations (int): number of random portfolios to compute, default 25000
        year (int): year we are evaluating
    returns:
        three dictionaries: max_sharpe_weights, max_return_weights, max_volatility_weights
                 dict keys: 'Returns', 'Volatility', 'Sharpe Ratio' and a key for every stock included ie: 'SPY weight', 'MMM weight', etc
    """
    # create empty lists for each portfolio we will generate and store data for
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
                    edgecolors='black', figsize=(12, 8), grid=True)
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    #plt.savefig('ef_{}.jpg'.format(year))
    plt.show()
    
    # find min Volatility & max sharpe values in the dataframe (df)
    min_volatility = df['Volatility'].min()
    max_sharpe = df['Sharpe Ratio'].max()
    max_return = df['Returns'].max()
    
    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    min_variance_port = df.loc[df['Volatility'] == min_volatility]
    
    # plot frontier, max sharpe & min Volatility values with a scatterplot
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(12, 8), grid=True)
    plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
    plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    #plt.savefig('ef2_{}.jpg'.format(year))
    plt.show()
    
    """
    SECONDARY PLOT DATA: find highest sharpe ratio, then the corresponding return and volatility
    """
    max_sharpe_row_ = df.iloc[df['Sharpe Ratio'].idxmax()]
    max_sharpe_val = max_sharpe_row_['Sharpe Ratio']
    max_sharpe_RTN_value = max_sharpe_row_['Returns']
    max_sharpe_VOL_value = max_sharpe_row_['Volatility']
    
    min_vol_row_ = df.iloc[df['Volatility'].idxmin()]
    min_vol_RTN_value = min_vol_row_['Returns']
    min_vol_VOL_value = min_vol_row_['Volatility']
    efficient_frontier_plot(port_returns, port_volatility, sharpe_ratio, 
                            max_sharpe_VOL_value, max_sharpe_RTN_value,
                            min_vol_VOL_value, min_vol_RTN_value, 
                            year)
    
    print("Efficient Frontier Complete.")
    
    # convert our max sharpe ratio to a dictionary
    max_sharpe_weights = max_sharpe_row_.to_dict()
    # finds the maximum Return and converts the row of data to a dictionary
    max_return_weights = df.iloc[df['Returns'].idxmax()].to_dict()
    # finds the maximum Volatility and converts the row of data to a dictionary
    max_volatility_weights = df.iloc[df['Volatility'].idxmax()].to_dict()
    min_volatility_weights = df.iloc[df['Volatility'].idxmin()].to_dict()
    
    return max_sharpe_weights, max_return_weights, max_volatility_weights, min_volatility_weights


def efficient_frontier_plot(p_rtns, p_vol, s_ratio, max_sharpe_VOL, max_sharpe_RTN, min_vol_VOL, min_vol_RTN, year):
    """
    args:
        p_rtns: list of portfolio returns, type float or integer 
                      (len(port_returns) MUST MATCH len(port_volatility))
        p_vol: list of portfolio standard deviations, type float or integer
                      (read note on length matching in 'port_returns' variable)
        s_ratio: list of sharpe ratios, type float or integer
        max_sharpe_VOL: float or integer, representing the standard deviation of your max sharpe portfolio
        max_sharpe_RTN: float or integer, representing the return value of your max sharpe portfolio
        min_vol_VOL: float or integer, representing the standard deviation of your min vol portfolio
        min_vol_RTN: float or integer, representing the return value of your min vol portfolio
        year (int): year of the data we are evaluating
    returns:
        NoneType 
    """
    plt.figure(figsize=(12,8))
    plt.scatter(p_vol, p_rtns, c=s_ratio, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.scatter(max_sharpe_VOL, max_sharpe_RTN, c='red', s=50) # red dot
    plt.scatter(min_vol_VOL, min_vol_RTN, c='blue', s=50) # red dot
    plt.title('Efficient Frontier')
    plt.savefig('efficient_frontier_{}.jpg'.format(year))
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

def write_dict_to_text(f_name, python_dict):
    """
    procedure to write python dictionary to text file .txt, comma separated between key and value
    args:
        f_name (string): desired file name
        python_dict (dict): python dictionary
    returns:
        
    """
    with open(f_name, 'w') as f:
        for key, value in python_dict.items():
            f.write('%s, %s\n' % (key, value))

def main():
    db_credential_info_p = "\\" + "database_info.txt"
    db_host, db_user, db_password, db_name = CM.load_db_credential_info(db_credential_info_p)
    conn = psycopg2.connect(host=db_host,database=db_name, user=db_user, password=db_password)
    
    
    year_list = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    
    for year_ in year_list:
        
        # find the last trading day for a given year
        curr_year = year_
        start_year = curr_year - 3
        last_tr_day_start = CM.fetch_last_day_mth(start_year, conn)
        last_tr_day_end = CM.fetch_last_day_mth(curr_year, conn)
        
        # create datetime objects of the date range    
        start_date = datetime.date(start_year,12,last_tr_day_start)
        end_date = datetime.date(curr_year,12,last_tr_day_end)
        
#        # retrieve a unique list of our tickers
#        stock_tuple = CM.load_db_tickers_start_date(start_date, conn)
#        stock_list = []
#        
#        for row in stock_tuple:
#            stock_list.append(row[0])
#            
#        # find SPY ETF index in our stock_list & remove it
#        spy_idx = stock_list.index('SPY')
#        stock_list.pop(spy_idx)
    
        # DEFINE OUR STOCK LIST FOR THE EFFICIENT FRONTIER
        #stock_list = ['SPY','IEF','GSG', 'AAPL', 'MSFT', 'MMM']
        stock_list = ['NOC', 'AAPL', 'MSFT', 'MMM']
        
        # load each ticker with the dates provided
        loaded_data = CM.load_df_stock_data_array_index_date(stock_list, start_date, end_date, conn)
        
        # merge our list of pd dataframes (each index is a DF of a stock) into one df
        merged_df = pd.concat(loaded_data, axis=1)
        daily_rtns = merged_df.pct_change()
        daily_rtns_log = np.log(1 + daily_rtns)
        correlation_df = daily_rtns_log.corr()
        sns.heatmap(correlation_df, 
                    annot = True, 
                    xticklabels=correlation_df.columns.values,
                    yticklabels=correlation_df.columns.values,
                    cmap = "Greens")
        
    
        max_sharpe_weights, max_return_weights, max_volatility_weights, min_volatility_weights = efficient_frontier(merged_df, stock_list, curr_year, iterations = 50000)
    
        # let's write our dict returned data to individual text files
        write_dict_to_text("max_sharpe/max_sharpe_weights_{}.txt".format(curr_year), max_sharpe_weights)
        write_dict_to_text("max_rtn/max_return_weights_{}.txt".format(curr_year), max_return_weights)
        write_dict_to_text("max_vol/max_volatility_weights_{}.txt".format(curr_year), max_volatility_weights)
        write_dict_to_text("min_vol/min_volatility_weights_{}.txt".format(curr_year), min_volatility_weights)
    
if __name__ == "__main__":
    main()