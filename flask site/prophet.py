# -*- coding: utf-8 -*-
from __future__ import division

import pandas as pd
import numpy as np
import pandas_datareader.data as web

from fbprophet import Prophet
import datetime
from flask import Flask, render_template
from flask import request, redirect
from pathlib import Path
import os
import os.path
import csv
from itertools import zip_longest

import pandas_datareader as web1
from pandas import Series,DataFrame


app = Flask(__name__)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
    
@app.route("/")
def first_page():
    tmp = Path("static/prophet.png")
    tmp_csv = Path("static/numbers.csv")
    if tmp.is_file():
        os.remove(tmp)
    if tmp_csv.is_file():
        os.remove(tmp_csv)
    return render_template("index.html")

#function to get stock data
def yahoo_stocks(symbol, start, end):
    return web.DataReader(symbol, 'yahoo', start, end)

def get_historical_stock_price(stock):
    print ("Getting historical stock prices for stock ", stock)
    
    startDate = datetime.datetime(2018, 1, 1)
    endDate = datetime.datetime.now().date()
    stockData = web.DataReader(stock,'yahoo', startDate, endDate)
    return stockData


@app.route("/plot" , methods = ['POST', 'GET'] )
def main():
    if request.method == 'POST':
        stock = request.form['companyname']
        df_whole = get_historical_stock_price(stock)
        
        df = df_whole.filter(['Close'])
        

        df['ds'] = df.index
        #log transform the ‘Close’ variable to convert non-stationary data to stationary.
        df['y'] = np.log(df['Close'])

        stockprices = df[['ds','Close']]

        original_end = df['Close'][-1]
        
        model = Prophet()
        model.fit(df)

        #num_days = int(input("Enter no of days to predict stock price for: "))
        
        num_days = 10
        future = model.make_future_dataframe(periods=num_days)
        forecast = model.predict(future)
        
        print (forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
        pre = forecast[['ds', 'yhat']]

        pre['yhat'] = np.exp(pre['yhat'])
        
        Stockpre = pd.merge(stockprices, pre, on ='ds')


        #Prophet plots 
        
        #vizualization
        df.set_index('ds', inplace=True)
        forecast.set_index('ds', inplace=True)
        #date = df['ds'].tail(plot_num)
        
        viz_df = df.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
        viz_df['yhat_scaled'] = np.exp(viz_df['yhat'])


        close_data = viz_df.Close
        forecasted_data = viz_df.yhat_scaled
        date = future['ds']
        forecast_start = forecasted_data[-num_days]

        d = [date, close_data, forecasted_data]
        export_data = zip_longest(*d, fillvalue = '')
        with open('static/numbers.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(("Date", "Actual", "Forecasted"))
            wr.writerows(export_data)
        myfile.close()



#Trader Sim


        Stockpre['Weekday'] = Stockpre['ds'].dt.day_name()
        
        Stockpre =  Stockpre[(Stockpre['Weekday'] == 'Monday')]

        Stockpre["tom"] = Stockpre["yhat"].shift(-1)

        yhat = delta = total = cost = 0.0
        status = "cash"
        tradecount = 0

        log = pd.DataFrame(columns=['Date', 'Action', 'Stock Price', 'Predicted Value','Profit', 'Trade Count'])
        

        for index, row in Stockpre.iterrows():
            print(index)
            print(f" On {row['ds']}")
            #check prediction value for next week "tom" vs the current ticker value
            if row['tom'] >= row['Close']:
                #If holding cash, initiate buy, set cost to current stock price 'Close' 
                #reset "current" for comparison, change status to holding stock 
                if status == "cash":
                    tradecount =  tradecount + 1
                    cost = row['Close']
                    print(f" bought stock @ {cost}: price {row['Close']} vs {row['tom']} ")
                        
                    log = log.append({'Date': row['ds'], 'Action': 'BUY', 'Stock Price': row['Close'], 'Profit' : total,
                                    'Predicted Value': row['tom'],'Trade Count': tradecount}, ignore_index=True)

                    status = "hold"
                
                elif status == "hold":
                    print(f" held stock @ cost {cost} : price {row['Close']} vs {row['tom']}")
                    log = log.append({'Date': row['ds'], 'Action': 'HOLD STOCK', 'Stock Price': row['Close'], 'Profit' : total,
                                    'Predicted Value': row['tom'],'Trade Count': tradecount}, ignore_index=True)

            elif row['tom'] < row['Close']:
                if status == "hold":
                    tradecount =  tradecount + 1
                    print(f" sold @ cost {cost} : price {row['Close']} vs {row['tom']}")           
                    delta = row['Close'] - cost
                    total = total + delta
                    status = "cash"
                    cost = 0.0
                    print(f" profit {delta}")
                        
                    log = log.append({'Date': row['ds'], 'Action': 'SELL', 'Stock Price': row['Close'], 'Profit' : total,
                                    'Predicted Value': row['tom'],'Trade Count': tradecount}, ignore_index=True)
                    
                        
                elif status == "cash" :
                    
                    print(f" held cash : price {row['Close']} vs {row['tom']}")
                                            
                    log = log.append({'Date': row['ds'], 'Action': 'HOLD CASH', 'Stock Price': row['Close'], 'Profit' : total,
                                    'Predicted Value': row['tom'],'Trade Count': tradecount}, ignore_index=True)

        if status == "hold":
            delta = final - cost
            total = total + delta 
            print(f"Final Networth :  cash {total}")
            
            log = log.append({'Date': Stockpre['ds'].iloc[-1], 'Action': 'FINAL SELL', 'Stock Price': Stockpre['Close'].iloc[-1], 'Profit' : total,
                            'Predicted Value': row['tom'],'Trade Count': tradecount}, ignore_index=True)
                        
        if status == "cash":
            print(f"Final Networth :  cash {total}")
            
            log = log.append({'Date': Stockpre['ds'].iloc[-1], 'Action': 'HELD CASH', 'Stock Price': Stockpre['Close'].iloc[-1], 'Profit' : total,
                            'Predicted Value': row['tom'],'Trade Count': tradecount}, ignore_index=True)

        log['Date'] = log['Date'].dt.strftime('%m/%d/%Y')
        log['Stock Price'] = log['Stock Price'].apply(lambda x: "${:.2f}".format((x)))
        log['Predicted Value'] = log['Predicted Value'].apply(lambda x: "${:.2f}".format((x)))
        log['Profit'] = log['Profit'].apply(lambda x: "${:.2f}".format((x)))
        
        return render_template("plot.html", original = round(original_end,2), forecast = round(forecast_start,2), stock_tinker = stock.upper(), 
            column_names=log.columns.values, row_data=list(log.values.tolist()), zip=zip)

'''
if __name__ == "__main__":
    main()
'''

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
