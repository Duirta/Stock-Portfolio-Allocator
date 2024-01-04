from levy import *
import yfinance as yf
from scipy.stats import levy_stable
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import interpolate
from pykalman import KalmanFilter
from time import perf_counter

# Ler lista de companies
f = open('Stocks.txt')
f = f.readlines()
companies = [i.replace('\n','') for i in f]

# Extract price & r history
start_date = '2016-01-01'
end_date = '2021-01-01'
datas = array((yf.download('AAPL',start_date,end_date,progress=False)).index)

def get_stock(company_name,start_date,end_date):
  company = yf.download(company_name,start_date,end_date,progress=False)
  datas = array(company.index)
  company_price = array(company['Adj Close'])
  y = log10(company_price)
  r = array([y[n]-y[n-1] for n in range(1,len(y)) if str(y[n]-y[n-1]) != 'nan'])
  return company_price, r

if 'stock_prices.csv' in os.listdir():
  df_stock = pd.read_csv('stock_prices.csv', sep = ';', index_col=0)
  bol = False
else:
  stock_price = dict()
  for i, company in enumerate(companies):
    print(f'{i+1}/{len(companies)}: {company}')
    stock_company, r_company = get_stock(company,start_date,end_date)
    if len(stock_company) == len(datas):
      stock_price[company] = stock_company
  df_stock = pd.DataFrame(data=stock_price, index=datas)
  df_stock.to_csv('stock_prices.csv', sep = ';', index = True)

r = dict()
for company in df_stock.columns:
  prices = log(array(df_stock[company]))
  r[company] = [prices[i]-prices[i-1] for i in range(1, len(prices))]
df_r = pd.DataFrame(data=r, index=datas[1:])
df_r.to_csv('stock_r.csv', sep = ';', index = True)

def random_w(size):
    w = random.uniform(size=(size,))
    w = w/sum(w)
    return w

def return_distribution(w, time = 1, n_simulations = 1500):
    x = array(df_stock.iloc[0])
    price_evolution = array([sum(i) for i in [w/x*df_stock.iloc[i] for i in range(df_stock.shape[0])]])
    r_w = log(price_evolution[1:]/price_evolution[:-1]) 
    alpha, beta, mu, c = fit_levy(r_w, mu = mean(r_w))[0].get()

    time *= 252

    returns = array([(exp(sum(levy_stable.rvs(alpha,beta,mu,c,size=(time,))))-1)*100 for i in range(n_simulations)])
    
    return r_w, alpha, beta, mu, c, returns

n_simulations = 20
lista = []
tempos = []
for i in range(1, n_simulations+1):
  init = perf_counter()
  w = random_w(df_stock.shape[1])
  w[argsort(w)[:-20]]=0
  w = w/sum(w)

  r_w, alpha, beta, mu, c, returns = return_distribution(w)

  media_anual = (exp(252*mu)-1)*100
  risco = sum(returns < 0)/len(returns)
  lista.append([alpha,beta,mu,c,risco])
  print(i, media_anual, risco, c, perf_counter()-init)
  tempos.append(perf_counter()-init)

df_risco = pd.DataFrame(lista, columns = ['alpha','beta','mu','c','Risco'])
df_risco.to_csv('risco.csv', sep = ';', index = True)

x_alpha = [1.39423397, 1.43119823, 1.45318794, 1.47075627, 1.48653173, 1.50180071, 1.51757617, 1.5351445 , 1.55713421, 1.59409847]
x_beta = [-0.13698385, -0.10759157, -0.09010637, -0.07613684, -0.06359292, -0.05145173, -0.03890781, -0.02493829, -0.00745309,  0.02193919]
x_mu = linspace(-0.0004,0.00115,11)
x_c = [0.00434918, 0.00478501, 0.00504428, 0.00525142, 0.00543742, 0.00561745, 0.00580345, 0.00601059, 0.00626986, 0.00670568]

n_simulations = 2000
for i_alpha in x_alpha:
  for i_beta in x_beta:
    for i_mu in x_mu:
      for i_c in x_c:
        returns = array([(exp(sum(levy_stable.rvs(alpha,beta,mu,c,size=(252,))))-1)*100 for i in range(n_simulations)])
        risco = sum(returns < 0)/len(returns)
        lista.append([i_alpha,i_beta,i_mu,i_c,risco])
        df_risco = pd.DataFrame(lista, columns = ['alpha','beta','mu','c','Risco'])
        df_risco.to_csv('risco.csv', sep = ';', index = True)

















