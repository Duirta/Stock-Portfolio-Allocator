import pandas as pd
from numpy import *
import yfinance as yf
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

companies = ['NVO','LLY','IWDA.AS','BY6.F','SON.LS']

# Extract price & r history
start_date = '2020-01-03'
end_date = '2023-11-15'
datas = array((yf.download('AAPL',start_date,end_date,progress=False)).index)

def get_stock(company_name,start_date,end_date):
    company = yf.download(company_name,start_date,end_date,progress=False)
    dates = array(company.index)
    company_price = array(company['Adj Close'])
    if len(company_price) == len(datas):
        pass
    else:
        if datas[0]==dates[0] and datas[-1]== dates[-1]:
            x_data = ((datas - datas[0])/86400000000000).astype(int)
            x_date = ((dates - datas[0])/86400000000000).astype(int)
            i_data = {j: i for i,j in enumerate(x_data)}
            i_date = {j: i for i,j in enumerate(x_date)}
            precos = {i: nan for i in x_data}
            for j in x_data:
                if j in x_date:
                    precos[j]=company_price[i_date[j]]
                else:
                    d_min = x_date[max(where(x_date<j)[0])]
                    d_max = x_date[min(where(x_date>j)[0])]
                    p_min = company_price[i_date[d_min]]
                    p_max = company_price[i_date[d_max]]
                    precos[j] = sqrt(p_min*p_max)
            company_price = array(list(precos.values()))
    return company_price

stock_price = dict()
for i, company in enumerate(companies):
    print(f'{i+1}/{len(companies)}: {company}')
    stock_company = get_stock(company,start_date,end_date)
    if len(stock_company) == len(datas):
      stock_price[company] = stock_company
df_stock = pd.DataFrame(data=stock_price, index=datas)
#df_stock.to_csv('stock_prices.csv', sep = ';', index = True)

n_moving_average = 21
if n_moving_average > 1:
    df_stock = df_stock.rolling(n_moving_average).mean().iloc[n_moving_average-1:]
    datas = datas[n_moving_average-1:]

r = dict()
for company in df_stock.columns:
    prices = log(array(df_stock[company]))
    r[company] = [prices[i]-prices[i-1] for i in range(1, len(prices))]
df_r = pd.DataFrame(data=r, index=datas[1:])

mu = array(df_r.mean())
cov = array(df_r.cov())

class PortfolioProblem(ElementwiseProblem):
    def __init__(self, mu, cov, risk_free_rate=0, **kwargs):
        super().__init__(n_var=len(df_r.columns), n_obj=2, xl=0.0, xu=1.0, **kwargs)
        self.mu = mu
        self.cov = cov
        self.risk_free_rate = 0

    def _evaluate(self, x, out, *args, **kwargs):
        exp_return = x @ self.mu
        exp_risk = sqrt(x.T @ self.cov @ x)
        sharpe = (exp_return - self.risk_free_rate) / exp_risk

        out["F"] = array([exp_risk, -exp_return])
        out["sharpe"] = sharpe

class PortfolioRepair(Repair):

    def _do(self, problem, X, **kwargs):

        for x in X:
            x.__dict__['X'][x.__dict__['X'] < 1e-3] = 0
            if x.__dict__['X'][2]/sum(x.__dict__['X']) < 0.49855 or x.__dict__['X'][1]/sum(x.__dict__['X']) < 0.3:
                for _ in range(100):
                    x.__dict__['X'] = x.__dict__['X']/sum(x.__dict__['X'])
                    x.__dict__['X'][[1,2]] = 0.599/2, 0.49855
            x.__dict__['X'] = x.__dict__['X']/sum(x.__dict__['X'])
        return X

problem = PortfolioProblem(mu, cov)
algorithm = NSGA2(repair=PortfolioRepair())
res = minimize(problem, algorithm, seed=1, verbose=False)

X, F, sharpe = res.opt.get("X", "F", "sharpe")
F = F * [1, -1]
max_sharpe = sharpe.argmax()

def media_anual(sigma, mu):
    mu_n = 252*mu
    s_n = sqrt(252)*sigma
    return (exp(mu_n+s_n**2/2)-1)*100

def std_anual(sigma, mu):
    mu_n = 252*mu
    s_n = sqrt(252*n_moving_average)*sigma
    return (sqrt((exp(s_n**2)-1)*exp(2*mu_n+s_n**2)))*100

print('\nSharpe portfolio')
for i in range(len(companies)):
    print(f'{companies[i]}: {X[max_sharpe][i]*2000:.1f}%')
print(f'\nRetorno anual: {media_anual(F[max_sharpe, 0], F[max_sharpe, 1]):.1f}%')
print(f'Risco anual: {std_anual(F[max_sharpe, 0], F[max_sharpe, 1]):.1f}%')
print(f'Sharpe ratio: {sharpe[max_sharpe][0]*sqrt(252/n_moving_average):.3f}')

plt.scatter(std_anual(F[:, 0], F[:, 1]), media_anual(F[:, 0], F[:, 1]), facecolor="none", edgecolors="blue", alpha=0.5, label="Pareto-Optimal Portfolio")
plt.scatter(std_anual(cov.diagonal() ** 0.5, mu), media_anual(cov.diagonal() ** 0.5, mu), facecolor="none", edgecolors="black", s=30, label="Asset")
plt.scatter(std_anual(F[max_sharpe, 0], F[max_sharpe, 1]), media_anual(F[max_sharpe, 0], F[max_sharpe, 1]), marker="x", s=100, color="red", label="Max Sharpe Portfolio")
plt.legend()
plt.xlabel("expected annual volatility (%)")
plt.ylabel("expected annual return (%)")
plt.grid()
plt.show()


















