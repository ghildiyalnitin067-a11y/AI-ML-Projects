import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

stocks = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"]   # add more if you want
start = "2023-01-01"
end = "2025-10-10"

data = yf.download(stocks, start=start, end=end)["Close"]
data.to_csv("Multi_Stock_Closing_Prices.csv", index=True)

data.head()
data.info()
a = data.info()
print(a)
print("CSV created: Multi_Stock_Closing_Prices.csv")

# closing price comparison plot


plt.figure(figsize=(12, 6))
for col in data.columns:
    plt.plot(data[col], label=col)

plt.title("Closing Price Trend Comparison")
plt.xlabel("Date"); plt.ylabel("Price")
plt.grid(); plt.legend()
plt.show()


# commulative price comaparison plot on 1 rupee investment
daily_returns = data.pct_change()
cumulative_returns = (1 + daily_returns).cumprod()

plt.figure(figsize=(12, 6))
for col in cumulative_returns.columns:
    plt.plot(cumulative_returns[col], label=col)

plt.title("Cumulative Returns — ₹1 Invested")
plt.xlabel("Date"); plt.ylabel("Growth of ₹1")
plt.grid(); plt.legend()
plt.show()

# summery table
summary = pd.DataFrame({
    "Mean Daily Return (%)": daily_returns.mean() * 100,
    "Volatility (%)": daily_returns.std() * 100,
    "Total Return (%)": (cumulative_returns.iloc[-1] - 1) * 100,
})
print("\n📌 STOCK PERFORMANCE SUMMARY\n")
print(summary)


best = summary["Total Return (%)"].idxmax()
worst = summary["Total Return (%)"].idxmin()

print(f"\n🔥 BEST Performer: {best} with return {round(summary.loc[best, 'Total Return (%)'], 2)} %")
print(f"❌ WORST Performer: {worst} with return {round(summary.loc[worst, 'Total Return (%)'], 2)} %")
