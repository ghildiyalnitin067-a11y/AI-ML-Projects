import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120



df = pd.read_excel("INDIA_RETAIL_DATA.xlsx")
print(df.head())

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Month"] = df["Order Date"].dt.month_name()

# Create Revenue if needed
df["Revenue"] = df["Sales"]    # Already available as 'Sales'

print("Total Revenue:", df["Revenue"].sum())
print("Total Orders:", len(df))
print("Total Quantity Sold:", df["QtyOrdered"].sum())


print(df.groupby("Product Type")["Revenue"].sum())

print(df.groupby("Product Sub-Category")["Revenue"].sum())

print(df.groupby("State")["Revenue"].sum())

print(df.groupby("Month")["Revenue"].sum())

print(df.groupby("Segment")["Profit"].sum())

plt.style.use("seaborn-v0_8")  


plt.figure(figsize=(9,5))
product_rev = df.groupby("Product Type")["Sales"].sum().sort_values(ascending=False)

sns.barplot(x=product_rev.index, y=product_rev.values, palette="Blues")
plt.title("Revenue by Product Type")
plt.xlabel("Product Type")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# by states
plt.figure(figsize=(10,5))
state_rev = df.groupby("State")["Sales"].sum().sort_values(ascending=False)

sns.barplot(x=state_rev.index, y=state_rev.values, palette="Greens")
plt.title("Revenue by State")
plt.xlabel("State")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# sale trend monthly

plt.figure(figsize=(9,5))
monthly = df.groupby("Month")["Sales"].sum()

sns.lineplot(x=monthly.index, y=monthly.values, marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# quantity vs revenue

plt.figure(figsize=(8,5))
sns.scatterplot(x=df["QtyOrdered"], y=df["Sales"], alpha=0.7)
plt.title("Quantity Ordered vs Revenue")
plt.xlabel("Quantity Ordered")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()




