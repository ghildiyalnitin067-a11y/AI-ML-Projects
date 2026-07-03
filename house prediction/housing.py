import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Housing.csv")


print(data.head())

#basic info of data set
print(data.shape)
print(data.info())
print(data.describe())

# using specific features for house price prediction 
#likke beadroom , sale , area etc

data_set =["mainroad","guestroom","hotwaterheating","basement","airconditioning","prefarea"]


#converting yes no features of coloums into 1 and 0
for col in data_set:
    data[col] = data[col].map({"yes":1,"no":0})

data = pd.get_dummies(data, columns=["furnishingstatus"], drop_first=True)
#checking null values
print("null values",data.isnull().sum())
data.dropna(inplace=True) #removing the rows with null values

#data visualization
sns.scatterplot(x="area",y="price",data=data)
plt.title("house price vs area")
plt.xlabel("Area (sqft)")
plt.ylabel("price")
plt.show()

#heat graph
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


#histogram of price distribution
plt.figure()
sns.histplot(data["price"], bins=20)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

#bedroom vs price
plt.figure()
sns.boxplot(x="bedrooms", y="price", data=data)
plt.title("Bedrooms vs Price")
plt.show()

#bathroom vs price
plt.figure()
sns.boxplot(x="bathrooms", y="price", data=data)
plt.title("Bathrooms vs Price")
plt.show()

#stories vs price
plt.figure()
sns.boxplot(x="stories", y="price", data=data)
plt.title("Stories vs Price")
plt.show()

#parking vs price
plt.figure()
sns.boxplot(x="parking", y="price", data=data)
plt.title("Parking vs Price")
plt.show()

#pairing
sns.pairplot(
    data,
    vars=["area", "bedrooms", "bathrooms", "price"]
)
plt.show()




#training data set
X = data[
    [
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "parking",
        "prefarea",
        "furnishingstatus_semi-furnished",
        "furnishingstatus_unfurnished"
    ]
]
X = data.drop("price",axis=1)
Y = data["price"] #it focus on price now

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)

#now predicting the values of price

y_pred = model.predict(x_test)

from sklearn.metrics import mean_absolute_error,r2_score

print("R2 score :",r2_score(y_test,y_pred))

#now predicting the price of new houses

new_houses = []

while True:
    choice = input("\nDo you want to add a new house? (y/n): ").lower()
    if choice != "y":
        break

    area = int(input("Area (sqft): "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    stories = int(input("Stories: "))

    mainroad = int(input("Mainroad (1=yes, 0=no): "))
    guestroom = int(input("Guestroom (1=yes, 0=no): "))
    basement = int(input("Basement (1=yes, 0=no): "))
    hotwaterheating = int(input("Hot water heating (1=yes, 0=no): "))
    airconditioning = int(input("Air conditioning (1=yes, 0=no): "))

    parking = int(input("Parking spaces: "))
    prefarea = int(input("Preferred area (1=yes, 0=no): "))

    furnishing = input(
        "Furnishing (furnished / semi / unfurnished): "
    ).lower()

    semi = 1 if furnishing == "semi" else 0
    unfurnished = 1 if furnishing == "unfurnished" else 0

    new_houses.append([
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        parking,
        prefarea,
        semi,
        unfurnished
    ])


#prediction

if len(new_houses) > 0:
    new_houses = np.array(new_houses)
    predictions = model.predict(new_houses)

    print("\nPredicted House Prices:")
    for i, price in enumerate(predictions):
        print(f"House {i+1}: ₹{int(price)}")
else:
    print("\nNo house data entered. Prediction skipped.")

#end of the program


