import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Dataset .csv")

df = df.dropna(subset=['Aggregate rating', 'Cuisines', 'City', 'Votes', 'Price range'])

df['Cuisine'] = df['Cuisines'].astype(str).apply(lambda x: x.split(',')[0].strip())

sns.set(style='whitegrid')

plt.figure(figsize=(8, 5))
plt.hist(df['Aggregate rating'], bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Aggregate Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Restaurants")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Aggregate rating', data=df, palette='pastel')
plt.title("Count of Restaurants per Rating")
plt.xlabel("Aggregate Rating")
plt.ylabel("Count")
plt.show()

top_cuisines = df['Cuisine'].value_counts().head(5).index
df_cuisine = df[df['Cuisine'].isin(top_cuisines)]

plt.figure(figsize=(8, 5))
sns.barplot(x='Cuisine', y='Aggregate rating', data=df_cuisine, ci=None, palette='muted')
plt.title("Average Rating by Cuisine")
plt.xlabel("Cuisine")
plt.ylabel("Average Rating")
plt.show()

top_cities = df['City'].value_counts().head(5).index
df_city = df[df['City'].isin(top_cities)]

plt.figure(figsize=(8, 5))
sns.barplot(x='City', y='Aggregate rating', data=df_city, ci=None, palette='coolwarm')
plt.title("Average Rating by City")
plt.xlabel("City")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Votes', y='Aggregate rating', data=df)
plt.title("Votes vs Rating")
plt.xlabel("Votes")
plt.ylabel("Aggregate Rating")
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(x='Price range', y='Aggregate rating', data=df)
plt.title("Rating Across Price Ranges")
plt.xlabel("Price Range")
plt.ylabel("Aggregate Rating")
plt.show()