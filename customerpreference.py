import pandas as pd

df = pd.read_csv("Dataset .csv")

df = df.dropna(subset=['Cuisines', 'Aggregate rating', 'Votes'])

df['Main Cuisine'] = df['Cuisines'].astype(str).apply(lambda x: x.split(',')[0].strip())

avg_rating_by_cuisine = df.groupby('Main Cuisine')['Aggregate rating'].mean().sort_values(ascending=False)

print("\n Average Rating by Cuisine:")
print(avg_rating_by_cuisine.head(10))  

votes_by_cuisine = df.groupby('Main Cuisine')['Votes'].sum().sort_values(ascending=False)

print("\n Most Popular Cuisines (by total votes):")
print(votes_by_cuisine.head(10))  

print("\n Cuisines That Tend to Receive Higher Ratings:")
for cuisine in avg_rating_by_cuisine.head(10).index:
    rating = avg_rating_by_cuisine[cuisine]
    print(f"{cuisine}: {rating:.2f}")