Piece of unused code in `2. MovieLens-IMDb Merging.ipynb`, this comes after cell 2 and before cell 3

```python
# MovieLens imdbId doesn't include the "tt" prefix that IMDb uses

def map_movie_ids():
    # Dictionary mapping IMDb IDs to merged dataset rows
    imdb_to_movie = {row['tconst']: row for _, row in df_merged.iterrows()}
    
    # New dataframe with mapped data
    mapped_data = []
    for _, row in df_links.iterrows():
        # Format to match IMDb's tt0000000 format
        imdb_id = f"tt{row['imdbId']:07d}"  
        
        if imdb_id in imdb_to_movie:
            movie_data = imdb_to_movie[imdb_id]
            # Create a row with data from both sources
            mapped_data.append({
                'movieId': row['movieId'], 
                'imdbId': imdb_id,        
                'title': movie_data['primaryTitle'],
            })
    
    return pd.DataFrame(mapped_data)

```


cell 3 before:

```python
movielens_imdb_ids = set()
    
for _, row in df_links.iterrows():
    # Convert float to int to string, then format
    imdb_id = int(row['imdbId'])
    formatted_id = f"tt{imdb_id:07d}"
    movielens_imdb_ids.add(formatted_id)

# Filter the IMDb dataset
filtered_imdb = df_merged[df_merged['tconst'].isin(movielens_imdb_ids)]
```

however although it has clear step-by-step transformation it is slow for large datasets (as it uses .iterrows(), which is inefficient in pandas) and has more lines of code than necessary

so i replaced it with 

```python
# Create a set of formatted IMDb IDs
movielens_imdb_ids = set(df_links['imdbId'].apply(lambda x: f"tt{int(x):07d}"))

# Filter the IMDb dataset using the set of IMDb IDs
filtered_imdb = df_merged[df_merged['tconst'].isin(movielens_imdb_ids)]

print(f"Total IMDb movies matched with MovieLens: {len(filtered_imdb)}")
    
display(filtered_imdb.head(5))
```

Although it is slightly less readable/understandable for beginners/novice programmers like myself, it is more efficient (as it avoids .iterrows() and uses vectorized operations) and more concise (single line replaces entire loop)
