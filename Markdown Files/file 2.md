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


Same for cell 6, original code:

```python

# First, create a mapping between MovieLens IDs and IMDb IDs
movielens_to_imdb = {}
for _, row in df_links.iterrows():
    imdb_id = f"tt{int(row['imdbId']):07d}"
    movielens_to_imdb[row['movieId']] = imdb_id

# Create a DataFrame from this mapping
id_mapping = pd.DataFrame({
    'movieId': list(movielens_to_imdb.keys()),
    'tconst': list(movielens_to_imdb.values())
})

# Merge the tags with the mapping
tags_with_imdb = pd.merge(tags_grouped, id_mapping, on='movieId', how='inner')

# Merge with the filtered IMDb dataset
result = pd.merge(filtered_imdb, tags_with_imdb[['tconst', 'tags']], on='tconst', how='left')

# Fill NaN tags with an empty string
result['tags'] = result['tags'].fillna('')

# Save the result
output_file = os.path.join(script_dir, "Cleaned Datasets", "Final_Movie_Data.tsv")
result.to_csv(output_file, sep="\t", index=False)

print(f"Added tags to movie data. Saved to: {output_file}")
display(result.head())

```
Evaluating the original code:
❌ Uses .iterrows(), which is slow
❌ Manually creates a dictionary and converts it to a DataFrame
✅ Correct logic but inefficient implementation

Refactored code:

```python

# Convert IMDb IDs to formatted strings directly in the DataFrame
df_links['tconst'] = df_links['imdbId'].apply(lambda x: f"tt{int(x):07d}")

# Create a mapping DataFrame
id_mapping = df_links[['movieId', 'tconst']]

# Merge the tags with the mapping
tags_with_imdb = pd.merge(tags_grouped, id_mapping, on='movieId', how='inner')

# Merge with the filtered IMDb dataset
result = pd.merge(filtered_imdb, tags_with_imdb[['tconst', 'tags']], on='tconst', how='left')

# Fill NaN tags with an empty string
result['tags'] = result['tags'].fillna('')

# Save the result
output_file = os.path.join(script_dir, "Cleaned Datasets", "Final_Movie_Data.tsv")
result.to_csv(output_file, sep="\t", index=False)

print(f"Added tags to movie data. Saved to: {output_file}")
display(result.head())


```
Code evaluation:

✅ Avoids .iterrows() → Uses apply(), which is faster
✅ Avoids creating a dictionary manually → Directly processes df_links
✅ Faster execution → Works better for large datasets

# Code Optimization: Mapping MovieLens IDs to IMDb IDs

## Comparison of Loop vs. Vectorized Approach

| Feature          | Original (Loop)                | Optimized (`apply`)            |
|-----------------|--------------------------------|--------------------------------|
| **Performance** | Slow (loops through rows)     | Fast (vectorized operations)  |
| **Code Simplicity** | Creates dictionary manually | Directly modifies DataFrame  |
| **Lines of Code** | More lines                   | Fewer lines                  |
| **Scalability** | Struggles with large data     | Handles large data efficiently |

## Explanation of Changes
### **Original Approach (Using Loop)**
- Used `.iterrows()`, which is slow because it iterates through each row individually.
- Manually created a dictionary to map `movieId` to `tconst` (IMDb ID).
- Converted the dictionary into a DataFrame before merging.

### **Optimized Approach (Using `apply()`)**
- Directly applied a lambda function to format IMDb IDs inside the DataFrame.
- Eliminated the need for a dictionary by leveraging pandas' built-in functions.
- Achieved better scalability and performance, especially for large datasets.

## **Why This Matters?**
✅ **Faster Execution** – Avoiding loops significantly speeds up processing.  
✅ **Cleaner Code** – Reduced unnecessary steps and manual dictionary creation.  
✅ **Better Scalability** – Handles large datasets more efficiently.  

By implementing these optimizations, the code becomes more efficient and runs significantly faster, making it suitable for handling larger movie datasets.
