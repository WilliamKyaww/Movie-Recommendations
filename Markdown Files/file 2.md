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
