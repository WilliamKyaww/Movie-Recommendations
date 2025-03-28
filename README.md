# Welcome to my Movie Recommendation System!

This project was not completed by following a tutorial but was actively self-learned with the help of LLMs.

That being said, the AI generated sections of the code has been rigously checked, understood, documented and re-factored multiple times to my satisfaction. This ensures I fully understand what the code cell does and why, and avoids unnecessary noise sometimes created by LLMs. 

As a novice programmer and AI/ML enthusiast, I took this opportunity to deepen my understanding of machine learning technologies, techniques, and approaches. 


Additionally, I included markdown explanations before relevant code sections. These help me document my thought process and reasoning, as welll as providing better context for anyone reviewing the code. I aimed to be concise, to avoid over-explaining simple concepts while ensuring clarity for more complex topics.

I used a cell-specific import strategy, introducing libraries where needed instead of consolidating them at the beginning. This helps me understand which libraries are required for specific functions and makes dependencies clearer.



**Technology used**:
- Visual Studio Code
- Jupyter Notebook
- Google Colab
- Github
- Chatgpt o3/Claude 3.7

AI has been utilised to learn, refactor and generate some portions of my project. 

**Libraries imported**
- Os
- Numpy
- Pandas
- Sklearn

# Files

## IMDB Movie Dataset

Movie files downloaded from the IMDB database
Files downloaded from: https://datasets.imdbws.com/
Dataset information: https://developer.imdb.com/non-commercial-datasets/


### **title.basics.tsv.gz**
-   tconst (string) - alphanumeric unique identifier of the title
-   titleType (string) – the type/format of the title (movie, short, tvseries, tvepisode, video, etc)
-   primaryTitle (string) – the more popular title; the title used by the filmmakers on promotional materials at the point of release
-   originalTitle (string) - original title, in the original language
-   isAdult (boolean) - 0: non-adult title; 1: adult title
-   startYear (YYYY) – represents the release year of a title or series start year of a TV Series
-   endYear (YYYY) – TV Series end year. '\N' for all other title types
-   runtimeMinutes – primary runtime in minutes
-   genres (string array) – up to three genres 

### **title.crew.tsv.gz**
-   tconst (string) - alphanumeric unique identifier
-   directors (array of nconsts) - director(s) of the given title
-   writers (array of nconsts) – writer(s) of the given title

### **title.ratings.tsv.gz**
-   tconst (string) - alphanumeric unique identifier
-   averageRating – weighted average of all the individual user ratings
-   numVotes - number of votes the title has received


This document provides an overview of our merged movie dataset containing information from IMDB.

## MovieLens Dataset


Regarding the dataset I used from MovieLens, I downloaded the small "100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users file", as it is "recommended for education and development", which suits my purpose.

Files downloaded from: 
https://grouplens.org/datasets/movielens/


### **links.csv**

- movieId (string) - alphanumeric unique identifier of the title
- imdbId (array of nconsts) - director(s) of the given title
- tmdbId (array of nconsts) – writer(s) of the given title


### **tags.csv**

### **ratings.csv**



# 1. Data Cleaning

include tables



The dataset combines movie basics, ratings, and crew information with the following columns:

| Column | Description |
|--------|-------------|
| tconst | Unique identifier for each title |
| isAdult | Flag indicating adult content (0 = no, 1 = yes) |
| startYear | Year of release |
| genres | Comma-separated list of genres |
| runtimeMinutes | Length of the movie in minutes |
| averageRating | Average user rating (scale of 1-10) |
| directors | Comma-separated list of director IDs |
| writers | Comma-separated list of writer IDs |

## Sample Data

Below is a sample of the merged dataset:

| tconst | isAdult | startYear | genres | runtimeMinutes | averageRating | directors | writers |
|--------|---------|-----------|--------|---------------|--------------|-----------|---------|
| tt0002130 | 0 | 1911 | Adventure,Drama,Fantasy | 71 | 7.0 | nm0078205,nm0655824,nm0209738 | nm0019604 |
| tt0002423 | 0 | 1919 | Biography,Drama,Romance | 113 | 6.6 | nm0523932 | nm0266183,nm0473134 |
| tt0002844 | 0 | 1913 | Crime,Drama | 54 | 6.9 | nm0275421 | nm0019855,nm0275421,nm0816232 |
| tt0003014 | 0 | 1913 | Drama | 96 | 7.0 | nm0803705 | nm0472236,nm0803705 |
| tt0003037 | 0 | 1913 | Crime,Drama | 61 | 6.9 | nm0275421 | nm0019855,nm0275421,nm0816232 |
| tt0003165 | 0 | 1913 | Crime,Drama,Mystery | 90 | 6.9 | nm0275421 | nm0019855,nm0275421,nm0816232 |

## Notes

- Director and writer IDs (starting with "nm") can be cross-referenced with the IMDB name database to get actual names
- The dataset has been cleaned and merged from multiple IMDB source files
- All adult content is flagged for appropriate filtering

# 2. Data Merging 

include tables

# 3. Collaborative Filtering

include graphs

cell 3 original code:

```python

from numpy.linalg import norm

# Compute cosine similarity manually
def cosine_similarity(movie1, movie2):
    dot_product = np.dot(movie1, movie2)
    norm_product = norm(movie1) * norm(movie2)
    return dot_product / norm_product if norm_product != 0 else 0

# Create similarity matrix based on the training set
num_movies = train_array.shape[1]
similarity_matrix_train = np.zeros((num_movies, num_movies))

for i in range(num_movies):
    for j in range(num_movies):
        similarity_matrix_train[i, j] = cosine_similarity(train_array[:, i], train_array[:, j])

# Convert to DataFrame
movie_similarity_train_df = pd.DataFrame(similarity_matrix_train, index=train_matrix.columns, columns=train_matrix.columns)

'''
The above cell block takes around 30 minutes to run which is far too long, while the code below uses the cosine_similarity function from the sklearn library. This does the exact same thing while only taking just over a second to execute. 

My original code manually computes cosine similarity using loops, which is inefficient for large datasets.
cosine_similarity(train_array.T) from sklearn performs the same computation in a highly optimised way using matrix operations.

```

# 4. Content Based Filtering

# 5. Hybrid Filtering

# 6. GUI

include screenshots

# Markdown Explanations

My files come with markdown cells to explain the code in an attempt to help the reviewer as well as myself understand why I did what I did. This way I am not just blindly copy pasting the code sections which are generated by chatGPT


> **Note:** You can **bold** your text in your note section.


