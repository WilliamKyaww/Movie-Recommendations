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

last cell of content based filtering:
```python
# Optional: Save the TF-IDF model and matrix for future use
# This allows you to load the model later without recalculating
import pickle

# Save the TF-IDF vectorizer
with open('movie_tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Save only features (not the full similarity matrix) to save space
# For very large datasets, consider not saving the matrix at all
sample_size_to_save = min(5000, len(df_movie))
if len(df_movie) > sample_size_to_save:
    # Save a sample of the matrix
    indices_to_save = np.random.choice(len(df_movie), size=sample_size_to_save, replace=False)
    np.save('movie_tfidf_sample_matrix.npy', tfidf_matrix[indices_to_save].toarray())
    np.save('movie_sample_indices.npy', indices_to_save)
else:
    # Save the full matrix if it's reasonably sized
    np.save('movie_tfidf_matrix.npy', tfidf_matrix.toarray())

print("Models saved for future use.")
```

## **What is TF-IDF?**  

TF-IDF (Term Frequency-Inverse Document Frequency) is a technique used to **convert text into numerical values**, making it useful for tasks like search engines, document similarity, and recommendation systems.

---

### **1. Term Frequency (TF)**  
- Measures how **frequent** a word appears in a document.
- Formula:  
  
  TF = (Number of times a word appears in a document) / (Total number of words in the document)
  
- Example:  
  - In the movie **"Action Thriller Adventure"**, the word **"Action"** appears **once**.
  - If the total words in the movie description are **3**, then:  

    TF(Action) = 1 / 3 = 0.33


### **2. Inverse Document Frequency (IDF)**  
- Measures how **important** a word is by checking how many documents contain it.
- Formula:  
  ```
  IDF = log(Total number of documents / Number of documents containing the word)
  ```
- If a word appears in **many** documents, its importance is **low** (common words like *"the"* or *"movie"* should not be weighted heavily).
- If a word appears in **fewer** documents, its importance is **high** (unique words like *"sci-fi"* are more meaningful).

---

### **Final TF-IDF Score**
The final **TF-IDF score** is calculated by multiplying **TF × IDF**:
```  
TF-IDF = TF × IDF  
```
- **High TF-IDF** means the word is **important** in this document but **rare across all documents**.
- **Low TF-IDF** means the word is **common across many documents**, so it's **less important**.

---

### **Why Use TF-IDF?**
- Filters out common words and keeps meaningful ones 
- Improves search engines
- Essential for recommendation systems

---

### **Example:**
If there are movies with **genres, directors, and writers**, TF-IDF will assign **higher importance** to unique words like *"Sci-Fi"* or *"Christopher Nolan"* while **reducing the weight** of common words like *"Drama"*.

### Cell 4:

**1. Importing TF-IDF Vectorizer**

```python 
from sklearn.feature_extraction.text import TfidfVectorizer
```
```TfidfVectorizer``` is used to convert text data (```content_features```) into a matrix of numerical values for similarity calculations.

**2. Initialising the TF-IDF Vectorizer**
   
```python
tfidf = TfidfVectorizer(stop_words='english')
```

```stop_words='english```' removes common English words (such as "the", "and", "is") that don’t add value for similarity.

**3. Transforming Content into a TF-IDF Matrix**

```python
tfidf_matrix = tfidf.fit_transform(df_movie['content_features'].fillna(''))
```

- ```fit_transform()``` learns the vocabulary and creates vectors for each movie.
- ```fillna('')``` ensures missing values (NaN) are replaced with empty strings (honestly this bit isn't needed).

**4. Printing Matrix Shape**

```python
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
```
For example, if the output is (10000, 5000), then 10000 would be the number of movies while 5000 wold be the number of unique words (features).

**5. Printing Feature Count**
```python
print(f"Number of features: {len(tfidf.get_feature_names_out())}")
```
Retrieves the number of unique words used in the TF-IDF matrix.

### Conclusion
Converts text data into numerical form, allowing us to calculate movie similarity scores.

TF-IDF gives higher weight to important words and lower weight to common words.

This matrix will be used next for cosine similarity calculations (to recommend similar movies).

# 5. Hybrid Filtering

# 6. GUI

include screenshots

# Markdown Explanations

My files come with markdown cells to explain the code in an attempt to help the reviewer as well as myself understand why I did what I did. This way I am not just blindly copy pasting the code sections which are generated by chatGPT


> **Note:** You can **bold** your text in your note section.


