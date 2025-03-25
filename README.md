# Welcome to StackEdit!

This is NOT completed by following any tutorial, but completed and actively self-learned with the help of LLMs.

As a novice programmer and an AI/ML enthusiast I have used my opportunity to enrich my understand of machine learning technologies 
The reason why there are a lot of markdown cells is so that I understand more of why I did what I did. 

 As this project serves primarily as an exploration and learning exercise, this approach provides better context for anyone reviewing or learning from the code.

Throughout the project, I've adopted a cell-specific import strategy, where libraries are introduced at the point of use rather than consolidating all imports at the beginning. This is for self learninig/educational value for myself so I can understand which libraries are required for which specific functionality. This makes it easier to understand the dependencies for individual operations.

I've also incorporated mark down cells before to explain the function and rationale of the revelant code that comes before or after these explanatory sections. I tried my best to be concise while also explaining the relevant information to avoid underexplaining important and hard-to-understand topics, as well as not to overexplain easy to understand basic topics.  



**Technology used**:
- Visual Studio Code
- Jupyter Notebook
- Google Colab
- Github
- Chatgpt o3/Claude 3.7

AI has been utilised to learn, refactor and generate some portions of my project. 

**Libraries imported**
- Numpy
- Pandas
- Sklearn

# Files

## IMDB Movie Dataset

Movie files downloaded from the IMDB database
Files downloaded from: https://datasets.imdbws.com/
Dataset information: https://developer.imdb.com/non-commercial-datasets/


### **title.basics.tsv.gz**[![](https://developer.imdb.com/icons/anchorIcon.svg)](https://developer.imdb.com/non-commercial-datasets/#titlebasicstsvgz)

-   tconst (string) - alphanumeric unique identifier of the title
-   titleType (string) – the type/format of the title (e.g. movie, short, tvseries, tvepisode, video, etc)
-   primaryTitle (string) – the more popular title / the title used by the filmmakers on promotional materials at the point of release
-   originalTitle (string) - original title, in the original language
-   isAdult (boolean) - 0: non-adult title; 1: adult title
-   startYear (YYYY) – represents the release year of a title. In the case of TV Series, it is the series start year
-   endYear (YYYY) – TV Series end year. '\N' for all other title types
-   runtimeMinutes – primary runtime of the title, in minutes
-   genres (string array) – includes up to three genres associated with the title

### **title.crew.tsv.gz**[![](https://developer.imdb.com/icons/anchorIcon.svg)](https://developer.imdb.com/non-commercial-datasets/#titlecrewtsvgz)

-   tconst (string) - alphanumeric unique identifier of the title
-   directors (array of nconsts) - director(s) of the given title
-   writers (array of nconsts) – writer(s) of the given title

### **title.ratings.tsv.gz**[![](https://developer.imdb.com/icons/anchorIcon.svg)](https://developer.imdb.com/non-commercial-datasets/#titleratingstsvgz)

-   tconst (string) - alphanumeric unique identifier of the title
-   averageRating – weighted average of all the individual user ratings
-   numVotes - number of votes the title has received


This document provides an overview of our merged movie dataset containing information from IMDB.

## MovieLens Dataset


### **links.csv**

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

# 4. Content Based Filtering

# 5. Hybrid Filtering

# 6. GUI

include screenshots

# Markdown Explanations

My fiels come with markdown cells to explain the code in an attempt to help the reviewer as well as myself understand why I did what I did. This way I am not just blindly copy pasting the code sections which are generated by chatGPT


> **Note:** You can **bold** your text in your note section.


