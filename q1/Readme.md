
## Jaccard coefficient between a given query and the document


Jaccard Coefficient = Intersection of (doc,query) / Union of (doc,query)

The high the value of the Jaccard coefficient, the more the document is relevant for the query.


```bash
  • Use the same data given in assignment 1 and carry out the same preprocessing steps as mentioned
before.
  • To calculate this make set of the document token and query token and perform intersection and union
between the query and each document.
  • Report the top 5 relevant documents based on the value of the Jaccard coefficient.s
```

##  TF-IDF matrix and obtain a TF-IDF score of query

```bash
• Computing Term Frequency involves calculating the raw count of the word in each document and
stored as a nested dictionary for each document.

• To calculate the document frequency of each word, find the postings list of each word and subsequently
find the no. of documents in each posting list of each word.

• The IDF value of each word is calculated using the formula as mention below:
Using smoothing:-
IDF(word)=log(total no. of documents/document frequency(word)+1)

```
