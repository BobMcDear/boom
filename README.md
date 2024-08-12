# boom

• **[Introduction](#introduction)**<br>
• **[Installation](#installation)**<br>
• **[Preprocessing](#preprocessing)**<br>
• **[Embeddings Database](#embeddings-database)**<br>
• **[Web App](#web-app)**<br>

## Introduction

boom is a toolbox for handling feature embeddings for [Open Library (OL)](https://openlibrary.org/) books, consisting of three components:

* Preprocessing utilities that can extract clean information from [OL dumps](https://openlibrary.org/developers/dumps), including titles, author names, and descriptions.

* A minimal embeddings database for books with similarity search functionality.

* A web app for performing nearest neighbours or semantic search on the database.

Rather than being a highly-configurable and hyper-optimized platform that can leverage various backends and can scale to billions of data points, boom is a small project with few dependencies that can nevertheless get the job done in a timely fashion. For instance, in lieu of relying on bulky libraries such as [Chroma](https://github.com/chroma-core/chroma) or [Faiss](https://github.com/facebookresearch/faiss), a basic database is implemented from scratch, and nearest neighbours is done in plain PyTorch. boom can therefore be run in easily on most setups while achieving acceptable performance. In addition to the codebase, this repository also comes with preprocessed OL data for over 200,000 books as well as a pre-built database, so users can skip through the preprocessing and/or database construction phases.

## Installation

Please install boom's dependencies by executing ```pip install -r requirements.txt```. You can then directly run the scripts in [```boom/```](https://github.com/BobMcDear/boom/tree/main/boom) as indicated below or import the relevant modules.

## Preprocessing

Open Library supplies monthly dumps of its data, ranging from book metadata to user ratings and reading logs. The relevant dumps for boom are the works dump and authors dump. Given these two, the preprocessing stage extracts the titles, authors, and descriptions of OL entries, ignoring those that have unnamed authors or lack an associated description. A few data cleaning techniques are applied on the descriptions, such as the removal of HTML tags, but they are conservative and aim to minimize the amount of information discarded at the expense of potentially noisy entries.

Provided the paths to the works and authors dumps, [```boom/preprocessing.py```](https://github.com/BobMcDear/boom/blob/main/boom/preprocessing.py) will create a CSV file of titles, authors, and descriptions. The extracted data corresponding to the 2024-07-31 dumps is available at [this Hugging Face dataset](https://huggingface.co/datasets/BobMcDear/boom).

## Embeddings Database

Using the preprocessed data from the previous part, boom can construct a very basic embeddings database tailored specifically for books. It extracts the vector representations of books by embedding their titles & descriptions using [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) and can conduct batched similarity search for determining the nearest neighbours of an embedding vector. Additionally, it tracks the language of each entry using [```langdetect```](https://github.com/Mimino666/langdetect) to allow users to filter works by language since OL is a multilingual catalog.

Running [```boom/database.py```](https://github.com/BobMcDear/boom/blob/main/boom/database.py) will build a database of the preprocessed books and save it for later use. The pre-built database corresponding to the 2024-07-31 dumps is available as at [this Hugging Face dataset](https://huggingface.co/datasets/BobMcDear/boom).

## Web App

Finally, boom offers a lightweight web app implemented in Flask for interacting with the embeddings database. It supports two kinds of search: nearest neighbours search and semantic search. In nearest neighbours search, the user can look up a particular title and browse through similar works. This is done by comparing the embedding of the queried book with those of other works. Semantic search, on the other hand, generates the embedding vector of the query, essentially treating it as a book description, and yields the most similar matches.

For instance, a nearest neighbours search on [_Mathematics for Machine Learning_](https://mml-book.github.io/book/mml-book.pdf) by Deisenroth _et al._ brings up entries whose embedding vectors are most similar to the embedding of this _book_. A _semantic_ search given the query "Mathematics for Machine Learning" calculates the embedding of this _text_ and finds works whose embeddings are most similar to it. With semantic search, it is best to be descriptive since embedding models tend to struggle with short inputs.

[```boom/app.py```](https://github.com/BobMcDear/boom/blob/main/boom/app.py) launches the web app given the path to the saved database. To limit the results to certain languages, the ```--allowed-languages``` option can be set as desired.
