"""
Embeddings database for books.
"""


import json
from argparse import ArgumentParser
from operator import add
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from langdetect import LangDetectException, detect
from safetensors.torch import load_file, save_file
from sentence_transformers import SentenceTransformer


class EmbeddingsDatabase:
    """
    Embeddings database for books.

    Attributes:
        model_name: Name of embedding model to use.
        model: Embedding model.
        batch_size: Batch size used for computing embeddings and
            similarity searches.
        prompt_title: Prompt to prepend before the books' titles.
        prompt_description: Prompt to prepend before books' descriptions.
        title_to_ind: Dictionary mapping titles to their indices in the database.
        embeddings: Normalized embeddings of books in the database.
        titles: Titles of books in the database.
        authors: Authors of books in the database.
        descriptions: Descriptions of books in the database.
        languages: Languages of the descriptions of books in the database.
    """
    def __init__(
        self,
        model_name: str = 'intfloat/multilingual-e5-base',
        batch_size: int = 32,
        prompt_model: str = 'query: ',
        prompt_title: str = 'Title of book: ',
        prompt_description: str = 'Description of book: ',
        ) -> None:
        """
        Initializes the embedding model and database.

        Args:
            model_name: Name of embedding model to use.
            batch_size: Batch size used for computing embeddings and
                similarity searches.
            prompt_model: Prompt to prepend before the model's inputs.
            prompt_title: Prompt to prepend before the books' titles.
            prompt_description: Prompt to prepend before books' descriptions.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name,
                                         prompts={'prompt': prompt_model},
                                         default_prompt_name='prompt')
        self.batch_size = batch_size
        self.prompt_title = prompt_title
        self.prompt_description = prompt_description

        dim = self.model.get_sentence_embedding_dimension()
        self.title_to_ind = {}
        self.embeddings = torch.empty((0, dim), device=self.device)
        self.titles = []
        self.authors = []
        self.descriptions = []
        self.languages = []

    def __str__(self) -> str:
        """
        Returns a string representation of the books in the database.
        """
        books = ''
        for ind, (title, author, description) in enumerate(zip(self.titles,
                                                               self.authors,
                                                               self.descriptions)):
            books += (f'Index: {ind}\n'
                      f'Title: {title}\n'
                      f'Author: {author}\n'
                      f'Description: {description}\n')
        return books


    @property
    def device(self) -> str:
        """
        Returns the device on which the database resides.
        """
        return str(self.model.device)


    def prepend_prompts(
        self,
        titles: Optional[List[str]] = None,
        descriptions: Optional[List[str]] = None,
        ) -> List[str]:
        """
        Prepends prompts to titles and descriptions, then concatenates them.

        Args:
            titles: Titles to prepend prompts to.
            descriptions: Descriptions to prepend prompts to.

        Returns:
            Concatenated titles and descriptions with prepended prompts.
        """
        assert not (titles is None and descriptions is None), \
            'At least one of titles and descriptions must be provided.'
        assert len(titles or descriptions) == len(descriptions or titles), \
            'The number of titles and descriptions must be identical.'

        empty = len(descriptions or titles) * ['']
        titles = (empty if titles is None else
                  [self.prompt_title + title + '\n' for title in titles])
        descriptions = (empty if descriptions is None else
                        [self.prompt_description + description
                         for description in descriptions])
        return list(map(add, titles, descriptions))

    def generate_embeddings(
        self,
        titles: Optional[List[str]] = None,
        descriptions: Optional[List[str]] = None,
        ) -> torch.Tensor:
        """
        Generates embeddings for books given their titles and descriptions.

        Args:
            titles: Titles of books.
            descriptions: Descriptions of books.

        Returns:
            Normalized embeddings of books.
        """
        return self.model.encode(self.prepend_prompts(titles, descriptions),
                                 batch_size=self.batch_size,
                                 normalize_embeddings=True,
                                 convert_to_tensor=True,
                                 show_progress_bar=True)

    def add(
        self,
        titles: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        descriptions: Optional[List[str]] = None,
        ) -> None:
        """
        Adds books to the database.

        Args:
            titles: Titles of books.
            authors: Authors of books. Only the titles and descriptions are
                used for generating embeddings.
            descriptions: Descriptions of books.
        """
        assert authors is None or len(titles or descriptions) == len(authors), \
            'The number of authors and titles or descriptions must be identical.'

        embeddings = self.generate_embeddings(titles, descriptions)
        empty = len(embeddings) * [None]

        for ind, title in enumerate(titles):
            self.title_to_ind[title.lower()] = len(self.embeddings) + ind

        self.embeddings = torch.concat((self.embeddings, embeddings), dim=0)
        self.titles += titles or empty
        self.authors += authors or empty
        self.descriptions += descriptions or empty

        for description in descriptions:
            try:
                self.languages.append(detect(description))

            # langdetect can fail on some texts.
            except LangDetectException:
                self.languages.append('unk')

    def query(
        self,
        titles: Optional[List[str]] = None,
        inds: Optional[List[int]] = None,
        ) -> List[Dict]:
        """
        Queries the database given indices or titles of books.

        Args:
            titles: Titles of books to return.
                If provided, inds cannot be provided.
            inds: Indices of books to return.
                If provided,titles cannot be provided.

        Returns:
            Embeddings, titles, descriptions, and authors of the sought books.
        """
        assert (inds is None) ^ (titles is None), \
            'Exactly one of inds or titles must be provided.'

        inds = ([self.title_to_ind[title.lower()] for title in titles]
                if inds is None else inds)
        return [{'embedding': self.embeddings[ind],
                 'title': self.titles[ind],
                 'authors': self.authors[ind],
                 'description': self.descriptions[ind],
                 'language': self.languages[ind]}
                 for ind in inds]

    @torch.no_grad
    def find_nearest_neighbours(
        self,
        embeddings: torch.Tensor,
        k: int = 5,
        allowed_languages: Optional[List[str]] = None,
        ) -> torch.Tensor:
        """
        Finds the k nearest neighbours of query embeddings in the database.

        Args:
            embeddings: Embeddings whose nearest neighbours in the database
                are found. Must be normalized.
            k: Number of nearest neighbours to find.
            allowed_languages: If not None, only books whose descriptions
                are in these languages are considered.

        Returns:
            Nearest neighbours of the query embeddings in the database.
        """
        embeddings = embeddings.to(self.device)
        nearest_neighbours = []

        if allowed_languages is not None:
            allowed_languages = set(allowed_languages)
            allowed = torch.tensor([language in allowed_languages
                                    for language in self.languages],
                                    device=self.device)

        for ind in range(0, len(embeddings), self.batch_size):
            sim = torch.tensordot(embeddings[ind:ind+self.batch_size],
                                  self.embeddings,
                                  dims=([1], [1]))

            if allowed_languages is not None:
                sim *= allowed

            nearest_neighbours.append(torch.topk(sim, k, dim=-1).indices)

        nearest_neighbours = torch.concat(nearest_neighbours, dim=0)

        return [self.query(inds=inds) for inds in nearest_neighbours]

    def save(self, path: str = 'database/') -> None:
        """
        Saves the database.

        Args:
            path: Path at which to save the database.
        """
        Path(path).mkdir(parents=True)

        save_file({'embeddings': self.embeddings},
                  f'{path}/embeddings.safetensor')

        with open(f'{path}/metadata.json', 'w') as file:
            json.dump({'model_name': self.model_name,
                       'prompt_model': self.model.prompts,
                       'prompt_title': self.prompt_title,
                       'prompt_description': self.prompt_description,
                       'titles': self.titles,
                       'authors': self.authors,
                       'descriptions': self.descriptions,
                       'languages': self.languages},
                      file)

    def load(self, path: str = 'database/') -> None:
        """
        Loads a saved database.

        Args:
            path: Path containing the saved database.
        """
        self.embeddings = load_file(f'{path}/embeddings.safetensor',
                                    device=str(self.device))['embeddings']

        with open(f'{path}/metadata.json') as file:
            metadata = json.load(file)

        self.model = SentenceTransformer(metadata['model_name'],
                                         prompts=metadata['prompt_model'],
                                         default_prompt_name='prompt')
        self.prompt_title = metadata['prompt_title']
        self.prompt_description = metadata['prompt_description']
        self.titles = metadata['titles']
        self.authors = metadata['authors']
        self.descriptions = metadata['descriptions']
        self.languages = metadata['languages']

        for ind, title in enumerate(self.titles):
            self.title_to_ind[title.lower()] = ind


def main(
    path_books: str = 'books.csv',
    path_database: str = 'database/',
    model_name: str = 'intfloat/multilingual-e5-base',
    batch_size: int = 32,
    prompt_model: str = 'query: ',
    prompt_title: str = 'Title of book: ',
    prompt_description: str = 'Description of book: ',
    ) -> None:
    """
    Creates and saves an embeddings database of preprocessed OL books.

    Args:
        path_books: Path to CSV file of preprocessed books.
        path_database: Path at which to save the database.
        model_name: Name of embedding model to use.
        batch_size: Batch size used for computing embeddings and
            similarity searches.
        prompt_model: Prompt to prepend before the model's inputs.
        prompt_title: Prompt to prepend before the books' titles.
        prompt_description: Prompt to prepend before books' descriptions.
    """
    books = pd.read_csv(path_books)
    database = EmbeddingsDatabase(model_name, batch_size, prompt_model,
                                  prompt_title, prompt_description)
    database.add(list(books.index), list(books['authors']),
                 list(books['description']))
    database.save(path_database)


if __name__ == '__main__':
    parser = ArgumentParser(description='Generates embeddings of books,\
                                         saving them as a SafeTensor.')
    parser.add_argument('--path-books',
                        type=str,
                        default='books.csv',
                        help='Path to CSV file of preprocessed books.')
    parser.add_argument('--path-database',
                        type=str,
                        default='database/',
                        help='Path at which to save the database.')
    parser.add_argument('--model-name',
                        type=str,
                        default='intfloat/multilingual-e5-base',
                        help='Name of embedding model to use.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='Batch size used for computing embeddings\
                              and similarity searches.')
    parser.add_argument('--prompt-model',
                        type=str,
                        default='query: ',
                        help='Prompt to prepend before the model\'s inputs.')
    parser.add_argument('--prompt-title',
                        type=str,
                        default='Title of book: ',
                        help='Prompt to prepend before the books\' titles.')
    parser.add_argument('--prompt-description',
                        type=str,
                        default='Description of book: ',
                        help='Prompt to prepend before the books\' descriptions.')
    args = parser.parse_args()

    main(args.path_books, args.path_database, args.model_name, args.batch_size,
         args.prompt_model, args.prompt_title, args.prompt_description)
