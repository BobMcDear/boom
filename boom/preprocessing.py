"""
Preprocessing utilities to extract author names and descriptions from OL works.
"""


import json
import regex as re
import string
from argparse import ArgumentParser
from typing import Any, Dict

import pandas as pd


def count_leaves(tree: Any) -> int:
    """
    Counts the leaves in a tree-like list or dictionary.

    Args:
        tree: Tree whose leaves are counted.

    Returns:
        Number of leaves in tree.
    """
    if isinstance(tree, list):
        return len(tree) + sum(map(count_leaves, tree))

    elif isinstance(tree, dict):
        return len(tree) + sum(map(count_leaves, tree.values()))

    else:
        return 1


def build_ol_author_keys(path: str) -> Dict:
    """
    Builds a mapping of OL author keys to names.

    Args:
        path: Path to OL authors dump.

    Returns:
        Mapping of OL author keys to names.
    """
    key_to_name = {}

    with open(path) as file:
        for line in file:
            author = json.loads(line.split('\t')[-1])
            if 'name' in author:
                key_to_name[author['key']] = author['name']

    return key_to_name


def remove_html(input: str) -> str:
    """
    Removes HTML tags and entities from the input.

    Args:
        input: Input whose HTML tags and entities are removed.

    Returns:
        Input with its HTML tags and entities removed.
    """
    return re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', input)


def remove_markdown_links(input: str) -> str:
    """
    Removes Markdown-style links from the input.

    Args:
        input: Input whose Markdown-style links are removed.

    Returns:
        Input with its Markdown-style links removed.
    """
    return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', input)


def punc_ratio(input: str) -> float:
    """
    Calculates the punctuation-to-length ratio of the input.

    Args:
        input: Input whose punctuation-to-length ratio is calculated.

    Returns:
        Punctuation-to-length ratio of the input.
    """
    return sum([1 for c in input if c in set(string.punctuation)]) / len(input)


def extract_ol_books(
    path_authors: str,
    path_works: str,
    keep_description_html: bool = False,
    keep_description_markdown_links: bool = False,
    min_description_len: int = 640,
    max_description_len: int = 3_840,
    max_description_punc_ratio: float = 0.075,
    ) -> Dict:
    """
    Extracts author names and descriptions from OL works dump.

    Args:
        path_authors: Path to OL authors dump.
        path_works: Path to OL works dump.
        keep_description_html: Flag for keeping HTML tags and entities
            in descriptions.
        keep_description_markdown_links: Flag for keeping Markdown-style links
            in descriptions.
        min_description_len: Only works with at least this many characters
            in their descriptions are included.
        max_description_len: Only works with at most this many characters
            in their descriptions are included.
        max_description_punc_ratio: Only works whose descriptions have at most
            this punctuation-to-length ratio are included.

    Returns:
        Mapping of OL titles to author names and descriptions.
    """
    # Many works might have the same title.
    # To resolve such potential conflicts,
    # the sample with the highest number of leaves
    # in its JSON data is picked.
    # The rationale is that entries with
    # more information (leaves) associated with them
    # are more popular and should thus be prioritized.
    num_leaves = {}
    key_to_name = build_ol_author_keys(path_authors)
    books = {}

    with open(path_works) as file:
        for i, line in enumerate(file):
            if i%250_000 == 0:
                print(i)

            work = json.loads(line.split('\t')[-1])

            if 'description' in work and 'authors' in work:
                title = work['title'].lower()
                leaves = count_leaves(work)

                if title not in num_leaves or num_leaves[title] < leaves:
                    description = work['description']
                    description = (description if isinstance(description, str)
                                    else description['value'])

                    if not keep_description_html:
                        description = remove_html(description)

                    if not keep_description_markdown_links:
                        description = remove_markdown_links(description)

                    if (min_description_len <= len(description) <= max_description_len
                        and punc_ratio(description) <= max_description_punc_ratio):
                        try:
                            authors = map(lambda x: key_to_name[x['author']['key']],
                                          work['authors'])
                            books[work['title']] = {'authors': ', '.join(authors),
                                                    'description': description}
                            num_leaves[title] = leaves

                        # This exception is raised when an author's name is unknown,
                        # in which case the work is discarded.
                        except KeyError:
                            pass

    return books


def main(
    path_authors: str,
    path_works: str,
    path_books: str = 'books.csv',
    keep_description_html: bool = False,
    keep_description_markdown_links: bool = False,
    min_description_len: int = 640,
    max_description_len: int = 3840,
    max_description_punc_ratio: float = 0.075,
    ) -> None:
    """
    Converts OL works dump into a CSV file containing titles, authors,
    and descriptions of books.

    Args:
        path_authors: Path to OL authors dump.
        path_works: Path to OL works dump.
        path_books: Path to save the preprocessed books at as a CSV file.
        keep_description_html: Flag for keeping HTML tags and entities
            in descriptions.
        keep_description_markdown_links: Flag for keeping Markdown-style links
            in descriptions.
        min_description_len: Only works with at least this many characters
            in their descriptions are included.
        max_description_len: Only works with at most this many characters
            in their descriptions are included.
        max_description_punc_ratio: Only works whose descriptions have at most
            this punctuation-to-length ratio are included.
    """
    books = extract_ol_books(path_authors, path_works,
                             keep_description_html,
                             keep_description_markdown_links,
                             min_description_len, max_description_len,
                             max_description_punc_ratio)
    pd.DataFrame(books).dropna().T.to_csv(path_books, index_label=False)


if __name__ == '__main__':
    parser = ArgumentParser(description='Converts OL dumps into a CSV file\
                                         containing titles, authors, and \
                                         descriptions of books.')
    parser.add_argument('--path-authors',
                        type=str,
                        default='authors.txt',
                        help='Path to OL authors dump.')
    parser.add_argument('--path-works',
                        type=str,
                        default='works.txt',
                        help='Path to OL works dump.')
    parser.add_argument('--path-books',
                        type=str,
                        default='books.csv',
                        help='Path to save the preprocessed books at as a CSV file.')
    parser.add_argument('--keep-description-html',
                        action='store_true',
                        help='Flag for keeping HTML tags and entities\
                              in descriptions.')
    parser.add_argument('--keep-description-markdown-links',
                        action='store_true',
                        help='Flag for keeping Markdown-style links\
                              in descriptions.')
    parser.add_argument('--min-description-len',
                        type=int,
                        default=640,
                        help='Only works with at least this many characters in\
                              their descriptions are included.')
    parser.add_argument('--max-description-len',
                        type=int,
                        default=3_840,
                        help='Only works with at most this many characters in\
                              their descriptions are included.')
    parser.add_argument('--max-description-punc-ratio',
                        type=float,
                        default=0.075,
                        help='Only works whose descriptions have at most\
                              this punctuation-to-length ratio are included.')
    args = parser.parse_args()

    main(args.path_authors, args.path_works, args.path_books,
         args.keep_description_html, args.keep_description_markdown_links,
         args.min_description_len, args.max_description_len,
         args.max_description_punc_ratio)
