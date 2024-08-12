"""
Search engine web app for looking up books and similar entries.
"""


from argparse import ArgumentParser

from flask import Flask, render_template, request

from database import EmbeddingsDatabase


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    k = app.config.get('k')
    allowed_languages = app.config.get('allowed_languages')

    if 'nearest-neighbour-search' in request.form:
        try:
            embeddings = database.query([query])[0]['embedding'].unsqueeze(0)

        # This exception is raised when a title does not exist.
        except KeyError:
            embeddings = None

    elif 'semantic-search' in request.form:
        embeddings = database.generate_embeddings(descriptions=[query])

    else:
        return 'Invalid button', 400

    results = (None if embeddings is None else
               database.find_nearest_neighbours(embeddings,
                                                k=k,
                                                allowed_languages=allowed_languages)[0])
    return render_template('results.html', query=query, results=results)


if __name__ == '__main__':
    parser = ArgumentParser(description='Runs the search engine web app.')
    parser.add_argument('--path-database',
                        type=str,
                        default='database/',
                        help='Path containing the saved database')
    parser.add_argument('--k',
                        type=int,
                        default=5,
                        help='Number of nearest neighbours to find.')
    parser.add_argument('--allowed-languages',
                        nargs='+',
                        type=str,
                        default=None,
                        help='If not None, only books whose descriptions are\
                              in these languages are considered.')
    args = parser.parse_args()

    database = EmbeddingsDatabase()
    database.load(args.path_database)

    app.config['k'] = args.k
    app.config['allowed_languages'] = args.allowed_languages
    app.run()
