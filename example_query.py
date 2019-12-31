"""
An example how the transactions can be queried from permaweb by identity.
"""
from os import listdir

import requests

from config import Config

if __name__ == '__main__':
    # 1) Get the path to dataset, where the names of directories
    # are labels/identities
    path_to_dataset = Config().DATASET

    # 2) Iterate through the identities
    for identity in listdir(path_to_dataset):
        # 3) Construct the ArQL object
        # (see https://github.com/ArweaveTeam/arweave-js#arql)
        arql_obj = {
            'op': 'and',
            'expr1': {
                'op': 'equals',
                'expr1': 'Feed-Name',
                'expr2': 'VideoPresenceTracker',
            },
            'expr2': {
                'op': 'equals',
                'expr1': identity,
                'expr2': '1'
            }
        }

        print(f'Quering transactions containing videos with {identity}')

        # 4) Convert the object to JSON and send it as a POST request
        r = requests.post('https://arweave.net/arql', json=arql_obj)

        # 5) Print the transactions
        print('Transactions:')
        print(r.content)
