"""
Contains commong functions useful throughout the application
"""
import os
import time


def create_if_not(path):
    """
    Creates a folder at the given path if one doesnt exist before
    :param path: destination to check for existense
    """
    if not os.path.exists(path):
        os.makedirs(path)


def print_progress(start, iteration, total, prefix=None, bar_length=25):
    if iteration > 0:
        percent = '{0:.2f}'.format(100 * (iteration / total))
        filled_length = int(bar_length * iteration // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print('\r{}|{}| {}% - {:.0f} sec'.format(prefix + ' - ' if prefix else '', bar, percent, time.time() - start),
              end='', flush=True)
