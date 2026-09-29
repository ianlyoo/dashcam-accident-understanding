import os


def revision_work(base):
    revision = os.environ.get('S172_REVISION', '')
    if revision not in ('', 'revision2', 'revision3'):
        raise ValueError('Unknown S172 revision: ' + revision)
    return base / revision if revision else base
