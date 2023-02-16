import warnings


class IgnoreWarnings(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        warnings.filterwarnings("ignore", message=f".*{self.message}.*")

    def __exit__(self, *_):
        warnings.filterwarnings("default", message=f".*{self.message}.*")