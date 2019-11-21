class MissingFlagsError(Exception):
    def __init__(self, method, missing_flags):
        self.method = method
        self.missing_flags = missing_flags

    def __str__(self):
        return (
            f'For the method "{self.method}" the flag(s) {self.missing_flags} is required.'
        )
