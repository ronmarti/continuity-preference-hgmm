class CachingError(RuntimeError):
    pass


class NanError(RuntimeError):
    pass


class NotPSDError(RuntimeError):
    pass


__all__ = ["CachingError", "NanError", "NotPSDError"]
