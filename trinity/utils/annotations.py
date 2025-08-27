"""Some annotations."""


def Experimental(func):
    """Decorator to mark a function/class as experimental.

    Experimental features may change or be removed in future releases.
    """
    func._is_experimental = True
    return func


def Deprecated(func):
    """Decorator to mark a function/class as deprecated.

    Deprecated features should not be used in new code and may be removed in future releases.
    """
    func._is_deprecated = True
    return func
