def singleton(cls, *args, **kw):

    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls(*args, **kw)
        return _instance[cls]
    
    return inner


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

