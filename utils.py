
import sys
import functools
import itertools
import inspect


import operator
if not hasattr(operator, '__call__'): # sys.version_info < (3,11)
    operator.__call__ = lambda obj, *args, **kwargs: obj(*args, **kwargs)
    operator.call = operator.__call__

class Args(dict):
    """Wrapper around *args, **kwargs"""
    # author: Jean-Raphael Gaglione
    def __init__(self, *args, **kwargs):
        assert sys.version_info >= (3, 6), "need python 3.6+ to preserve dict order"
        super().__init__({key:arg for key,arg in itertools.chain(enumerate(args), kwargs.items())})
    @classmethod
    def fromiterable(cls, iterable):
        "Create an Args object from an iterable over (key, value) pairs."
        ans = cls()
        for key,arg in iterable: ans[key] = arg
        return ans
    def extract(self):
        "Efficient method to get (self.args, self.kwargs)."
        args, kwargs = [], {}
        for key, arg in self.items():
            args.append(arg) if isinstance(key, int) else kwargs.setdefault(key, arg)
        return args, kwargs
    @property
    def args(self): return [arg for key,arg in self.items() if isinstance(key, int)]
    @property
    def kwargs(self): return {key:arg for key,arg in self.items() if isinstance(key, str)}
    def __getattr__(self, *args,**kwargs): return self.__getitem__(*args,**kwargs)
    def __setattr__(self, *args,**kwargs): return self.__setitem__(*args,**kwargs)
    def __delattr__(self, *args,**kwargs): return self.__delitem__(*args,**kwargs)
    def insert(self, key, arg):
        if isinstance(key, str):
            if key in self: raise KeyError(f"This key is already taken: {key!r}")
            self[key] = arg
        elif isinstance(key, int):
            if key != 0 and abs(key)-1 not in self:
                raise KeyError(f"Not enough positional arguments: {key!r}")
            args, kwargs = self.extract()
            args.insert(key, arg)
            self.clear()
            super().update(Args(*args, **kwargs))
        else:
            raise TypeError(f"Unexpected key type: {type(key)!r}")
    def pop(self, key, *default):
        if isinstance(key, int):
            args, kwargs = self.extract()
            try:
                ans = args.pop(key)
            except KeyError as err:
                if default: return default[0]
                else: raise err
            self.clear()
            super().update(Args(*args, **kwargs))
            return ans
        super().pop(key, *default)

    def items(self, argspec=None):
        "if argspec is given, iterates over (key,value,annotation)"
        if argspec is None: return super().items()
        if not isinstance(argspec, inspect.FullArgSpec):
            try: argspec = inspect.getfullargspec(argspec)
            except TypeError: argspec = None
        def gen():
            for key, arg in self.items():
                if argspec is None:
                    yield key, arg, None
                    continue
                if isinstance(key, int):
                    if key < len(argspec.args): varname = argspec.args[key]
                    else: varname = argspec.varargs
                else:
                    if key in argspec.args or key in argspec.kwonlyargs: varname = key
                    else: varname = argspec.varkw
                yield key, arg, argspec.annotations.get(varname)
        return gen()
    def map(self, func, argspec=None):
        "map func(key,value,[annotation])"
        return self.__class__.fromiterable((item[0],func(*item)) for item in self.items(argspec=argspec))
    def zip(self):
        "Assuming that every value is an iterable, return an iterable of Args."
        keys, args_iter = zip(*self.items())
        for args in zip(*args_iter):
            yield self.__class__.fromiterable(zip(keys, args))
    def zip_longest(self, fillvalue=None):
        "Similar to `Args.zip` but with the behavior of `itertools.zip_longest`."
        keys, args_iter = zip(*self.items())
        for args in itertools.zip_longest(*args_iter, fillvalue=fillvalue):
            yield self.__class__.fromiterable(zip(keys, args))
    def product(self):
        "Assuming that every value is an iterable, cartesian product of inputs (see itertools.product)."
        keys, args_iter = zip(*self.items())
        for args in itertools.product(*args_iter):
            yield self.__class__.fromiterable(zip(keys, args))
    def __repr__(self):
        args = itertools.chain(
            (f"{arg!r}" for arg in self.args),
            (f"{kw}={arg!r}" for kw,arg in self.kwargs.items()),
        )
        return f"({', '.join(args)})"