

class FlagEnv(object):
    counter = 0
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): 
            setattr(self, k, v)

    @classmethod
    def get_counter(cls): 
        current_counter = cls.counter
        cls.counter += 1
        return current_counter
     
    def __setattr__(self, name, value):
        if not hasattr(self, "_keys"):
            super().__setattr__("_keys", [])
        if name != "_keys": 
            if name not in self._keys:
                self._keys.append(name)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def __iter__(self):
        for k in self._keys: 
            yield f"--{k}", getattr(self, k)