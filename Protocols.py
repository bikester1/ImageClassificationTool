from typing import Protocol, Callable, TypeVar
from abc import abstractmethod


_funcT = TypeVar("_funcT")


class Observable(Protocol):
    
    @property
    def callbacks(self) -> dict[str, list[Callable[[str], None]]]:
        """ Dictionary with a list of callbacks for each variable name """
        if "_callbacks" not in self.__dict__:
            self._callbacks = {"__all_updates__": []}
        
        return self._callbacks
    
    def attach(self, func: Callable[[str], None], variable: str = "__all_updates__") -> None:
        if variable not in self.callbacks:
            self.callbacks[variable] = []
        
        self.callbacks[variable].append(func)
    
    def detach(self, func: Callable[[str], None], variable: str = "__all_updates__") -> None:
        self.callbacks[variable].remove(func)
    
    def dispatch_update(self, variable: str = "__all_updates__"):
        if variable not in self.callbacks:
            return
        
        for func in self.callbacks[variable]:
            func(variable)


def updates(*updates_args) -> Callable[[_funcT], _funcT]:
    
    def decorator(function: _funcT) -> _funcT:
        
        def func(self: Observable, *args, **kwargs):
            ret_value = function(self, *args, **kwargs)
            for variable in updates_args:
                self.dispatch_update(variable)
            self.dispatch_update("__all_updates__")
            return ret_value
        
        return func
    
    return decorator


class Dynamic(Protocol):
    
    @property
    @abstractmethod
    def update_callbacks(self) -> dict[str:callable]:
        pass


class lazy_init_property:
    def __init__(self, func: callable):
        self.function = func
    
    def __set_name__(self, owner: type, name: str):
        self.attrib_name = name
        self.attrib_private_name = f"_{name}"
    
    def __get__(self, instance: object, owner: type):
        
        if self.attrib_private_name not in instance.__dict__:
            instance.__setattr__(self.attrib_private_name, None)
        
        current_value = instance.__getattribute__(self.attrib_private_name)
        if current_value is None:
            value = self.function(instance)
            setattr(instance, self.attrib_private_name, value)
            return value
        
        return current_value
    
    def __set__(self, instance: object, value: any):
        setattr(instance, self.attrib_private_name, value)
