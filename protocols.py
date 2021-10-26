"""
This module holds abstract interfaces and decorators for the app.
"""
from typing import Protocol, Callable, TypeVar


_funcT = TypeVar("_funcT")


# pylint: disable-msg=W0201
class Observable(Protocol):
    """Class/Protocol for defining default implementation of an observable class."""

    def __default_init_implementation__(self):
        self.callbacks: dict[str, list[Callable[[str], None]]] = {"__all_updates__": []}

    def attach(self, func: Callable[[str], None], variable: str = "__all_updates__") -> None:
        """Attach an observer function to a specific variable name.

        :param func: Function to be called when variable is updated.
        :param variable: Variable name to observe.
        :return: None.
        """
        if variable not in self.callbacks:
            self.callbacks[variable] = []

        self.callbacks[variable].append(func)

    def detach(self, func: Callable[[str], None], variable: str = "__all_updates__") -> None:
        """Detaches an observer function from a specific variable name.

        :param func: Function to be removed.
        :param variable: Variable name being observed.
        :return: None.
        """
        self.callbacks[variable].remove(func)

    def dispatch_update(self, variable: str = "__all_updates__"):
        """Default implementation for dispatching updates to observers.

        :param variable: Variable being updated.
        :return: None.
        """
        if variable not in self.callbacks:
            return

        for func in self.callbacks[variable]:
            func(variable)


def updates(*updates_args) -> Callable[[_funcT], _funcT]:
    """Decorator for dispatching updates to specified variable observers."""

    def decorator(function: _funcT) -> _funcT:

        def func(self: Observable, *args, **kwargs):
            ret_value = function(self, *args, **kwargs)
            for variable in updates_args:
                self.dispatch_update(variable)
            self.dispatch_update("__all_updates__")
            return ret_value

        return func

    return decorator


# pylint: disable-msg=W0201
class LazyInitProperty:
    """Decorator for a property that is initialized on first call instead of instantiation. Saves
    resources by only running expensive or space inefficient code when it is first needed."""

    def __init__(self, func: callable):
        self.function = func

    def __set_name__(self, owner: type, name: str):
        """Specifies a private variable for the property to store data once calculated.

        :param owner: Instance that owns the property.
        :param name: Name of the property.
        :return: None.
        """
        self.attrib_name = name
        self.attrib_private_name = f"_{name}"

    def __get__(self, instance: object, owner: type):
        """On the first call to get the property the current value is set.
        If the property wasn't set to None or a desired value then the private property is
        created and filled with the output of the specified function.

        :param instance: instance of the object that owns this property.
        :param owner: the type of the
        :return:
        """
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
