"""Lightweight registry for user-friendly component factories."""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    """Named callable registry with decorator and safe lookup."""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if not key:
            raise ValueError("registry key must be non-empty")

        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if key in self._items:
                raise ValueError(f"{self.name} registry already has key {key!r}")
            self._items[key] = fn
            return fn

        return _decorator

    def build(self, key: str, **kwargs: Any) -> Any:
        if key not in self._items:
            available = ", ".join(sorted(self._items.keys())) or "<empty>"
            raise KeyError(f"Unknown {self.name} key {key!r}. Available: {available}")
        return self._items[key](**kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items.keys())
