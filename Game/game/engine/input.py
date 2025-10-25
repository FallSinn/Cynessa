"""Basic input handler with key bindings and state caching."""
from __future__ import annotations

import pygame


DEFAULT_BINDINGS = {
    "move_up": pygame.K_w,
    "move_down": pygame.K_s,
    "move_left": pygame.K_a,
    "move_right": pygame.K_d,
    "dash": pygame.K_LSHIFT,
    "interact": pygame.K_e,
}


class InputManager:
    def __init__(self) -> None:
        self.bindings = DEFAULT_BINDINGS.copy()
        self._keys = pygame.key.get_pressed()
        self._mouse_buttons = pygame.mouse.get_pressed()
        self._mouse_pos = pygame.mouse.get_pos()
        self._events = []

    def process_event(self, event: pygame.event.Event) -> None:
        self._events.append(event)

    def update(self) -> None:
        self._keys = pygame.key.get_pressed()
        self._mouse_buttons = pygame.mouse.get_pressed()
        self._mouse_pos = pygame.mouse.get_pos()
        self._events.clear()

    def is_pressed(self, action: str) -> bool:
        key = self.bindings.get(action)
        return bool(key and self._keys[key])

    @property
    def mouse_buttons(self):
        return self._mouse_buttons

    @property
    def mouse_pos(self):
        return self._mouse_pos
