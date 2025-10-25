"""Asset loading helpers with caching and graceful fallbacks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pygame

LOGGER = logging.getLogger(__name__)


class ResourceManager:
    def __init__(self, root: Path):
        self.root = root
        self.images: Dict[str, pygame.Surface] = {}
        self.sounds: Dict[str, pygame.mixer.Sound] = {}

    def image(self, name: str) -> pygame.Surface:
        if name not in self.images:
            path = self.root / "assets" / "images" / name
            self.images[name] = self._load_image(path)
        return self.images[name]

    def sound(self, name: str) -> pygame.mixer.Sound:
        if name not in self.sounds:
            path = self.root / "assets" / "sounds" / name
            self.sounds[name] = self._load_sound(path)
        return self.sounds[name]

    def _load_image(self, path: Path) -> pygame.Surface:
        if path.exists():
            try:
                return pygame.image.load(path.as_posix()).convert_alpha()
            except pygame.error as exc:  # pragma: no cover - diagnostics
                LOGGER.warning("Failed to load image %s: %s", path, exc)
        LOGGER.debug("Generating placeholder surface for %s", path)
        surface = pygame.Surface((32, 32))
        surface.fill(pygame.Color(255, 0, 255))
        return surface

    def _load_sound(self, path: Path) -> pygame.mixer.Sound:
        if path.exists():
            try:
                return pygame.mixer.Sound(path.as_posix())
            except pygame.error as exc:  # pragma: no cover
                LOGGER.warning("Failed to load sound %s: %s", path, exc)
        return pygame.mixer.Sound(buffer=b"\x00" * 4)
