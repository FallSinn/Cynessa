"""Audio manager for music and sound effects."""
from __future__ import annotations

import logging
from typing import Dict

import pygame

LOGGER = logging.getLogger(__name__)


class AudioManager:
    def __init__(self, settings: Dict[str, object]):
        self.music_volume = float(settings.get("music_volume", 0.8))
        self.sfx_volume = float(settings.get("sfx_volume", 0.8))
        pygame.mixer.music.set_volume(self.music_volume)
        self.sounds = {}

    def load_sound(self, identifier: str, path: str) -> None:
        try:
            sound = pygame.mixer.Sound(path)
            sound.set_volume(self.sfx_volume)
            self.sounds[identifier] = sound
        except pygame.error as exc:  # pragma: no cover - file I/O error reporting
            LOGGER.warning("Unable to load sound %s: %s", path, exc)

    def play_sound(self, identifier: str) -> None:
        sound = self.sounds.get(identifier)
        if sound:
            sound.play()

    def play_music(self, path: str, loops: int = -1) -> None:
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play(loops=loops)
        except pygame.error as exc:  # pragma: no cover
            LOGGER.warning("Failed to play music %s: %s", path, exc)
