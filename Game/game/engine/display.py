"""Display subsystem wrapping the pygame window and frame pacing."""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import pygame

LOGGER = logging.getLogger(__name__)


class DisplayManager:
    """Manage display creation, frame begin/end and vsync toggling."""

    def __init__(self, settings: Dict[str, object]):
        resolution = settings.get("resolution", [1280, 720])
        self.fullscreen = bool(settings.get("fullscreen", False))
        self.vsync = bool(settings.get("vsync", True))
        self.target_fps = int(settings.get("target_fps", 60))
        flags = pygame.SCALED
        if self.fullscreen:
            flags |= pygame.FULLSCREEN
        if self.vsync:
            flags |= pygame.SCALED  # pygame respects vsync through driver settings
        LOGGER.info(
            "Initialising display %sx%s fullscreen=%s vsync=%s", *resolution, self.fullscreen, self.vsync
        )
        self.surface = pygame.display.set_mode(resolution, flags, vsync=1 if self.vsync else 0)
        pygame.display.set_caption("Cynessa Portable Shooter")
        self.background_color = pygame.Color(12, 12, 18)

    def begin_frame(self) -> pygame.Surface:
        self.surface.fill(self.background_color)
        return self.surface

    def end_frame(self) -> None:
        pygame.display.flip()

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.surface.get_size()
