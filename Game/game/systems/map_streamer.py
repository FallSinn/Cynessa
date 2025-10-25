"""Chunk streaming and procedural map data."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, Tuple, Iterable

import pygame

LOGGER = logging.getLogger(__name__)


class Chunk:
    def __init__(self, position: Tuple[int, int], tile_size: int, chunk_size: int):
        self.position = position
        self.tile_size = tile_size
        self.chunk_size = chunk_size
        self.rect = pygame.Rect(
            position[0] * tile_size * chunk_size,
            position[1] * tile_size * chunk_size,
            tile_size * chunk_size,
            tile_size * chunk_size,
        )
        self.color = pygame.Color(
            random.randint(40, 80), random.randint(40, 80), random.randint(40, 80)
        )
        self.spawn_points = [
            pygame.Vector2(
                self.rect.left + random.randint(0, self.rect.width),
                self.rect.top + random.randint(0, self.rect.height),
            )
            for _ in range(5)
        ]


class MapStreamer:
    def __init__(self, config_path: Path, resources) -> None:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.tile_size = int(config.get("tile_size", 32))
        self.chunk_size = int(config.get("chunk_size", 64))
        self.radius = int(config.get("load_radius", 2))
        self.loaded: Dict[Tuple[int, int], Chunk] = {}

    def ensure_chunks(self, position: pygame.Vector2) -> None:
        chunk_coords = self._chunk_coords(position)
        for cx in range(chunk_coords[0] - self.radius, chunk_coords[0] + self.radius + 1):
            for cy in range(chunk_coords[1] - self.radius, chunk_coords[1] + self.radius + 1):
                if (cx, cy) not in self.loaded:
                    self.loaded[(cx, cy)] = Chunk((cx, cy), self.tile_size, self.chunk_size)
        keys = list(self.loaded.keys())
        for key in keys:
            if abs(key[0] - chunk_coords[0]) > self.radius + 1 or abs(key[1] - chunk_coords[1]) > self.radius + 1:
                del self.loaded[key]

    def render(self, surface: pygame.Surface, camera_rect: pygame.Rect) -> None:
        for chunk in self.loaded.values():
            if camera_rect.colliderect(chunk.rect):
                screen_rect = chunk.rect.move(-camera_rect.left, -camera_rect.top)
                pygame.draw.rect(surface, chunk.color, screen_rect, 0)

    def is_blocked(self, position: pygame.Vector2) -> bool:
        return False

    def _chunk_coords(self, position: pygame.Vector2) -> Tuple[int, int]:
        x = int(position.x // (self.tile_size * self.chunk_size))
        y = int(position.y // (self.tile_size * self.chunk_size))
        return x, y

    def iter_spawn_points(self) -> Iterable[pygame.Vector2]:
        for chunk in self.loaded.values():
            for point in chunk.spawn_points:
                yield point
