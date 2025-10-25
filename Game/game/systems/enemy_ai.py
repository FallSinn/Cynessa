"""Enemy behaviour management."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List

import pygame

from systems.map_streamer import MapStreamer
from systems.projectiles import ProjectileManager
from systems.weapon_base import WeaponDatabase

LOGGER = logging.getLogger(__name__)


@dataclass
class Enemy:
    position: pygame.Vector2
    velocity: pygame.Vector2
    health: float
    state: str
    target: pygame.Vector2


class EnemyDirector:
    def __init__(
        self,
        map_streamer: MapStreamer,
        projectiles: ProjectileManager,
        weapons: WeaponDatabase,
    ) -> None:
        self.map_streamer = map_streamer
        self.projectiles = projectiles
        self.weapons = weapons
        self.enemies: List[Enemy] = []
        self.spawn_cooldown = 0.0

    def update(self, dt: float, player, map_streamer: MapStreamer) -> None:
        self.spawn_cooldown -= dt
        if self.spawn_cooldown <= 0:
            self.spawn_wave()
            self.spawn_cooldown = 10.0
        for enemy in list(self.enemies):
            direction = player.position - enemy.position
            distance = direction.length()
            if distance > 0:
                direction = direction.normalize()
            enemy.velocity = direction * 120
            enemy.position += enemy.velocity * dt
            if distance < 30:
                player.apply_damage(10 * dt)
            if enemy.health <= 0:
                self.enemies.remove(enemy)

    def render(self, surface: pygame.Surface, camera_rect: pygame.Rect) -> None:
        for enemy in self.enemies:
            screen_pos = enemy.position - pygame.Vector2(camera_rect.topleft)
            pygame.draw.circle(surface, (200, 80, 80), screen_pos, 12)

    def spawn_wave(self) -> None:
        for spawn in self.map_streamer.iter_spawn_points():
            if random.random() < 0.2:
                enemy = Enemy(spawn.copy(), pygame.Vector2(), 60.0, "chase", spawn.copy())
                self.enemies.append(enemy)

    def try_hit(self, position: pygame.Vector2, damage: float) -> None:
        for enemy in self.enemies:
            if enemy.position.distance_to(position) < 20:
                enemy.health -= damage

    @property
    def enemy_count(self) -> int:
        return len(self.enemies)
