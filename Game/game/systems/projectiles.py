"""Projectile spawning and pooling."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

import pygame

from systems.weapon_base import WeaponStats


@dataclass
class Projectile:
    position: pygame.Vector2
    velocity: pygame.Vector2
    ttl: float
    damage: float
    projectile_type: str


class ProjectileManager:
    def __init__(self) -> None:
        self.active: List[Projectile] = []
        self.pool: List[Projectile] = []

    def spawn_projectile(self, owner, stats: WeaponStats) -> None:
        direction = pygame.Vector2(1, 0)
        mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
        world_pos = pygame.Vector2(owner.position)
        offset = mouse_pos + owner.get_camera_offset() - world_pos
        if offset.length_squared() > 0:
            direction = offset.normalize()
        spread = random.uniform(-2, 2)
        direction.rotate_ip(spread)
        speed = 600
        if stats.projectile in {"rocket", "arc", "gravity"}:
            speed = 300
        velocity = direction * speed
        projectile = self._acquire()
        projectile.position = world_pos.copy()
        projectile.velocity = velocity
        projectile.ttl = 1.5 if stats.projectile == "hitscan" else 3.0
        projectile.damage = stats.damage
        projectile.projectile_type = stats.projectile
        self.active.append(projectile)

    def update(self, dt: float, map_streamer, enemy_director) -> None:
        for projectile in list(self.active):
            projectile.ttl -= dt
            if projectile.projectile_type == "hitscan":
                self._resolve_hitscan(projectile, enemy_director)
                projectile.ttl = 0
            else:
                projectile.position += projectile.velocity * dt
                if map_streamer.is_blocked(projectile.position):
                    projectile.ttl = 0
                else:
                    enemy_director.try_hit(projectile.position, projectile.damage)
            if projectile.ttl <= 0:
                self._release(projectile)

    def render(self, surface: pygame.Surface, camera_rect: pygame.Rect) -> None:
        for projectile in self.active:
            if projectile.projectile_type == "hitscan":
                continue
            screen_pos = projectile.position - pygame.Vector2(camera_rect.topleft)
            pygame.draw.circle(surface, (255, 255, 64), screen_pos, 3)

    # ------------------------------------------------------------------
    def _acquire(self) -> Projectile:
        if self.pool:
            return self.pool.pop()
        return Projectile(pygame.Vector2(), pygame.Vector2(), 0.0, 0.0, "hitscan")

    def _release(self, projectile: Projectile) -> None:
        if projectile in self.active:
            self.active.remove(projectile)
        self.pool.append(projectile)

    def _resolve_hitscan(self, projectile: Projectile, enemy_director) -> None:
        enemy_director.try_hit(projectile.position, projectile.damage)
