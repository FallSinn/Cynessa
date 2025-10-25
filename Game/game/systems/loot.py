"""Loot generation and handling."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List

import pygame

from systems.weapon_base import WeaponDatabase

LOGGER = logging.getLogger(__name__)


@dataclass
class LootPickup:
    position: pygame.Vector2
    loot_type: str
    amount: float
    weapon_id: str | None = None


class LootManager:
    def __init__(self, map_streamer) -> None:
        self.map_streamer = map_streamer
        self.pickups: List[LootPickup] = []
        self.spawn_timer = 0.0

    def update(self, dt: float, player) -> None:
        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self.spawn_loot()
            self.spawn_timer = 8.0
        for pickup in list(self.pickups):
            if pickup.position.distance_to(player.position) < 20:
                if pickup.loot_type == "health":
                    player.heal(pickup.amount)
                elif pickup.loot_type == "armor":
                    player.add_armor(pickup.amount)
                elif pickup.loot_type == "ammo" and pickup.weapon_id:
                    stats = player.weapons.get(pickup.weapon_id)
                    player.add_ammo(stats.ammo, int(pickup.amount))
                self.pickups.remove(pickup)

    def render(self, surface: pygame.Surface, camera_rect: pygame.Rect) -> None:
        for pickup in self.pickups:
            screen_pos = pickup.position - pygame.Vector2(camera_rect.topleft)
            color = (80, 255, 120)
            if pickup.loot_type == "armor":
                color = (120, 180, 255)
            elif pickup.loot_type == "ammo":
                color = (255, 220, 120)
            pygame.draw.circle(surface, color, screen_pos, 8)

    def handle_pickups(self, player) -> None:
        # handled inside update to keep logic centralised
        pass

    def spawn_loot(self) -> None:
        for spawn in self.map_streamer.iter_spawn_points():
            if random.random() < 0.1:
                loot_type = random.choice(["health", "armor", "ammo"])
                amount = 20 if loot_type != "ammo" else 30
                weapon_id = None
                if loot_type == "ammo":
                    weapon_id = random.choice(list(player_weapon_ids()))
                pickup = LootPickup(spawn.copy(), loot_type, amount, weapon_id)
                self.pickups.append(pickup)


def player_weapon_ids() -> List[str]:
    return [
        "pistol",
        "smg",
        "assault",
        "shotgun",
        "sniper",
        "rocket",
    ]
