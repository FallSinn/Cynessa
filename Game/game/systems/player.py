"""Player controller implementation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import pygame

from engine.input import InputManager
from engine.resources import ResourceManager
from systems.weapon_base import WeaponDatabase, WeaponInstance
from systems.projectiles import ProjectileManager
from systems.map_streamer import MapStreamer
from systems.loot import LootManager
from systems.weapon_base import WeaponStats
from engine.audio import AudioManager

LOGGER = logging.getLogger(__name__)


@dataclass
class PlayerState:
    position: pygame.Vector2
    velocity: pygame.Vector2
    health: float
    armor: float


class PlayerController:
    def __init__(
        self,
        resources: ResourceManager,
        weapons: WeaponDatabase,
        projectiles: ProjectileManager,
        audio: AudioManager,
    ) -> None:
        self.resources = resources
        self.weapons = weapons
        self.projectiles = projectiles
        self.audio = audio
        self.state = PlayerState(pygame.Vector2(128, 128), pygame.Vector2(), 100.0, 50.0)
        self.speed = 220.0
        self.dash_speed = 360.0
        self.camera_rect = pygame.Rect(0, 0, 640, 360)
        self.current_weapon: WeaponInstance = self.weapons.spawn("pistol")
        self.inventory = {self.current_weapon.weapon_id: self.current_weapon}
        self.ammo_reserve = {"9mm": 90}
        self.marked_for_exit = False

    # ------------------------------------------------------------------
    def update(
        self,
        dt: float,
        inputs: InputManager,
        map_streamer: MapStreamer,
        loot_manager: LootManager,
    ) -> None:
        move_vec = pygame.Vector2(0, 0)
        if inputs.is_pressed("move_up"):
            move_vec.y -= 1
        if inputs.is_pressed("move_down"):
            move_vec.y += 1
        if inputs.is_pressed("move_left"):
            move_vec.x -= 1
        if inputs.is_pressed("move_right"):
            move_vec.x += 1
        if move_vec.length_squared() > 0:
            move_vec = move_vec.normalize()
        speed = self.dash_speed if inputs.is_pressed("dash") else self.speed
        self.state.velocity = move_vec * speed
        self.state.position += self.state.velocity * dt
        self.camera_rect.center = self.state.position.xy

        self.current_weapon.update(dt, inputs, self, self.projectiles)
        map_streamer.ensure_chunks(self.state.position)
        loot_manager.handle_pickups(self)

        if self.state.health <= 0:
            LOGGER.info("Player defeated, exiting loop")
            self.marked_for_exit = True

    def render(self, surface: pygame.Surface) -> None:
        rect = pygame.Rect(0, 0, 32, 32)
        rect.center = self.camera_rect.center
        pygame.draw.rect(surface, (80, 200, 255), rect)

    # ------------------------------------------------------------------
    def apply_damage(self, amount: float) -> None:
        if self.state.armor > 0:
            mitigated = min(self.state.armor, amount * 0.6)
            self.state.armor -= mitigated
            amount -= mitigated
        self.state.health -= amount

    def heal(self, amount: float) -> None:
        self.state.health = min(100.0, self.state.health + amount)

    def add_armor(self, amount: float) -> None:
        self.state.armor = min(100.0, self.state.armor + amount)

    # ------------------------------------------------------------------
    def grant_weapon(self, weapon_id: str, stats: WeaponStats) -> None:
        if weapon_id not in self.inventory:
            self.inventory[weapon_id] = self.weapons.spawn(weapon_id)
        self.current_weapon = self.inventory[weapon_id]
        self.ammo_reserve.setdefault(stats.ammo, stats.mag * 3)

    def add_ammo(self, ammo_type: str, amount: int) -> None:
        self.ammo_reserve[ammo_type] = self.ammo_reserve.get(ammo_type, 0) + amount

    @property
    def position(self) -> pygame.Vector2:
        return self.state.position

    @property
    def world_rect(self) -> pygame.Rect:
        rect = pygame.Rect(0, 0, 32, 32)
        rect.center = self.state.position
        return rect

    def consume_ammo(self, ammo_type: str, amount: int) -> bool:
        if ammo_type == "--":
            return True
        current = self.ammo_reserve.get(ammo_type, 0)
        if current >= amount:
            self.ammo_reserve[ammo_type] = current - amount
            return True
        return False

    def get_camera_offset(self) -> Tuple[int, int]:
        return self.camera_rect.left, self.camera_rect.top
