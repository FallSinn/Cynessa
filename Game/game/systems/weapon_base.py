"""Weapon database and runtime behaviour."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import pygame

from engine.input import InputManager
from systems.projectiles import ProjectileManager


@dataclass
class WeaponStats:
    weapon_id: str
    type: str
    ammo: str
    mag: int
    damage: float
    fire_rate: float
    reload: float
    projectile: str
    special: str = ""


class WeaponInstance:
    def __init__(self, stats: WeaponStats):
        self.weapon_id = stats.weapon_id
        self.stats = stats
        self.fire_timer = 0.0
        self.reload_timer = 0.0
        self.magazine = stats.mag

    def update(
        self,
        dt: float,
        inputs: InputManager,
        owner,
        projectile_manager: ProjectileManager,
    ) -> None:
        self.fire_timer = max(0.0, self.fire_timer - dt)
        self.reload_timer = max(0.0, self.reload_timer - dt)
        if self.reload_timer > 0:
            return
        if self.magazine <= 0 and self.stats.mag > 0:
            if owner.consume_ammo(self.stats.ammo, self.stats.mag):
                self.magazine = self.stats.mag
                self.reload_timer = self.stats.reload
            return
        mouse_pressed = inputs.mouse_buttons[0]
        if mouse_pressed and self.fire_timer <= 0:
            self.fire_timer = 1.0 / max(0.01, self.stats.fire_rate)
            self.magazine = max(0, self.magazine - 1)
            projectile_manager.spawn_projectile(owner, self.stats)


class WeaponDatabase:
    def __init__(self, weapon_path: Path):
        with weapon_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.stats: Dict[str, WeaponStats] = {}
        for entry in data:
            type_value = entry.get("type") or entry.get(".type") or "unknown"
            stats = WeaponStats(
                weapon_id=entry["id"],
                type=type_value,
                ammo=entry["ammo"],
                mag=int(entry.get("mag", 0)),
                damage=float(entry.get("damage", 0)),
                fire_rate=float(entry.get("fire_rate", 1.0)),
                reload=float(entry.get("reload", 1.0)),
                projectile=entry.get("projectile", "hitscan"),
                special=entry.get("special", ""),
            )
            self.stats[stats.weapon_id] = stats

    def spawn(self, weapon_id: str) -> WeaponInstance:
        return WeaponInstance(self.stats[weapon_id])

    def get(self, weapon_id: str) -> WeaponStats:
        return self.stats[weapon_id]
