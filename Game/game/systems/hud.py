"""Heads up display rendering."""
from __future__ import annotations

import pygame


class HudRenderer:
    def __init__(self) -> None:
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 18)

    def render(self, surface: pygame.Surface, player, enemy_director, current_fps: float, settings) -> None:
        health_text = self.font.render(f"HP: {player.state.health:03.0f}", True, (255, 255, 255))
        armor_text = self.font.render(f"AR: {player.state.armor:03.0f}", True, (100, 200, 255))
        ammo_text = self.font.render(
            f"Ammo: {player.current_weapon.magazine}/{player.ammo_reserve.get(player.current_weapon.stats.ammo, 0)}",
            True,
            (255, 255, 120),
        )
        enemy_text = self.font.render(f"Enemies: {enemy_director.enemy_count}", True, (255, 180, 180))
        surface.blit(health_text, (10, 10))
        surface.blit(armor_text, (10, 32))
        surface.blit(ammo_text, (10, 54))
        surface.blit(enemy_text, (10, 76))
        if settings.get("show_fps"):
            fps_text = self.font.render(f"FPS: {current_fps:.0f}", True, (180, 255, 180))
            surface.blit(fps_text, (10, 98))
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.circle(surface, (255, 255, 255), mouse_pos, 4, 1)
