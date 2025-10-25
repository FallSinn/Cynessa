"""Main entry point for the Cynessa portable shooter."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import pygame

from engine.display import DisplayManager
from engine.input import InputManager
from engine.audio import AudioManager
from engine.resources import ResourceManager
from systems.player import PlayerController
from systems.weapon_base import WeaponDatabase
from systems.projectiles import ProjectileManager
from systems.map_streamer import MapStreamer
from systems.enemy_ai import EnemyDirector
from systems.loot import LootManager
from systems.hud import HudRenderer
from systems.save import SaveManager

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
LOG_DIR = ROOT / "logs"
SAVE_DIR = ROOT / "saves"
MOD_DIR = ROOT / "mods"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
WEAPONS_FILE = CONFIG_DIR / "weapons.json"
MAP_FILE = CONFIG_DIR / "map_config.json"


# ---------------------------------------------------------------------------
# bootstrap logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "game.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
LOGGER = logging.getLogger("main")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return default


def load_settings() -> Dict[str, Any]:
    defaults = {
        "resolution": [1280, 720],
        "fullscreen": False,
        "vsync": True,
        "target_fps": 60,
        "music_volume": 0.8,
        "sfx_volume": 0.8,
        "language": "en",
        "show_fps": False,
    }
    settings = load_json(SETTINGS_FILE, defaults)
    # ensure defaults for missing keys
    for key, value in defaults.items():
        settings.setdefault(key, value)
    return settings


# ---------------------------------------------------------------------------
# Mod loader
# ---------------------------------------------------------------------------
def load_mods(mod_dir: Path) -> None:
    sys.path.append(str(mod_dir))
    if not mod_dir.exists():
        return
    for mod_file in mod_dir.glob("*.py"):
        try:
            __import__(mod_file.stem)
            LOGGER.info("Loaded mod module %s", mod_file.stem)
        except Exception as exc:  # pragma: no cover - diagnostic output
            LOGGER.exception("Failed to load mod %s: %s", mod_file.stem, exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    os.chdir(ROOT.parent)  # ensures relative paths for assets
    LOGGER.info("Starting Cynessa portable shooter")
    settings = load_settings()

    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()

    display = DisplayManager(settings)
    input_manager = InputManager()
    audio = AudioManager(settings)
    resources = ResourceManager(ROOT)
    weapons = WeaponDatabase(WEAPONS_FILE)
    projectiles = ProjectileManager()
    map_streamer = MapStreamer(MAP_FILE, resources)
    enemy_director = EnemyDirector(map_streamer, projectiles, weapons)
    loot_manager = LootManager(map_streamer)
    hud = HudRenderer()
    save_manager = SaveManager(SAVE_DIR, settings)

    player = PlayerController(resources, weapons, projectiles, audio)

    load_mods(MOD_DIR)

    clock = pygame.time.Clock()
    running = True
    accumulator = 0.0
    target_dt = 1.0 / float(settings.get("target_fps", 60))

    while running:
        dt = clock.tick(display.target_fps) / 1000.0
        accumulator += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            input_manager.process_event(event)

        while accumulator >= target_dt:
            accumulator -= target_dt
            running = running and update_simulation(
                target_dt,
                input_manager,
                player,
                projectiles,
                map_streamer,
                enemy_director,
                loot_manager,
                save_manager,
            )

        render_scene(
            display,
            resources,
            player,
            projectiles,
            map_streamer,
            enemy_director,
            loot_manager,
            hud,
            settings,
            clock.get_fps(),
        )

    save_manager.auto_save(player, map_streamer, enemy_director)
    pygame.quit()
    LOGGER.info("Shutdown complete")


def update_simulation(
    dt: float,
    input_manager: InputManager,
    player: PlayerController,
    projectiles: ProjectileManager,
    map_streamer: MapStreamer,
    enemy_director: EnemyDirector,
    loot_manager: LootManager,
    save_manager: SaveManager,
) -> bool:
    input_manager.update()
    player.update(dt, input_manager, map_streamer, loot_manager)
    projectiles.update(dt, map_streamer, enemy_director)
    enemy_director.update(dt, player, map_streamer)
    loot_manager.update(dt, player)
    save_manager.tick(dt, player)
    return not player.marked_for_exit


def render_scene(
    display: DisplayManager,
    resources: ResourceManager,
    player: PlayerController,
    projectiles: ProjectileManager,
    map_streamer: MapStreamer,
    enemy_director: EnemyDirector,
    loot_manager: LootManager,
    hud: HudRenderer,
    settings: Dict[str, Any],
    current_fps: float,
) -> None:
    surface = display.begin_frame()
    map_streamer.render(surface, player.camera_rect)
    loot_manager.render(surface, player.camera_rect)
    projectiles.render(surface, player.camera_rect)
    enemy_director.render(surface, player.camera_rect)
    player.render(surface)
    hud.render(surface, player, enemy_director, current_fps, settings)
    display.end_frame()


if __name__ == "__main__":
    main()
