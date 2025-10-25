"""JSON save management."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


class SaveManager:
    def __init__(self, save_dir: Path, settings: Dict[str, Any]) -> None:
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.timer = 0.0

    def tick(self, dt: float, player) -> None:
        self.timer += dt
        if self.timer >= 30.0:
            self.auto_save(player, None, None)
            self.timer = 0.0

    def auto_save(self, player, map_streamer, enemy_director) -> None:
        save_data = {
            "player": {
                "position": [player.position.x, player.position.y],
                "health": player.state.health,
                "armor": player.state.armor,
                "ammo": player.ammo_reserve,
                "current_weapon": player.current_weapon.weapon_id,
            },
            "settings": self.settings,
        }
        save_path = self.save_dir / "autosave.json"
        with save_path.open("w", encoding="utf-8") as handle:
            json.dump(save_data, handle, indent=2)
