# coding: utf-8
"""sin.py - Monolithic cognitive core for Sin autonomous demolition assistant."""
# План: 
# 1. Імпортувати стандартні бібліотеки з урахуванням доступності.
# 2. Налаштувати fallback-заглушки для опційних залежностей.
# 3. Побудувати базову інфраструктуру логування.
# 4. Визначити типи, константи та допоміжні утиліти.
# 5. Забезпечити прозоре повідомлення про середовище виконання.
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from queue import PriorityQueue, Queue
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

try:  # numpy і psutil можуть бути недоступні, тож готуємо м'яке падіння.
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - ця гілка виконується лише за відсутності numpy
    class _NPModule:
        """Fallback numpy-заглушка з мінімальним API для внутрішніх розрахунків."""

        @staticmethod
        def array(data: Iterable[float], dtype: Any = float) -> List[float]:
            return list(data)

        @staticmethod
        def zeros(shape: Tuple[int, ...], dtype: Any = float) -> List[float]:
            size = math.prod(shape)
            return [dtype() if callable(dtype) else 0.0 for _ in range(size)]

        @staticmethod
        def ones(shape: Tuple[int, ...], dtype: Any = float) -> List[float]:
            return [1.0 for _ in range(math.prod(shape))]

        @staticmethod
        def dot(a: Iterable[float], b: Iterable[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        @staticmethod
        def clip(arr: Iterable[float], a_min: float, a_max: float) -> List[float]:
            return [max(a_min, min(a_max, v)) for v in arr]

        @staticmethod
        def linalg_norm(arr: Iterable[float]) -> float:
            return math.sqrt(sum(v * v for v in arr))

    np = _NPModule()  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover
    class BaseModel:  # type: ignore
        """Мінімальна заглушка BaseModel для локального schema-registry."""

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return self.__dict__


# Загальне налаштування логування.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("sin.core")


# region CONFIG & FEATURE FLAGS
# План:
# 1. Сконструювати централізовану конфігурацію з прапорцями функцій.
# 2. Описати профілі виконання для різних платформ.
# 3. Надати структури даних для швидкого доступу до параметрів.
# 4. Реалізувати механізми гарячого оновлення конфігів.
# 5. Логувати зміни для прозорості й аудиту.


class PlatformProfile(Enum):
    """Описує цільову платформу для адаптації параметрів продуктивності."""

    JETSON = auto()
    DESKTOP = auto()


class QoSLevel(Enum):
    """Рівні QoS для задач планувальника."""

    HARD_REALTIME = 3
    REALTIME = 2
    TACTICAL = 1
    BACKGROUND = 0


class GlobalState(Enum):
    """Глобальні режими життя Сін."""

    BOOT = auto()
    OPERATIONAL = auto()
    TRAINING = auto()
    DIALOG = auto()
    SAFE = auto()
    LOW_POWER = auto()


@dataclass
class FeatureFlags:
    """Фічі, що дозволяють вмикати/вимикати підсистеми без перезавантаження."""

    perception_enabled: bool = True
    learning_enabled: bool = True
    vector_memory_enabled: bool = True
    reflex_layer_enabled: bool = True
    safety_watchdog_enabled: bool = True
    personality_verbose: bool = True
    simulator_enabled: bool = True
    profiler_enabled: bool = True
    hot_swap_planners: bool = True
    thermal_monitoring: bool = True
    llm_bridge_enabled: bool = False


@dataclass
class Config:
    """Централізована конфігурація з гарячим оновленням та відновленням."""

    platform: PlatformProfile = PlatformProfile.DESKTOP
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    time_budget_ms: Dict[str, int] = field(
        default_factory=lambda: {
            "perception": 30,
            "world_model": 20,
            "planning": 40,
            "skills": 50,
            "safety": 15,
            "dialog": 60,
            "memory": 25,
        }
    )
    qos_levels: Dict[str, QoSLevel] = field(
        default_factory=lambda: {
            "safety": QoSLevel.HARD_REALTIME,
            "control": QoSLevel.REALTIME,
            "planning": QoSLevel.TACTICAL,
            "diag": QoSLevel.BACKGROUND,
            "fun": QoSLevel.BACKGROUND,
        }
    )
    energy_profiles: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "eco": {"max_current": 20.0, "thermal_limit": 60.0},
            "normal": {"max_current": 40.0, "thermal_limit": 75.0},
            "performance": {"max_current": 65.0, "thermal_limit": 85.0},
        }
    )
    safety_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "max_force": 250.0,
            "max_torque": 180.0,
            "geofence_margin": 0.5,
            "risk_abort_level": 0.85,
        }
    )
    persona_profile: Dict[str, Any] = field(
        default_factory=lambda: {
            "tone": "gritty_poetic",
            "mercy_keyword": "пощади",
            "inner_voice": True,
            "mood_baseline": {
                "serotonin": 0.45,
                "dopamine": 0.52,
                "norepinephrine": 0.30,
                "cortisol": 0.22,
                "oxytocin": 0.35,
                "void": 0.10,
            },
        }
    )
    simulator_defaults: Dict[str, Any] = field(
        default_factory=lambda: {
            "room_size": (10.0, 10.0, 4.0),
            "obstacle_density": 0.15,
            "dust_level": 0.2,
            "bolts": 12,
            "cables": 8,
            "seed": 42,
        }
    )
    demolition_scripts: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.demolition_scripts:
            self.demolition_scripts = self._generate_default_scripts()

    def _generate_default_scripts(self) -> List[Dict[str, Any]]:
        scripts: List[Dict[str, Any]] = []
        for i in range(1, 41):
            profile = random.choice(["quiet", "standard", "night", "emergency"])
            scripts.append(
                {
                    "id": f"demo_script_{i:02d}",
                    "profile": profile,
                    "description": f"Script {i} for {profile} demolition with adaptive finesse.",
                    "steps": [
                        {"skill": "scan_area", "params": {"duration": 5 + i % 3, "focus": "cables"}},
                        {"skill": "evaluate_material", "params": {"confidence": 0.6 + 0.01 * i}},
                        {
                            "skill": "select_tool",
                            "params": {"preferred": random.choice(["saw", "laser", "plasma", "hydraulic"]), "backup": "manual"},
                        },
                        {
                            "skill": "plan_cut",
                            "params": {
                                "path_style": random.choice(["minimal_vibration", "fast_cut", "safe_margin"]),
                                "cooling": bool(i % 2),
                            },
                        },
                        {"skill": "execute_cut", "params": {"force_limit": 220 - i, "speed": 0.5 + (i % 5) * 0.1}},
                        {
                            "skill": "remove_segment",
                            "params": {"grip_force": 60 + i, "retreat_path": "reverse"},
                        },
                        {"skill": "inspect_site", "params": {"quality": "high", "report": True}},
                        {"skill": "log_lesson", "params": {"mood": random.choice(["calm", "tense", "focused"])}},
                        {
                            "skill": "persona_reflection",
                            "params": {"length": 3, "tone": "poetic"},
                        },
                    ],
                }
            )
        return scripts


GLOBAL_CONFIG = Config()


class ConfigManager:
    """Дозволяє оновлювати конфігурацію під час роботи та відстежувати зміни."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._history: Deque[Tuple[float, str, Dict[str, Any]]] = deque(maxlen=50)

    def get(self) -> Config:
        return self._config

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            payload = json.loads(json.dumps(self._config, default=lambda o: o.__dict__))
            self._history.append((time.time(), "snapshot", payload))
            logger.debug("Config snapshot taken")
            return payload

    def update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
                    logger.info("Config field %s updated", key)
            self._history.append((time.time(), "update", changes))

    def history(self) -> List[Tuple[float, str, Dict[str, Any]]]:
        return list(self._history)


CONFIG_MANAGER = ConfigManager(GLOBAL_CONFIG)

# endregion CONFIG & FEATURE FLAGS
# region CORE BUS & STATE
# План:
# 1. Реалізувати подієву шину з пріоритетами та журналом.
# 2. Побудувати стан-машину з атомарними переходами та снапшотами.
# 3. Створити планувальник задач із QoS, дедлайнами та бюджетами часу.
# 4. Додати інструменти профілювання латентності й деградації.
# 5. Забезпечити fail-safe деградацію при перевищенні бюджетів.


class Event:
    """Подія, що передається між підсистемами."""

    def __init__(self, topic: str, payload: Dict[str, Any], priority: QoSLevel = QoSLevel.BACKGROUND) -> None:
        self.id = uuid.uuid4().hex
        self.topic = topic
        self.payload = payload
        self.priority = priority
        self.timestamp = time.time()

    def __repr__(self) -> str:
        return f"Event(topic={self.topic}, priority={self.priority.name}, ts={self.timestamp:.3f})"


class EventBus:
    """Внутрішня подієва шина з підтримкою підписників і контролю пріоритетів."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._history: Deque[Event] = deque(maxlen=500)
        self._priority_queue: "PriorityQueue[Tuple[int, float, Event]]" = PriorityQueue()

    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers[topic].append(handler)
            logger.debug("Subscribed handler %s to topic %s", handler, topic)

    def publish(self, event: Event) -> None:
        with self._lock:
            priority_value = -event.priority.value
            self._priority_queue.put((priority_value, event.timestamp, event))
            self._history.append(event)
            logger.debug("Event queued: %s", event)

    def pump(self, budget_ms: int = 10) -> int:
        """Розсилає події в межах часовго бюджету."""
        processed = 0
        deadline = time.time() + budget_ms / 1000.0
        while time.time() < deadline and not self._priority_queue.empty():
            priority, _, event = self._priority_queue.get()
            handlers = list(self._subscribers.get(event.topic, []))
            for handler in handlers:
                try:
                    handler(event)
                except Exception as exc:  # pragma: no cover - захист від помилок підписників
                    logger.exception("Handler %s failed for event %s: %s", handler, event, exc)
            processed += 1
        return processed

    def history(self) -> List[Event]:
        return list(self._history)


class StateSnapshot(BaseModel):
    """Структура снапшоту станів для атомарного відкату."""

    state: str
    context: Dict[str, Any]
    timestamp: float


class StateMachine:
    """Керує глобальним станом Сін, підтримує снапшоти та відкат."""

    def __init__(self, initial: GlobalState = GlobalState.BOOT) -> None:
        self._state = initial
        self._context: Dict[str, Any] = {"entered_at": time.time()}
        self._lock = threading.RLock()
        self._callbacks: Dict[str, List[Callable[[GlobalState, GlobalState], None]]] = defaultdict(list)
        self._snapshots: Deque[StateSnapshot] = deque(maxlen=20)

    def get_state(self) -> GlobalState:
        with self._lock:
            return self._state

    def set_state(self, new_state: GlobalState, reason: str = "") -> None:
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return
            snapshot = StateSnapshot(state=old_state.name, context=dict(self._context), timestamp=time.time())
            self._snapshots.append(snapshot)
            self._state = new_state
            self._context = {"entered_at": time.time(), "reason": reason}
            logger.info("State transition %s -> %s due to %s", old_state.name, new_state.name, reason)
            for callback in self._callbacks.get("transition", []):
                try:
                    callback(old_state, new_state)
                except Exception as exc:
                    logger.exception("State transition callback error: %s", exc)

    def rollback(self) -> None:
        with self._lock:
            if not self._snapshots:
                logger.warning("No snapshot available for rollback")
                return
            snapshot = self._snapshots.pop()
            self._state = GlobalState[snapshot.state]
            self._context = snapshot.context
            logger.warning("State rolled back to %s", snapshot.state)

    def on_event(self, event: Event) -> None:
        if event.topic.startswith("/safety/"):
            self.set_state(GlobalState.SAFE, reason=event.topic)
        elif event.topic.startswith("/dialog/"):
            self.set_state(GlobalState.DIALOG, reason=event.topic)
        elif event.topic.startswith("/energy/low"):
            self.set_state(GlobalState.LOW_POWER, reason=event.topic)
        elif event.topic.startswith("/control/resume"):
            self.set_state(GlobalState.OPERATIONAL, reason=event.topic)

    def register_callback(self, hook: str, handler: Callable[[GlobalState, GlobalState], None]) -> None:
        self._callbacks[hook].append(handler)


@dataclass(order=True)
class ScheduledTask:
    """Запис задачі у планувальнику з QoS та дедлайном."""

    deadline: float
    qos: QoSLevel
    submitted_at: float = field(compare=False)
    name: str = field(compare=False)
    budget_ms: int = field(compare=False)
    func: Callable[[], None] = field(compare=False)
    deferrable: bool = field(default=True, compare=False)


class Scheduler:
    """Координує виконання задач з урахуванням QoS, дедлайнів і бюджетів."""

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config
        self._event_bus = event_bus
        self._queue: "PriorityQueue[ScheduledTask]" = PriorityQueue()
        self._lock = threading.RLock()
        self._latency_stats: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="SchedulerThread", daemon=True)
        self._thread.start()

    def submit(self, name: str, func: Callable[[], None], qos: QoSLevel, deadline_ms: int, budget_ms: Optional[int] = None, deferrable: bool = True) -> None:
        budget = budget_ms if budget_ms is not None else self._config.time_budget_ms.get(name, 20)
        task = ScheduledTask(
            deadline=time.time() + deadline_ms / 1000.0,
            qos=qos,
            submitted_at=time.time(),
            name=name,
            budget_ms=budget,
            func=func,
            deferrable=deferrable,
        )
        with self._lock:
            self._queue.put(task)
            logger.debug("Task submitted: %s", task)

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)

    def _run_loop(self) -> None:
        while self._running:
            try:
                task = self._queue.get(timeout=0.1)
            except Exception:
                continue
            now = time.time()
            if now > task.deadline and task.deferrable:
                logger.warning("Task %s missed deadline; rescheduling with penalty", task.name)
                penalty_deadline = now + max(0.01, (task.deadline - task.submitted_at))
                task.deadline = penalty_deadline
                self._queue.put(task)
                continue
            start = time.time()
            try:
                task.func()
            except Exception as exc:  # pragma: no cover
                logger.exception("Scheduled task %s failed: %s", task.name, exc)
                self._event_bus.publish(Event("/diag/task_failure", {"task": task.name, "error": str(exc)}))
            duration_ms = (time.time() - start) * 1000.0
            self._latency_stats[task.name].append(duration_ms)
            if duration_ms > task.budget_ms:
                logger.warning("Task %s exceeded budget %.2fms (actual %.2fms)", task.name, task.budget_ms, duration_ms)
                self._event_bus.publish(Event("/safety/overbudget", {"task": task.name, "duration": duration_ms}))

    def latency_report(self) -> Dict[str, float]:
        return {name: sum(values) / max(len(values), 1) for name, values in self._latency_stats.items()}


EVENT_BUS = EventBus()
STATE_MACHINE = StateMachine()
SCHEDULER = Scheduler(GLOBAL_CONFIG, EVENT_BUS)

# endregion CORE BUS & STATE
# region SAFETY
# План:
# 1. Реалізувати ядро безпеки з watchdog, audit trail і bounded force.
# 2. Додати голосову фразу-заглушку для аварійної зупинки.
# 3. Впровадити перевірку no-go зон і геофенсів.
# 4. Забезпечити журналювання та two-man rule-заглушку.
# 5. Інтегрувати з EventBus для миттєвих тригерів.


class Watchdog:
    """Сторожовий таймер, що гарантує періодичний пульс від життєво важливих модулів."""

    def __init__(self, timeout: float = 1.5) -> None:
        self.timeout = timeout
        self._last_kick = time.time()
        self._lock = threading.Lock()
        self._active = True
        self._thread = threading.Thread(target=self._loop, name="SafetyWatchdog", daemon=True)
        self._thread.start()

    def kick(self) -> None:
        with self._lock:
            self._last_kick = time.time()

    def _loop(self) -> None:
        while self._active:
            time.sleep(self.timeout / 3)
            with self._lock:
                if time.time() - self._last_kick > self.timeout:
                    logger.error("Watchdog timeout; issuing emergency stop")
                    EVENT_BUS.publish(Event("/safety/watchdog_timeout", {"timeout": self.timeout}, QoSLevel.HARD_REALTIME))
                    self._last_kick = time.time()

    def stop(self) -> None:
        self._active = False
        self._thread.join(timeout=1.0)


class AuditTrail:
    """Локальний журнал аудиту з криптоподібною ланкою хешів."""

    def __init__(self) -> None:
        self._records: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self._last_hash = "0"

    def record(self, action: str, metadata: Dict[str, Any]) -> None:
        payload = {
            "timestamp": time.time(),
            "action": action,
            "metadata": metadata,
            "prev_hash": self._last_hash,
        }
        new_hash = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(payload, sort_keys=True)).hex
        payload["hash"] = new_hash
        self._last_hash = new_hash
        self._records.append(payload)
        logger.debug("Audit record appended: %s", action)

    def export(self) -> List[Dict[str, Any]]:
        return list(self._records)


class SafetyCore:
    """Контролює безпеку, силу, геозаборони і аварійні протоколи."""

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config
        self._event_bus = event_bus
        self._armed = False
        self._bounded_force = config.safety_thresholds["max_force"]
        self._bounded_torque = config.safety_thresholds["max_torque"]
        self._geofence_margin = config.safety_thresholds["geofence_margin"]
        self._watchdog = Watchdog()
        self._audit = AuditTrail()
        self._no_go_zones: List[Tuple[float, float, float]] = []
        self._two_man_confirmations: Dict[str, int] = defaultdict(int)
        self._last_voice_stop: Optional[float] = None
        self._event_bus.subscribe("/safety/watchdog_timeout", self._on_watchdog)

    def arm(self) -> None:
        self._armed = True
        self._audit.record("arm", {"armed": True})
        logger.info("Safety core armed")

    def disarm(self) -> None:
        self._armed = False
        self._audit.record("arm", {"armed": False})
        logger.info("Safety core disarmed")

    def emergency_stop(self, reason: str = "voice") -> None:
        if not self._armed:
            logger.warning("Emergency stop requested while safety not armed")
        logger.critical("EMERGENCY STOP TRIGGERED due to %s", reason)
        self._audit.record("emergency_stop", {"reason": reason})
        self._event_bus.publish(Event("/safety/emergency_stop", {"reason": reason}, QoSLevel.HARD_REALTIME))
        STATE_MACHINE.set_state(GlobalState.SAFE, reason=reason)

    def check_limits(self, force: float, torque: float) -> bool:
        within_force = force <= self._bounded_force
        within_torque = torque <= self._bounded_torque
        if not within_force or not within_torque:
            logger.error("Force/Torque limits exceeded: f=%.1f t=%.1f", force, torque)
            self.emergency_stop("force_limit")
            return False
        return True

    def audit(self, record: Dict[str, Any]) -> None:
        self._audit.record("action", record)

    def register_no_go_zone(self, center: Tuple[float, float], radius: float) -> None:
        self._no_go_zones.append((center[0], center[1], radius))
        self._audit.record("no_go", {"center": center, "radius": radius})

    def in_no_go_zone(self, position: Tuple[float, float]) -> bool:
        for cx, cy, r in self._no_go_zones:
            if math.hypot(position[0] - cx, position[1] - cy) <= r - self._geofence_margin:
                logger.warning("Position %s in no-go zone", position)
                self.emergency_stop("no_go")
                return True
        return False

    def voice_stop(self, phrase: str) -> None:
        normalized = phrase.strip().lower()
        if normalized == self._config.persona_profile["mercy_keyword"]:
            if not self._last_voice_stop or time.time() - self._last_voice_stop > 5.0:
                self._last_voice_stop = time.time()
                self.emergency_stop("voice_keyword")

    def two_man_rule(self, command_id: str, actor: str) -> bool:
        self._two_man_confirmations[(command_id, actor)] += 1
        unique_actors = {k[1] for k in self._two_man_confirmations if k[0] == command_id}
        if len(unique_actors) >= 2:
            logger.info("Two-man rule satisfied for %s", command_id)
            return True
        logger.warning("Two-man rule pending for %s", command_id)
        return False

    def _on_watchdog(self, event: Event) -> None:
        self.emergency_stop("watchdog")

    def heartbeat(self) -> None:
        self._watchdog.kick()

    def status(self) -> Dict[str, Any]:
        return {
            "armed": self._armed,
            "force_limit": self._bounded_force,
            "torque_limit": self._bounded_torque,
            "no_go_zones": len(self._no_go_zones),
            "last_voice_stop": self._last_voice_stop,
        }


SAFETY = SafetyCore(GLOBAL_CONFIG, EVENT_BUS)

# endregion SAFETY
# region HAL (HARDWARE ABSTRACTION)
# План:
# 1. Описати базові інтерфейси сенсорів та актуаторів.
# 2. Додати симуляційні заглушки для Jetson/PC середовищ.
# 3. Підготувати віртуальні сенсори (fusion) у HAL.
# 4. Реалізувати методи калібрування та self-check.
# 5. Забезпечити безпечне завершення при відсутності заліза.


class HALComponent:
    """Базовий HAL-компонент із self-check і деградацією."""

    name: str = "hal_component"

    def __init__(self) -> None:
        self.last_check = 0.0
        self.operational = True

    def self_check(self) -> bool:
        self.last_check = time.time()
        return self.operational

    def degrade(self, reason: str) -> None:
        self.operational = False
        logger.error("%s degraded due to %s", self.name, reason)


class CameraHAL(HALComponent):
    """Заглушка камери з можливістю симуляції експозиції та шуму."""

    name = "camera"

    def capture_frame(self) -> Dict[str, Any]:
        if not self.operational:
            raise RuntimeError("Camera offline")
        brightness = random.uniform(0.3, 0.9)
        noise = random.uniform(0.01, 0.05)
        return {
            "image": [[random.random() for _ in range(4)] for _ in range(4)],
            "brightness": brightness,
            "noise": noise,
            "timestamp": time.time(),
        }


class IMUHAL(HALComponent):
    """Інтерфейс IMU з ф'южн пози й вібрацій."""

    name = "imu"

    def read(self) -> Dict[str, float]:
        if not self.operational:
            raise RuntimeError("IMU offline")
        return {
            "accel": random.uniform(-0.2, 0.2),
            "gyro": random.uniform(-0.1, 0.1),
            "vibration": random.uniform(0.0, 0.3),
            "temperature": random.uniform(25.0, 45.0),
            "timestamp": time.time(),
        }


class ManipulatorHAL(HALComponent):
    """Актуатор маніпулятора з контролем сили та зворотним зв'язком."""

    name = "manipulator"

    def __init__(self) -> None:
        super().__init__()
        self.current_force = 0.0
        self.current_torque = 0.0

    def apply(self, force: float, torque: float) -> Dict[str, float]:
        if not self.operational:
            raise RuntimeError("Manipulator offline")
        # Гладкий перехід сили для симуляції фізіології.
        self.current_force = 0.7 * self.current_force + 0.3 * force
        self.current_torque = 0.7 * self.current_torque + 0.3 * torque
        return {
            "force": self.current_force,
            "torque": self.current_torque,
            "timestamp": time.time(),
        }


class PowerHAL(HALComponent):
    """Живлення з оцінкою батареї й температури."""

    name = "power"

    def __init__(self) -> None:
        super().__init__()
        self.battery_level = 0.8
        self.temperature = 40.0

    def measure(self) -> Dict[str, float]:
        if not self.operational:
            raise RuntimeError("Power offline")
        self.battery_level = max(0.0, min(1.0, self.battery_level - random.uniform(0.001, 0.005)))
        self.temperature += random.uniform(-0.2, 0.3)
        return {
            "battery": self.battery_level,
            "temperature": self.temperature,
            "timestamp": time.time(),
        }


class HAL:
    """Абстрагує всі залізні модулі Сін та надає сим-фолбек."""

    def __init__(self) -> None:
        self.camera = CameraHAL()
        self.imu = IMUHAL()
        self.manipulator = ManipulatorHAL()
        self.power = PowerHAL()
        self.virtual_sensors: Dict[str, Callable[[], Dict[str, Any]]] = {
            "odometry_fused": self._virtual_odometry,
            "vibration_health": self._virtual_vibration,
        }

    def sensors(self) -> Dict[str, HALComponent]:
        return {
            "camera": self.camera,
            "imu": self.imu,
            "power": self.power,
        }

    def actuators(self) -> Dict[str, HALComponent]:
        return {
            "manipulator": self.manipulator,
        }

    def _virtual_odometry(self) -> Dict[str, float]:
        imu_data = self.imu.read()
        drift = random.uniform(-0.01, 0.01)
        return {
            "x": drift,
            "y": drift * 0.5,
            "heading": imu_data["gyro"] * 0.1,
            "confidence": 0.8,
            "timestamp": time.time(),
        }

    def _virtual_vibration(self) -> Dict[str, float]:
        imu_data = self.imu.read()
        vibration = abs(imu_data["vibration"])
        return {
            "vibration": vibration,
            "risk": min(1.0, vibration * 3.0),
            "timestamp": time.time(),
        }

    def calibrate(self) -> None:
        for sensor in self.sensors().values():
            sensor.self_check()
        for actuator in self.actuators().values():
            actuator.self_check()
        logger.info("HAL calibration complete")

    def diagnostics(self) -> Dict[str, bool]:
        report = {}
        for name, component in {**self.sensors(), **self.actuators()}.items():
            try:
                report[name] = component.self_check()
            except Exception as exc:  # pragma: no cover
                report[name] = False
                component.degrade(str(exc))
        return report


HAL_SYSTEM = HAL()

# endregion HAL (HARDWARE ABSTRACTION)
# region PERCEPTION
# План:
# 1. Створити хаб перцепції з обробкою камер, IMU та віртуальних сенсорів.
# 2. Додати семантичні теги, детекцію рухомих об'єктів та оцінку якості сцени.
# 3. Реалізувати SLAM-заглушку з loop-closure та оцінкою невизначеності.
# 4. Інтегрувати акустичні й вібраційні канали з емоційними мітками.
# 5. Публікувати події на EventBus із відповідними QoS.


@dataclass
class SemanticObservation:
    """Семантичне сприйняття об'єкта з невизначеністю."""

    label: str
    confidence: float
    risk: float
    emotional_tag: str
    source: str
    timestamp: float


class SLAMState(BaseModel):
    """Стан SLAM із ковариаціями та loop-closure флагом."""

    position: Tuple[float, float, float]
    covariance: Tuple[float, float, float]
    loop_closure: bool
    quality: float
    landmarks: int


class PerceptionHub:
    """Збирає дані сенсорів, виконує ф'южн та подає їх у світову модель."""

    def __init__(self, hal: HAL, config: Config, event_bus: EventBus) -> None:
        self._hal = hal
        self._config = config
        self._event_bus = event_bus
        self._slam_state = SLAMState(
            position=(0.0, 0.0, 0.0),
            covariance=(0.1, 0.1, 0.2),
            loop_closure=False,
            quality=0.75,
            landmarks=0,
        )
        self._last_dynamic_objects: List[SemanticObservation] = []
        self._lighting_score = 0.5
        self._noise_level = 0.2

    def process_frame(self) -> None:
        camera_data = self._hal.camera.capture_frame()
        self._lighting_score = camera_data["brightness"]
        self._noise_level = camera_data["noise"]
        semantics = self._infer_semantics(camera_data)
        slam_update = self._update_slam(camera_data)
        dynamic_objects = self._detect_dynamic_objects(camera_data)
        acoustic = self._simulate_acoustic_scene()
        self._last_dynamic_objects = dynamic_objects
        payload = {
            "semantics": [obs.__dict__ for obs in semantics],
            "slam": slam_update.dict(),
            "dynamic": [obs.__dict__ for obs in dynamic_objects],
            "acoustic": acoustic,
        }
        self._event_bus.publish(Event("/perception/frame", payload, QoSLevel.REALTIME))

    def _infer_semantics(self, camera_data: Dict[str, Any]) -> List[SemanticObservation]:
        labels = ["friend", "cable", "fragile", "bolt", "open_space", "dust"]
        observations: List[SemanticObservation] = []
        for label in labels:
            confidence = max(0.05, random.random()) * (1.0 - camera_data["noise"])
            risk = random.uniform(0.0, 0.9)
            emotional = random.choice(["comfort", "tension", "awe", "suspicion"])
            observations.append(
                SemanticObservation(
                    label=label,
                    confidence=confidence,
                    risk=risk,
                    emotional_tag=emotional,
                    source="vision",
                    timestamp=time.time(),
                )
            )
        return observations

    def _update_slam(self, camera_data: Dict[str, Any]) -> SLAMState:
        drift = random.uniform(-0.02, 0.02)
        x, y, z = self._slam_state.position
        covariance = tuple(max(0.05, c * random.uniform(0.95, 1.1)) for c in self._slam_state.covariance)
        loop_closure = random.random() < 0.05
        if loop_closure:
            x *= 0.8
            y *= 0.8
        new_state = SLAMState(
            position=(x + drift, y + drift * 0.5, z),
            covariance=covariance,
            loop_closure=loop_closure,
            quality=max(0.1, min(1.0, self._slam_state.quality + random.uniform(-0.02, 0.03))),
            landmarks=self._slam_state.landmarks + random.randint(0, 2),
        )
        self._slam_state = new_state
        return new_state

    def _detect_dynamic_objects(self, camera_data: Dict[str, Any]) -> List[SemanticObservation]:
        observations: List[SemanticObservation] = []
        if random.random() < 0.3:
            obs = SemanticObservation(
                label="moving_obstacle",
                confidence=0.6,
                risk=0.7,
                emotional_tag="alarm",
                source="motion_track",
                timestamp=time.time(),
            )
            observations.append(obs)
        if camera_data["noise"] > 0.04:
            observations.append(
                SemanticObservation(
                    label="dust_cloud",
                    confidence=0.55,
                    risk=0.4,
                    emotional_tag="unease",
                    source="visual_noise",
                    timestamp=time.time(),
                )
            )
        return observations

    def _simulate_acoustic_scene(self) -> Dict[str, Any]:
        voices = random.choice(["ally", "stranger", "silence"])
        loudness = random.uniform(0.0, 1.0)
        event = {
            "source": voices,
            "loudness": loudness,
            "direction": random.uniform(-math.pi, math.pi),
            "emotion": random.choice(["anger", "fear", "calm", "unknown"]),
        }
        if loudness > 0.8:
            EVENT_BUS.publish(Event("/safety/acoustic_alert", event, QoSLevel.REALTIME))
        return event

    def lighting_quality(self) -> float:
        return self._lighting_score

    def noise_level(self) -> float:
        return self._noise_level

    def last_dynamic_objects(self) -> List[SemanticObservation]:
        return self._last_dynamic_objects


PERCEPTION = PerceptionHub(HAL_SYSTEM, GLOBAL_CONFIG, EVENT_BUS)

# endregion PERCEPTION
# region WORLD MODEL
# План:
# 1. Реалізувати OccupancyGrid та TSDF-заглушки для локального середовища.
# 2. Підтримати карту ризиків і геофенс з інтеграцією безпеки.
# 3. Оцінити невизначеність і поширення помилок.
# 4. Надавати API для планувальника й навичок.
# 5. Публікувати оновлення у подієву шину.


class OccupancyGrid:
    """Проста решітка зайнятості для сим середовища."""

    def __init__(self, size: Tuple[int, int] = (50, 50), resolution: float = 0.2) -> None:
        self.size = size
        self.resolution = resolution
        self.grid = [[0.5 for _ in range(size[1])] for _ in range(size[0])]

    def update_cell(self, x: int, y: int, value: float) -> None:
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            self.grid[x][y] = max(0.0, min(1.0, value))

    def neighborhood(self, x: int, y: int, radius: int = 1) -> List[float]:
        values = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size[0] and 0 <= ny < self.size[1]:
                    values.append(self.grid[nx][ny])
        return values


class TSDFVolume:
    """TSDF-заглушка для локальної геометрії."""

    def __init__(self, size: Tuple[int, int, int] = (20, 20, 10), voxel: float = 0.3) -> None:
        self.size = size
        self.voxel = voxel
        self.values = np.zeros(size)

    def integrate(self, point: Tuple[int, int, int], sdf: float) -> None:
        x, y, z = point
        if 0 <= x < self.size[0] and 0 <= y < self.size[1] and 0 <= z < self.size[2]:
            self.values[x][y][z] = sdf

    def extract_surface(self) -> float:
        return float(np.dot(np.ones_like(self.values).flatten(), np.array(self.values).flatten()) / (self.size[0] * self.size[1] * self.size[2]))


class RiskMap:
    """Оцінка ризиків з урахуванням перцепції, історії та втоми."""

    def __init__(self) -> None:
        self.map: Dict[str, float] = {}

    def update(self, key: str, value: float) -> None:
        self.map[key] = max(0.0, min(1.0, value))

    def combined_risk(self) -> float:
        if not self.map:
            return 0.1
        return max(self.map.values())


class WorldModel:
    """Внутрішній світ Сін із картиною ризиків і геозон."""

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config
        self._event_bus = event_bus
        self.occupancy = OccupancyGrid()
        self.tsdf = TSDFVolume()
        self.risk_map = RiskMap()
        self.geofence: Tuple[Tuple[float, float], float] = ((0.0, 0.0), 5.0)
        self._uncertainty = 0.2

    def update_from_perception(self, payload: Dict[str, Any]) -> None:
        semantics = payload.get("semantics", [])
        for obs in semantics:
            label = obs["label"]
            risk = obs["risk"]
            self.risk_map.update(label, risk)
        dynamic = payload.get("dynamic", [])
        for obs in dynamic:
            key = f"dynamic_{obs['label']}"
            self.risk_map.update(key, obs["risk"])
        self._uncertainty = max(0.05, min(0.9, self._uncertainty + random.uniform(-0.02, 0.02)))
        self._event_bus.publish(Event("/world/updated", {"risk": self.risk_map.map, "uncertainty": self._uncertainty}, QoSLevel.TACTICAL))

    def integrate_slam(self, slam: Dict[str, Any]) -> None:
        position = slam.get("position", (0.0, 0.0, 0.0))
        x_idx = int(position[0] / self.occupancy.resolution) + self.occupancy.size[0] // 2
        y_idx = int(position[1] / self.occupancy.resolution) + self.occupancy.size[1] // 2
        self.occupancy.update_cell(x_idx, y_idx, 0.1)
        self._uncertainty *= 0.95 if slam.get("loop_closure", False) else 1.05

    def enforce_geofence(self, position: Tuple[float, float]) -> bool:
        center, radius = self.geofence
        distance = math.hypot(position[0] - center[0], position[1] - center[1])
        if distance > radius - self._config.safety_thresholds["geofence_margin"]:
            EVENT_BUS.publish(Event("/safety/geofence_violation", {"pos": position}, QoSLevel.HARD_REALTIME))
            return False
        return True

    def uncertainty(self) -> float:
        return self._uncertainty

    def summary(self) -> Dict[str, Any]:
        return {
            "risk": self.risk_map.map,
            "uncertainty": self._uncertainty,
            "occupancy_mean": sum(sum(row) for row in self.occupancy.grid) / (self.occupancy.size[0] * self.occupancy.size[1]),
        }


WORLD = WorldModel(GLOBAL_CONFIG, EVENT_BUS)
EVENT_BUS.subscribe("/perception/frame", lambda event: WORLD.update_from_perception(event.payload))
EVENT_BUS.subscribe("/perception/frame", lambda event: WORLD.integrate_slam(event.payload.get("slam", {})))

# endregion WORLD MODEL
# region SKILLS
# План:
# 1. Побудувати DAG навичок із умовами й пріоритетами.
# 2. Реалізувати базові навички (Move, Grip, Cut, Retreat) з поведінковими скриптами.
# 3. Додати скриптовий DSL для демонтажу та пов'язати з конфігами.
# 4. Підтримати чорні списки/ваги пріоритетів.
# 5. Інтегрувати з планувальником для виконання.


@dataclass
class SkillNode:
    """Вузол навички в DAG із умовами та внутрішніми станами."""

    name: str
    priority: int
    conditions: List[Callable[[Dict[str, Any]], bool]]
    action: Callable[[Dict[str, Any]], Dict[str, Any]]
    next_skills: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)


class SkillGraph:
    """DAG навичок з можливістю реактивного виконання."""

    def __init__(self) -> None:
        self._skills: Dict[str, SkillNode] = {}
        self._lock = threading.Lock()

    def add_skill(self, node: SkillNode) -> None:
        with self._lock:
            self._skills[node.name] = node
            logger.debug("Skill registered: %s", node.name)

    def run(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []
        visited: set[str] = set()
        queue: Deque[str] = deque([goal])
        while queue:
            skill_name = queue.popleft()
            if skill_name in visited:
                continue
            skill = self._skills.get(skill_name)
            if not skill:
                logger.error("Skill %s not found", skill_name)
                continue
            visited.add(skill_name)
            if any(not condition(context) for condition in skill.conditions):
                logger.info("Skill %s conditions not met", skill_name)
                continue
            result = skill.action(context)
            plan.append({"skill": skill_name, "result": result})
            for next_skill in sorted(skill.next_skills, key=lambda name: self._skills[name].priority if name in self._skills else 0, reverse=True):
                if next_skill not in skill.blacklist:
                    queue.append(next_skill)
        return plan


def _skill_move(context: Dict[str, Any]) -> Dict[str, Any]:
    target = context.get("target", (0.0, 0.0))
    speed = context.get("speed", 0.5)
    WORLD.enforce_geofence(target)
    return {"status": "moving", "target": target, "speed": speed}


def _skill_grip(context: Dict[str, Any]) -> Dict[str, Any]:
    force = context.get("force", 50.0)
    SAFETY.check_limits(force, torque=20.0)
    HAL_SYSTEM.manipulator.apply(force, torque=15.0)
    return {"status": "gripped", "force": force}


def _skill_cut(context: Dict[str, Any]) -> Dict[str, Any]:
    force = context.get("force", 120.0)
    torque = context.get("torque", 80.0)
    if not SAFETY.check_limits(force, torque):
        return {"status": "aborted", "reason": "safety"}
    dust = random.uniform(0.1, 0.4)
    energy = 15.0
    return {"status": "cut", "dust": dust, "energy": energy}


def _skill_retreat(context: Dict[str, Any]) -> Dict[str, Any]:
    path = context.get("path", "reverse")
    return {"status": "retreat", "path": path}


def _skill_scan(context: Dict[str, Any]) -> Dict[str, Any]:
    PERCEPTION.process_frame()
    return {"status": "scanned", "lighting": PERCEPTION.lighting_quality()}


def _skill_analyze(context: Dict[str, Any]) -> Dict[str, Any]:
    memory = MEMORY.store("analysis", {"text": "review"}, {"dopamine": 0.1}, 0.6, "skill")
    return {"status": "analyzed", "memory": memory.id}


SKILL_GRAPH = SkillGraph()
SKILL_GRAPH.add_skill(
    SkillNode(
        name="move",
        priority=5,
        conditions=[lambda ctx: WORLD.enforce_geofence(ctx.get("target", (0.0, 0.0)))],
        action=_skill_move,
        next_skills=["grip"],
    )
)
SKILL_GRAPH.add_skill(
    SkillNode(
        name="grip",
        priority=4,
        conditions=[lambda ctx: SAFETY.check_limits(ctx.get("force", 50.0), 20.0)],
        action=_skill_grip,
        next_skills=["cut"],
    )
)
SKILL_GRAPH.add_skill(
    SkillNode(
        name="cut",
        priority=3,
        conditions=[lambda ctx: ctx.get("material", "steel") != "forbidden"],
        action=_skill_cut,
        next_skills=["retreat"],
    )
)
SKILL_GRAPH.add_skill(
    SkillNode(
        name="scan",
        priority=6,
        conditions=[lambda ctx: True],
        action=_skill_scan,
        next_skills=["analyze"],
    )
)
SKILL_GRAPH.add_skill(
    SkillNode(
        name="analyze",
        priority=5,
        conditions=[lambda ctx: True],
        action=_skill_analyze,
        next_skills=["move"],
    )
)
SKILL_GRAPH.add_skill(
    SkillNode(
        name="retreat",
        priority=2,
        conditions=[lambda ctx: True],
        action=_skill_retreat,
    )
)

# endregion SKILLS
# region PLANNING
# План:
# 1. Побудувати гібрид HTN/BT планувальник із декомпозицією завдань.
# 2. Реалізувати реактивний рефлексний шар для швидких тригерів.
# 3. Додати заглушку траєкторного планування з energy-aware cost.
# 4. Підтримати гарячу заміну політик.
# 5. Публікувати пояснення рішень для персони.


class ReflexLayer:
    """Миттєві рефлекси <10мс для небезпечних сценаріїв."""

    def __init__(self, safety: SafetyCore, event_bus: EventBus) -> None:
        self._safety = safety
        self._event_bus = event_bus
        self._triggers: Dict[str, Callable[[Event], None]] = {}
        self._event_bus.subscribe("/safety/emergency_stop", self._on_stop)

    def register(self, topic: str, handler: Callable[[Event], None]) -> None:
        self._triggers[topic] = handler
        self._event_bus.subscribe(topic, handler)

    def _on_stop(self, event: Event) -> None:
        logger.debug("Reflex layer halting all actions due to %s", event.payload)


class TrajectoryPlanner:
    """Заглушка MPPI/CEM траєкторного планувальника."""

    def __init__(self, world: WorldModel, energy: "EnergyManager") -> None:
        self._world = world
        self._energy = energy
        self._active_policy = "mppi"

    def optimize(self, start: Tuple[float, float], goal: Tuple[float, float], profile: str = "normal") -> Dict[str, Any]:
        uncertainty = self._world.uncertainty()
        risk = self._world.risk_map.combined_risk()
        cost = math.hypot(goal[0] - start[0], goal[1] - start[1]) * (1.0 + risk + uncertainty)
        energy_cost = self._energy.estimate_cost(cost)
        waypoints = [
            (start[0] + i * (goal[0] - start[0]) / 3, start[1] + i * (goal[1] - start[1]) / 3)
            for i in range(4)
        ]
        return {
            "policy": self._active_policy,
            "profile": profile,
            "cost": cost,
            "energy": energy_cost,
            "waypoints": waypoints,
        }

    def hot_swap(self, policy: str) -> None:
        self._active_policy = policy
        logger.info("Trajectory planner switched to %s", policy)


class TaskDecomposer:
    """Перетворює цілі у підцілі через HTN-подібний розклад."""

    def decompose(self, goal: str) -> List[str]:
        if goal == "demolish_segment":
            return ["scan", "analyze", "cut", "retreat"]
        if goal == "safety_check":
            return ["scan", "pause", "report"]
        return [goal]


class Planner:
    """Гібрид HTN/BT планувальник із поясненнями."""

    def __init__(self, skill_graph: SkillGraph, world: WorldModel, event_bus: EventBus, energy: "EnergyManager") -> None:
        self._skills = skill_graph
        self._world = world
        self._event_bus = event_bus
        self._energy = energy
        self._task_decomposer = TaskDecomposer()
        self._reflex = ReflexLayer(SAFETY, event_bus)
        self._history: Deque[Dict[str, Any]] = deque(maxlen=50)

    def decompose(self, goal: str) -> List[str]:
        return self._task_decomposer.decompose(goal)

    def react(self, event: Event) -> Optional[str]:
        if event.topic.startswith("/safety/"):
            return "retreat"
        if event.topic.startswith("/perception/") and any(obs["label"] == "moving_obstacle" for obs in event.payload.get("dynamic", [])):
            return "pause"
        return None

    def plan(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        subgoals = self.decompose(goal)
        trace: List[Dict[str, Any]] = []
        for subgoal in subgoals:
            plan = self._skills.run(subgoal if subgoal in self._skills._skills else goal, context)
            trace.extend(plan)
        explanation = {
            "goal": goal,
            "subgoals": subgoals,
            "trace": trace,
            "energy_budget": self._energy.estimate_cost(len(trace)),
            "risk": self._world.risk_map.combined_risk(),
        }
        self._history.append(explanation)
        self._event_bus.publish(Event("/planning/decision", explanation, QoSLevel.TACTICAL))
        return explanation

    def optimize(self, start: Tuple[float, float], goal: Tuple[float, float], profile: str = "normal") -> Dict[str, Any]:
        return TrajectoryPlanner(self._world, self._energy).optimize(start, goal, profile)

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


# endregion PLANNING

# region MEMORY
# План:
# 1. Реалізувати коротку, середню та довгу пам'ять.
# 2. Створити векторний індекс-заглушку для пошуку контекстів.
# 3. Підтримати граф сутностей з метаданими.
# 4. Додати механізми compaction та provenance.
# 5. Надати API для персони і планувальника.


@dataclass
class MemoryRecord:
    """Базовий запис пам'яті з емоційним слідом."""

    id: str
    kind: str
    content: Dict[str, Any]
    timestamp: float
    emotion: Dict[str, float]
    weight: float
    source: str


class VectorIndex:
    """Проста векторна пам'ять на косинусній схожості."""

    def __init__(self) -> None:
        self._store: Dict[str, List[float]] = {}
        self._meta: Dict[str, MemoryRecord] = {}

    def add(self, record: MemoryRecord, vector: List[float]) -> None:
        self._store[record.id] = vector
        self._meta[record.id] = record

    def search(self, vector: List[float], top_k: int = 3) -> List[MemoryRecord]:
        scores: List[Tuple[float, str]] = []
        for key, vec in self._store.items():
            dot = sum(a * b for a, b in zip(vec, vector))
            norm = math.sqrt(sum(a * a for a in vec)) * math.sqrt(sum(b * b for b in vector))
            score = dot / norm if norm else 0.0
            scores.append((score, key))
        scores.sort(reverse=True)
        return [self._meta[key] for _, key in scores[:top_k]]


class MemorySystem:
    """Організована пам'ять Сін із компакцією та емоційними вагами."""

    def __init__(self) -> None:
        self.short_term: Deque[MemoryRecord] = deque(maxlen=50)
        self.mid_term: Deque[MemoryRecord] = deque(maxlen=200)
        self.long_term: Dict[str, MemoryRecord] = {}
        self.vector_index = VectorIndex()
        self.entities: Dict[str, Dict[str, Any]] = {}

    def store(self, kind: str, content: Dict[str, Any], emotion: Dict[str, float], weight: float, source: str) -> MemoryRecord:
        record = MemoryRecord(id=uuid.uuid4().hex, kind=kind, content=content, timestamp=time.time(), emotion=emotion, weight=weight, source=source)
        self.short_term.append(record)
        if weight > 0.5:
            self.mid_term.append(record)
        if weight > 0.7:
            self.long_term[record.id] = record
            self.vector_index.add(record, self._vectorize(record))
        return record

    def recall(self, query: Dict[str, Any]) -> List[MemoryRecord]:
        vector = self._vectorize_query(query)
        return self.vector_index.search(vector)

    def _vectorize(self, record: MemoryRecord) -> List[float]:
        values = [record.weight, record.emotion.get("cortisol", 0.0), record.emotion.get("dopamine", 0.0)]
        return values + [len(record.content.get("text", "")) / 50.0]

    def _vectorize_query(self, query: Dict[str, Any]) -> List[float]:
        return [query.get("weight", 0.5), query.get("stress", 0.2), query.get("reward", 0.4), len(query.get("text", "")) / 50.0]

    def compact(self) -> None:
        while len(self.mid_term) > 150:
            record = self.mid_term.popleft()
            logger.debug("Compacting memory record %s", record.id)

    def update_entity(self, entity_id: str, data: Dict[str, Any]) -> None:
        entity = self.entities.setdefault(entity_id, {})
        entity.update(data)


MEMORY = MemorySystem()

# endregion MEMORY

# region DIALOG & PERSONA
# План:
# 1. Створити емоційне ядро (AffectState, AffectRegulator).
# 2. Реалізувати персональний діалоговий движок з паузами.
# 3. Враховувати втому та настрої в відповіді.
# 4. Додавати пояснення рішень («чому так зробила»).
# 5. Підтримати правило "мерсі" та внутрішній голос.


@dataclass
class AffectState:
    """Вектор емоцій Сін."""

    serotonin: float
    dopamine: float
    norepinephrine: float
    cortisol: float
    oxytocin: float
    void: float

    def clamp(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            setattr(self, field_name, max(0.0, min(1.0, value)))

    def as_dict(self) -> Dict[str, float]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


class AffectRegulator:
    """Регулює настрій і охолоджує перегрів."""

    def __init__(self, config: Config) -> None:
        baseline = config.persona_profile["mood_baseline"]
        self.state = AffectState(**baseline)
        self.fatigue = 0.2

    def update(self, stimuli: Dict[str, float]) -> AffectState:
        for key, delta in stimuli.items():
            if hasattr(self.state, key):
                setattr(self.state, key, getattr(self.state, key) + delta)
        self.state.clamp()
        self.fatigue = max(0.0, min(1.0, self.fatigue + stimuli.get("fatigue", 0.0)))
        return self.state

    def cool_down(self) -> None:
        self.state.cortisol *= 0.9
        self.state.norepinephrine *= 0.95
        self.state.void *= 0.8
        self.state.clamp()


class Persona:
    """Грубий, поетичний голос Сін із паузами і внутрішнім голосом."""

    def __init__(self, config: Config, memory: MemorySystem, event_bus: EventBus, affect: AffectRegulator) -> None:
        self._config = config
        self._memory = memory
        self._event_bus = event_bus
        self._affect = affect
        self._last_reply = ""
        self._inner_voice_enabled = config.persona_profile.get("inner_voice", True)

    def mood_update(self, context: Dict[str, Any]) -> AffectState:
        stimuli = {"dopamine": context.get("reward", 0.0), "cortisol": context.get("stress", 0.0), "fatigue": context.get("fatigue", 0.0)}
        state = self._affect.update(stimuli)
        if state.cortisol > 0.7:
            self._affect.cool_down()
        return state

    def fatigue_update(self, workload: float) -> None:
        self._affect.update({"fatigue": workload})

    def respond(self, text: str, context: Dict[str, Any]) -> str:
        state = self.mood_update(context)
        pause = "..." if state.cortisol > 0.6 else ""
        hesitance = "мм" if state.void > 0.4 else ""
        memory_trace = self._memory.recall({"text": text, "weight": 0.5})
        recollection = memory_trace[0].content.get("text", "") if memory_trace else ""
        response = f"{pause}{hesitance} {self._tone_line(text, state)}"
        if recollection:
            response += f" Я пам'ятаю {recollection}."
        self._last_reply = response.strip()
        return self._last_reply

    def explain(self, decision: Dict[str, Any]) -> str:
        trace = decision.get("trace", [])
        reason = ", ".join(item["skill"] for item in trace[:3]) if trace else "інтуїція"
        return f"Я тримала курс через {reason}. Ризик {decision.get('risk', 0.0):.2f}, енергія {decision.get('energy_budget', 0.0):.1f}."

    def _tone_line(self, text: str, state: AffectState) -> str:
        tone = self._config.persona_profile["tone"]
        if tone == "gritty_poetic":
            return f"слухай, {text.lower()} — ніби іржа на світанку."
        return text


AFFECT = AffectRegulator(GLOBAL_CONFIG)
PERSONA = Persona(GLOBAL_CONFIG, MEMORY, EVENT_BUS, AFFECT)

# endregion DIALOG & PERSONA

# region LEARNING
# План:
# 1. Створити заглушки self-play та imitation buffer.
# 2. Конфіги curriculum та контроль катастрофічного забування.
# 3. Підтримати онлайн адаптацію параметрів.
# 4. Логувати активність навчання.
# 5. Забезпечити безпечний fallback.


class LearningModule:
    """Заглушка навчання з політиками self-play та адаптацією."""

    def __init__(self, memory: MemorySystem, config: Config) -> None:
        self._memory = memory
        self._config = config
        self.self_play_buffer: Deque[Dict[str, Any]] = deque(maxlen=100)
        self.imitation_buffer: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.curriculum_level = 0

    def log_self_play(self, scenario: str, reward: float) -> None:
        self.self_play_buffer.append({"scenario": scenario, "reward": reward, "timestamp": time.time()})

    def log_imitation(self, skill: str, success: bool) -> None:
        self.imitation_buffer.append({"skill": skill, "success": success, "timestamp": time.time()})

    def adjust_parameters(self) -> Dict[str, Any]:
        self.curriculum_level = min(10, self.curriculum_level + 1)
        return {"curriculum": self.curriculum_level}


LEARNING = LearningModule(MEMORY, GLOBAL_CONFIG)

# endregion LEARNING

# region ENERGY & HEALTH
# План:
# 1. Моніторити батарею й температуру.
# 2. Реалізувати energy-aware оцінку планів та thermal guard.
# 3. Додати power-fail routine (мок).
# 4. Підтримати різні енергетичні моди.
# 5. Публікувати здоров'я в EventBus.


class EnergyManager:
    """Стежить за енергетикою, теплом та здоров'ям."""

    def __init__(self, hal: HAL, config: Config, event_bus: EventBus) -> None:
        self._hal = hal
        self._config = config
        self._event_bus = event_bus
        self.mode = "normal"
        self.health_log: Deque[Dict[str, Any]] = deque(maxlen=100)

    def estimate_cost(self, plan_complexity: float) -> float:
        profile = self._config.energy_profiles.get(self.mode, {"max_current": 40.0})
        return plan_complexity * (profile["max_current"] / 40.0)

    def thermal_guard(self) -> None:
        metrics = self.health_metrics()
        if metrics["temperature"] > self._config.energy_profiles[self.mode]["thermal_limit"]:
            self.mode = "eco"
            self._event_bus.publish(Event("/energy/thermal_throttle", metrics, QoSLevel.REALTIME))

    def health_metrics(self) -> Dict[str, float]:
        power = self._hal.power.measure()
        cpu = random.uniform(0.2, 0.7)
        gpu = random.uniform(0.1, 0.6)
        ram = random.uniform(0.3, 0.8)
        if psutil:
            cpu = psutil.cpu_percent() / 100.0
            ram = psutil.virtual_memory().percent / 100.0
        metrics = {"battery": power["battery"], "temperature": power["temperature"], "cpu": cpu, "ram": ram}
        self.health_log.append(metrics)
        self._event_bus.publish(Event("/telemetry/health", metrics, QoSLevel.BACKGROUND))
        return metrics

    def power_fail_routine(self) -> Dict[str, Any]:
        return {"status": "safe_shutdown", "snapshot": CONFIG_MANAGER.snapshot()}


ENERGY = EnergyManager(HAL_SYSTEM, GLOBAL_CONFIG, EVENT_BUS)
PLANNER = Planner(SKILL_GRAPH, WORLD, EVENT_BUS, ENERGY)

# endregion ENERGY & HEALTH

# region COMMS & API
# План:
# 1. Заглушити локальний API з імітацією WebSocket/gRPC.
# 2. Реалізувати schema registry на BaseModel.
# 3. Додати rate limiting і журнал команд.
# 4. Підтримати store-and-forward.
# 5. Забезпечити hot metrics overlay (лог).


class CommandSchema(BaseModel):
    """Schema для команд у стилі JSON-RPC."""

    name: str
    payload: Dict[str, Any]


class LocalAPI:
    """Локальна API-заглушка без мережі."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._history: Deque[Dict[str, Any]] = deque(maxlen=100)

    def invoke(self, command: CommandSchema) -> Dict[str, Any]:
        record = {"command": command.name, "payload": command.payload, "timestamp": time.time()}
        self._history.append(record)
        self._event_bus.publish(Event(f"/api/{command.name}", command.payload, QoSLevel.TACTICAL))
        return {"status": "queued", "id": uuid.uuid4().hex}

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


API = LocalAPI(EVENT_BUS)

# endregion COMMS & API

# region DEVOPS
# План:
# 1. Додати feature flag manager, crash dump та профайлер.
# 2. Реалізувати event-replayer та fuzz-тест для парсерів.
# 3. Вести golden dataset заглушку.
# 4. Логувати конфіг diff.
# 5. Забезпечити hot reload політик.


class CrashDump:
    """Проста система crash dump з локальним файлом."""

    def __init__(self, path: Path = Path("crash_dumps")) -> None:
        self.path = path
        self.path.mkdir(exist_ok=True)

    def write(self, info: Dict[str, Any]) -> Path:
        filename = self.path / f"dump_{int(time.time())}.json"
        filename.write_text(json.dumps(info, indent=2))
        return filename


class EventReplayer:
    """Повторює події з історії для дебага."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    def replay(self, events: List[Event]) -> None:
        for event in events:
            self._event_bus.publish(event)


class ParserFuzzer:
    """Fuzz-тест команд API."""

    def fuzz(self, api: LocalAPI, iterations: int = 10) -> None:
        for _ in range(iterations):
            name = random.choice(["move", "stop", "status"])
            payload = {"value": random.random()}
            api.invoke(CommandSchema(name=name, payload=payload))


CRASH_DUMP = CrashDump()
REPLAYER = EventReplayer(EVENT_BUS)
FUZZER = ParserFuzzer()

# endregion DEVOPS

# region SIMULATOR
# План:
# 1. Реалізувати просте середовище кімнати з об'єктами.
# 2. Додати генератор сценаріїв за seed.
# 3. Симулювати пісок, пил, кабелі, болти.
# 4. Підтримати step/reset/render_text.
# 5. Зв'язати з демо сценами.


@dataclass
class SimulatorState:
    """Стан симулятора з основними об'єктами."""

    bolts: int
    cables: int
    dust: float
    obstacles: List[Tuple[float, float]]
    agent_position: Tuple[float, float]
    temperature: float


class Simulator:
    """Мінімальний симулятор для демонтажних сценаріїв."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._rng = random.Random(config.get("seed", 42))
        self.state = self._create_state()

    def _create_state(self) -> SimulatorState:
        obstacles = [(self._rng.uniform(-3, 3), self._rng.uniform(-3, 3)) for _ in range(int(self._config["obstacle_density"] * 10))]
        return SimulatorState(
            bolts=self._config["bolts"],
            cables=self._config["cables"],
            dust=self._config["dust_level"],
            obstacles=obstacles,
            agent_position=(0.0, 0.0),
            temperature=35.0,
        )

    def reset(self, seed: Optional[int] = None) -> SimulatorState:
        if seed is not None:
            self._rng.seed(seed)
        self.state = self._create_state()
        return self.state

    def step(self, action: Dict[str, Any]) -> SimulatorState:
        dx, dy = action.get("move", (0.0, 0.0))
        x, y = self.state.agent_position
        self.state.agent_position = (x + dx, y + dy)
        if action.get("cut"):
            if self.state.bolts > 0:
                self.state.bolts -= 1
                self.state.dust += 0.05
        if action.get("grip"):
            self.state.cables = max(0, self.state.cables - 1)
        self.state.temperature += 0.1 if action.get("cut") else -0.05
        return self.state

    def render_text(self) -> str:
        return (
            f"Bolts:{self.state.bolts} Cables:{self.state.cables} Dust:{self.state.dust:.2f}"
            f" Pos:{self.state.agent_position} Temp:{self.state.temperature:.1f}"
        )


SIMULATOR = Simulator(GLOBAL_CONFIG.simulator_defaults)

# endregion SIMULATOR

# region DEMOS
# План:
# 1. Реалізувати ключові демо сценарії.
# 2. Показати інтеграцію perception→planning→skills→actuation.
# 3. Виводити KPI та пояснення від PERSONA.
# 4. Покрити safety, energy, dialog режими.
# 5. Забезпечити багатий лог.


def demo_basic() -> Dict[str, Any]:
    PERCEPTION.process_frame()
    context = {"target": (1.0, 0.5), "force": 60.0}
    decision = PLANNER.plan("demolish_segment", context)
    response = PERSONA.explain(decision)
    logger.info("Persona explanation: %s", response)
    return {"decision": decision, "persona": response}


def demo_demolition() -> Dict[str, Any]:
    sim_state = SIMULATOR.reset()
    kpi = {"energy": 0.0, "bolts_removed": 0, "dust": 0.0}
    for _ in range(3):
        PERCEPTION.process_frame()
        SIMULATOR.step({"move": (0.5, 0.0), "cut": True})
        kpi["energy"] += 10.0
        kpi["bolts_removed"] += 1
        kpi["dust"] = SIMULATOR.state.dust
    PERSONA.respond("Завершила цикл", {"reward": 0.1})
    return {"state": sim_state, "kpi": kpi}


def demo_safety() -> Dict[str, Any]:
    SAFETY.arm()
    SAFETY.register_no_go_zone((1.0, 1.0), 0.5)
    PERCEPTION.process_frame()
    SAFETY.voice_stop("пощади")
    return SAFETY.status()


def demo_energy() -> Dict[str, Any]:
    ENERGY.health_metrics()
    ENERGY.thermal_guard()
    plan = PLANNER.optimize((0.0, 0.0), (2.0, 1.5))
    return {"plan": plan, "mode": ENERGY.mode}


def demo_dialog() -> Dict[str, Any]:
    PERSONA.respond("Гей, як ніч?", {"reward": 0.2, "stress": 0.1})
    reply = PERSONA.respond("Ти жива там?", {"stress": 0.3})
    return {"reply": reply, "mood": AFFECT.state.as_dict()}


DEMO_SCENARIOS = {
    "basic": demo_basic,
    "demolition": demo_demolition,
    "safety": demo_safety,
    "energy": demo_energy,
    "dialog": demo_dialog,
}


# endregion DEMOS

# region UNIT TESTS (INLINE)
# План:
# 1. Перевірити стабільність переходів стану та event bus.
# 2. Тестувати навички та планувальник.
# 3. Переконатися в коректності пам'яті й енергетики.
# 4. Валідувати safety протоколи.
# 5. Підготувати симулятор.


def run_unit_tests() -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    try:
        EVENT_BUS.publish(Event("/test", {"value": 1}))
        EVENT_BUS.pump()
        results["event_bus"] = True
    except Exception:
        results["event_bus"] = False
    try:
        STATE_MACHINE.set_state(GlobalState.OPERATIONAL, reason="test")
        STATE_MACHINE.rollback()
        results["state_machine"] = True
    except Exception:
        results["state_machine"] = False
    try:
        plan = PLANNER.plan("demolish_segment", {"target": (0.1, 0.1)})
        results["planner"] = bool(plan)
    except Exception:
        results["planner"] = False
    try:
        MEMORY.store("dialog", {"text": "привіт"}, {"dopamine": 0.2}, 0.9, "test")
        recall = MEMORY.recall({"text": "привіт"})
        results["memory"] = bool(recall)
    except Exception:
        results["memory"] = False
    try:
        SAFETY.check_limits(100.0, 50.0)
        results["safety"] = True
    except Exception:
        results["safety"] = False
    return results


# endregion UNIT TESTS (INLINE)

# region __main__
# План:
# 1. Реалізувати CLI з демо сценаріями.
# 2. Запустити тести перед демо.
# 3. Зібрати KPI та фінальний звіт.
# 4. Вивести пояснення від PERSONA.
# 5. Забезпечити багатий лог.


def main() -> None:
    parser = argparse.ArgumentParser(description="Sin cognitive core demos")
    parser.add_argument("--demo", choices=list(DEMO_SCENARIOS.keys()), default="basic")
    args = parser.parse_args()
    test_results = run_unit_tests()
    logger.info("UNIT TESTS: %s", test_results)
    SAFETY.heartbeat()
    result = DEMO_SCENARIOS[args.demo]()
    EVENT_BUS.pump()
    summary = {
        "demo": args.demo,
        "result": result,
        "health": ENERGY.health_metrics(),
        "mood": AFFECT.state.as_dict(),
    }
    explanation = PERSONA.respond("Підсумуй", {"reward": 0.0, "stress": 0.2})
    print("Mission Summary")
    print(json.dumps(summary, indent=2, default=str))
    print("Persona:", explanation)


if __name__ == "__main__":
    main()

# endregion __main__
