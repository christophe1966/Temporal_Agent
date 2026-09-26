# -*- coding: utf-8 -*-
"""
Temporal Agent Predictor 2D - V0.7 apprentissage multi-pas auto-régressif

Principe de la scène 2D :
    - NOIR   : trajectoire réelle ancienne, jusqu'à S - horizon.
    - BLEU   : trajectoire réelle récente, de S - horizon à S.
    - ORANGE : trajectoire prédite par le RN sur la même fenêtre horizon.

Dynamique réelle :
    - Agent balistique qui se déplace préférentiellement en ligne droite.
    - Vitesse modulée périodiquement : accélération / décélération.
    - Rebonds continus sur les murs, y compris les coins.

Réseau neuronal :
    - Apprentissage supervisé, pas PPO.
    - Entrée  : fenêtre de N états réels d'amorçage.
    - Sortie  : état suivant prédit.
    - Entraînement : prédiction multi-pas avec auto-réinjection.
    - Orange : auto-réinjection des états prédits sur horizon pas.

Mode entraînement :
    - La dynamique réelle bleue/noire avance.
    - Le RN apprend à prédire l'état suivant réel.
    - La scène montre la comparaison bleu/orange sur l'horizon récent.

Mode simulation :
    - L'entraînement est suspendu.
    - La dynamique réelle continue.
    - Le RN produit uniquement la trace orange de comparaison sur horizon.

Dépendances :
    pip install tensorflow numpy matplotlib
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import math
import time
import random
import threading
import queue
from pathlib import Path
from datetime import datetime
from collections import deque

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import patches

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


# =============================================================================
# Chemins locaux
# =============================================================================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

CONFIG_PATH = SCRIPT_DIR / "temporal_agent_config.json"
AUTOSAVES_DIR = SCRIPT_DIR / "autosaves"
AUTOSAVES_DIR.mkdir(exist_ok=True)


# =============================================================================
# TensorFlow
# =============================================================================

def configure_tensorflow():
    """Active la croissance mémoire GPU si TensorFlow voit un GPU."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return len(gpus)
    except Exception:
        return 0


# =============================================================================
# Meta-paramètres
# =============================================================================

DEFAULT_META_PARAMS = {
    "experiment": {
        "name": "temporal_agent_predictor_2d_v07_autoregressive_multistep",
        "seed": 42,
    },

    "world": {
        "width": 10.0,
        "height": 10.0,
        "dt": 0.08,
        "bounce_damping": 0.96,
        "max_speed": 4.5,
        "max_steps_per_episode": 5000,
        "spawn_margin": 1.0,
    },

    "agent": {
        "radius": 0.13,
        "start_random": True,
        "start_speed_min": 1.2,
        "start_speed_max": 2.2,
    },

    "motion_profile": {
        "enabled": True,
        "cycle_steps": 180,
        "speed_base": 2.2,
        "speed_amplitude": 1.1,
        "speed_min": 0.35,
        "speed_response": 2.4,
        "randomize_phase_on_reset": True,
        "real_noise_std": 0.0,
    },

    "predictor": {
        "warmup_real_steps": 5,
        "horizon_steps": 10,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "replay_maxlen": 30000,
        "min_replay_to_train": 512,
        "train_batches_per_loop": 2,
        "lstm_units": 64,
        "dense_units": 128,
        "rollout_train_steps": 3,
        "rollout_loss_weight": 0.50,
    },

    "ui": {
        "refresh_ms": 35,
        "train_steps_per_loop": 8,
        "sim_steps_per_tick": 4,
        "trail_len": 1200,
        "display_real_steps": 50,
        "graph_history_len": 1200,
        "queue_max_messages_per_frame": 300,
    },

    "autosave": {
        "enabled": True,
        "every_minutes": 15,
        "keep_last": 30,
    },
}


def deep_merge(base, override):
    """Fusion récursive de dictionnaires."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_json(path, data):
    """Sauvegarde JSON indentée."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_or_create_config():
    """Charge la config locale ou crée une config par défaut."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_params = json.load(f)
            return deep_merge(DEFAULT_META_PARAMS, user_params)
        except Exception:
            return dict(DEFAULT_META_PARAMS)

    save_json(CONFIG_PATH, DEFAULT_META_PARAMS)
    return dict(DEFAULT_META_PARAMS)


# =============================================================================
# Dynamique réelle 2D
# =============================================================================

class RealDynamics2D:
    """Dynamique réelle bleue/noire.

    L'agent conserve sa direction entre les rebonds. Sa vitesse est modulée par
    un profil périodique. Les collisions murales sont traitées en temps continu
    pour gérer proprement les coins.
    """

    def __init__(self, meta_params, seed=None):
        self.meta = meta_params
        self.rng = np.random.default_rng(seed)

        self.world = self.meta["world"]
        self.agent_cfg = self.meta["agent"]
        self.motion_cfg = self.meta["motion_profile"]
        self.ui_cfg = self.meta["ui"]

        self.width = float(self.world["width"])
        self.height = float(self.world["height"])
        self.dt = float(self.world["dt"])
        self.agent_radius = float(self.agent_cfg["radius"])
        self.state_dim = 12

        self.global_step = 0
        self.episode_id = -1
        self.phase_offset = 0.0
        self.reset()

    def _random_pos(self):
        """Position initiale éloignée des murs."""
        margin = max(float(self.world.get("spawn_margin", 1.0)), self.agent_radius)
        x = self.rng.uniform(margin, self.width - margin)
        y = self.rng.uniform(margin, self.height - margin)
        return np.array([x, y], dtype=np.float32)

    def _random_velocity(self):
        """Vitesse initiale orientée aléatoirement."""
        speed_min = float(self.agent_cfg.get("start_speed_min", 1.2))
        speed_max = float(self.agent_cfg.get("start_speed_max", 2.2))
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        speed = self.rng.uniform(speed_min, speed_max)
        return np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)

    def reset(self):
        """Réinitialise la dynamique réelle."""
        self.episode_id += 1
        self.step_count = 0

        if self.motion_cfg.get("randomize_phase_on_reset", True):
            self.phase_offset = float(self.rng.uniform(0.0, 2.0 * math.pi))
        else:
            self.phase_offset = 0.0

        if self.agent_cfg.get("start_random", True):
            self.agent_pos = self._random_pos()
        else:
            self.agent_pos = np.array([self.width * 0.5, self.height * 0.5], dtype=np.float32)

        self.agent_vel = self._random_velocity()

        max_hist = max(10000, int(self.ui_cfg.get("trail_len", 1200)) * 2)
        self.trail = deque(maxlen=int(self.ui_cfg.get("trail_len", 1200)))
        self.state_history = deque(maxlen=max_hist)

        self.trail.append(self.agent_pos.copy())
        self.state_history.append(self.get_state_vector())

        return self.get_state_vector()

    def speed_phase(self):
        """Phase du cycle vitesse."""
        cycle = max(1, int(self.motion_cfg.get("cycle_steps", 180)))
        phase = 2.0 * math.pi * ((self.step_count % cycle) / float(cycle))
        return phase + self.phase_offset

    def profile_speed(self):
        """Vitesse cible périodique : accélération puis décélération."""
        if not self.motion_cfg.get("enabled", True):
            return float(self.motion_cfg.get("speed_base", 2.2))

        base = float(self.motion_cfg.get("speed_base", 2.2))
        amp = float(self.motion_cfg.get("speed_amplitude", 1.1))
        speed_min = float(self.motion_cfg.get("speed_min", 0.35))
        value = base + amp * math.sin(self.speed_phase())
        return max(speed_min, value)

    def _advance_with_bounces(self, dt):
        """Avance avec rebonds continus, y compris les coins.

        Si tx et ty coïncident, l'agent touche un coin : vx et vy sont inversés.
        """
        eps = 1e-9
        remaining_dt = float(dt)
        wall_hits = 0

        radius = self.agent_radius
        damping = float(self.world.get("bounce_damping", 0.96))

        x_min = radius
        x_max = self.width - radius
        y_min = radius
        y_max = self.height - radius

        max_bounces_per_step = 8

        for _ in range(max_bounces_per_step):
            if remaining_dt <= eps:
                break

            vx = float(self.agent_vel[0])
            vy = float(self.agent_vel[1])
            x = float(self.agent_pos[0])
            y = float(self.agent_pos[1])

            if abs(vx) <= eps and abs(vy) <= eps:
                break

            if vx > eps:
                tx = (x_max - x) / vx
            elif vx < -eps:
                tx = (x_min - x) / vx
            else:
                tx = math.inf

            if vy > eps:
                ty = (y_max - y) / vy
            elif vy < -eps:
                ty = (y_min - y) / vy
            else:
                ty = math.inf

            t_hit = min(tx, ty)

            if t_hit == math.inf or t_hit > remaining_dt:
                self.agent_pos = self.agent_pos + self.agent_vel * remaining_dt
                remaining_dt = 0.0
                break

            t_hit = max(0.0, float(t_hit))
            self.agent_pos = self.agent_pos + self.agent_vel * t_hit
            remaining_dt -= t_hit

            hit_x = abs(tx - t_hit) < 1e-7
            hit_y = abs(ty - t_hit) < 1e-7

            self.agent_pos[0] = np.clip(self.agent_pos[0], x_min, x_max)
            self.agent_pos[1] = np.clip(self.agent_pos[1], y_min, y_max)

            if hit_x:
                self.agent_vel[0] = -self.agent_vel[0] * damping
                wall_hits += 1

            if hit_y:
                self.agent_vel[1] = -self.agent_vel[1] * damping
                wall_hits += 1

            if remaining_dt > eps:
                micro_dt = min(remaining_dt, 1e-6)
                self.agent_pos = self.agent_pos + self.agent_vel * micro_dt
                remaining_dt -= micro_dt

        self.agent_pos[0] = np.clip(self.agent_pos[0], x_min, x_max)
        self.agent_pos[1] = np.clip(self.agent_pos[1], y_min, y_max)

        return int(wall_hits)

    def real_step(self):
        """Avance la dynamique réelle d'un pas."""
        self.step_count += 1
        self.global_step += 1

        max_speed = float(self.world["max_speed"])
        dt = float(self.world["dt"])

        old_speed = float(np.linalg.norm(self.agent_vel))
        if old_speed < 1e-8:
            self.agent_vel = self._random_velocity()
            old_speed = float(np.linalg.norm(self.agent_vel))

        direction = self.agent_vel / (old_speed + 1e-8)
        target_speed = self.profile_speed()

        response = float(self.motion_cfg.get("speed_response", 2.4))
        new_speed = old_speed + (target_speed - old_speed) * response * dt
        new_speed = max(0.01, min(max_speed, new_speed))

        self.agent_vel = (direction * new_speed).astype(np.float32)

        noise_std = float(self.motion_cfg.get("real_noise_std", 0.0))
        if noise_std > 0.0:
            self.agent_vel = self.agent_vel + self.rng.normal(0.0, noise_std, size=2).astype(np.float32)
            speed = float(np.linalg.norm(self.agent_vel))
            if speed > max_speed:
                self.agent_vel = self.agent_vel / (speed + 1e-8) * max_speed

        wall_hits = self._advance_with_bounces(dt)

        self.trail.append(self.agent_pos.copy())
        self.state_history.append(self.get_state_vector())

        done = self.step_count >= int(self.world["max_steps_per_episode"])

        info = {
            "speed": float(np.linalg.norm(self.agent_vel)),
            "profile_speed": float(target_speed),
            "wall_hits": int(wall_hits),
            "step_count": int(self.step_count),
            "global_step": int(self.global_step),
        }

        return self.get_state_vector(), done, info

    def get_state_vector(self):
        """État normalisé utilisé par le réseau."""
        max_speed = float(self.world["max_speed"])
        cx = self.width * 0.5
        cy = self.height * 0.5

        speed = float(np.linalg.norm(self.agent_vel))
        prof_speed = self.profile_speed()
        phase = self.speed_phase()

        dist_l = self.agent_pos[0]
        dist_r = self.width - self.agent_pos[0]
        dist_b = self.agent_pos[1]
        dist_t = self.height - self.agent_pos[1]

        state = np.array([
            (self.agent_pos[0] - cx) / cx,
            (self.agent_pos[1] - cy) / cy,
            self.agent_vel[0] / max_speed,
            self.agent_vel[1] / max_speed,
            speed / max_speed,
            prof_speed / max_speed,
            math.sin(phase),
            math.cos(phase),
            dist_l / self.width,
            dist_r / self.width,
            dist_b / self.height,
            dist_t / self.height,
        ], dtype=np.float32)

        return np.clip(state, -3.0, 3.0)

    def state_to_position(self, state):
        """Convertit un état normalisé en position monde."""
        state = np.asarray(state, dtype=np.float32)
        cx = self.width * 0.5
        cy = self.height * 0.5
        x = float(state[0] * cx + cx)
        y = float(state[1] * cy + cy)
        return np.array([x, y], dtype=np.float32)

    def sanitize_predicted_state(self, state):
        """Stabilise un état prédit avant auto-réinjection."""
        state = np.asarray(state, dtype=np.float32).copy()

        max_speed = float(self.world["max_speed"])
        cx = self.width * 0.5
        cy = self.height * 0.5

        x = float(state[0] * cx + cx)
        y = float(state[1] * cy + cy)
        x = float(np.clip(x, self.agent_radius, self.width - self.agent_radius))
        y = float(np.clip(y, self.agent_radius, self.height - self.agent_radius))

        vx_n = float(np.clip(state[2], -1.2, 1.2))
        vy_n = float(np.clip(state[3], -1.2, 1.2))
        speed_n = min(1.2, math.sqrt(vx_n * vx_n + vy_n * vy_n))

        state[0] = (x - cx) / cx
        state[1] = (y - cy) / cy
        state[2] = vx_n
        state[3] = vy_n
        state[4] = speed_n
        state[5] = float(np.clip(state[5], 0.0, 1.5))

        phase_norm = math.sqrt(float(state[6] * state[6] + state[7] * state[7]))
        if phase_norm > 1e-8:
            state[6] = state[6] / phase_norm
            state[7] = state[7] / phase_norm
        else:
            state[6] = 0.0
            state[7] = 1.0

        state[8] = x / self.width
        state[9] = (self.width - x) / self.width
        state[10] = y / self.height
        state[11] = (self.height - y) / self.height

        return np.clip(state, -3.0, 3.0).astype(np.float32)

    def copy_state(self):
        """État dynamique sérialisable."""
        return {
            "agent_pos": self.agent_pos.copy(),
            "agent_vel": self.agent_vel.copy(),
            "phase_offset": float(self.phase_offset),
            "step_count": int(self.step_count),
            "global_step": int(self.global_step),
            "episode_id": int(self.episode_id),
        }

    def set_state(self, state):
        """Restaure l'état dynamique."""
        self.agent_pos = np.asarray(state["agent_pos"], dtype=np.float32)
        self.agent_vel = np.asarray(state["agent_vel"], dtype=np.float32)
        self.phase_offset = float(state["phase_offset"])
        self.step_count = int(state["step_count"])
        self.global_step = int(state["global_step"])
        self.episode_id = int(state["episode_id"])

        max_hist = max(10000, int(self.ui_cfg.get("trail_len", 1200)) * 2)
        self.trail = deque(maxlen=int(self.ui_cfg.get("trail_len", 1200)))
        self.state_history = deque(maxlen=max_hist)
        self.trail.append(self.agent_pos.copy())
        self.state_history.append(self.get_state_vector())


# =============================================================================
# Replay buffer supervisé
# =============================================================================

class SupervisedReplayBuffer:
    """Buffer de séquences réelles pour apprentissage supervisé.

    Chaque échantillon stocke maintenant :
        - sequence : fenêtre d'amorçage réelle, shape=(warmup, state_dim)
        - targets  : états futurs réels, shape=(rollout_train_steps, state_dim)
    """

    def __init__(self, maxlen=30000):
        self.maxlen = int(maxlen)
        self.samples = deque(maxlen=self.maxlen)

    def __len__(self):
        return len(self.samples)

    def clear(self):
        self.samples.clear()

    def store(self, sequence, targets):
        self.samples.append((
            np.asarray(sequence, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
        ))

    def sample(self, batch_size):
        batch_size = min(int(batch_size), len(self.samples))
        indices = np.random.choice(len(self.samples), size=batch_size, replace=False)

        xs = []
        ys = []
        samples_list = list(self.samples)
        for idx in indices:
            x, y = samples_list[int(idx)]
            xs.append(x)
            ys.append(y)

        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


# =============================================================================
# Réseau prédictif temporel
# =============================================================================

class TemporalPredictor:
    """Réseau supervisé : fenêtre d'états réels -> état suivant."""

    def __init__(self, warmup_steps, state_dim, meta_params):
        self.warmup_steps = int(warmup_steps)
        self.state_dim = int(state_dim)
        self.meta = meta_params
        self.cfg = self.meta["predictor"]

        self.model = self._build_model()
        self.optimizer = optimizers.Adam(float(self.cfg["learning_rate"]))
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.train_count = 0

    def _build_model(self):
        inputs = layers.Input(shape=(self.warmup_steps, self.state_dim), name="state_window")
        x = layers.LSTM(int(self.cfg.get("lstm_units", 64)), name="lstm")(inputs)
        x = layers.Dense(int(self.cfg.get("dense_units", 128)), activation="relu")(x)
        x = layers.Dense(int(self.cfg.get("dense_units", 128)), activation="relu")(x)
        outputs = layers.Dense(self.state_dim, activation="linear", name="next_state")(x)
        return Model(inputs, outputs, name="temporal_predictor")

    def predict_next(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32).reshape(1, self.warmup_steps, self.state_dim)
        pred = self.model(sequence, training=False).numpy()[0].astype(np.float32)
        return pred

    def train_on_batch(self, x, y):
        """Entraîne le RN en auto-réinjection sur plusieurs pas.

        x : shape=(batch, warmup, state_dim)
            Séquence réelle d'amorçage.
        y : shape=(batch, rollout_steps, state_dim)
            États réels futurs à reconstruire.

        Principe :
            pas 1 : x réel -> S+1 prédit, comparé à S+1 réel
            pas 2 : on réinjecte S+1 prédit -> S+2 prédit, comparé à S+2 réel
            etc.
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        if len(y.shape) == 2:
            y = tf.expand_dims(y, axis=1)

        rollout_steps = int(y.shape[1])
        rollout_weight = float(self.cfg.get("rollout_loss_weight", 0.50))

        with tf.GradientTape() as tape:
            sequence = x
            step_losses = []

            for step_idx in range(rollout_steps):
                pred = self.model(sequence, training=True)
                target = y[:, step_idx, :]
                step_loss = self.loss_fn(target, pred)
                step_losses.append(step_loss)

                # Auto-réinjection différentiable : la sortie prédite devient
                # une partie de l'entrée suivante.
                pred_for_next = tf.clip_by_value(pred, -3.0, 3.0)
                sequence = tf.concat(
                    [sequence[:, 1:, :], tf.expand_dims(pred_for_next, axis=1)],
                    axis=1,
                )

            one_step_loss = step_losses[0]
            if rollout_steps > 1:
                future_loss = tf.reduce_mean(tf.stack(step_losses[1:]))
                loss = one_step_loss + rollout_weight * future_loss
            else:
                future_loss = tf.constant(0.0, dtype=tf.float32)
                loss = one_step_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_count += 1
        return {
            "loss": float(loss.numpy()),
            "one_step_loss": float(one_step_loss.numpy()),
            "rollout_loss": float(future_loss.numpy()),
            "rollout_steps": int(rollout_steps),
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "temporal_predictor.keras")
        save_json(directory / "predictor_state.json", {
            "warmup_steps": int(self.warmup_steps),
            "state_dim": int(self.state_dim),
            "train_count": int(self.train_count),
        })

    def load(self, directory):
        directory = Path(directory)
        model_path = directory / "temporal_predictor.keras"
        state_path = directory / "predictor_state.json"

        if not model_path.exists():
            raise FileNotFoundError("temporal_predictor.keras manquant.")

        self.model = tf.keras.models.load_model(model_path, compile=False)
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.train_count = int(state.get("train_count", 0))
        else:
            self.train_count = 0

        self.optimizer = optimizers.Adam(float(self.cfg["learning_rate"]))


# =============================================================================
# IHM
# =============================================================================

class TemporalAgentApp:
    """Application Tkinter principale."""

    def __init__(self, root):
        self.root = root
        self.root.title("Temporal Agent 2D - apprentissage multi-pas")

        self.meta = load_or_create_config()
        self.seed = int(self.meta["experiment"].get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self.gpu_count = configure_tensorflow()

        self.env = RealDynamics2D(self.meta, seed=self.seed)
        warmup = int(self.meta["predictor"].get("warmup_real_steps", 5))
        self.predictor = TemporalPredictor(warmup, self.env.state_dim, self.meta)
        self.replay = SupervisedReplayBuffer(int(self.meta["predictor"].get("replay_maxlen", 30000)))

        self.training_state = "stopped"  # running, paused, stopped
        self.simulation_enabled = False
        self.last_autosave_time = time.time()

        self.training_thread = None
        self.training_stop_event = threading.Event()
        self.training_pause_event = threading.Event()
        self.training_stop_event.set()
        self.training_pause_event.set()

        self.ui_queue = queue.Queue()
        self.env_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.replay_lock = threading.Lock()

        maxlen = int(self.meta["ui"]["graph_history_len"])
        self.metrics = {
            "loss": deque(maxlen=maxlen),
            "one_step_loss": deque(maxlen=maxlen),
            "rollout_loss": deque(maxlen=maxlen),
            "pred_error": deque(maxlen=maxlen),
            "speed": deque(maxlen=maxlen),
            "profile_speed": deque(maxlen=maxlen),
            "wall_hits": deque(maxlen=maxlen),
            "replay_size": deque(maxlen=maxlen),
            "traj_mean_error": deque(maxlen=maxlen),
            "traj_final_error": deque(maxlen=maxlen),
            "traj_max_error": deque(maxlen=maxlen),
        }

        # Évite d'ajouter plusieurs fois la même mesure d'écart trajectoire
        # lorsque l'IHM se rafraîchit sans nouveau pas de dynamique.
        self.last_trajectory_error_step = -1

        self._build_ui()
        self.log("Temporal Agent Predictor V0.7 initialisé.")
        self.log("Scène : noir=réel ancien, bleu=réel horizon, orange=RN horizon.")
        self.log("Thread entraînement : non lancé au démarrage. Utiliser Start / Resume.")
        self.log("GPU détectés : %d" % self.gpu_count)
        self.log("Config : %s" % CONFIG_PATH)
        self.log("Autosaves : %s" % AUTOSAVES_DIR)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.main_loop)

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    def _build_ui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=2)

        scene_frame = ttk.LabelFrame(self.root, text="Scène 2D")
        scene_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        scene_frame.rowconfigure(0, weight=1)
        scene_frame.columnconfigure(0, weight=1)

        self.fig_scene = Figure(figsize=(7, 6), dpi=100)
        self.ax_scene = self.fig_scene.add_subplot(111)
        self.canvas_scene = FigureCanvasTkAgg(self.fig_scene, master=scene_frame)
        self.canvas_scene.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        right_frame = ttk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        controls = ttk.LabelFrame(right_frame, text="Contrôles")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        self.btn_start_resume = ttk.Button(controls, text="Start / Resume", command=self.start_resume_training)
        self.btn_start_resume.grid(row=0, column=0, sticky="ew", padx=3, pady=3)

        self.btn_pause = ttk.Button(controls, text="Pause", command=self.pause_training)
        self.btn_pause.grid(row=0, column=1, sticky="ew", padx=3, pady=3)

        self.btn_stop = ttk.Button(controls, text="Stop", command=self.stop_training)
        self.btn_stop.grid(row=0, column=2, sticky="ew", padx=3, pady=3)

        self.btn_sim = ttk.Button(controls, text="Simulation : OFF", command=self.toggle_simulation)
        self.btn_sim.grid(row=1, column=0, columnspan=3, sticky="ew", padx=3, pady=3)

        ttk.Button(controls, text="Reset scène", command=self.reset_scene).grid(
            row=2, column=0, columnspan=3, sticky="ew", padx=3, pady=3
        )

        self.pred_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Afficher RN orange", variable=self.pred_enabled_var).grid(
            row=3, column=0, columnspan=3, sticky="w", padx=3, pady=3
        )

        predictor_frame = ttk.LabelFrame(controls, text="Fenêtre horizon")
        predictor_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=3, pady=5)
        predictor_frame.columnconfigure(1, weight=1)

        ttk.Label(predictor_frame, text="Amorçage réel").grid(row=0, column=0, sticky="w", padx=3, pady=2)
        self.entry_warmup = ttk.Entry(predictor_frame)
        self.entry_warmup.grid(row=0, column=1, sticky="ew", padx=3, pady=2)
        self.entry_warmup.insert(0, str(int(self.meta["predictor"].get("warmup_real_steps", 5))))

        ttk.Label(predictor_frame, text="Horizon").grid(row=1, column=0, sticky="w", padx=3, pady=2)
        self.entry_horizon = ttk.Entry(predictor_frame)
        self.entry_horizon.grid(row=1, column=1, sticky="ew", padx=3, pady=2)
        self.entry_horizon.insert(0, str(int(self.meta["predictor"].get("horizon_steps", 10))))

        ttk.Label(predictor_frame, text="Historique affiché").grid(row=2, column=0, sticky="w", padx=3, pady=2)
        self.entry_display_steps = ttk.Entry(predictor_frame)
        self.entry_display_steps.grid(row=2, column=1, sticky="ew", padx=3, pady=2)
        self.entry_display_steps.insert(0, str(int(self.meta["ui"].get("display_real_steps", 50))))

        save_frame = ttk.Frame(controls)
        save_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=3, pady=5)
        save_frame.columnconfigure(0, weight=1)
        save_frame.columnconfigure(1, weight=1)
        save_frame.columnconfigure(2, weight=1)

        ttk.Button(save_frame, text="Sauver", command=self.manual_save).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(save_frame, text="Charger", command=self.manual_load).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(save_frame, text="Sauver config", command=self.save_config_from_ui).grid(row=0, column=2, sticky="ew", padx=2)

        status_frame = ttk.LabelFrame(controls, text="État")
        status_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=3, pady=5)
        status_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="-")
        ttk.Label(status_frame, textvariable=self.status_var, justify="left").grid(row=0, column=0, sticky="w", padx=4, pady=4)

        logs_frame = ttk.LabelFrame(right_frame, text="Logs")
        logs_frame.grid(row=1, column=0, sticky="nsew")
        logs_frame.rowconfigure(0, weight=1)
        logs_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(logs_frame, height=12, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(logs_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        graphs_frame = ttk.LabelFrame(self.root, text="Graphiques")
        graphs_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)
        graphs_frame.rowconfigure(0, weight=1)
        graphs_frame.columnconfigure(0, weight=1)

        self.fig_metrics = Figure(figsize=(12, 4.5), dpi=100)
        self.ax_loss = self.fig_metrics.add_subplot(221)
        self.ax_speed = self.fig_metrics.add_subplot(222)
        self.ax_error = self.fig_metrics.add_subplot(223)
        self.ax_replay = self.fig_metrics.add_subplot(224)

        self.canvas_metrics = FigureCanvasTkAgg(self.fig_metrics, master=graphs_frame)
        self.canvas_metrics.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # -------------------------------------------------------------------------
    # Utilitaires UI
    # -------------------------------------------------------------------------

    def log(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", "[%s] %s\n" % (stamp, message))
        self.log_text.see("end")

    def read_int_from_entry(self, entry, default, min_value, max_value):
        """Lit un entier sans réécrire le champ pendant la frappe."""
        raw = entry.get().strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except Exception:
            return default
        return max(min_value, min(max_value, value))

    def force_int_entry_value(self, entry, value):
        entry.delete(0, "end")
        entry.insert(0, str(int(value)))

    def get_ui_values(self):
        warmup = self.read_int_from_entry(self.entry_warmup, 5, 1, 50)
        horizon = self.read_int_from_entry(self.entry_horizon, 10, 1, 2000)
        display_steps = self.read_int_from_entry(self.entry_display_steps, 50, 5, 5000)
        return warmup, horizon, display_steps

    def sync_meta_from_ui(self, rewrite_entries=True):
        warmup, horizon, display_steps = self.get_ui_values()
        self.meta["predictor"]["warmup_real_steps"] = warmup
        self.meta["predictor"]["horizon_steps"] = horizon
        self.meta["ui"]["display_real_steps"] = display_steps

        if rewrite_entries:
            self.force_int_entry_value(self.entry_warmup, warmup)
            self.force_int_entry_value(self.entry_horizon, horizon)
            self.force_int_entry_value(self.entry_display_steps, display_steps)

    def apply_predictor_shape_if_needed(self):
        """Reconstruit le RN si l'amorçage a changé.

        Le nombre d'états d'amorçage fait partie de la forme d'entrée LSTM.
        Si on le change, il faut reconstruire le modèle et vider le replay.
        """
        warmup, _, _ = self.get_ui_values()
        if warmup == self.predictor.warmup_steps:
            return

        with self.model_lock:
            self.predictor = TemporalPredictor(warmup, self.env.state_dim, self.meta)
        with self.replay_lock:
            self.replay.clear()

        self.log("Amorçage modifié : RN reconstruit, replay vidé. warmup=%d" % warmup)

    def save_config_from_ui(self):
        self.sync_meta_from_ui(rewrite_entries=True)
        save_json(CONFIG_PATH, self.meta)
        self.log("Config sauvegardée : %s" % CONFIG_PATH)

    def update_status(self):
        with self.env_lock:
            speed = float(np.linalg.norm(self.env.agent_vel))
            profile = self.env.profile_speed()
            global_step = int(self.env.global_step)
            step_count = int(self.env.step_count)

        with self.model_lock:
            train_count = int(self.predictor.train_count)
            active_warmup = int(self.predictor.warmup_steps)

        with self.replay_lock:
            replay_size = len(self.replay)

        thread_alive = self.training_thread is not None and self.training_thread.is_alive()
        _, horizon, display_steps = self.get_ui_values()

        self.status_var.set(
            "training=%s | simulation=%s | thread=%s\n"
            "global_step=%d | episode_step=%d | train_count=%d\n"
            "replay=%d | warmup actif=%d | horizon=%d | affichage=%d\n"
            "speed=%.3f | profile=%.3f"
            % (
                self.training_state,
                "ON" if self.simulation_enabled else "OFF",
                "ON" if thread_alive else "OFF",
                global_step,
                step_count,
                train_count,
                replay_size,
                active_warmup,
                horizon,
                display_steps,
                speed,
                profile,
            )
        )

    # -------------------------------------------------------------------------
    # Contrôles
    # -------------------------------------------------------------------------

    def start_resume_training(self):
        self.sync_meta_from_ui(rewrite_entries=True)
        self.apply_predictor_shape_if_needed()

        self.simulation_enabled = False
        self.training_state = "running"
        self.training_stop_event.clear()
        self.training_pause_event.clear()

        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(
                target=self.training_worker,
                daemon=True,
                name="TemporalPredictorTrainingThread",
            )
            self.training_thread.start()
            self.log("Entraînement RN : thread démarré.")
        else:
            self.log("Entraînement RN : reprise.")

        self.refresh_buttons()

    def pause_training(self):
        if self.training_state == "running":
            self.training_state = "paused"
            self.training_pause_event.set()
            self.log("Entraînement RN : pause demandée.")
        self.refresh_buttons()

    def stop_training(self):
        self.training_state = "stopped"
        self.training_stop_event.set()
        self.training_pause_event.clear()
        self.log("Entraînement RN : stop demandé. Réseau conservé.")
        self.refresh_buttons()

    def toggle_simulation(self):
        self.sync_meta_from_ui(rewrite_entries=True)
        self.apply_predictor_shape_if_needed()

        self.simulation_enabled = not self.simulation_enabled
        if self.simulation_enabled:
            if self.training_state == "running":
                self.training_state = "paused"
                self.training_pause_event.set()
            self.log("Simulation ON : entraînement suspendu.")
        else:
            self.log("Simulation OFF.")
        self.refresh_buttons()

    def refresh_buttons(self):
        self.btn_sim.configure(text="Simulation : ON" if self.simulation_enabled else "Simulation : OFF")

    def reset_scene(self):
        if self.training_state == "running":
            self.training_state = "paused"
            self.training_pause_event.set()
            self.log("Reset scène : entraînement mis en pause.")

        with self.env_lock:
            self.env.reset()

        self.log("Scène réinitialisée.")
        self.refresh_buttons()

    # -------------------------------------------------------------------------
    # Thread entraînement
    # -------------------------------------------------------------------------

    def training_worker(self):
        """Boucle d'entraînement supervisé dans un thread séparé."""
        self.ui_queue.put(("log", "Thread RN actif."))

        while not self.training_stop_event.is_set():
            if self.training_pause_event.is_set() or self.simulation_enabled:
                time.sleep(0.03)
                continue

            try:
                steps_per_loop = int(self.meta["ui"].get("train_steps_per_loop", 8))
                for _ in range(steps_per_loop):
                    if self.training_stop_event.is_set() or self.training_pause_event.is_set() or self.simulation_enabled:
                        break
                    self.training_step_threadsafe()

                self.train_predictor_batches()

            except Exception as exc:
                self.ui_queue.put(("error", "Erreur thread RN : %s" % exc))
                self.training_pause_event.set()
                break

            time.sleep(0.001)

        self.ui_queue.put(("log", "Thread RN terminé."))

    def training_step_threadsafe(self):
        """Avance la dynamique réelle et stocke un échantillon supervisé multi-pas."""
        with self.env_lock:
            _, done, info = self.env.real_step()
            history = list(self.env.state_history)

        warmup = int(self.predictor.warmup_steps)
        rollout_steps = max(1, int(self.meta["predictor"].get("rollout_train_steps", 3)))

        if len(history) >= warmup + rollout_steps:
            # Exemple avec warmup=5 et rollout_steps=3 :
            # entrée = [S0, S1, S2, S3, S4]
            # cibles = [S5, S6, S7]
            sequence = np.asarray(
                history[-warmup-rollout_steps:-rollout_steps],
                dtype=np.float32,
            )
            targets = np.asarray(
                history[-rollout_steps:],
                dtype=np.float32,
            )

            with self.replay_lock:
                self.replay.store(sequence, targets)
                replay_size = len(self.replay)

            # Diagnostic local : erreur à un pas seulement.
            with self.model_lock:
                pred = self.predictor.predict_next(sequence)
            pred_error = float(np.mean((pred - targets[0]) ** 2))

            self.ui_queue.put(("metric", {
                "speed": info["speed"],
                "profile_speed": info["profile_speed"],
                "wall_hits": info["wall_hits"],
                "pred_error": pred_error,
                "replay_size": replay_size,
            }))

        if done:
            self.ui_queue.put(("log", "Fin épisode dynamique réelle | steps=%d" % info["step_count"]))
            with self.env_lock:
                self.env.reset()

    def train_predictor_batches(self):
        """Entraîne le réseau sur des mini-batches multi-pas auto-régressifs."""
        batch_size = int(self.meta["predictor"].get("batch_size", 128))
        min_replay = int(self.meta["predictor"].get("min_replay_to_train", 512))
        train_batches = int(self.meta["predictor"].get("train_batches_per_loop", 2))

        with self.replay_lock:
            replay_size = len(self.replay)

        if replay_size < min_replay:
            return

        stats_list = []
        for _ in range(train_batches):
            with self.replay_lock:
                x, y = self.replay.sample(batch_size)

            with self.model_lock:
                stats = self.predictor.train_on_batch(x, y)

            stats_list.append(stats)

        if stats_list:
            self.ui_queue.put(("train_loss", {
                "loss": float(np.mean([s["loss"] for s in stats_list])),
                "one_step_loss": float(np.mean([s["one_step_loss"] for s in stats_list])),
                "rollout_loss": float(np.mean([s["rollout_loss"] for s in stats_list])),
                "rollout_steps": int(stats_list[-1].get("rollout_steps", 1)),
                "train_count": int(self.predictor.train_count),
                "replay_size": int(replay_size),
            }))

    def process_ui_queue(self):
        """Traite les messages du thread RN depuis le thread Tkinter."""
        max_messages = int(self.meta["ui"].get("queue_max_messages_per_frame", 300))
        count = 0

        while count < max_messages:
            try:
                kind, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            count += 1

            if kind == "log":
                self.log(str(payload))

            elif kind == "metric":
                self.metrics["speed"].append(payload["speed"])
                self.metrics["profile_speed"].append(payload["profile_speed"])
                self.metrics["wall_hits"].append(payload["wall_hits"])
                self.metrics["pred_error"].append(payload["pred_error"])
                self.metrics["replay_size"].append(payload["replay_size"])

            elif kind == "train_loss":
                self.metrics["loss"].append(payload["loss"])
                self.metrics["one_step_loss"].append(payload.get("one_step_loss", payload["loss"]))
                self.metrics["rollout_loss"].append(payload.get("rollout_loss", 0.0))
                self.metrics["replay_size"].append(payload["replay_size"])
                self.log(
                    "Train RN #%d | loss=%.6f | one=%.6f | rollout=%.6f | steps=%d | replay=%d"
                    % (
                        payload["train_count"],
                        payload["loss"],
                        payload.get("one_step_loss", payload["loss"]),
                        payload.get("rollout_loss", 0.0),
                        payload.get("rollout_steps", 1),
                        payload["replay_size"],
                    )
                )

            elif kind == "error":
                self.training_state = "paused"
                self.simulation_enabled = False
                self.training_pause_event.set()
                self.refresh_buttons()
                self.log(str(payload))
                messagebox.showerror("Erreur thread RN", str(payload))

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def simulation_step(self):
        """Mode simulation : la dynamique réelle avance, le RN ne s'entraîne pas."""
        with self.env_lock:
            _, done, info = self.env.real_step()

        self.metrics["speed"].append(info["speed"])
        self.metrics["profile_speed"].append(info["profile_speed"])
        self.metrics["wall_hits"].append(info["wall_hits"])

        if done:
            with self.env_lock:
                self.env.reset()
            self.log("Simulation : épisode relancé après max_steps.")

    # -------------------------------------------------------------------------
    # Scène : noir / bleu / orange
    # -------------------------------------------------------------------------

    def get_real_scene_segments(self):
        """Retourne les segments réel ancien/noir et réel horizon/bleu.

        Soit S l'état courant :
            noir = trajectoire réelle jusqu'à S - horizon
            bleu = trajectoire réelle de S - horizon à S
        Le total affiché est limité par ui.display_real_steps.
        """
        _, horizon, display_steps = self.get_ui_values()

        with self.env_lock:
            history = list(self.env.state_history)

            if len(history) < 2:
                return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

            positions = np.asarray([self.env.state_to_position(s) for s in history], dtype=np.float32)

        end_idx = len(positions) - 1
        start_display = max(0, end_idx - display_steps)
        split_idx = max(start_display, end_idx - horizon)

        black = positions[start_display:split_idx + 1]
        blue = positions[split_idx:end_idx + 1]
        return black, blue

    def compute_prediction_on_horizon_window(self):
        """Calcule l'orange sur la même fenêtre que le bleu.

        Départ orange : état réel à S - horizon.
        Amorçage RN : warmup états réels se terminant à S - horizon.
        Déroulé RN : horizon états auto-prédits.
        """
        if not bool(self.pred_enabled_var.get()):
            return np.zeros((0, 2), dtype=np.float32)

        _, horizon, _ = self.get_ui_values()
        warmup = int(self.predictor.warmup_steps)

        with self.env_lock:
            history = list(self.env.state_history)

        if len(history) < warmup + horizon + 1:
            return np.zeros((0, 2), dtype=np.float32)

        end_idx = len(history) - 1
        start_idx = end_idx - horizon
        warmup_start = start_idx - warmup + 1

        if warmup_start < 0:
            return np.zeros((0, 2), dtype=np.float32)

        with self.env_lock:
            sequence = np.asarray(history[warmup_start:start_idx + 1], dtype=np.float32)
            start_pos = self.env.state_to_position(history[start_idx])

        points = [start_pos.copy()]

        with self.model_lock:
            for _ in range(horizon):
                pred_state = self.predictor.predict_next(sequence)

                with self.env_lock:
                    pred_state = self.env.sanitize_predicted_state(pred_state)
                    pred_pos = self.env.state_to_position(pred_state)

                points.append(pred_pos.copy())
                sequence = np.concatenate([sequence[1:], pred_state.reshape(1, -1)], axis=0).astype(np.float32)

        return np.asarray(points, dtype=np.float32)

    def update_trajectory_error_metrics(self, blue, orange):
        """Mesure l'écart entre réel horizon bleu et RN horizon orange.

        blue et orange contiennent chacun le point de départ puis horizon points.
        On compare point à point sur la longueur commune.
        """
        if len(blue) < 2 or len(orange) < 2:
            return

        with self.env_lock:
            current_step = int(self.env.global_step)

        # Ne pas remplir le graphe avec des doublons si la scène ne bouge pas.
        if current_step == self.last_trajectory_error_step:
            return
        self.last_trajectory_error_step = current_step

        n = min(len(blue), len(orange))
        if n <= 1:
            return

        blue_cmp = np.asarray(blue[:n], dtype=np.float32)
        orange_cmp = np.asarray(orange[:n], dtype=np.float32)
        dists = np.linalg.norm(blue_cmp - orange_cmp, axis=1)

        self.metrics["traj_mean_error"].append(float(np.mean(dists)))
        self.metrics["traj_final_error"].append(float(dists[-1]))
        self.metrics["traj_max_error"].append(float(np.max(dists)))

    # -------------------------------------------------------------------------
    # Boucle principale Tkinter
    # -------------------------------------------------------------------------

    def main_loop(self):
        try:
            self.process_ui_queue()

            if self.simulation_enabled:
                for _ in range(int(self.meta["ui"].get("sim_steps_per_tick", 4))):
                    self.simulation_step()

            self.render_all()
            self.maybe_autosave()
        except Exception as exc:
            self.training_state = "paused"
            self.simulation_enabled = False
            self.training_pause_event.set()
            self.refresh_buttons()
            self.log("ERREUR : %s" % exc)
            messagebox.showerror("Erreur", str(exc))

        self.root.after(int(self.meta["ui"].get("refresh_ms", 35)), self.main_loop)

    # -------------------------------------------------------------------------
    # Rendu
    # -------------------------------------------------------------------------

    def render_all(self):
        self.render_scene()
        self.render_graphs()
        self.update_status()

    def render_scene(self):
        with self.env_lock:
            width = float(self.env.width)
            height = float(self.env.height)
            agent_pos = self.env.agent_pos.copy()
            agent_radius = float(self.env.agent_radius)

        _, horizon, display_steps = self.get_ui_values()
        black, blue = self.get_real_scene_segments()
        orange = self.compute_prediction_on_horizon_window()
        self.update_trajectory_error_metrics(blue, orange)

        self.ax_scene.clear()
        self.ax_scene.set_title(
            "Noir=réel ancien | Bleu=réel horizon | Orange=RN horizon | H=%d | affichage=%d"
            % (horizon, display_steps)
        )
        self.ax_scene.set_xlim(0.0, width)
        self.ax_scene.set_ylim(0.0, height)
        self.ax_scene.set_aspect("equal", adjustable="box")
        self.ax_scene.grid(True, alpha=0.25)

        border = patches.Rectangle((0, 0), width, height, fill=False, linewidth=1.5)
        self.ax_scene.add_patch(border)

        if len(black) > 1:
            self.ax_scene.plot(
                black[:, 0],
                black[:, 1],
                color="black",
                linewidth=1.1,
                alpha=0.75,
                label="réel ancien",
            )

        if len(blue) > 1:
            self.ax_scene.plot(
                blue[:, 0],
                blue[:, 1],
                color="blue",
                linewidth=2.0,
                alpha=0.95,
                label="réel horizon",
            )

        if len(orange) > 1:
            self.ax_scene.plot(
                orange[:, 0],
                orange[:, 1],
                color="orange",
                linestyle="--",
                linewidth=2.3,
                alpha=0.95,
                label="RN horizon",
            )
            self.ax_scene.scatter(orange[-1, 0], orange[-1, 1], color="orange", s=28, alpha=0.95)

        agent = patches.Circle(agent_pos, agent_radius, fill=True, color="blue", alpha=0.95)
        self.ax_scene.add_patch(agent)

        self.ax_scene.legend(loc="upper right", fontsize=8)
        self.canvas_scene.draw_idle()

    def render_graphs(self):
        self.ax_loss.clear()
        if self.metrics["loss"]:
            self.ax_loss.plot(
                list(self.metrics["loss"]),
                linewidth=1.1,
                label="loss totale",
            )
        if self.metrics["one_step_loss"]:
            self.ax_loss.plot(
                list(self.metrics["one_step_loss"]),
                linewidth=1.0,
                linestyle="--",
                label="loss 1 pas",
            )
        if self.metrics["rollout_loss"]:
            self.ax_loss.plot(
                list(self.metrics["rollout_loss"]),
                linewidth=1.0,
                linestyle=":",
                label="loss rollout",
            )
        self.ax_loss.set_title("Loss RN multi-pas")
        self.ax_loss.grid(True, alpha=0.25)
        self.ax_loss.legend(fontsize=7, loc="best")

        self.ax_speed.clear()
        if self.metrics["speed"]:
            self.ax_speed.plot(list(self.metrics["speed"]), linewidth=1.0, label="speed réelle")
        if self.metrics["profile_speed"]:
            self.ax_speed.plot(list(self.metrics["profile_speed"]), linewidth=1.0, linestyle="--", label="profil vitesse")
        self.ax_speed.set_title("Vitesse cyclique")
        self.ax_speed.grid(True, alpha=0.25)
        self.ax_speed.legend(fontsize=7, loc="best")

        self.ax_error.clear()
        if self.metrics["pred_error"]:
            self.ax_error.plot(list(self.metrics["pred_error"]), linewidth=1.0, label="erreur préd. 1 pas")
        self.ax_error.set_title("Erreur prédiction")
        self.ax_error.grid(True, alpha=0.25)
        self.ax_error.legend(fontsize=7, loc="best")

        self.ax_replay.clear()
        if self.metrics["traj_mean_error"]:
            self.ax_replay.plot(
                list(self.metrics["traj_mean_error"]),
                linewidth=1.2,
                label="écart moyen",
            )
        if self.metrics["traj_final_error"]:
            self.ax_replay.plot(
                list(self.metrics["traj_final_error"]),
                linewidth=1.0,
                linestyle="--",
                label="écart final",
            )
        if self.metrics["traj_max_error"]:
            self.ax_replay.plot(
                list(self.metrics["traj_max_error"]),
                linewidth=1.0,
                linestyle=":",
                label="écart max",
            )
        self.ax_replay.set_title("Écart trajectoire RN vs réel")
        self.ax_replay.grid(True, alpha=0.25)
        self.ax_replay.legend(fontsize=7, loc="best")

        self.fig_metrics.tight_layout()
        self.canvas_metrics.draw_idle()

    # -------------------------------------------------------------------------
    # Sauvegarde / restauration
    # -------------------------------------------------------------------------

    def _timestamp_dir(self):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return AUTOSAVES_DIR / stamp

    def _jsonable(self, value):
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        if isinstance(value, deque):
            return [self._jsonable(v) for v in list(value)]
        if isinstance(value, list):
            return [self._jsonable(v) for v in value]
        if isinstance(value, dict):
            return {k: self._jsonable(v) for k, v in value.items()}
        return value

    def _metrics_as_jsonable(self):
        return {key: list(value) for key, value in self.metrics.items()}

    def _env_state_as_jsonable(self):
        with self.env_lock:
            state = self.env.copy_state()
            state["trail"] = [p.copy() for p in list(self.env.trail)]
            state["state_history"] = [s.copy() for s in list(self.env.state_history)[-1000:]]
        return self._jsonable(state)

    def _restore_env_state(self, state):
        if not isinstance(state, dict):
            return

        required = ["agent_pos", "agent_vel", "phase_offset", "step_count", "global_step", "episode_id"]
        if not all(key in state for key in required):
            return

        with self.env_lock:
            self.env.set_state({
                "agent_pos": np.asarray(state["agent_pos"], dtype=np.float32),
                "agent_vel": np.asarray(state["agent_vel"], dtype=np.float32),
                "phase_offset": float(state["phase_offset"]),
                "step_count": int(state["step_count"]),
                "global_step": int(state["global_step"]),
                "episode_id": int(state["episode_id"]),
            })

            self.env.trail = deque(maxlen=int(self.meta["ui"].get("trail_len", 1200)))
            for p in state.get("trail", []):
                self.env.trail.append(np.asarray(p, dtype=np.float32))
            if not self.env.trail:
                self.env.trail.append(self.env.agent_pos.copy())

            max_hist = max(10000, int(self.meta["ui"].get("trail_len", 1200)) * 2)
            self.env.state_history = deque(maxlen=max_hist)
            for s in state.get("state_history", []):
                self.env.state_history.append(np.asarray(s, dtype=np.float32))
            if len(self.env.state_history) == 0:
                self.env.state_history.append(self.env.get_state_vector())

    def _restore_metrics(self, data):
        if not isinstance(data, dict):
            return
        maxlen = int(self.meta["ui"].get("graph_history_len", 1200))
        for key in self.metrics.keys():
            self.metrics[key] = deque(data.get(key, []), maxlen=maxlen)

    def manual_save(self):
        self.sync_meta_from_ui(rewrite_entries=True)
        save_dir = self._timestamp_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        with self.model_lock:
            self.predictor.save(save_dir)

        save_json(save_dir / "meta_params.json", self.meta)
        save_json(save_dir / "metrics.json", self._metrics_as_jsonable())
        save_json(save_dir / "env_state.json", self._env_state_as_jsonable())

        with self.replay_lock:
            replay_size = len(self.replay)

        with self.env_lock:
            global_step = int(self.env.global_step)
            episode_id = int(self.env.episode_id)
            episode_step = int(self.env.step_count)

        training_state = {
            "training_state": self.training_state,
            "simulation_enabled": bool(self.simulation_enabled),
            "global_step": global_step,
            "episode_id": episode_id,
            "episode_step": episode_step,
            "replay_size": int(replay_size),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(save_dir / "training_state.json", training_state)

        # Pas de last_autosave / last_autosave.json.
        save_json(CONFIG_PATH, self.meta)
        self.log("Sauvegarde créée : %s" % save_dir)
        self.cleanup_old_autosaves()

    def manual_load(self):
        self.training_state = "paused"
        self.training_pause_event.set()
        self.simulation_enabled = False
        self.refresh_buttons()

        directory = filedialog.askdirectory(
            title="Sélectionner un dossier autosave horodaté",
            initialdir=str(AUTOSAVES_DIR),
        )
        if not directory:
            return

        directory = Path(directory)
        try:
            meta_path = directory / "meta_params.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    loaded_meta = json.load(f)
                self.meta = deep_merge(DEFAULT_META_PARAMS, loaded_meta)
                save_json(CONFIG_PATH, self.meta)

            with self.env_lock:
                self.env = RealDynamics2D(self.meta, seed=self.seed)

            warmup = int(self.meta["predictor"].get("warmup_real_steps", 5))
            with self.model_lock:
                self.predictor = TemporalPredictor(warmup, self.env.state_dim, self.meta)
                self.predictor.load(directory)

            env_state_path = directory / "env_state.json"
            if env_state_path.exists():
                with open(env_state_path, "r", encoding="utf-8") as f:
                    self._restore_env_state(json.load(f))

            metrics_path = directory / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    self._restore_metrics(json.load(f))

            with self.replay_lock:
                self.replay.clear()

            self.apply_loaded_meta_to_ui()
            self.training_state = "paused"
            self.simulation_enabled = False
            self.refresh_buttons()
            self.log("Restauration effectuée : %s" % directory)

        except Exception as exc:
            self.log("Échec restauration : %s" % exc)
            messagebox.showerror("Restauration impossible", str(exc))

    def apply_loaded_meta_to_ui(self):
        self.entry_warmup.delete(0, "end")
        self.entry_warmup.insert(0, str(int(self.meta["predictor"].get("warmup_real_steps", 5))))

        self.entry_horizon.delete(0, "end")
        self.entry_horizon.insert(0, str(int(self.meta["predictor"].get("horizon_steps", 10))))

        self.entry_display_steps.delete(0, "end")
        self.entry_display_steps.insert(0, str(int(self.meta["ui"].get("display_real_steps", 50))))

    def maybe_autosave(self):
        if not bool(self.meta["autosave"].get("enabled", True)):
            return
        every_minutes = float(self.meta["autosave"].get("every_minutes", 15))
        now = time.time()
        if now - self.last_autosave_time >= every_minutes * 60.0:
            self.manual_save()
            self.last_autosave_time = now

    def cleanup_old_autosaves(self):
        keep_last = int(self.meta["autosave"].get("keep_last", 30))
        if keep_last <= 0:
            return

        dirs = [p for p in AUTOSAVES_DIR.iterdir() if p.is_dir()]
        dirs.sort(key=lambda p: p.name)
        excess = dirs[:-keep_last]

        for d in excess:
            try:
                for child in d.iterdir():
                    if child.is_file():
                        child.unlink()
                d.rmdir()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Fermeture
    # -------------------------------------------------------------------------

    def on_close(self):
        self.training_state = "stopped"
        self.simulation_enabled = False
        self.training_stop_event.set()
        self.training_pause_event.clear()
        self.sync_meta_from_ui(rewrite_entries=True)
        save_json(CONFIG_PATH, self.meta)
        self.root.destroy()


# =============================================================================
# Main
# =============================================================================

def main():
    root = tk.Tk()
    root.geometry("1450x900")
    TemporalAgentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
