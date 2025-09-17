import math
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class STKRewardShaping(gym.Wrapper):
	"""
	Reward shaping pour SuperTuxKart (pystk2_gymnasium) visant à:
	- Récompenser le progrès le long de la piste (delta de « progress » 0..1)
	- Récompenser la vitesse orientée vers l'avant (si dispo)
	- Pénaliser off-track, glisse latérale, marche arrière, steering saccadé
	- Bonus de checkpoints / finish si dispo

	Hypothèses douces sur les clés de info; le wrapper est défensif et s'adapte:
	- progress:    info["progress"] in [0,1] (sinon: info["distance"] croissante)
	- forward_speed/speed: vitesse vers l'avant (>=0) (ou vitesse approximative)
	- lateral_speed: norme vitesse latérale (>=0)
	- off_track/is_off_track: bool indiquant sortie de piste
	- finished/is_finished: bool quand la course est finie
	- position/num_karts: rang & nb de karts (optionnel, bonus léger)

	Le wrapper ajoute info["rew_components"] pour le monitoring.
	"""

	def __init__(self, env: gym.Env,
				 scale_progress: float = 1.0,
				 w_forward_speed: float = 0.02,
				 w_lateral: float = 0.05,
				 p_offtrack: float = 0.2,
				 p_reverse: float = 0.5,
				 p_steer_jerk: float = 0.01,
				 w_throttle_align: float = 0.1,
				 w_steer_mag: float = 0.01,
				 bonus_checkpoint: float = 1.0,
				 bonus_finish: float = 50.0,
				 use_position_bonus: bool = False,
				 max_progress_step: float = 0.1,
				 stuck_window: int = 20,
				 stuck_threshold: float = 0.005,
				 stuck_penalty: float = 0.5):
		super().__init__(env)
		self.prev_progress: Optional[float] = None
		self.prev_distance: Optional[float] = None
		self.prev_action: Optional[np.ndarray] = None
		self.progress_hist: deque = deque(maxlen=stuck_window)

		# Coefficients
		self.scale_progress = scale_progress      # applied after clipping
		self.w_forward_speed = w_forward_speed
		self.w_lateral = w_lateral
		self.p_offtrack = p_offtrack
		self.p_reverse = p_reverse
		self.p_steer_jerk = p_steer_jerk
		self.w_throttle_align = w_throttle_align
		self.w_steer_mag = w_steer_mag
		self.bonus_checkpoint = bonus_checkpoint
		self.bonus_finish = bonus_finish
		self.use_position_bonus = use_position_bonus
		self.max_progress_step = max_progress_step
		self.stuck_threshold = stuck_threshold
		self.stuck_penalty = stuck_penalty

	# -------- helpers
	@staticmethod
	def _get_bool(info: Dict[str, Any], *keys: str) -> bool:
		for k in keys:
			v = info.get(k, None)
			if isinstance(v, (bool, np.bool_)):
				return bool(v)
			if isinstance(v, (int, np.integer)):
				return bool(int(v))
		return False

	@staticmethod
	def _get_float(info: Dict[str, Any], default: float, *keys: str) -> float:
		for k in keys:
			v = info.get(k, None)
			if v is None:
				continue
			try:
				return float(v)
			except Exception:
				continue
		return default

	def _extract_progress(self, info: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
		"""
		Returns (progress_0_1, distance_raw). One or both may be None.
		"""
		prog = info.get("progress", None)
		if prog is not None:
			try:
				return float(prog), None
			except Exception:
				pass
		# fallback: monotonic distance
		dist = info.get("distance", None)
		if dist is not None:
			try:
				return None, float(dist)
			except Exception:
				pass
		# some envs report lap_progress
		lap_prog = info.get("lap_progress", None)
		if lap_prog is not None:
			try:
				return float(lap_prog), None
			except Exception:
				pass
		return None, None

	def _progress_delta(self, info: Dict[str, Any]) -> float:
		p, d = self._extract_progress(info)
		delta = 0.0
		if p is not None:
			if self.prev_progress is None:
				delta = 0.0
			else:
				delta = p - self.prev_progress
			self.prev_progress = p
		elif d is not None:
			if self.prev_distance is None:
				delta = 0.0
			else:
				delta = d - self.prev_distance
			self.prev_distance = d
		else:
			delta = 0.0
		# Clamp extreme deltas to be safe (teleports/checkpoints)
		return float(np.clip(delta, -self.max_progress_step, self.max_progress_step))

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		# initialize trackers
		self.prev_progress = None
		self.prev_distance = None
		self.prev_action = None
		self.progress_hist.clear()
		# Prime progress trackers with first info
		self._progress_delta(info)
		return obs, info

	def step(self, action):
		obs, env_reward, terminated, truncated, info = self.env.step(action)

		# Compute deltas & components
		d_prog = self._progress_delta(info)
		self.progress_hist.append(max(0.0, d_prog))

		forward_speed = self._get_float(info, 0.0, "forward_speed", "speed")
		lateral_speed = abs(self._get_float(info, 0.0, "lateral_speed"))
		off_track = self._get_bool(info, "off_track", "is_off_track")
		finished = self._get_bool(info, "finished", "is_finished")

		# Steering jerk (requires prev action), assume steering at index 0 if continuous
		steer_jerk = 0.0
		if isinstance(action, (np.ndarray, list, tuple)):
			try:
				steer = float(action[0])
				prev_steer = float(self.prev_action[0]) if self.prev_action is not None else steer
				steer_jerk = abs(steer - prev_steer)
			except Exception:
				steer_jerk = 0.0
		self.prev_action = np.array(action, dtype=float) if isinstance(action, (np.ndarray, list, tuple)) else None

		# Reverse detection: negative forward projection if available
		reverse = self._get_bool(info, "reverse")
		if not reverse:
			# Heuristic: very small or decreasing progress with some speed backwards
			reverse = (forward_speed < 0.0)

		# Progress reward: scale a clipped delta to be ~[-1,1]
		dnorm = float(np.clip(d_prog / max(self.max_progress_step, 1e-6), -1.0, 1.0))
		r_progress = dnorm * self.scale_progress

		# Speed alignment (only forward component rewarded)
		r_forward = self.w_forward_speed * max(0.0, forward_speed)

		# Penalties
		r_off = -self.p_offtrack * (1.0 if off_track else 0.0)
		r_lat = -self.w_lateral * float(lateral_speed)
		r_rev = -self.p_reverse * (1.0 if reverse else 0.0)
		r_jerk = -self.p_steer_jerk * float(steer_jerk)

		# Action-aligned shaping (safe without env-specific signals)
		steer_val = None
		accel_val = None
		if isinstance(action, (np.ndarray, list, tuple)) and len(action) >= 1:
			try:
				steer_val = float(action[0])
			except Exception:
				steer_val = None
		if isinstance(action, (np.ndarray, list, tuple)) and len(action) >= 2:
			try:
				accel_val = float(action[1])
			except Exception:
				accel_val = None

		# Encourage throttle only when moving forward (positive progress)
		r_throttle_align = 0.0
		if accel_val is not None:
			# Map accel into [0,1] approximately (works for either [0,1] or [-1,1])
			a_go = 0.5 * (accel_val + 1.0)
			a_go = float(np.clip(a_go, 0.0, 1.0))
			r_throttle_align = self.w_throttle_align * a_go * max(0.0, dnorm)

		# Penalize strong steering when not progressing (discourage spinning)
		r_steer_mag = 0.0
		if steer_val is not None:
			r_steer_mag = -self.w_steer_mag * abs(steer_val) * (1.0 - max(0.0, dnorm))

		# Stuck penalty over a small window of steps
		stuck = (len(self.progress_hist) == self.progress_hist.maxlen and sum(self.progress_hist) < self.stuck_threshold)
		r_stuck = -self.stuck_penalty * (1.0 if stuck else 0.0)

		# Optional mild position shaping (disabled by default to avoid bias)
		r_pos = 0.0
		if self.use_position_bonus:
			pos = self._get_float(info, 1.0, "position")  # 1..K
			karts = self._get_float(info, 1.0, "num_karts", "num_kart")
			if karts > 0:
				r_pos = 0.05 * ((karts - pos) / karts)  # [-~0.05, +~0.05]

		# Terminal bonuses
		r_finish = self.bonus_finish if finished else 0.0
		# Checkpoint detection (coarse): large progress jump
		r_checkpoint = self.bonus_checkpoint if d_prog > 0.02 else 0.0

		shaped = (
			r_progress + r_forward + r_off + r_lat + r_rev + r_jerk + r_stuck + r_pos + r_checkpoint + r_finish
			+ r_throttle_align + r_steer_mag
		)

		# Expose components for logging/diagnostics
		info = dict(info)  # shallow copy
		info.setdefault("rew_components", {})
		info["rew_components"].update({
			"progress": r_progress,
			"forward": r_forward,
			"offtrack": r_off,
			"lateral": r_lat,
			"reverse": r_rev,
			"jerk": r_jerk,
			"stuck": r_stuck,
			"position": r_pos,
			"checkpoint": r_checkpoint,
			"finish": r_finish,
			"env_reward": float(env_reward),
			"d_progress": float(d_prog),
			"forward_speed": float(forward_speed),
			"lateral_speed": float(lateral_speed),
			"off_track": bool(off_track),
			"throttle_align": float(r_throttle_align),
			"steer_mag": float(r_steer_mag),
		})

		# Return shaped reward (we replace env reward)
		return obs, shaped, terminated, truncated, info

