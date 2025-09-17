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
				  stuck_penalty: float = 0.5,
				  obs_index_map: Optional[Dict[str, int]] = None,
				  auto_calibrate: bool = True,
				  calib_steps: int = 256):
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
		# Optional mapping for extracting signals directly from observations
		# Keys supported: 'forward_speed', 'lateral_speed', 'off_track', 'yaw_sin', 'yaw_cos'
		self.obs_index_map: Optional[Dict[str, int]] = dict(obs_index_map) if obs_index_map else None
		# Auto-calibration of observation indices
		self.auto_calibrate: bool = bool(auto_calibrate)
		self.calib_steps: int = int(max(32, calib_steps))
		self._calib_done: bool = False
		self._calib_X: Optional[list] = None
		self._calib_dprog: Optional[list] = None
		self._calib_off_info: Optional[list] = None

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
		# Reset calibration buffers
		self._calib_done = (self.obs_index_map is not None) and not self.auto_calibrate
		self._calib_X = [] if self.auto_calibrate else None
		self._calib_dprog = [] if self.auto_calibrate else None
		self._calib_off_info = [] if self.auto_calibrate else None
		# Prime progress trackers with first info
		self._progress_delta(info)
		return obs, info

	def _finalize_calibration(self):
		"""
		Infers indices for forward_speed, lateral_speed (best-effort) and off_track
		from the collected continuous observations and d_progress signal.
		"""
		if self._calib_done or not self.auto_calibrate:
			return
		X = np.asarray(self._calib_X, dtype=float) if self._calib_X else None
		dp = np.asarray(self._calib_dprog, dtype=float) if self._calib_dprog else None
		if X is None or X.ndim != 2 or X.shape[0] < 10 or dp is None or dp.shape[0] != X.shape[0]:
			self._calib_done = True
			return

		# Forward speed index: highest positive correlation with d_progress
		corrs = []
		for i in range(X.shape[1]):
			xi = X[:, i]
			std = float(np.std(xi))
			if std < 1e-8:
				c = 0.0
			else:
				try:
					c = float(np.corrcoef(xi, dp)[0, 1])
				except Exception:
					c = 0.0
			corrs.append((i, c, std))
		corrs.sort(key=lambda t: abs(t[1]), reverse=True)
		fwd_idx = next((i for i, c, s in corrs if c > 0.2 and s > 1e-6), None)

		# Off-track index: near-binary and (if info provided) correlates with off_track
		off_idx = None
		if self._calib_off_info and len(self._calib_off_info) == X.shape[0] and any(self._calib_off_info):
			off = np.asarray(self._calib_off_info, dtype=float)
			best = (-1e9, None)
			for i in range(X.shape[1]):
				xi = X[:, i]
				# Normalize to [0,1] and measure binariness
				xn = (xi - np.min(xi)) / (max(1e-9, np.max(xi) - np.min(xi)))
				bin_score = float(np.mean((xn < 0.1) | (xn > 0.9)))
				try:
					c = float(np.corrcoef((xn > 0.5).astype(float), off)[0, 1])
				except Exception:
					c = 0.0
				score = bin_score + abs(c)
				if score > best[0]:
					best = (score, i)
			off_idx = best[1]
		else:
			# fallback: pick index with most near-binary behaviour and negative corr with d_progress
			best = (-1e9, None)
			for i in range(X.shape[1]):
				xi = X[:, i]
				xn = (xi - np.min(xi)) / (max(1e-9, np.max(xi) - np.min(xi)))
				bin_score = float(np.mean((xn < 0.1) | (xn > 0.9)))
				try:
					c = float(np.corrcoef((xn > 0.5).astype(float), (dp > 1e-4).astype(float))[0, 1])
				except Exception:
					c = 0.0
				score = bin_score + (0.2 if c < 0 else 0.0)
				if score > best[0]:
					best = (score, i)
			off_idx = best[1]

		# Lateral speed: heuristic — choose index with high variance, low |corr| with d_progress
		lat_idx = None
		for i, c, s in corrs:
			if abs(c) < 0.15 and s > 1e-3:
				lat_idx = i
				break

		new_map = {}
		if fwd_idx is not None:
			new_map["forward_speed"] = int(fwd_idx)
		if lat_idx is not None:
			new_map["lateral_speed"] = int(lat_idx)
		if off_idx is not None:
			new_map["off_track"] = int(off_idx)

		# Merge with any provided mapping but let auto-calibration override if different
		base = dict(self.obs_index_map) if isinstance(self.obs_index_map, dict) else {}
		base.update(new_map)
		self.obs_index_map = base if base else None
		self._calib_done = True

	def step(self, action):
		obs, env_reward, terminated, truncated, info = self.env.step(action)

		# Compute deltas & components
		d_prog = self._progress_delta(info)
		self.progress_hist.append(max(0.0, d_prog))

		# Accumulate calibration data from raw observations
		if self.auto_calibrate and not self._calib_done and isinstance(obs, dict) and "continuous" in obs:
			try:
				vec = np.asarray(obs["continuous"], dtype=float).reshape(-1)
				self._calib_X.append(vec)
				self._calib_dprog.append(float(d_prog))
				self._calib_off_info.append(1.0 if self._get_bool(info, "off_track", "is_off_track") else 0.0)
				if len(self._calib_X) >= self.calib_steps:
					self._finalize_calibration()
			except Exception:
				pass

		forward_speed = self._get_float(info, 0.0, "forward_speed", "speed")
		lateral_speed = abs(self._get_float(info, 0.0, "lateral_speed"))
		off_track = self._get_bool(info, "off_track", "is_off_track")
		# More robust finished detection
		finished = self._get_bool(info, "finished", "is_finished")
		if not finished:
			p, _ = self._extract_progress(info)
			if p is not None and p >= 0.999:
				finished = True

		# If mapping provided and observation is available, override missing values
		try:
			if self.obs_index_map and isinstance(obs, dict) and "continuous" in obs:
				vec = np.asarray(obs["continuous"], dtype=float)
				# forward_speed from obs
				ix = self.obs_index_map.get("forward_speed") if isinstance(self.obs_index_map, dict) else None
				if ix is not None and 0 <= ix < vec.shape[0]:
					fv = float(vec[ix])
					# Only override if info didn't provide anything useful
					if abs(forward_speed) < 1e-6:
						forward_speed = fv
				# lateral_speed from obs
				ix = self.obs_index_map.get("lateral_speed") if isinstance(self.obs_index_map, dict) else None
				if ix is not None and 0 <= ix < vec.shape[0]:
					lv = abs(float(vec[ix]))
					if abs(lateral_speed) < 1e-6:
						lateral_speed = lv
				# off_track from obs (treat >0.5 as True for boolean flags)
				ix = self.obs_index_map.get("off_track") if isinstance(self.obs_index_map, dict) else None
				if ix is not None and 0 <= ix < vec.shape[0]:
					ot = bool(float(vec[ix]) > 0.5)
					# If info lacks off_track, use obs flag
					if not off_track:
						off_track = ot
		except Exception:
			# be defensive, never break env stepping due to mapping
			pass

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
		# Expose calibration diagnostics periodically
		if self.auto_calibrate:
			info.setdefault("obs_index_map", dict(self.obs_index_map) if self.obs_index_map else None)
			info.setdefault("calibration_done", bool(self._calib_done))

		# Return shaped reward (we replace env reward)
		return obs, shaped, terminated, truncated, info

