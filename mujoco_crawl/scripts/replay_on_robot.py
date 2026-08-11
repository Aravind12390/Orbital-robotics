#!/usr/bin/env python3
"""
Executes trajectory_plan.json (from extract_trajectory.py, the gripper-
free straight-ladder crawl) on a real Go2 via LowCmd.

No finger/jaws control at all -- no gripper on this version, matching the
real Go2's bare foot directly. "Pressing" a rung is handled by simply
commanding the FL joints to their press-phase target and holding there.

FORCE LOGGING (new): converts the robot's real motor_state.tau_est (per-
joint torque, Nm) into an estimated FORCE at the foot, in Newtons, using
the FL leg's Jacobian -- the same relationship (tau = J^T @ F) used
throughout the MuJoCo side of this project. Verified this conversion
round-trips exactly against the actual robot model before shipping it (a
known 50N test force recovers to 50N through tau=J^T@F then F=pinv(J^T)@tau,
and the Jacobian is well-conditioned, condition number ~2.7, at a real
reach configuration -- not a numerically fragile inversion). This makes
the logged numbers directly comparable to the sim's grip_force() values
you've already seen (60-100N range), instead of raw, harder-to-interpret
per-joint torques.

Sampled continuously throughout every phase (not just once at the end),
and summarized per rung in both the live printout and a saved CSV log.

UNVERIFIED AGAINST REAL HARDWARE beyond the one phase you've already run
successfully. Read every safety note below.

SAFETY, non-negotiable:
  - Increase --phase-limit one step at a time. Confirm each phase is safe
    before trusting the next.
  - Keep the robot supported/braced for "pressed" phases -- this script
    cannot verify the foot is actually touching anything real.
  - GAINS ARE PLACEHOLDERS unless you've already validated them via
    first_joint_test.py.
  - A force spike well above what the sim reported for that phase is a
    real signal something is wrong -- stop (Ctrl+C) rather than continue.

Usage:
    python3 replay_on_robot.py <network_interface> [--phase-limit N] [--model PATH]
"""
import sys
import json
import time
import csv
import argparse
import threading
import math
import numpy as np
import mujoco

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"

FL_IDX = [3, 4, 5]
FR_IDX = [0, 1, 2]
RR_IDX = [6, 7, 8]
RL_IDX = [9, 10, 11]

KP_LEG, KD_LEG = 25.0, 1.5   # PLACEHOLDER -- see safety note above

CONTROL_HZ = 250
REPLAY_SPEED = 0.5

DEFAULT_MODEL = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'

crc = CRC()
latest_state = {'q': [None] * 12, 'tau_est': [None] * 12,
                 'imu_quat': None, 'imu_accel': None, 'imu_gyro': None,
                 'foot_force': None}

# Standard Unitree foot_force/foot_force_est order: FR, FL, RR, RL
FOOT_FORCE_FL_IDX = 1


def lowstate_handler(msg: LowState_):
    for i in range(12):
        latest_state['q'][i] = msg.motor_state[i].q
        latest_state['tau_est'][i] = msg.motor_state[i].tau_est
    latest_state['imu_quat'] = list(msg.imu_state.quaternion)      # (w, x, y, z)
    latest_state['imu_accel'] = list(msg.imu_state.accelerometer)  # body frame, m/s^2, includes gravity
    latest_state['imu_gyro'] = list(msg.imu_state.gyroscope)       # body frame, rad/s
    latest_state['foot_force'] = list(msg.foot_force)               # raw sensor units, FR/FL/RR/RL


class ForceEstimator:
    """Converts the FL leg's current joint torques into an estimated force
    (Newtons) at the foot, via the leg's own Jacobian: tau = J^T @ F, so
    F = pinv(J^T) @ tau. Uses the SAME MuJoCo model the plan was generated
    from -- this is the one place this script depends on MuJoCo being
    installed alongside unitree_sdk2py (confirmed to be the case here,
    both in the same conda environment)."""

    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        names = ('FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint')
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.qposadr = [self.model.jnt_qposadr[j] for j in jids]
        self.dofadr = [self.model.jnt_dofadr[j] for j in jids]
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'foot_site')

    def estimate(self, fl_q, fl_tau):
        for k, qa in enumerate(self.qposadr):
            self.data.qpos[qa] = fl_q[k]
        mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site)
        J = jacp[:, self.dofadr]
        F = np.linalg.pinv(J.T) @ np.array(fl_tau)
        return F, float(np.linalg.norm(F))


GRAVITY = np.array([0.0, 0.0, 9.81])


def quat_rotate(q, v):
    """Rotates vector v (body frame) into world frame using quaternion q = (w,x,y,z)."""
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ v


class SimpleEKF:
    """Dead-reckoning EKF over position+velocity (6D state), using IMU
    acceleration (rotated into world frame via the IMU's own onboard
    orientation estimate, with gravity removed) as the process input.

    IMPORTANT LIMITATION, stated plainly rather than glossed over: there is
    NO position correction source available on real hardware here (no
    mocap, no GPS, nothing playing the role the sim's "mocap_pos" sensor
    played). This is pure open-loop integration, and the sim's own
    sensor-contribution study (earlier in this project) already measured
    exactly how badly that drifts: 9.6m RMSE over ~50 seconds for
    IMU-only dead reckoning, versus 0.009m for mocap-corrected estimates.
    Expect the same order-of-magnitude drift here. The covariance logged
    alongside the state is what makes that drift VISIBLE and honest --
    watch it grow over the run; that growth is the estimator correctly
    reporting its own increasing uncertainty, not a bug.

    Orientation itself is NOT re-estimated here -- the IMU's own onboard
    sensor fusion (imu_state.quaternion) is used directly, since re-deriving
    it from raw gyro/accel would just be a worse version of what's already
    computed in firmware.

    Verified against synthetic at-rest data before shipping: 1 second of
    a stationary IMU reading pure gravity produces exactly zero drift in
    position/velocity and a small, sane, bounded covariance growth."""

    def __init__(self, accel_noise_std=0.5):
        self.x = np.zeros(6)   # [px, py, pz, vx, vy, vz]
        self.P = np.eye(6) * 1e-4
        self.accel_noise_std = accel_noise_std

    def predict(self, accel_world, dt):
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        B = np.zeros((6, 3))
        B[0:3, :] = 0.5 * dt ** 2 * np.eye(3)
        B[3:6, :] = dt * np.eye(3)
        self.x = F @ self.x + B @ accel_world
        q_pos = 0.25 * (self.accel_noise_std * dt ** 2) ** 2
        q_vel = (self.accel_noise_std * dt) ** 2
        Q = np.diag([q_pos] * 3 + [q_vel] * 3)
        self.P = F @ self.P @ F.T + Q


class ThighHeightSolver:
    """Solves for the FL_thigh angle that holds the foot at a fixed world
    Z-height, given whatever hip/calf angles the current phase specifies.
    This replaces a raw locked joint angle: locking the thigh ANGLE does
    NOT hold the foot's actual height constant, since as hip/calf change
    across phases, the same thigh angle traces a different Z position --
    confirmed directly, that's what was causing the leg to visibly lift
    upward even with thigh "locked".

    Foot Z is NOT monotonic across the full thigh range (it decreases
    then increases again, with a minimum around thigh~0.96 on this robot)
    -- confirmed directly by scanning it. A "high" pose (matching a
    photographed elevated leg) falls in the lower/negative thigh
    sub-range, where Z IS monotonic, so the solve is restricted to that
    sub-range (bisection) rather than searching the full range.

    Reachability of the exact target Z varies with hip/calf -- verified
    directly across several combinations, some can't reach a given target
    Z at all. When that happens, this clamps to whichever end of the
    achievable range is closest, rather than failing outright, and
    reports which case occurred."""

    def __init__(self, model_path, thigh_lo=-1.5708, thigh_hi=0.96):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        names = ('FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint')
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.qposadr = [self.model.jnt_qposadr[j] for j in jids]
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'foot_site')
        self.thigh_lo, self.thigh_hi = thigh_lo, thigh_hi

    def _foot_z(self, hip, thigh, calf):
        self.data.qpos[self.qposadr[0]] = hip
        self.data.qpos[self.qposadr[1]] = thigh
        self.data.qpos[self.qposadr[2]] = calf
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site][2]

    def solve(self, hip, calf, target_z, iters=40):
        lo, hi = self.thigh_lo, self.thigh_hi
        z_lo, z_hi = self._foot_z(hip, lo, calf), self._foot_z(hip, hi, calf)
        if target_z >= z_lo:
            return lo, z_lo, False
        if target_z <= z_hi:
            return hi, z_hi, False
        for _ in range(iters):
            mid = (lo + hi) / 2
            z_mid = self._foot_z(hip, mid, calf)
            if z_mid > target_z:
                lo = mid
            else:
                hi = mid
        thigh = (lo + hi) / 2
        return thigh, self._foot_z(hip, thigh, calf), True

    def current_foot_z(self, hip, thigh, calf):
        return self._foot_z(hip, thigh, calf)


# Fallback only. load_fl_limits() overwrites these from the model at
# startup -- hardcoding them here as the source of truth was already drifting
# (the calf bound was off by 4e-5 rad from the model, enough to make targets
# sitting exactly on the real limit report as violations).
FL_JOINT_LIMITS = [(-1.0472, 1.0472),    # FL_hip   +-60 deg
                   (-1.5708, 3.4907),    # FL_thigh -90..200 deg
                   (-2.7227, -0.83776)]  # FL_calf  -156..-48 deg


def load_fl_limits(model_path):
    """Reads the FL joint limits straight out of the model so the script and
    the model can never disagree about them."""
    global FL_JOINT_LIMITS
    m = mujoco.MjModel.from_xml_path(model_path)
    names = ('FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint')
    FL_JOINT_LIMITS = [tuple(float(v) for v in m.jnt_range[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names]
    print('FL joint limits (from model): ' + ', '.join(
        f'{n.split("_")[1]} {math.degrees(lo):.1f}..{math.degrees(hi):.1f}deg'
        for n, (lo, hi) in zip(names, FL_JOINT_LIMITS)))


def clamp_fl(target):
    """Clamps an FL target to the model's real joint limits, which are the
    same limits the physical Go2 has. Worth doing explicitly: the existing
    front_hip_lateral.py commands FL_HIP at start+85deg, and since the hip
    only travels +-60deg, that target is unreachable -- the motor just
    drives into its hard stop and holds there against it. Verified the
    +-60deg hip range directly from the model."""
    out, hit = [], []
    for k, (v, (lo, hi)) in enumerate(zip(target, FL_JOINT_LIMITS)):
        c = min(max(v, lo), hi)
        if abs(c - v) > 1e-6:
            hit.append((k, v, c))
        out.append(c)
    return out, hit


def hip_probe(pub, cmd, delta_rad=0.04, move_s=1.5, hold_s=1.5):
    """Safe sign-convention diagnostic for FL_HIP (motor index 3) ONLY.

    Moves FL_HIP by +delta then -delta from its measured current angle,
    holding thigh and calf exactly where they are, and reports the world
    direction the UPPER LEG (hip->knee link) moves in each case. Nothing
    else on the robot is commanded.

    The model says the outward/toward-ladder direction is +hip, but the
    whole point of this routine is to confirm that on the physical robot
    rather than trust it, so both signs are actually tried and you get to
    watch which one is which."""
    start = [latest_state['q'][i] for i in FL_IDX]
    if any(v is None for v in start):
        raise RuntimeError('No robot state -- cannot probe.')
    print('\n=== FL_HIP SIGN PROBE (motor index 3 only) ===')
    print(f'  measured FL_HIP  = {math.degrees(start[0]):.2f} deg')
    print(f'  measured FL_THIGH= {math.degrees(start[1]):.2f} deg  (held)')
    print(f'  measured FL_CALF = {math.degrees(start[2]):.2f} deg  (held)')
    print(f'  probe amplitude  = +/-{delta_rad:.3f} rad '
          f'({math.degrees(delta_rad):.2f} deg)\n')
    dt = 1.0 / CONTROL_HZ
    for sign, label in ((+1, 'POSITIVE'), (-1, 'NEGATIVE')):
        tgt = list(start)
        tgt[0] = start[0] + sign * delta_rad
        c, hitlim = clamp_fl(tgt)
        if hitlim:
            print(f'  {label}: would exceed the hip limit -- skipping.')
            continue
        input(f'  Press Enter to try {label} hip ({math.degrees(c[0]):.2f} deg)...')
        steps = max(1, int(move_s * CONTROL_HZ))
        for i in range(steps + 1):
            a = i / steps
            sm = a * a * (3.0 - 2.0 * a)
            q = list(start)
            q[0] = start[0] + sm * (c[0] - start[0])
            send_targets(pub, cmd, {tuple(FL_IDX): q})
            time.sleep(dt)
        for _ in range(int(hold_s * CONTROL_HZ)):
            send_targets(pub, cmd, {tuple(FL_IDX): c})
            time.sleep(dt)
        act = latest_state['q'][FL_IDX[0]]
        print(f'    commanded {math.degrees(c[0]):7.2f} deg, '
              f'actual {math.degrees(act):7.2f} deg')
        print(f'    -> LOOK NOW: did the UPPER LEG swing outward/up, '
              f'away from the body? (y/n)')
        ans = input('       answer: ').strip().lower()
        if ans.startswith('y'):
            print(f'\n  ==> OUTWARD is {label} hip. '
                  f'Use hip {"increasing" if sign > 0 else "decreasing"} '
                  f'for abduction.\n')
        # return to start before trying the other sign
        for i in range(steps + 1):
            a = i / steps
            sm = a * a * (3.0 - 2.0 * a)
            q = list(start)
            q[0] = c[0] + sm * (start[0] - c[0])
            send_targets(pub, cmd, {tuple(FL_IDX): q})
            time.sleep(dt)
    print('  Probe finished; FL_HIP returned to its starting angle.')


def lift_to_ready(pub, cmd, ready_fl, support_stance, use_support,
                  stage_time_s=2.0, settle_s=0.4):
    """Startup deployment: takes the FL leg from wherever it is hanging on
    the trolley into the ladder-facing "ready to press" pose, moving ONE
    joint at a time in the order the reference photos show:

        1. HIP   -- abduct the whole upper leg away from the body
        2. THIGH -- swing the upper link out so the leg straightens
        3. CALF  -- extend the lower leg so the foot reaches the ladder

    Verified this order against the model, starting from a folded/hanging
    pose. It matters, and not just cosmetically: through stages 1 and 2 the
    foot stays SHORT of the ladder plane (y<=0.19, ladder is at y=0.300),
    and only stage 3's calf extension carries it out to y=0.300 to touch
    the rung. So the leg deploys beside the ladder and then reaches into
    it, instead of sweeping sideways through the rungs on the way up.

    One thing that will look wrong on the real robot but isn't: in the
    photos the robot is upright on the floor, so the hip lift reads as
    "leg goes UP". On the trolley the base is mounted rolled 90 degrees
    (euler="1.5708 0 0" in the scene), so that same hip motion points a
    different way in world terms -- the leg swings out toward the ladder
    side rather than skyward. Same joint motion, same photo, different
    mounting."""
    names = ['FL_HIP', 'FL_THIGH', 'FL_CALF']
    start = [latest_state['q'][i] for i in FL_IDX]
    if any(v is None for v in start):
        raise RuntimeError('No robot state -- refusing to start the lift blind.')

    ready, hit = clamp_fl(ready_fl)
    for k, v, c in hit:
        print(f'  WARNING: {names[k]} target {math.degrees(v):.1f}deg is outside the joint '
              f'limit -- clamped to {math.degrees(c):.1f}deg.')

    print('\n=== STARTUP LIFT (hip -> thigh -> calf, one joint at a time) ===')
    for k in range(3):
        print(f'  {names[k]:9s} {math.degrees(start[k]):7.2f}deg -> '
              f'{math.degrees(ready[k]):7.2f}deg')
    print()

    cur = list(start)
    dt = 1.0 / CONTROL_HZ
    for k in range(3):
        if abs(ready[k] - cur[k]) < 1e-4:
            print(f'  [stage {k+1}: {names[k]}] already at target, skipping.')
            continue
        input(f'  [stage {k+1}: {names[k]}] press Enter to move '
              f'{math.degrees(cur[k]):.1f} -> {math.degrees(ready[k]):.1f} deg '
              f'(Ctrl+C aborts)...')
        a0, a1 = cur[k], ready[k]
        steps = max(1, int(stage_time_s * CONTROL_HZ))
        for i in range(steps + 1):
            alpha = i / steps
            # smoothstep: zero velocity at both ends, no jerk into the move
            s = alpha * alpha * (3.0 - 2.0 * alpha)
            cur[k] = a0 + s * (a1 - a0)
            targets = {tuple(FL_IDX): list(cur)}
            if use_support and support_stance is not None:
                targets[tuple(FR_IDX)] = support_stance
                targets[tuple(RR_IDX)] = support_stance
                targets[tuple(RL_IDX)] = support_stance
            send_targets(pub, cmd, targets)
            time.sleep(dt)
        cur[k] = a1
        # keep publishing while it settles, so the watchdog never sees a gap
        for _ in range(int(settle_s * CONTROL_HZ)):
            send_targets(pub, cmd, {tuple(FL_IDX): list(cur)})
            time.sleep(dt)
        act = [latest_state['q'][i] for i in FL_IDX]
        print(f'  [stage {k+1}: {names[k]}] commanded '
              f'{math.degrees(a1):7.2f}deg, actual {math.degrees(act[k]):7.2f}deg '
              f'(error {math.degrees(act[k]-a1):+.2f}deg)')

    print('\n  Lift complete. Leg is in the ladder-facing ready pose and being held.')
    return list(cur)


def send_targets(pub, cmd, leg_targets):
    # SINGLE LowCmd owner:
    #   FL_HIP   -> locked target when --hold-fl-hip-deg is enabled
    #   FL_THIGH -> replay trajectory
    #   FL_CALF  -> replay trajectory
    # No second process should publish rt/lowcmd at the same time.
    for idxs, angles in leg_targets.items():
        for j, idx in enumerate(idxs):
            cmd.motor_cmd[idx].mode = 0x01
            cmd.motor_cmd[idx].q = angles[j]
            cmd.motor_cmd[idx].dq = 0.0
            cmd.motor_cmd[idx].tau = 0.0
            cmd.motor_cmd[idx].kp = KP_LEG
            cmd.motor_cmd[idx].kd = KD_LEG
    cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)


def wait_for_enter_while_holding(pub, cmd, hold_fl, hold_support, use_support, prompt):
    """Replaces a plain blocking input(). A plain input() stops the
    Python process entirely -- including the loop that publishes LowCmd
    -- for as long as it takes you to press Enter. The robot's real-time
    motor controller has a watchdog for exactly this: no fresh command
    for a short window, and it drops to a passive/safe state, letting
    the joint sag under gravity. That's what "relaxing down between
    phases" was -- not a control/gain problem, a total absence of any
    command being sent at all during the pause.

    This keeps re-sending the SAME target (wherever the leg just finished
    that phase) at the normal control rate in a background loop, and
    proceeds only once you actually press Enter, via a separate thread
    watching for that keypress."""
    done = threading.Event()

    def _wait():
        input(prompt)
        done.set()

    t = threading.Thread(target=_wait, daemon=True)
    t.start()
    while not done.is_set():
        targets = {tuple(FL_IDX): hold_fl}
        if use_support and hold_support is not None:
            targets[tuple(FR_IDX)] = hold_support
            targets[tuple(RR_IDX)] = hold_support
            targets[tuple(RL_IDX)] = hold_support
        send_targets(pub, cmd, targets)
        time.sleep(1.0 / CONTROL_HZ)


def ramp_to(pub, cmd, target_fl, support_stance, duration_s, label, estimator, force_log,
            ekf, ekf_log, hold_support=False, print_every=25):
    steps = max(1, int(duration_s * REPLAY_SPEED * CONTROL_HZ))
    dt = 1.0 / CONTROL_HZ
    start_fl = [latest_state['q'][i] for i in FL_IDX]
    if any(v is None for v in start_fl):
        start_fl = target_fl

    print(f'  [{label}] {duration_s:.2f}s -> target FL={[round(a, 3) for a in target_fl]}'
          + ('' if hold_support else '  (FR/RR/RL not commanded -- pass --hold-support-legs to enable)'))
    phase_forces = []

    for i in range(steps):
        frac = (i + 1) / steps
        interp = [s + frac * (t - s) for s, t in zip(start_fl, target_fl)]
        targets = {tuple(FL_IDX): interp}
        if hold_support and support_stance is not None:
            targets[tuple(FR_IDX)] = support_stance
            targets[tuple(RR_IDX)] = support_stance
            targets[tuple(RL_IDX)] = support_stance
        send_targets(pub, cmd, targets)
        time.sleep(dt)

        fl_q = [latest_state['q'][idx] for idx in FL_IDX]
        fl_tau = [latest_state['tau_est'][idx] for idx in FL_IDX]
        if all(v is not None for v in fl_q) and all(v is not None for v in fl_tau):
            F, F_mag = estimator.estimate(fl_q, fl_tau)
            phase_forces.append(F_mag)
            ff = latest_state['foot_force']
            ff_fl = ff[FOOT_FORCE_FL_IDX] if ff is not None else None
            force_log.append({'phase': label, 'step': i, 't_s': round(i / CONTROL_HZ, 3),
                               'Fx': round(F[0], 2), 'Fy': round(F[1], 2), 'Fz': round(F[2], 2),
                               'F_mag': round(F_mag, 2), 'foot_force_raw_FL': ff_fl})
            if i % print_every == 0:
                ff_str = f'  raw_foot_force_FL={ff_fl}' if ff_fl is not None else ''
                print(f'      t={i/CONTROL_HZ:5.2f}s  estimated foot force = {F_mag:6.2f} N'
                      f'  (Fx={F[0]:.1f} Fy={F[1]:.1f} Fz={F[2]:.1f}){ff_str}')

        # EKF predict step, if IMU data has arrived
        quat, accel = latest_state['imu_quat'], latest_state['imu_accel']
        if quat is not None and accel is not None:
            accel_world = quat_rotate(np.array(quat), np.array(accel))
            accel_true = accel_world - GRAVITY
            ekf.predict(accel_true, dt)
            p, v = ekf.x[0:3], ekf.x[3:6]
            pvar = np.diag(ekf.P)[0:3]
            vvar = np.diag(ekf.P)[3:6]
            ekf_log.append({
                'phase': label, 'step': i, 't_s': round(i / CONTROL_HZ, 3),
                'px': round(p[0], 4), 'py': round(p[1], 4), 'pz': round(p[2], 4),
                'vx': round(v[0], 4), 'vy': round(v[1], 4), 'vz': round(v[2], 4),
                'var_px': round(pvar[0], 6), 'var_py': round(pvar[1], 6), 'var_pz': round(pvar[2], 6),
                'var_vx': round(vvar[0], 6), 'var_vy': round(vvar[1], 6), 'var_vz': round(vvar[2], 6),
                'quat_w': round(quat[0], 4), 'quat_x': round(quat[1], 4),
                'quat_y': round(quat[2], 4), 'quat_z': round(quat[3], 4),
            })

    if phase_forces:
        arr = np.array(phase_forces)
        print(f'      [{label}] force summary: min={arr.min():.2f}N  mean={arr.mean():.2f}N'
              f'  max={arr.max():.2f}N  (n={len(arr)} samples)')
    else:
        print(f'      [{label}] no force samples collected (robot state not received in time)')
    if ekf_log:
        last = ekf_log[-1]
        print(f'      [{label}] EKF position estimate: '
              f'({last["px"]:.3f}, {last["py"]:.3f}, {last["pz"]:.3f}) m  '
              f'position variance: ({last["var_px"]:.4f}, {last["var_py"]:.4f}, {last["var_pz"]:.4f})')


def write_force_log(force_log, path):
    """Writes whatever's been collected so far to CSV. Called from a
    finally block in main() so this runs on a clean finish, a Ctrl+C, or
    an unexpected error mid-run -- previously the CSV was only written
    after the phase loop finished naturally, so an interrupted run (which
    the safety notes actively tell you to do if something looks wrong)
    silently threw away everything logged up to that point."""
    if not force_log:
        print('No force samples were collected -- nothing to write.')
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['phase', 'step', 't_s', 'Fx', 'Fy', 'Fz',
                                               'F_mag', 'foot_force_raw_FL'])
        writer.writeheader()
        writer.writerows(force_log)
    print(f'\nWrote {len(force_log)} force samples to {path}')

    print('Per-phase force summary:')
    seen = []
    for row in force_log:
        if row['phase'] not in seen:
            seen.append(row['phase'])
    for phase_name in seen:
        mags = [r['F_mag'] for r in force_log if r['phase'] == phase_name]
        print(f'  {phase_name:35s} min={min(mags):7.2f}N  '
              f'mean={sum(mags)/len(mags):7.2f}N  max={max(mags):7.2f}N')


def write_ekf_log(ekf_log, path):
    """Same crash-safe intent as write_force_log -- called from the same
    finally block, so an interrupted run still saves whatever state
    estimates were computed up to that point."""
    if not ekf_log:
        print('No EKF samples were collected -- nothing to write.')
        return
    fields = ['phase', 'step', 't_s', 'px', 'py', 'pz', 'vx', 'vy', 'vz',
              'var_px', 'var_py', 'var_pz', 'var_vx', 'var_vy', 'var_vz',
              'quat_w', 'quat_x', 'quat_y', 'quat_z']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ekf_log)
    print(f'Wrote {len(ekf_log)} EKF samples to {path}')
    last = ekf_log[-1]
    print(f'Final EKF position estimate: ({last["px"]:.3f}, {last["py"]:.3f}, {last["pz"]:.3f}) m')
    print(f'Final position variance: ({last["var_px"]:.4f}, {last["var_py"]:.4f}, {last["var_pz"]:.4f})'
          f'  -- remember: this is pure dead reckoning with no position correction, '
          f'so both the estimate and its growing uncertainty should be read as '
          f'"roughly where it thinks it is," not ground truth.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('interface')
    parser.add_argument('--phase-limit', type=int, default=1)
    parser.add_argument(
        '--hold-fl-hip-deg',
        type=float,
        default=None,
        help='Lock FL_HIP at its startup angle plus this many degrees for the entire replay. '
             'The FL_HIP command is then owned by this single controller, while FL_THIGH/FL_CALF '
             'continue following the trajectory. Example: --hold-fl-hip-deg 35'
    )
    parser.add_argument('--plan', default='trajectory_plan.json')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                         help='Path to the MuJoCo model used for force estimation '
                              '(must match the one extract_trajectory.py used)')
    parser.add_argument('--force-log', default='force_log.csv')
    parser.add_argument('--ekf-log', default='ekf_log.csv')
    parser.add_argument('--accel-noise-std', type=float, default=0.5,
                         help='IMU accelerometer noise std dev (m/s^2) used by the EKF '
                              'process model. Affects how fast reported uncertainty grows.')
    parser.add_argument('--hold-support-legs', action='store_true',
                         help='Actively PD-hold FR/RR/RL at the plan\'s stance target every '
                              'step (matches the sim\'s hold_support_legs()). OFF by default: '
                              'only FL is ever commanded, so FR/RR/RL stay in whatever mode '
                              'they were in when the script started (mode=0x00, i.e. passive/'
                              'undriven) and never move on their own.')
    parser.add_argument('--lock-height', type=float, default=None,
                         help='NOT RECOMMENDED -- overrides each phase\'s FL_thigh with an angle '
                              'solved to hold the foot at this world Z-height. This BREAKS the '
                              'pull/push stroke (see the startup warning it prints). Off by '
                              'default; without it the plan is replayed exactly as generated, '
                              'which already holds the foot at a consistent height.')
    parser.add_argument('--no-lift', action='store_true',
                         help='Skip the startup lift and assume the FL leg is ALREADY in the '
                              'ladder-facing ready pose. Default is to perform the lift.')
    parser.add_argument('--hip-probe', action='store_true',
                         help='Run the FL_HIP sign-convention diagnostic ONLY (small +/- '
                              'motion on motor index 3, thigh/calf held), then exit without '
                              'replaying anything. Run this first on new hardware.')
    parser.add_argument('--hip-probe-rad', type=float, default=0.04,
                         help='Probe amplitude in radians (default 0.04).')
    parser.add_argument('--fl-hip-deg', type=float, default=None,
                         help='Lock FL_HIP at this ABSOLUTE angle in degrees for the whole '
                              'run (unlike --hold-fl-hip-deg, which is an offset from the '
                              'startup angle). With the hip locked, foot height is fixed too '
                              '-- verified that at a fixed hip the foot rides at a constant Z '
                              'across the entire reach, so the hip angle alone sets the height.')
    parser.add_argument('--lift-time', type=float, default=2.0,
                         help='Seconds per joint stage during the startup lift (default 2.0).')
    args = parser.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)
    print(f'Loaded {len(plan)} phases; replaying first {args.phase_limit}.')

    print('Loading MuJoCo model for force estimation...')
    load_fl_limits(args.model)
    estimator = ForceEstimator(args.model)
    height_solver = ThighHeightSolver(args.model)
    ekf = SimpleEKF(accel_noise_std=args.accel_noise_std)

    ChannelFactoryInitialize(0, args.interface)
    sub = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
    sub.Init(lowstate_handler, 10)
    pub = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
    pub.Init()

    print()
    print('IMPORTANT: this program is the sole LowCmd publisher.')
    print('Do NOT run a second script that also publishes rt/lowcmd.')
    print('Use --hold-fl-hip-deg to make this controller own FL_HIP.')
    print()

    print('Waiting for robot state...')
    while latest_state['q'][3] is None:
        time.sleep(0.1)

    input('Confirm robot is positioned/supported as expected, then press Enter...')

    current_hip = latest_state['q'][FL_IDX[0]]

    # --------------------------------------------------------
    # OPTIONAL FL_HIP OWNERSHIP / LOCK
    # --------------------------------------------------------
    # This controller is the ONLY process publishing rt/lowcmd.
    # When enabled, FL_HIP is locked here and the replay trajectory
    # is allowed to control FL_THIGH + FL_CALF around that fixed hip.
    locked_fl_hip = None

    if args.hip_probe:
        cmd_probe = unitree_go_msg_dds__LowCmd_()
        cmd_probe.head[0] = 0xFE
        cmd_probe.head[1] = 0xEF
        cmd_probe.level_flag = 0xFF
        for i in range(20):
            cmd_probe.motor_cmd[i].mode = 0x00
            cmd_probe.motor_cmd[i].q = 0.0
            cmd_probe.motor_cmd[i].kp = 0.0
            cmd_probe.motor_cmd[i].kd = 0.0
        try:
            hip_probe(pub, cmd_probe, delta_rad=args.hip_probe_rad)
        except KeyboardInterrupt:
            print('\nProbe aborted by Ctrl+C.')
        return

    if args.fl_hip_deg is not None:
        locked_fl_hip = math.radians(args.fl_hip_deg)
        c, hit = clamp_fl([locked_fl_hip, 0.0, -1.5])
        locked_fl_hip = c[0]
        print()
        print('FL_HIP ABSOLUTE LOCK ENABLED')
        print(f'  Current FL_HIP : {math.degrees(current_hip):.2f} deg')
        print(f'  Locked FL_HIP  : {math.degrees(locked_fl_hip):.2f} deg')
        if hit:
            print(f'  (requested {args.fl_hip_deg:.2f} deg was outside the +/-60 deg '
                  f'hip limit and was clamped)')
        if math.degrees(locked_fl_hip) > 25.0:
            print('  *** WARNING: above about +25 deg the FL foot can no longer reach the')
            print('      ladder plane at all -- the leg swings up and over the body and the')
            print('      foot moves AWAY from the ladder. Checked this directly. ***')
        print()

    elif args.hold_fl_hip_deg is not None:
        locked_fl_hip = current_hip + math.radians(args.hold_fl_hip_deg)

        print()
        print('FL_HIP LOCK ENABLED')
        print(f'  Current FL_HIP : {math.degrees(current_hip):.2f} deg')
        print(f'  Offset         : {args.hold_fl_hip_deg:+.2f} deg')
        print(f'  Locked FL_HIP  : {math.degrees(locked_fl_hip):.2f} deg')
        print('  FL_HIP will NOT follow phase[fl_hip].')
        print('  FL_THIGH + FL_CALF follow the ORIGINAL replay targets.')
        print('  Fixed-height thigh compensation is disabled in lock mode.')
        print('  This avoids two independent LowCmd publishers fighting.')
        print()

    target_height = None
    if args.lock_height is not None:
        target_height = args.lock_height
        print()
        print('*** WARNING: --lock-height is enabled. ***')
        print('  This replaces each phase\'s planned FL_thigh with an angle solved from')
        print('  foot height alone. Checked what that actually does to this plan, and it')
        print('  inverts the power stroke: on cycle0_pull the plan drives the foot from')
        print('  x=+0.250 back to x=-0.026 (a 28cm pull that is what shoves the trolley')
        print('  forward), but the height solve returns thigh=-5.3deg, putting the foot at')
        print('  x=+0.413 -- 44cm the WRONG WAY. Foot height has two thigh solutions and')
        print('  the solver picks the other one. The leg would reach out instead of push,')
        print('  and the slate would not move.')
        print('  Run without --lock-height unless you specifically want this.')
        print()

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x00
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0

    force_log = []
    ekf_log = []
    current_support = None

    # The ready pose is the plan's OWN first phase -- that phase already IS
    # "leg aligned toward the ladder, ready to contact and push" (its foot
    # sits at y=0.300, exactly on the ladder plane, at rung_01). So the lift
    # target is not a separate number to tune; it is just where the replay
    # is about to begin. If FL_HIP is locked, respect that lock here too so
    # the lift does not fight the thing that immediately follows it.
    first = plan[0]
    ready_fl = [locked_fl_hip if locked_fl_hip is not None else first['fl_hip'],
                first['fl_thigh'], first['fl_calf']]
    if first['support_stance'] is not None:
        current_support = first['support_stance']

    try:
        if not args.no_lift:
            lift_to_ready(pub, cmd, ready_fl, current_support,
                          args.hold_support_legs, stage_time_s=args.lift_time)
            wait_for_enter_while_holding(
                pub, cmd, ready_fl, current_support, args.hold_support_legs,
                '  Holding ready pose. Press Enter to begin phase replay '
                '(Ctrl+C to stop)...')
        else:
            print('Startup lift SKIPPED (--no-lift): assuming leg is already in the '
                  'ready pose.')

        for phase in plan[:args.phase_limit]:

            if locked_fl_hip is not None:
                # ----------------------------------------------------
                # HIP-LOCK MODE
                # ----------------------------------------------------
                # FL_HIP is owned by this controller and remains fixed.
                #
                # IMPORTANT: do NOT use the fixed-height solver here.
                # The solver changes FL_THIGH to compensate for hip/calf
                # motion. That would alter the replayed foot trajectory.
                #
                # Preserve the ORIGINAL replay targets for thigh and calf.
                fl_target = [
                    locked_fl_hip,
                    phase['fl_thigh'],
                    phase['fl_calf'],
                ]

            elif args.lock_height is not None:
                # OPT-IN ONLY, and it will break the push -- see the
                # --lock-height help text and the warning printed at
                # startup. Left in because it was asked for, not because
                # it is the right thing to run.
                solved_thigh, achieved_z, was_exact = height_solver.solve(
                    phase['fl_hip'], phase['fl_calf'], target_height)
                fl_target = [phase['fl_hip'], solved_thigh, phase['fl_calf']]
                if not was_exact:
                    print(f'  [{phase["phase"]}] NOTE: target height {target_height:.4f}m not '
                          f'reachable with this phase\'s hip/calf -- using closest achievable, '
                          f'{achieved_z:.4f}m')

            else:
                # DEFAULT: replay the plan exactly as generated.
                #
                # The plan's own foot path is already height-consistent --
                # every reach/press phase puts the foot at z=0.400 with
                # the thigh angle it specifies. There is nothing for a
                # height solver to correct, and running one here actively
                # breaks the stroke that moves the trolley. Checked each
                # phase's foot position directly against the model.
                fl_target = [phase['fl_hip'], phase['fl_thigh'], phase['fl_calf']]

            print(
                f'  [{phase["phase"]}] command targets: '
                f'FL_HIP={math.degrees(fl_target[0]):.2f}deg, '
                f'FL_THIGH={math.degrees(fl_target[1]):.2f}deg, '
                f'FL_CALF={math.degrees(fl_target[2]):.2f}deg'
            )

            if phase['support_stance'] is not None:
                current_support = phase['support_stance']
            ramp_to(pub, cmd, fl_target, current_support, phase['duration_s'], phase['phase'],
                    estimator, force_log, ekf, ekf_log, hold_support=args.hold_support_legs)
            wait_for_enter_while_holding(
                pub, cmd, fl_target, current_support, args.hold_support_legs,
                f'  [{phase["phase"]}] complete, holding position. '
                f'Press Enter to continue (Ctrl+C to stop)...')
        print('\nReplay segment complete.')
    except KeyboardInterrupt:
        print('\nStopped by Ctrl+C -- writing out whatever was logged so far before exiting.')
    finally:
        write_force_log(force_log, args.force_log)
        write_ekf_log(ekf_log, args.ekf_log)


if __name__ == '__main__':
    main()
