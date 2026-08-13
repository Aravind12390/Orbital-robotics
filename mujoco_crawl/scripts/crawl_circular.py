#!/usr/bin/env python3
import mujoco, mujoco.viewer
import numpy as np
import math, time, sys, os

MODEL = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'

TOTAL_RUNGS    = 42     # must match build_scene.py -- reverted from 28 (that
                         # widening broke the calibrated ~0.30m chord spacing)
LADDER_RADIUS  = 2.00
TROLLEY_RADIUS = 1.43
ANGLE_SPACING  = 2 * math.pi / TOTAL_RUNGS
GRIP_Z         = 0.450   # matches RUNG_Z / build_scene.py -- see that file's comment
WHEEL_RADIUS   = 0.045

CRAWL_STEPS = 76   # measured real progress averages ~5.04 deg/cycle (verified
                     # over a clean 42-cycle run reaching 211.65 deg with zero
                     # divergence), so ~72 cycles covers a full 360 deg lap --
                     # 76 gives a little margin. Increase further for multiple
                     # laps; rung indexing wraps for free either way.

PULL_STEPS    = 2000
SWING_STEPS   = 400
SETTLE_STEPS  = 350
KP, KV = 700.0, 20.0

FINGER_OPEN, FINGER_CLOSED = 0.0, 132.0
IK_WARN_THRESHOLD = 0.01

# Retracting a FULL ANGLE_SPACING (one whole rung's worth) pins the thigh
# joint exactly at its own range limit -- confirmed directly: IK residual
# is 0 up through 70% of ANGLE_SPACING, then grows sharply as thigh clips
# at -1.5708 beyond that. Capping the retraction here (with margin below
# that 70% ceiling) keeps every PULL phase kinematically achievable rather
# than stalling partway through against a hard limit it can never actually
# reach -- which is what was happening before (trolley advancing ~9.7 deg
# out of a 12.86 deg target, then completely stalling, omega=0, for the
# rest of the phase).
RETRACT_FRACTION = 0.6

ACT, IDS = {}, {}


def build_maps(model):
    for n in ['FL_hip', 'FL_thigh', 'FL_calf',
              'FR_hip', 'FR_thigh', 'FR_calf',
              'RL_hip', 'RL_thigh', 'RL_calf',
              'RR_hip', 'RR_thigh', 'RR_calf',
              'fingers_actuator']:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if i >= 0:
            ACT[n] = i
    IDS['fl_joint_names'] = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint']
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in IDS['fl_joint_names']]
    IDS['fl_qposadr'] = [model.jnt_qposadr[j] for j in jids]
    IDS['fl_dofadr']  = [model.jnt_dofadr[j] for j in jids]
    IDS['fl_ranges']  = [model.jnt_range[j].copy() for j in jids]
    IDS['fl_act']     = [ACT['FL_hip'], ACT['FL_thigh'], ACT['FL_calf']]

    IDS['site']    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pinch')
    IDS['gripper_base'] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_base')
    IDS['ladder']  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ladder')
    IDS['eq']      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'fl_grip')
    IDS['fingers'] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'fingers_actuator')
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'trolley_hinge')
    IDS['tjnt'] = j
    IDS['qadr'] = model.jnt_qposadr[j]
    IDS['tdof'] = model.jnt_dofadr[j]

    wheel_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                  for n in ['wfl_j', 'wfr_j', 'wrl_j', 'wrr_j']]
    IDS['wheel_qposadr'] = [model.jnt_qposadr[j] for j in wheel_jids if j >= 0]

    print(f'  {len(ACT)} actuators  trolley_hinge={j}  eq={IDS["eq"]}')


def c(data, n, v):
    if n in ACT:
        data.ctrl[ACT[n]] = float(v)


def trolley_angle(data):
    return float(data.qpos[IDS['qadr']])


def sync_wheels(data):
    if IDS.get('wheel_qposadr'):
        spin = trolley_angle(data) * TROLLEY_RADIUS / WHEEL_RADIUS
        for qa in IDS['wheel_qposadr']:
            data.qpos[qa] = spin


def rung_target(rung_idx):
    theta = rung_idx * ANGLE_SPACING
    return np.array([LADDER_RADIUS * math.cos(theta),
                      LADDER_RADIUS * math.sin(theta),
                      GRIP_Z])


def ik_fl(model, data, target, iters=400, damp=1e-2):
    qposadr, dofadr, ranges = IDS['fl_qposadr'], IDS['fl_dofadr'], IDS['fl_ranges']
    site = IDS['site']
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[site]
        if np.linalg.norm(err) < 1e-5:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site)
        J = jacp[:, dofadr]
        dq = np.linalg.solve(J.T @ J + damp * np.eye(3), J.T @ err)
        for k, qa in enumerate(qposadr):
            data.qpos[qa] += dq[k]
            lo, hi = ranges[k]
            data.qpos[qa] = np.clip(data.qpos[qa], lo, hi)
    return [data.qpos[qa] for qa in qposadr]


def solve_ik_from_current(model, data, target, **kwargs):
    qpos_snapshot = data.qpos.copy()
    qvel_snapshot = data.qvel.copy()
    angles = ik_fl(model, data, target, **kwargs)
    residual = float(np.linalg.norm(data.site_xpos[IDS['site']] - target))
    data.qpos[:] = qpos_snapshot
    data.qvel[:] = qvel_snapshot
    mujoco.mj_forward(model, data)
    if residual > IK_WARN_THRESHOLD:
        print(f'  !! IK did not fully converge: {residual*100:.2f} cm residual '
              f'at target {np.round(target, 3).tolist()}')
    return angles


def grasp_rung(model, data, rung_idx):
    ideal_target = rung_target(rung_idx)
    gbase, ladder, eq, site = IDS['gripper_base'], IDS['ladder'], IDS['eq'], IDS['site']
    actual_pos = data.site_xpos[site].copy()
    R2 = data.xmat[ladder].reshape(3, 3)
    model.eq_data[eq, 0:3] = [0.0, 0.0, 0.145]
    model.eq_data[eq, 3:6] = R2.T @ (actual_pos - data.xpos[ladder])
    data.eq_active[eq] = 1

    residual = float(np.linalg.norm(actual_pos - ideal_target))
    if residual > IK_WARN_THRESHOLD:
        print(f'  !! grasping {residual*100:.2f} cm away from rung_'
              f'{rung_idx % TOTAL_RUNGS:02d}\'s true center')


def release_rung(data):
    data.eq_active[IDS['eq']] = 0


BRAKE_KP, BRAKE_KV = 4000.0, 200.0

def brake_trolley(model, data, hold_angle):
    """Holds the hinge at hold_angle via a stiff virtual torque. Needed
    during SWING: with the grip released, nothing else resists the FL
    leg's reaction torque on the trolley, and that was measurably undoing
    part of PULL's progress (confirmed directly: trolley advanced +5.11deg
    during PULL, then regressed to +4.42deg by the end of the very next
    SWING -- the same class of issue as an earlier project's "collides,
    pushes back" bug, here showing up as lost progress instead)."""
    tdof = IDS['tdof']
    qadr = IDS['qadr']
    tau = BRAKE_KP * (hold_angle - data.qpos[qadr]) - BRAKE_KV * data.qvel[tdof]
    data.qfrc_applied[tdof] = tau


def pd_drive(model, data, target_angles, n_steps, viewer=None, sync_every=10, ramp_steps=None,
             brake_angle=None):
    qposadr, dofadr, act = IDS['fl_qposadr'], IDS['fl_dofadr'], IDS['fl_act']
    tdof = IDS['tdof']
    start_angles = [data.qpos[qa] for qa in qposadr]
    if ramp_steps is None:
        ramp_steps = max(50, int(0.1 * n_steps))
    for i in range(n_steps):
        if i < ramp_steps:
            frac = (i + 1) / ramp_steps
            setpoint = [s + frac * (t - s) for s, t in zip(start_angles, target_angles)]
        else:
            setpoint = target_angles
        for k, (qa, dv, a) in enumerate(zip(qposadr, dofadr, act)):
            tau = KP * (setpoint[k] - data.qpos[qa]) - KV * data.qvel[dv]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = np.clip(tau, lo, hi)
        if brake_angle is not None:
            brake_trolley(model, data, brake_angle)
        else:
            data.qfrc_applied[tdof] = 0.0
        mujoco.mj_step(model, data)
        sync_wheels(data)
        if viewer and i % sync_every == 0:
            viewer.sync()


def jaws(data, closed_amount):
    if 'fingers' in IDS and IDS['fingers'] >= 0:
        data.ctrl[IDS['fingers']] = float(closed_amount)


def nearest_reachable_rung(data, current_idx):
    """Searches a FULL LAP of candidate rung indices and returns whichever
    is angularly closest to the TROLLEY's actual current rotation.

    Important distinction, found by tracing this directly: this must use
    the TROLLEY's angle, not the gripper's site position. The gripper is
    held FIXED in world space by the equality constraint for the entire
    PULL phase (that's the whole mechanism -- the constraint doesn't move,
    the body reacts around it), so measuring the gripper's position right
    after release (before the swing has moved the leg anywhere new) just
    finds wherever the OLD rung already was -- which a full-lap search
    then "correctly" identifies as the closest match, since the gripper
    genuinely hasn't moved yet. That produced current_idx jumping by
    exactly +TOTAL_RUNGS every cycle (a periodic alias of the same
    physical rung), stalling the crawl at ~10 degrees forever. The
    trolley's own angle, by contrast, actually reflects how far the body
    has rotated -- that's the quantity that should drive which rung to
    reach for next."""
    trolley_theta = trolley_angle(data)
    best_idx, best_dist = current_idx + 1, None
    for idx in range(current_idx + 1, current_idx + 1 + TOTAL_RUNGS):
        rung_theta = idx * ANGLE_SPACING
        d = abs(((rung_theta - trolley_theta + math.pi) % (2 * math.pi)) - math.pi)
        if best_dist is None or d < best_dist:
            best_idx, best_dist = idx, d
    return best_idx


def main():
    headless = '--headless' in sys.argv
    if not os.path.exists(MODEL):
        print(f'ERROR: {MODEL} not found. Run build_scene.py first.')
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    build_maps(model)
    mujoco.mj_resetData(model, data)

    def run(viewer=None):
        print('\n\u2501\u2501\u2501 PHASE 1: SPAWN & GRIP RUNG 0 \u2501\u2501\u2501')
        for leg in ['FR', 'RL', 'RR']:
            c(data, f'{leg}_hip', 0.0)
            c(data, f'{leg}_thigh', 1.2)
            c(data, f'{leg}_calf', -2.5)
        mujoco.mj_forward(model, data)

        reach_angles = solve_ik_from_current(model, data, rung_target(0))
        for k, qa in enumerate(IDS['fl_qposadr']):
            data.qpos[qa] = reach_angles[k]
        mujoco.mj_forward(model, data)

        jaws(data, FINGER_CLOSED)
        grasp_rung(model, data, 0)
        pd_drive(model, data, reach_angles, SETTLE_STEPS, viewer)
        print(f'  Gripping rung_00. Trolley angle='
              f'{math.degrees(trolley_angle(data)):.2f} deg')

        print(f'\n\u2501\u2501\u2501 PHASE 2: CRAWL ({CRAWL_STEPS} transitions, '
              f'{math.degrees(ANGLE_SPACING):.2f} deg/rung) \u2501\u2501\u2501')
        theta0 = trolley_angle(data)
        current_idx = 0
        for i in range(CRAWL_STEPS):
            next_idx = current_idx + 1   # provisional -- may be revised below
            print(f'\n  rung_{current_idx % TOTAL_RUNGS:02d} -> '
                  f'rung_{next_idx % TOTAL_RUNGS:02d} (provisional)')

            print('  [A] Pull')
            # Retract relative to the gripper's ACTUAL current angular
            # position, not current_idx * ANGLE_SPACING. That assumed
            # formula drifts out of sync with reality once
            # nearest_reachable_rung has reassigned indices a few times
            # (which is expected and normal -- see that function's
            # docstring), and a stale target is what was producing the
            # later stalls: PULL itself works correctly when given a
            # reachable target (confirmed directly -- no torque
            # saturation, reaches ~5.11deg of an intended 5.14deg
            # cleanly), so a wrong target was the more likely explanation
            # than PULL itself failing.
            gripper_pos = data.site_xpos[IDS['site']]
            actual_theta = math.atan2(gripper_pos[1], gripper_pos[0])
            retract_theta = actual_theta - RETRACT_FRACTION * ANGLE_SPACING
            retract_target = np.array([LADDER_RADIUS * math.cos(retract_theta),
                                        LADDER_RADIUS * math.sin(retract_theta), GRIP_Z])
            retract_angles = solve_ik_from_current(model, data, retract_target)
            angle_before = trolley_angle(data)
            pd_drive(model, data, retract_angles, PULL_STEPS, viewer)
            swept = math.degrees(trolley_angle(data) - angle_before)
            print(f'    Trolley angle={math.degrees(trolley_angle(data)):.2f} deg  '
                  f'(+{swept:.3f} deg this pull, target {math.degrees(ANGLE_SPACING):.2f} deg)  '
                  f'omega={data.qvel[IDS["tdof"]]:.4f} rad/s  '
                  f'arc={TROLLEY_RADIUS*trolley_angle(data):.4f} m')

            print('  [B] Release + swing')
            hold_angle = trolley_angle(data)
            release_rung(data)
            jaws(data, FINGER_OPEN)
            # Re-decide the target AFTER the pull, based on where the
            # gripper actually ended up -- not the provisional current_idx+1
            # from before the pull. See nearest_reachable_rung's docstring.
            next_idx = nearest_reachable_rung(data, current_idx)
            reach_target = rung_target(next_idx)
            reach_angles = solve_ik_from_current(model, data, reach_target)
            pd_drive(model, data, reach_angles, SWING_STEPS, viewer, brake_angle=hold_angle)

            dist = np.linalg.norm(data.site_xpos[IDS['site']] - reach_target)
            print(f'    Reach complete, {dist:.4f} m from rung_{next_idx % TOTAL_RUNGS:02d} target')

            jaws(data, FINGER_CLOSED)
            grasp_rung(model, data, next_idx)
            pd_drive(model, data, reach_angles, SETTLE_STEPS, viewer)
            print(f'    Gripping rung_{next_idx % TOTAL_RUNGS:02d}. '
                  f'Trolley angle={math.degrees(trolley_angle(data)):.2f} deg')
            current_idx = next_idx

        total_deg = math.degrees(trolley_angle(data) - theta0)
        total_arc = TROLLEY_RADIUS * (trolley_angle(data) - theta0)
        ideal_deg = CRAWL_STEPS * math.degrees(ANGLE_SPACING)
        print(f'\n  CRAWL COMPLETE')
        print(f'    Swept: {total_deg:.2f} deg  (ideal {ideal_deg:.2f} deg)  |  '
              f'arc length: {total_arc:.4f} m  |  transitions: {CRAWL_STEPS}')

    if headless:
        run()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 3.2
            viewer.cam.elevation = -35
            viewer.cam.azimuth = 90
            viewer.cam.lookat[:] = [0, 0, 0.4]
            run(viewer)
            print('\nDone. Close viewer to exit.')
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)


if __name__ == '__main__':
    main()
