#!/usr/bin/env python3
"""
Go2 MuJoCo Ladder Crawl — real physics, using the actual Robotiq 2F-85
gripper (from mujoco_menagerie) instead of a hand-built placeholder.

WHAT CHANGED FROM THE PREVIOUS VERSION:

1. The gripper is now the real Robotiq 2F-85 hardware model, mounted on
   FL_calf at the foot location with a 180-degree-about-X mounting rotation
   (found by empirically searching candidate orientations against IK
   reachability -- see build_scene.py). It has its own "pinch" site (the
   grasp point) and a single "fingers_actuator" (ctrl 0 = open, ctrl 255 =
   closed) that drives both fingers symmetrically via a coupling tendon.

2. grasp_rung() now anchors to "gripper_base" (the 2F-85's main body) using
   a FIXED local offset of (0, 0, 0.145) -- that's where the "pinch" site
   sits relative to that body, and it never changes (no joints between
   them), so unlike the ladder-side anchor it doesn't need to be
   recomputed at runtime.

3. Timing has been reworked, not just uniformly slowed: PULL (the phase
   where the leg is gripped and retracting, which is what actually drags
   the trolley+robot forward) is now long and slow -- it's the one motion
   that should read as "the pull." SWING (release, reach back out to the
   next rung) is short and quick -- it's a leg reset, not a second pull.

   Why there are two phases at all, physically: a single gripping limb
   can only move the body while it's anchored to something fixed (that's
   the PULL). To reach the NEXT rung it has to let go and extend back out
   first (SWING) -- there's no way to cover two rung-lengths of travel in
   one continuous same-direction motion with one gripper, the same way an
   inchworm or a person on a single overhead rail has to alternate
   contract/release rather than pull hand-over-hand with one hand. What
   changed here is making that asymmetric in time so it *reads* as one
   dominant pull per cycle instead of two equal-weight phases.

4. FINGER_CLOSED was 255 (fully closed), which drives the pads to a ~1.5cm
   gap -- well inside the rung's actual 5.6cm diameter. Since the rung has
   no collision geometry, the fingers visibly closed straight through it.
   Recalibrated to 132, which measured out to a ~5.6cm gap -- the fingers
   now stop right at the rung's surface instead of slamming shut past it.

5. Contact excludes were only covering the gripper's base body before; the
   finger linkage itself (which sweeps much closer to the leg and trolley
   during the pull/swing motion) wasn't excluded, which was producing
   stray "ghost" contact forces. Now the whole gripper body tree is
   excluded against FL_hip/FL_thigh/FL_calf/trolley (see build_scene.py).
"""
import mujoco, mujoco.viewer
import numpy as np
import time, sys, os

MODEL   = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'
RUNGS   = 8
RSPACE  = 0.30          # metres between rungs (must match build_scene.py)
LADDER_X, LADDER_Y = 1.619, 0.300   # must match build_scene.py
GRIP_Z  = 0.410

PULL_STEPS    = 2000    # ~4.0s -- the single, slow, dominant pull-to-next-rung motion
SWING_STEPS   = 400     # ~0.8s -- quick leg reset (release + reach back out), not a second "pull"
SETTLE_STEPS  = 350     # ~0.7s -- let it stabilize after each grasp
KP, KV = 200.0, 8.0     # PD gains for the FL leg during pull/swing

# Calibrated, not guessed: ctrl=255 (fully closed) drives the pads to a
# 1.5cm gap, well inside the rung's actual 5.6cm diameter -- since the rung
# has no collision geometry, that let the fingers visibly slam shut right
# through it instead of stopping at its surface. ctrl=132 measured (via a
# settle test) to leave a ~5.6cm gap, i.e. closes snugly onto the rung.
FINGER_OPEN, FINGER_CLOSED = 0.0, 132.0

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
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'trolley_slide')
    IDS['tjnt'] = j
    IDS['qadr'] = model.jnt_qposadr[j]
    print(f'  {len(ACT)} actuators  trolley_joint={j}  eq={IDS["eq"]}')


def c(data, n, v):
    if n in ACT:
        data.ctrl[ACT[n]] = float(v)


def tx(data):
    return float(data.qpos[IDS['qadr']])


# ── Inverse kinematics for the FL leg (damped least squares on the gripper site) ──
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


# ── Grasping: THIS is the actual fix. See module docstring. ──────────────
def grasp_rung(model, data, rung_idx):
    """Snap the gripper onto rung_idx's exact, known world location and
    activate the equality constraint. We write eq_data ourselves instead of
    trusting the compiler's baked-in reference pose (which is wrong the
    moment the leg has moved from qpos0).

    body1 is "gripper_base" (the Robotiq 2F-85's main body). Its local
    anchor is fixed at (0, 0, 0.145) -- that's where the gripper's built-in
    "pinch" site sits relative to it, and since nothing moves between them
    (no joints), that offset is a constant, not something to recompute."""
    world_target = np.array([LADDER_X - 1.05 + rung_idx * RSPACE, LADDER_Y, GRIP_Z])
    gbase, ladder, eq = IDS['gripper_base'], IDS['ladder'], IDS['eq']
    R2 = data.xmat[ladder].reshape(3, 3)
    model.eq_data[eq, 0:3] = [0.0, 0.0, 0.145]                        # fixed pinch offset
    model.eq_data[eq, 3:6] = R2.T @ (world_target - data.xpos[ladder])  # exact rung point, in ladder's frame
    data.eq_active[eq] = 1


def release_rung(data):
    data.eq_active[IDS['eq']] = 0


# ── PD drive of the FL leg toward a joint-angle target (real actuation, not qpos-forcing) ──
def pd_drive(model, data, target_angles, n_steps, viewer=None, sync_every=10):
    qposadr, dofadr, act = IDS['fl_qposadr'], IDS['fl_dofadr'], IDS['fl_act']
    for i in range(n_steps):
        for k, (qa, dv, a) in enumerate(zip(qposadr, dofadr, act)):
            tau = KP * (target_angles[k] - data.qpos[qa]) - KV * data.qvel[dv]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = np.clip(tau, lo, hi)
        mujoco.mj_step(model, data)
        if viewer and i % sync_every == 0:
            viewer.sync()


def jaws(data, closed_amount):
    """closed_amount: 0 = fully open, 255 = fully closed (fingers_actuator's
    native ctrlrange -- see build_scene.py, matches the real Robotiq 2F-85's
    control convention)."""
    if 'fingers' in IDS and IDS['fingers'] >= 0:
        data.ctrl[IDS['fingers']] = float(closed_amount)


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
        # ── PHASE 1: SPAWN, reach for rung 1, grasp it ───────────────────
        print('\n━━ PHASE 1: SPAWN & GRIP RUNG 1 ━━')
        for leg in ['FR', 'RL', 'RR']:
            c(data, f'{leg}_hip', 0.0)
            c(data, f'{leg}_thigh', 1.2)
            c(data, f'{leg}_calf', -2.5)

        reach_angles = ik_fl(model, data, np.array([LADDER_X - 1.05, LADDER_Y, GRIP_Z]))
        retract_angles = ik_fl(model, data, np.array([LADDER_X - 1.05 - RSPACE, LADDER_Y, GRIP_Z]))
        for k, qa in enumerate(IDS['fl_qposadr']):
            data.qpos[qa] = reach_angles[k]
        mujoco.mj_forward(model, data)

        jaws(data, FINGER_CLOSED)  # closed
        grasp_rung(model, data, 0)
        pd_drive(model, data, reach_angles, SETTLE_STEPS, viewer)
        print(f'  Gripping rung_01. Trolley X={tx(data):.4f}')

        # ── PHASE 2: CRAWL ────────────────────────────────────────────
        print(f'\n━━ PHASE 2: CRAWL ({RUNGS - 1} transitions) ━━')
        x0 = tx(data)
        for i in range(RUNGS - 1):
            print(f'\n  ┌── rung_{i+1:02d} → rung_{i+2:02d} ─────────')

            # A: PULL — still gripping rung i, retract the leg. Since the
            #    grip point is fixed in world space, this drags the trolley
            #    (and the whole robot) forward instead of moving the foot.
            print('  │ [A] Pull (retract leg while gripped)')
            pd_drive(model, data, retract_angles, PULL_STEPS, viewer)
            print(f'  │     Trolley X={tx(data):.4f}')

            # B: RELEASE + SWING — let go, open the jaws, reach back out.
            #    Because the body advanced in step A, "reach_angles" now
            #    lines up with the NEXT rung, not the one we just held.
            print('  │ [B] Release + swing to next rung')
            release_rung(data)
            jaws(data, FINGER_OPEN)  # open
            pd_drive(model, data, reach_angles, SWING_STEPS, viewer)

            dist = np.linalg.norm(
                data.site_xpos[IDS['site']] -
                np.array([LADDER_X - 1.05 + (i + 1) * RSPACE, LADDER_Y, GRIP_Z]))
            print(f'  │     Reach complete, {dist:.4f} m from rung_{i+2:02d} target')

            # C: GRASP — snap onto the next rung's known location.
            jaws(data, FINGER_CLOSED)
            grasp_rung(model, data, i + 1)
            pd_drive(model, data, reach_angles, SETTLE_STEPS, viewer)
            print(f'  └── Gripping rung_{i+2:02d}. Trolley X={tx(data):.4f}')

        total = tx(data) - x0
        print(f'\n  ✓ CRAWL COMPLETE')
        print(f'    Distance: {total:.4f} m  |  ideal: {(RUNGS-1)*RSPACE:.4f} m'
              f'  |  transitions: {RUNGS-1}')

    if headless:
        run()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 130
            viewer.cam.lookat[:] = [LADDER_X, LADDER_Y, 0.4]
            run(viewer)
            print('\nDone. Close viewer to exit.')
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)


if __name__ == '__main__':
    main()
