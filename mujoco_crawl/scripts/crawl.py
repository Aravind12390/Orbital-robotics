#!/usr/bin/env python3
"""
Go2 MuJoCo Ladder Crawl — real contact-based grasping.

WHAT CHANGED FROM THE EQUALITY-CONSTRAINT VERSION:

Grasping used to be a MuJoCo "connect" equality constraint that snapped the
gripper onto a rung's exact location -- robust, but not a real grip: it
couldn't slip, and it worked regardless of whether the fingers were
actually around the rung. That's now gone. build_scene.py made the rungs
collidable (previously contype/conaffinity=0, purely cosmetic) and
restricted that collision to ONLY the gripper's 4 real fingertip pad geoms
(right_pad1/2, left_pad1/2 -- via an isolated collision bitmask group), so
the rest of the robot/trolley/floor never touches a rung.

Holding a rung is now genuinely: fingers close on a rung's surface,
friction (already realistic on the Robotiq's own pads -- mu=0.6-0.7,
priority=1, tuned solref/solimp, all from the real hardware model) resists
whatever load is on the grip. If the leg is positioned wrong or the load
is too high, it can now actually slip -- that's the tradeoff for realism
you asked for.

close_until_gripped() is closed-loop: it ramps the finger ctrl up while
watching the actual contact force between the pads and whatever rung is in
range, and stops once a target force is reached (measured: this system's
fingers saturate at ~111N of normal force well before ctrl maxes out --
that's the Robotiq's own driver joint hitting its mechanical limit against
the rung, not something we imposed).

grip_force() now reads real contact force (via mj_contactForce on any
pad-vs-rung contact) instead of an equality constraint's efc_force -- same
external API/purpose as before, different underlying physics.
"""
import mujoco, mujoco.viewer
import numpy as np
import time, sys, os

MODEL   = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'
RUNGS   = 8
RSPACE  = 0.30          # metres between rungs (must match build_scene.py)
# PULL consistently falls a little short of a full RSPACE per cycle (real
# dynamics, not perfectly repeatable) -- traced this directly: since each
# swing targets the NEXT rung's fixed absolute position, that shortfall
# doesn't compound as position error (fresh IK each cycle fixes that), but
# it DOES mean the leg has to reach further than the "comfortable"
# envelope to compensate, and that need grows cycle over cycle. Root-caused
# the final-cycle failure to exactly this: the calf joint was pinned at its
# hard limit (-0.838, vs range floor -0.838) trying to reach the last
# rung. Retracting slightly PAST one full RSPACE keeps that safety margin
# from eroding.
PULL_OVERSHOOT = 1.03
LADDER_X, LADDER_Y = 1.619, 0.300   # must match build_scene.py
GRIP_Z  = 0.410

PULL_STEPS    = 2000    # ~4.0s -- the single, slow, dominant pull-to-next-rung motion
SWING_STEPS   = 400     # ~0.8s -- quick leg reset (release + reach back out), not a second "pull"
SETTLE_STEPS  = 350     # ~0.7s -- let it stabilize after each grasp
KP, KV = 200.0, 8.0     # PD gains for the FL leg during pull/swing

FINGER_OPEN = 0.0

# The support-leg actuators (FR/RL/RR) are raw torque motors, same as FL --
# there is no such thing as "set a target angle" on them. Previously they
# were commanded once via a single ctrl value (meant as an angle, applied
# as a constant torque instead) and never touched again -- meaning they
# were drooping/drifting under a fixed torque bias through the ENTIRE
# simulation, not actually holding a pose. That was silently destabilizing
# everything built on top of it. Now PD-held continuously, every step,
# exactly like FL.
STANCE = (0.0, 1.2, -2.5)   # (hip, thigh, calf) target angles, same as before

def hold_support_legs(model, data):
    for leg, ids in IDS['support_legs'].items():
        for k, (qa, dv, a) in enumerate(zip(ids['qposadr'], ids['dofadr'], ids['act'])):
            tau = KP * (STANCE[k] - data.qpos[qa]) - KV * data.qvel[dv]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = np.clip(tau, lo, hi)


# The trolley has near-frictionless wheels by design (that's what lets PULL
# work at all) -- but that cuts both ways: with the grip released and
# nothing else resisting it, ANY reaction force from actively moving the
# unanchored FL leg drifts the whole body. Measured this directly and ruled
# out simpler explanations (rung-scraping, support-leg torque) -- even with
# confirmed zero contact force, an unbraked swing still drifted ~0.5-0.8m.
# This is what a real mobile manipulator does in the equivalent situation:
# brake the base while repositioning an unanchored arm, release the brake
# to actually drive/pull. BRAKE_KP is stiff enough to hold trolley position
# firmly against the leg's reaction without fighting the (separate, much
# larger) pull force during PULL -- it's only ever applied during SWING.
BRAKE_KP, BRAKE_KV = 4000.0, 200.0

def brake_trolley(model, data, hold_x):
    trolley_j = IDS['tjnt']
    qa, dv = IDS['qadr'], model.jnt_dofadr[trolley_j]
    tau = BRAKE_KP * (hold_x - data.qpos[qa]) - BRAKE_KV * data.qvel[dv]
    data.qfrc_applied[dv] = tau
# Measured directly: stopping the grasp early once some target force is
# reached (previously 80N) left too little margin -- as the leg's geometry
# shifts slightly during the pull, that force can drop and the grip slips.
# Always closing fully (to GRIP_MAX_CTRL) instead measured out to a solid,
# unwavering ~100-140N held through an ENTIRE pull cycle with zero drops.
# The Robotiq's own mechanical limit (driver joint hitting its stop against
# the rung) prevents over-squeezing, so there's no downside to just always
# closing all the way.
GRIP_TARGET_FORCE = 1e9   # effectively "never stop early" -- always reach GRIP_MAX_CTRL
GRIP_MAX_CTRL     = 255.0
GRIP_CTRL_RAMP    = 3.0   # ctrl units per step while closing

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

    IDS['support_legs'] = {}
    for leg in ('FR', 'RL', 'RR'):
        jn = [f'{leg}_hip_joint', f'{leg}_thigh_joint', f'{leg}_calf_joint']
        jd = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn]
        IDS['support_legs'][leg] = {
            'qposadr': [model.jnt_qposadr[j] for j in jd],
            'dofadr':  [model.jnt_dofadr[j] for j in jd],
            'act':     [ACT[f'{leg}_hip'], ACT[f'{leg}_thigh'], ACT[f'{leg}_calf']],
        }

    IDS['site']    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'pinch')
    IDS['fingers'] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'fingers_actuator')
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'trolley_slide')
    IDS['tjnt'] = j
    IDS['qadr'] = model.jnt_qposadr[j]

    IDS['pad_ids']  = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                        for n in ('right_pad1', 'right_pad2', 'left_pad1', 'left_pad2')}
    IDS['rung_ids'] = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'rung_{i:02d}')
                        for i in range(1, RUNGS + 1)}
    print(f'  {len(ACT)} actuators  trolley_joint={j}')


def c(data, n, v):
    if n in ACT:
        data.ctrl[ACT[n]] = float(v)


def tx(data):
    return float(data.qpos[IDS['qadr']])


# ── Inverse kinematics for the FL leg (damped least squares on the gripper site) ──
def ik_fl(model, data, target, iters=400, damp=1e-2):
    """Solves for FL joint angles that place the gripper site at `target`.
    Saves and restores data.qpos around the solve -- this mutates qpos
    directly via its internal gradient steps (and calls mj_forward
    repeatedly), so without restoring, calling this mid-simulation (e.g.
    for a swing waypoint) would silently teleport the live leg pose and
    corrupt the actual simulation state. Only the returned angles matter;
    the live state should be untouched by computing them."""
    qposadr, dofadr, ranges = IDS['fl_qposadr'], IDS['fl_dofadr'], IDS['fl_ranges']
    site = IDS['site']
    saved_qpos = data.qpos.copy()
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
    result = [data.qpos[qa] for qa in qposadr]
    data.qpos[:] = saved_qpos
    mujoco.mj_forward(model, data)
    return result


def jaws(data, ctrl_value):
    if 'fingers' in IDS and IDS['fingers'] >= 0:
        data.ctrl[IDS['fingers']] = float(ctrl_value)


def grip_force(model, data):
    """Real contact force (N), summed over every contact currently between
    a fingertip pad and any rung. Replaces the old equality-constraint-based
    version -- same purpose (a performance metric / grip-quality readout),
    now backed by an actual contact instead of a constraint."""
    pad_ids, rung_ids = IDS['pad_ids'], IDS['rung_ids']
    total = 0.0
    vec = np.zeros(3)
    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 in pad_ids and con.geom2 in rung_ids) or \
           (con.geom2 in pad_ids and con.geom1 in rung_ids):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f6)
            total += abs(f6[0])   # normal component, contact-local frame
            vec += con.frame[0:3] * f6[0]
    return total, vec


def n_rung_contacts(data):
    pad_ids, rung_ids = IDS['pad_ids'], IDS['rung_ids']
    n = 0
    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 in pad_ids and con.geom2 in rung_ids) or \
           (con.geom2 in pad_ids and con.geom1 in rung_ids):
            n += 1
    return n


# ── Closed-loop grasp: ramp the fingers closed while watching real contact
# force, stop once GRIP_TARGET_FORCE is reached (or ctrl maxes out). This
# is what "real" grasping means here -- it only succeeds if the fingers are
# actually positioned around a rung; there's no positional snapping.
def close_until_gripped(model, data, target_force=GRIP_TARGET_FORCE,
                         viewer=None, sync_every=10, max_steps=3000, brake_x=None):
    ctrl = 0.0
    force = 0.0
    steps_taken = 0
    for i in range(max_steps):
        ctrl = min(ctrl + GRIP_CTRL_RAMP, GRIP_MAX_CTRL)
        jaws(data, ctrl)
        hold_support_legs(model, data)
        if brake_x is not None:
            brake_trolley(model, data, brake_x)
        else:
            data.qfrc_applied[model.jnt_dofadr[IDS['tjnt']]] = 0.0
        mujoco.mj_step(model, data)
        steps_taken = i + 1
        force, _ = grip_force(model, data)
        if viewer and i % sync_every == 0:
            viewer.sync()
        if force >= target_force or ctrl >= GRIP_MAX_CTRL:
            break
    return ctrl, force, n_rung_contacts(data), steps_taken


# ── PD drive of the FL leg toward a joint-angle target (real actuation, not qpos-forcing) ──
def pd_drive(model, data, target_angles, n_steps, viewer=None, sync_every=10, brake_x=None):
    qposadr, dofadr, act = IDS['fl_qposadr'], IDS['fl_dofadr'], IDS['fl_act']
    for i in range(n_steps):
        for k, (qa, dv, a) in enumerate(zip(qposadr, dofadr, act)):
            tau = KP * (target_angles[k] - data.qpos[qa]) - KV * data.qvel[dv]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = np.clip(tau, lo, hi)
        hold_support_legs(model, data)
        if brake_x is not None:
            brake_trolley(model, data, brake_x)
        else:
            data.qfrc_applied[model.jnt_dofadr[IDS['tjnt']]] = 0.0
        mujoco.mj_step(model, data)
        if viewer and i % sync_every == 0:
            viewer.sync()


# ── Smooth-trajectory PD drive: interpolates the target gradually (smoothstep
# easing) instead of commanding the final pose immediately. This matters a
# lot now that grasping is real friction: jumping straight to a step target
# produces a large initial position error, which the PD turns into a large
# initial torque/force spike at the gripper -- and a real friction grip has
# a hard limit (~mu * normal_force) on how much tangential force it can
# resist before slipping. Measured directly: a step-target pull slipped the
# grip completely within 0.1s (300N -> 0N); the same motion smoothed out
# over the full duration held (~100-150N) the entire way.
def pd_drive_smooth(model, data, start_angles, end_angles, n_steps, viewer=None, sync_every=10,
                     tighten_steps=300, brake_x=None):
    qposadr, dofadr, act = IDS['fl_qposadr'], IDS['fl_dofadr'], IDS['fl_act']
    for i in range(n_steps):
        s = i / n_steps
        ease = s * s * (3 - 2 * s)   # smoothstep
        target = [(1 - ease) * a + ease * b for a, b in zip(start_angles, end_angles)]
        for k, (qa, dv, a) in enumerate(zip(qposadr, dofadr, act)):
            tau = KP * (target[k] - data.qpos[qa]) - KV * data.qvel[dv]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = np.clip(tau, lo, hi)
        hold_support_legs(model, data)
        if brake_x is not None:
            brake_trolley(model, data, brake_x)
        else:
            data.qfrc_applied[model.jnt_dofadr[IDS['tjnt']]] = 0.0
        mujoco.mj_step(model, data)
        if viewer and i % sync_every == 0:
            viewer.sync()
    # By this point velocity is already near zero (smoothstep ends with zero
    # slope), so a plain step-target burst here is safe -- no recoil risk --
    # and closes the residual tracking error smoothstep alone leaves behind
    # within a fixed step budget.
    if tighten_steps > 0:
        pd_drive(model, data, end_angles, tighten_steps, viewer, sync_every, brake_x=brake_x)


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

        current_idx = 0
        reach_angles = ik_fl(model, data, np.array([LADDER_X - 1.05, LADDER_Y, GRIP_Z]))
        for k, qa in enumerate(IDS['fl_qposadr']):
            data.qpos[qa] = reach_angles[k]
        mujoco.mj_forward(model, data)

        ctrl, force, ncon, _cs = close_until_gripped(model, data, viewer=viewer)
        print(f'  Gripping rung_01: ctrl={ctrl:.0f}  force={force:.2f} N  contacts={ncon}'
              f'  Trolley X={tx(data):.4f}')

        # ── PHASE 2: CRAWL ────────────────────────────────────────────
        # reach_angles is now recomputed fresh via IK every cycle, against
        # wherever the trolley ACTUALLY is -- not reused from rung1. PULL
        # doesn't reliably advance exactly RSPACE every time (measured:
        # 0.84 actual vs 0.90 ideal by the 3rd cycle already), and reusing
        # stale cached angles let that undershoot compound cycle over
        # cycle, which is why later rungs were increasingly missing. Fresh
        # IK each cycle means each swing targets the CORRECT rung for
        # wherever the robot actually ended up, regardless of how far off
        # the previous pull fell short.
        print(f'\n━━ PHASE 2: CRAWL ({RUNGS - 1} transitions) ━━')
        x0 = tx(data)
        successful_grips = 1   # rung_01 in Phase 1
        for i in range(RUNGS - 1):
            next_idx = current_idx + 1
            print(f'\n  ┌── rung_{current_idx+1:02d} → rung_{next_idx+1:02d} ─────────')

            # A: PULL — still gripping the current rung, retract the leg.
            #    The retract target is computed fresh too: "one rung-spacing
            #    behind the CURRENT rung", from wherever we actually are.
            current_target = np.array([LADDER_X - 1.05 + current_idx * RSPACE, LADDER_Y, GRIP_Z])
            retract_angles = ik_fl(model, data, current_target - np.array([RSPACE * PULL_OVERSHOOT, 0, 0]))
            print('  │ [A] Pull (retract leg while gripped, smooth trajectory)')
            pd_drive_smooth(model, data, reach_angles, retract_angles, PULL_STEPS, viewer)
            f_during_pull, _ = grip_force(model, data)
            print(f'  │     Trolley X={tx(data):.4f}   grip force now={f_during_pull:.2f} N'
                  f'   (0 here can mean it slipped off -- check contacts)')

            # B: RELEASE + SWING — open the jaws AND WAIT for them to
            #    actually clear the rung before moving the leg at all.
            #    Trolley is braked throughout this unanchored phase (B+C).
            hold_x = tx(data)
            print(f'  │ [B1] Open jaws and wait for clearance (trolley braked at {hold_x:.4f})')
            jaws(data, FINGER_OPEN)
            for _ in range(300):
                hold_support_legs(model, data)
                brake_trolley(model, data, hold_x)
                mujoco.mj_step(model, data)
                if viewer:
                    viewer.sync()
            n_open, _ = grip_force(model, data)
            print(f'  │      Contact force once open: {n_open:.2f} N (should be ~0)')

            # next_reach targets the next rung's ABSOLUTE world position,
            # via fresh IK against the actual current pose -- not a cached
            # offset assumed from rung1.
            next_target = np.array([LADDER_X - 1.05 + next_idx * RSPACE, LADDER_Y, GRIP_Z])
            next_reach = ik_fl(model, data, next_target)
            mid_x = (current_target[0] + next_target[0]) / 2
            clearance_angles = ik_fl(model, data, np.array([mid_x, LADDER_Y - 0.15, GRIP_Z]))
            print('  │ [B2] Swing to next rung (clearance arc)')
            pd_drive_smooth(model, data, retract_angles, clearance_angles, SWING_STEPS // 2,
                             viewer, tighten_steps=0, brake_x=hold_x)
            pd_drive_smooth(model, data, clearance_angles, next_reach, SWING_STEPS // 2,
                             viewer, tighten_steps=300, brake_x=hold_x)

            dist = np.linalg.norm(data.site_xpos[IDS['site']] - next_target)
            print(f'  │     Reach complete, {dist:.4f} m from rung_{next_idx+1:02d} target')

            # C: GRASP — real, closed-loop, force-based.
            ctrl, force, ncon, _cs = close_until_gripped(model, data, viewer=viewer, brake_x=hold_x)
            pd_drive(model, data, next_reach, SETTLE_STEPS, viewer, brake_x=hold_x)
            print(f'  └── Gripping rung_{next_idx+1:02d}: ctrl={ctrl:.0f}  force={force:.2f} N'
                  f'  contacts={ncon}  Trolley X={tx(data):.4f}')
            if ncon > 0:
                successful_grips += 1

            # Advance regardless of grip success -- IK is recomputed fresh
            # next cycle anyway, so a missed grasp doesn't compound; it
            # just means this rung wasn't actually held through the next
            # pull (that pull will find no resistance and free-drift
            # instead, which is visible in the printed grip force).
            reach_angles = next_reach
            current_idx = next_idx

        total = tx(data) - x0
        print(f'\n  ✓ CRAWL COMPLETE')
        print(f'    Distance: {total:.4f} m  |  ideal: {(RUNGS-1)*RSPACE:.4f} m'
              f'  |  successful grips: {successful_grips}/{RUNGS}')

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
