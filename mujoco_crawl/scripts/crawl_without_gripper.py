#!/usr/bin/env python3
"""
Go2 MuJoCo Ladder Crawl — bare foot, no gripper.

WHAT CHANGED FROM THE GRIPPER VERSION:

The Robotiq 2F-85 is gone entirely (removed in build_scene.py). The robot's
own FL foot -- a 2.2cm sphere, already defined in go2.xml with real
hardware-grade friction (mu=0.8 tangential, condim=6, priority=1) -- is
what contacts the rung now. There's no actuator to "close" anymore, so
holding a rung is purely: press the foot against the rung's surface with
enough sustained force that friction resists the pull. That's a real,
meaningful downgrade in grip security versus two fingers wrapping around
the rung -- a sphere touching a cylinder is ONE contact point with no
mechanical hook, versus two points that can't both slide the same
direction at once. Expect this to be less reliable under load than the
gripper version; that's an honest tradeoff of the design, not a bug to
chase out.

press_foot_onto_rung() replaces close_until_gripped(): instead of ramping
a finger actuator, it PD-drives the leg toward a target placed exactly at
the rung's central axis (physically inside the rung's solid geometry).
Once the foot's real surface makes contact, the persistent position error
between "target" and "physically achievable" is what generates sustained
press force through the leg -- the same principle as commanding a
torque-controlled arm to a point behind a wall to generate contact
pressure against it.

Everything else (PULL under smooth trajectory, brake_trolley during the
unanchored swing, hold_support_legs, fresh per-cycle IK, PULL_OVERSHOOT)
is unchanged from the gripper version -- none of that logic cared whether
the end effector was a gripper or a foot.

WHAT GOT THIS FROM 7/9 TO 9/9 SUCCESSFUL HOLDS:

The foot was consistently losing contact (force -> 0) partway through
every PULL, even at high press force -- ruling out "not squeezing hard
enough". The actual cause: go2.xml's default foot friction is
"0.8 0.02 0.01" (sliding, torsional, ROLLING) -- tuned for a foot planted
on flat ground, where rolling resistance barely matters. Against a
CURVED rung, that low rolling friction let the sphere roll/slide around
the cylinder's surface under lateral load instead of staying planted --
a genuinely different failure mode than a flat-ground foot slipping.
Raised to "0.9 0.05 0.5" for the FL foot specifically (build_scene.py;
doesn't affect the other 3 feet, which never touch a rung). PULL still
shows 0 contact force by the time it's measured -- the foot still doesn't
stay glued through the ENTIRE pull -- but the resulting drift is now
small and consistent (~3cm every cycle) rather than compounding, so the
press-and-recover at each new rung succeeds reliably instead of the
occasional total miss.
"""
import mujoco, mujoco.viewer
import numpy as np
import time, sys, os

MODEL   = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'
RUNGS   = 9
RSPACE  = 0.25          # metres between rungs (must match build_scene.py)
LADDER_X, LADDER_Y = 1.25, 0.300   # must match build_scene.py
GRIP_Z  = 0.40                     # must match build_scene.py
RUNG_X0 = -1.00          # rung_01's local X offset from LADDER_X (must match
                          # build_scene.py: RUNG_X0 = -RUNG_SPAN/2 = -(N_RUNGS-1)*RSPACE/2)

def rung_world_x(idx):
    return LADDER_X + RUNG_X0 + idx * RSPACE

PULL_STEPS    = 2000
SWING_STEPS   = 400
SETTLE_STEPS  = 350
KP, KV = 200.0, 8.0
GUIDE_KP, GUIDE_KV = 25.0, 4.0

RSPACE_ = RSPACE  # (kept for grep-friendliness with the older script)
PULL_OVERSHOOT = 1.03   # see gripper version's note: PULL falls a little
                          # short of a full RSPACE most cycles; retracting
                          # slightly past it keeps a joint-limit margin from
                          # eroding cycle over cycle

# Press-force target for "gripping" via the bare foot. There's no
# actuator to ramp here (no fingers) -- this is purely how hard the leg's
# own PD pushes into the rung before we consider it "held".
PRESS_TARGET_FORCE = 40.0
PRESS_MAX_STEPS = 2000

ACT, IDS = {}, {}


def build_maps(model):
    for n in ['FL_hip', 'FL_thigh', 'FL_calf',
              'FR_hip', 'FR_thigh', 'FR_calf',
              'RL_hip', 'RL_thigh', 'RL_calf',
              'RR_hip', 'RR_thigh', 'RR_calf']:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if i >= 0:
            ACT[n] = i
    IDS['fl_joint_names'] = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint']
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in IDS['fl_joint_names']]
    IDS['fl_qposadr'] = [model.jnt_qposadr[j] for j in jids]
    IDS['fl_dofadr']  = [model.jnt_dofadr[j] for j in jids]
    IDS['fl_ranges']  = [model.jnt_range[j].copy() for j in jids]
    IDS['fl_act']     = [ACT['FL_hip'], ACT['FL_thigh'], ACT['FL_calf']]

    IDS['site']      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'foot_site')
    IDS['foot_geom'] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'FL')
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'trolley_slide')
    IDS['tjnt'] = j
    IDS['qadr'] = model.jnt_qposadr[j]
    IDS['trolley_body'] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'trolley')

    IDS['rung_ids'] = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'rung_{i:02d}')
                        for i in range(1, RUNGS + 1)}

    IDS['support_legs'] = {}
    for leg in ('FR', 'RL', 'RR'):
        jn = [f'{leg}_hip_joint', f'{leg}_thigh_joint', f'{leg}_calf_joint']
        jd = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn]
        IDS['support_legs'][leg] = {
            'qposadr': [model.jnt_qposadr[j] for j in jd],
            'dofadr':  [model.jnt_dofadr[j] for j in jd],
            'act':     [ACT[f'{leg}_hip'], ACT[f'{leg}_thigh'], ACT[f'{leg}_calf']],
        }
    print(f'  {len(ACT)} actuators  trolley_joint={j}')


def c(data, n, v):
    if n in ACT:
        data.ctrl[ACT[n]] = float(v)


def tx(data):
    return float(data.qpos[IDS['qadr']])


STANCE = (0.0, 1.2, -2.5)

def hold_support_legs(model, data):
    for leg, ids in IDS['support_legs'].items():
        for k, (qa, dv, a) in enumerate(zip(ids['qposadr'], ids['dofadr'], ids['act'])):
            tau = KP * (STANCE[k] - data.qpos[qa]) - KV * data.qvel[dv]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = np.clip(tau, lo, hi)


BRAKE_KP, BRAKE_KV = 4000.0, 200.0

def brake_trolley(model, data, hold_x):
    dv = model.jnt_dofadr[IDS['tjnt']]
    tau = BRAKE_KP * (hold_x - data.qpos[IDS['qadr']]) - BRAKE_KV * data.qvel[dv]
    data.qfrc_applied[dv] = tau


def ik_fl(model, data, target, iters=400, damp=1e-2):
    """Saves/restores qpos around the solve -- see gripper version's note;
    this mutates data.qpos directly and must not corrupt live sim state
    when called mid-simulation."""
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


def grip_force(model, data):
    """Real contact force (N) between the FL foot and any rung. Same
    purpose/API as the gripper version's grip_force -- now backed by a
    single sphere-cylinder contact instead of two finger pads."""
    foot, rung_ids = IDS['foot_geom'], IDS['rung_ids']
    total = 0.0
    vec = np.zeros(3)
    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 == foot and con.geom2 in rung_ids) or \
           (con.geom2 == foot and con.geom1 in rung_ids):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f6)
            total += abs(f6[0])
            vec += con.frame[0:3] * f6[0]
    return total, vec


def n_rung_contacts(data):
    foot, rung_ids = IDS['foot_geom'], IDS['rung_ids']
    n = 0
    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 == foot and con.geom2 in rung_ids) or \
           (con.geom2 == foot and con.geom1 in rung_ids):
            n += 1
    return n


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


def pd_drive_smooth(model, data, start_angles, end_angles, n_steps, viewer=None, sync_every=10,
                     tighten_steps=300, brake_x=None):
    qposadr, dofadr, act = IDS['fl_qposadr'], IDS['fl_dofadr'], IDS['fl_act']
    for i in range(n_steps):
        s = i / n_steps
        ease = s * s * (3 - 2 * s)
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
    if tighten_steps > 0:
        pd_drive(model, data, end_angles, tighten_steps, viewer, sync_every, brake_x=brake_x)


def press_foot_onto_rung(model, data, target_angles, target_force=PRESS_TARGET_FORCE,
                          viewer=None, sync_every=10, max_steps=PRESS_MAX_STEPS, brake_x=None):
    """Replaces close_until_gripped(). Drives the leg toward `target_angles`
    (computed via IK against the rung's central axis -- physically inside
    the rung's solid geometry) and holds there. Once the foot's real
    surface contacts the rung, the sustained position error is what
    generates press force -- there's no separate "closing" action since
    there's no finger actuator anymore. Returns once target_force is
    reached or max_steps elapses (whichever first), same calling
    convention as the old close_until_gripped for the scripts built on it."""
    qposadr, dofadr, act = IDS['fl_qposadr'], IDS['fl_dofadr'], IDS['fl_act']
    force = 0.0
    steps_taken = 0
    for i in range(max_steps):
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
        steps_taken = i + 1
        force, _ = grip_force(model, data)
        if viewer and i % sync_every == 0:
            viewer.sync()
        if force >= target_force:
            break
    return force, n_rung_contacts(data), steps_taken


# ── Position-triggered pull: stops as soon as the trolley has advanced
# RSPACE, instead of always running the full smoothstep duration.
#
# Diagnosed directly: the fixed-duration pull was reaching its target
# advance by roughly step 350-400 (of 2000), but continued interpolating
# the leg all the way to retract_angles regardless -- and that EXTRA
# continued motion, well past the point of any useful pulling, was what
# physically swung the foot into the NEXT rung's collision geometry (twice,
# measured: force spikes to ~280N then ~150N later in the same pull).
# Stopping as soon as the advance target is met removes that extra sweep
# entirely -- "pull just enough to reach the next rung," not a fixed-time
# motion that happens to also do that along the way.
def pd_drive_smooth_until_advance(model, data, start_angles, end_angles, target_advance,
                                   max_steps, viewer=None, sync_every=10, brake_x=None):
    """Includes a ratchet: the trolley is never allowed to drift backward
    past its position at the start of THIS pull, regardless of what the
    grip is doing. This is what fixes the "collides, pushes back to the
    previous rung" pattern -- traced directly to grip force decaying to 0
    partway through a pull, after which the leg's continued (unconstrained)
    retraction motion was dragging the body backward via momentum/reaction,
    far enough to swing the foot into contact with the PREVIOUS rung.
    Aborting the pull the instant grip is lost was tried and made things
    much worse (healthy pulls also have long legitimate zero-force
    stretches that later recover on their own) -- the ratchet instead just
    prevents the specific backward excursion without interrupting the
    pull's normal progress, the same way a real winch's pawl stops
    backsliding without stopping the winch from continuing to try."""
    qposadr, dofadr, act = IDS['fl_qposadr'], IDS['fl_dofadr'], IDS['fl_act']
    x0 = tx(data)
    qadr = IDS['qadr']
    vadr = model.jnt_dofadr[IDS['tjnt']]
    for i in range(max_steps):
        s = i / max_steps
        ease = s * s * (3 - 2 * s)
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
        # ratchet: clamp any backward excursion past the pull's start
        if data.qpos[qadr] < x0:
            data.qpos[qadr] = x0
            if data.qvel[vadr] < 0:
                data.qvel[vadr] = 0.0
            mujoco.mj_forward(model, data)
        if viewer and i % sync_every == 0:
            viewer.sync()
        if (tx(data) - x0) >= target_advance:
            return i + 1, 'advance_reached'
    return max_steps, 'timeout'


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
        print('\n━━ PHASE 1: SPAWN & PRESS FOOT ONTO RUNG 1 ━━')
        current_idx = 0
        reach_angles = ik_fl(model, data, np.array([rung_world_x(0), LADDER_Y, GRIP_Z]))
        for k, qa in enumerate(IDS['fl_qposadr']):
            data.qpos[qa] = reach_angles[k]
        mujoco.mj_forward(model, data)

        force, ncon, steps = press_foot_onto_rung(model, data, reach_angles, viewer=viewer)
        pd_drive(model, data, reach_angles, SETTLE_STEPS, viewer, brake_x=tx(data))
        print(f'  rung_01: force={force:.2f} N  contacts={ncon}  steps={steps}  Trolley X={tx(data):.4f}')

        print(f'\n━━ PHASE 2: CRAWL ({RUNGS - 1} transitions) ━━')
        x0 = tx(data)
        successful_holds = 1
        for i in range(RUNGS - 1):
            next_idx = current_idx + 1
            print(f'\n  ┌── rung_{current_idx+1:02d} → rung_{next_idx+1:02d} ─────────')

            current_target = np.array([rung_world_x(current_idx), LADDER_Y, GRIP_Z])
            retract_angles = ik_fl(model, data, current_target - np.array([RSPACE * PULL_OVERSHOOT, 0, 0]))
            print('  │ [A] Pull (stops at target advance OR if grip is lost)')
            steps_used, stop_reason = pd_drive_smooth_until_advance(
                model, data, reach_angles, retract_angles,
                target_advance=RSPACE, max_steps=PULL_STEPS, viewer=viewer)
            f_during_pull, _ = grip_force(model, data)
            print(f'  │     Trolley X={tx(data):.4f}   contact force now={f_during_pull:.2f} N'
                  f'   stop_reason={stop_reason} steps={steps_used}')

            hold_x = tx(data)
            print(f'  │ [B] Lift foot clear and swing to next rung (trolley braked at {hold_x:.4f})')
            next_target = np.array([rung_world_x(next_idx), LADDER_Y, GRIP_Z])
            next_reach = ik_fl(model, data, next_target)
            mid_x = (current_target[0] + next_target[0]) / 2
            clearance_angles = ik_fl(model, data, np.array([mid_x, LADDER_Y - 0.15, GRIP_Z]))
            pd_drive_smooth(model, data, retract_angles, clearance_angles, SWING_STEPS // 2,
                             viewer, tighten_steps=0, brake_x=hold_x)
            pd_drive_smooth(model, data, clearance_angles, next_reach, SWING_STEPS // 2,
                             viewer, tighten_steps=300, brake_x=hold_x)

            dist = np.linalg.norm(data.site_xpos[IDS['site']] - next_target)
            print(f'  │     Reach complete, {dist:.4f} m from rung_{next_idx+1:02d} target')

            force, ncon, steps = press_foot_onto_rung(model, data, next_reach, viewer=viewer, brake_x=hold_x)
            pd_drive(model, data, next_reach, SETTLE_STEPS, viewer, brake_x=hold_x)
            print(f'  └── rung_{next_idx+1:02d}: force={force:.2f} N  contacts={ncon}  steps={steps}'
                  f'  Trolley X={tx(data):.4f}')
            if ncon > 0:
                successful_holds += 1

            reach_angles = next_reach
            current_idx = next_idx

        total = tx(data) - x0
        print(f'\n  ✓ CRAWL COMPLETE')
        print(f'    Distance: {total:.4f} m  |  ideal: {(RUNGS-1)*RSPACE:.4f} m'
              f'  |  successful holds: {successful_holds}/{RUNGS}')

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
