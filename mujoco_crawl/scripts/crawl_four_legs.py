#!/usr/bin/env python3
"""
Go2 MuJoCo Ladder Crawl -- ALL FOUR LEGS, bare feet, no gripper.

WHAT CHANGED FROM THE SINGLE-LEG (FL-ONLY) VERSION:

All four legs (FL, FR, RL, RR) now grip rungs and take turns pulling,
instead of only FL doing the work while FR/RL/RR hold a fixed static
stance. This is a genuine quadrupedal climbing gait: exactly ONE leg is
ever "active" (releasing, swinging, re-gripping) at a time, while the
other THREE stay anchored on their current rungs, holding the body
stable -- the same principle a real 4-limbed climber uses (a sloth, a
child on monkey bars): move one limb at a time, keep three points of
contact always.

KEY GEOMETRIC FACT this design is built on, verified directly via forward
kinematics before writing any of this: with the robot lying on its side
in this exact mount orientation, FL/RL share one hip height (world
Z=0.4165) and FR/RR share a different, lower one (world Z=0.3235) -- a
real ~9.3cm split baked into the robot's own geometry, not a modeling
choice. Each leg pair therefore has its OWN GRIP_Z, verified via IK grid
search to converge cleanly with comfortable joint margins:
  FL, RL -> GRIP_Z = 0.40  (matches the original single-leg value)
  FR, RR -> GRIP_Z = 0.30
All four still target gx=0.25-relative rung positions (same LADDER_X
calibration as before) and gy=LADDER_Y=0.30.

Front legs (FL, FR) share hip X=+0.1934; back legs (RL, RR) share hip
X=-0.1934 -- a ~0.387m front-back offset baked into the body. Rather than
hand-track which specific rung index each leg "should" be on, each leg
independently finds whichever rung is physically nearest to its OWN
foot's current position every cycle (nearest_rung_for_leg()) -- the same
"recompute fresh from actual state" principle that fixed drift issues in
earlier iterations of this project. This means different legs will
naturally end up gripping different, nearby rungs rather than all four
sharing one rung -- exactly what you'd expect given the body's own
length.

GAIT ORDER: FL, FR, RL, RR, repeating. Arbitrary but reasonable -- easy to
change (see LEG_ORDER below) if a different sequence turns out to matter
for stability once tested.

Everything else -- press-onto-rung via sustained PD force, the position-
triggered pull with a backward-drift ratchet, softened foot-rung contact,
raised rolling friction -- is unchanged in mechanism, just generalized
from a single hardcoded FL to work for any leg.
"""
import mujoco, mujoco.viewer
import numpy as np
import time, sys, os

MODEL   = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'
RUNGS   = 11   # extended from 9 -- see build_scene.py's comment: RL/RR
                # couldn't reach the old rung_01 at all
RSPACE  = 0.25
LADDER_X, LADDER_Y = 1.25, 0.300
RUNG_X0 = -1.25   # must match build_scene.py's extended span

LEG_ORDER = ['FL', 'FR', 'RL', 'RR']   # gait order -- change freely
LEG_GRIP_Z = {'FL': 0.40, 'RL': 0.40, 'FR': 0.30, 'RR': 0.30}
# Front legs (FL/FR, hip X=+0.1934) and back legs (RL/RR, hip X=-0.1934)
# need DIFFERENT initial rungs -- verified directly that a single shared
# initial target doesn't work for both: rung_01 (x=0.0) is comfortable for
# RL/RR but rung_02 (x=0.25) is what FL/FR were actually validated
# against. Using the wrong one for either pair leaves it unable to
# establish real contact on its own turn.
LEG_INITIAL_RUNG = {'FL': 1, 'FR': 1, 'RL': 0, 'RR': 0}

PULL_STEPS    = 2000
SWING_STEPS   = 400
SETTLE_STEPS  = 350
KP, KV = 200.0, 8.0

PULL_OVERSHOOT = 1.03
PRESS_TARGET_FORCE = 40.0
PRESS_MAX_STEPS = 2000
BRAKE_KP, BRAKE_KV = 4000.0, 200.0

ACT, IDS = {}, {}


def rung_world_x(idx):
    return LADDER_X + RUNG_X0 + idx * RSPACE


def build_maps(model):
    for leg in ['FL', 'FR', 'RL', 'RR']:
        for part in ['hip', 'thigh', 'calf']:
            n = f'{leg}_{part}'
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            if i >= 0:
                ACT[n] = i

    IDS['legs'] = {}
    for leg in ['FL', 'FR', 'RL', 'RR']:
        joint_names = [f'{leg}_hip_joint', f'{leg}_thigh_joint', f'{leg}_calf_joint']
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        IDS['legs'][leg] = {
            'qposadr': [model.jnt_qposadr[j] for j in jids],
            'dofadr':  [model.jnt_dofadr[j] for j in jids],
            'ranges':  [model.jnt_range[j].copy() for j in jids],
            'act':     [ACT[f'{leg}_hip'], ACT[f'{leg}_thigh'], ACT[f'{leg}_calf']],
            'site':    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'{leg.lower()}_foot_site'),
            'geom':    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg),
        }

    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'trolley_slide')
    IDS['tjnt'] = j
    IDS['qadr'] = model.jnt_qposadr[j]
    IDS['trolley_body'] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'trolley')
    IDS['rung_ids'] = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'rung_{i:02d}')
                        for i in range(1, RUNGS + 1)}
    print(f'  {len(ACT)} actuators  trolley_joint={j}')


def tx(data):
    return float(data.qpos[IDS['qadr']])


def brake_trolley(model, data, hold_x):
    dv = model.jnt_dofadr[IDS['tjnt']]
    tau = BRAKE_KP * (hold_x - data.qpos[IDS['qadr']]) - BRAKE_KV * data.qvel[dv]
    data.qfrc_applied[dv] = tau


def hold_leg(model, data, leg, target_angles):
    L = IDS['legs'][leg]
    for k, (qa, dv, a) in enumerate(zip(L['qposadr'], L['dofadr'], L['act'])):
        tau = KP * (target_angles[k] - data.qpos[qa]) - KV * data.qvel[dv]
        lo, hi = model.actuator_ctrlrange[a]
        data.ctrl[a] = np.clip(tau, lo, hi)


def ik_leg(model, data, leg, target, iters=400, damp=1e-2):
    L = IDS['legs'][leg]
    qposadr, dofadr, ranges, site = L['qposadr'], L['dofadr'], L['ranges'], L['site']
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


def grip_force(model, data, leg):
    L = IDS['legs'][leg]
    foot, rung_ids = L['geom'], IDS['rung_ids']
    total = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 == foot and con.geom2 in rung_ids) or \
           (con.geom2 == foot and con.geom1 in rung_ids):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f6)
            total += abs(f6[0])
    return total


def n_rung_contacts(data, leg):
    L = IDS['legs'][leg]
    foot, rung_ids = L['geom'], IDS['rung_ids']
    n = 0
    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 == foot and con.geom2 in rung_ids) or \
           (con.geom2 == foot and con.geom1 in rung_ids):
            n += 1
    return n


def nearest_rung_for_leg(data, leg):
    site = IDS['legs'][leg]['site']
    foot_x = data.site_xpos[site][0]
    idx = round((foot_x - LADDER_X - RUNG_X0) / RSPACE)
    return max(0, min(RUNGS - 1, idx))


def pd_drive_all(model, data, active_leg, active_target, other_targets, n_steps,
                  viewer=None, sync_every=10, brake_x=None):
    L = IDS['legs'][active_leg]
    qposadr, dofadr, act = L['qposadr'], L['dofadr'], L['act']
    for i in range(n_steps):
        for k, (qa, dv, a) in enumerate(zip(qposadr, dofadr, act)):
            tau = KP * (active_target[k] - data.qpos[qa]) - KV * data.qvel[dv]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = np.clip(tau, lo, hi)
        for leg, tgt in other_targets.items():
            hold_leg(model, data, leg, tgt)
        if brake_x is not None:
            brake_trolley(model, data, brake_x)
        else:
            data.qfrc_applied[model.jnt_dofadr[IDS['tjnt']]] = 0.0
        mujoco.mj_step(model, data)
        if viewer and i % sync_every == 0:
            viewer.sync()


def pd_drive_staged_reach(model, data, active_leg, start_angles, end_angles, other_targets,
                           steps_per_joint, viewer=None, sync_every=10, brake_x=None):
    """Moves the active leg's 3 joints SEQUENTIALLY -- hip first (the
    primary lifting/abduction motion), then thigh (extending the upper
    link toward horizontal), then calf (extending the lower link) --
    instead of all 3 simultaneously. This is what makes the leg visibly
    straighten out toward the target in stages, matching the reference
    motion (hip lift -> extend thigh -> extend calf -> reach position)
    rather than a blended, curled-looking simultaneous motion. Ends at
    exactly end_angles (the IK-solved target), so the final foot position
    is unaffected -- only the PATH taken to get there changes."""
    L = IDS['legs'][active_leg]
    qposadr, dofadr, act = L['qposadr'], L['dofadr'], L['act']
    current = list(start_angles)
    for joint_idx in range(3):   # 0=hip, 1=thigh, 2=calf
        joint_start = current[joint_idx]
        joint_end = end_angles[joint_idx]
        for i in range(steps_per_joint):
            s = i / steps_per_joint
            ease = s * s * (3 - 2 * s)
            target = list(current)
            target[joint_idx] = joint_start + ease * (joint_end - joint_start)
            for k, (qa, dv, a) in enumerate(zip(qposadr, dofadr, act)):
                tau = KP * (target[k] - data.qpos[qa]) - KV * data.qvel[dv]
                lo, hi = model.actuator_ctrlrange[a]
                data.ctrl[a] = np.clip(tau, lo, hi)
            for leg, tgt in other_targets.items():
                hold_leg(model, data, leg, tgt)
            if brake_x is not None:
                brake_trolley(model, data, brake_x)
            else:
                data.qfrc_applied[model.jnt_dofadr[IDS['tjnt']]] = 0.0
            mujoco.mj_step(model, data)
            if viewer and i % sync_every == 0:
                viewer.sync()
        current[joint_idx] = joint_end


def pd_drive_smooth_all(model, data, active_leg, start_angles, end_angles, other_targets,
                         n_steps, viewer=None, sync_every=10, tighten_steps=300, brake_x=None):
    L = IDS['legs'][active_leg]
    qposadr, dofadr, act = L['qposadr'], L['dofadr'], L['act']
    for i in range(n_steps):
        s = i / n_steps
        ease = s * s * (3 - 2 * s)
        target = [(1 - ease) * a + ease * b for a, b in zip(start_angles, end_angles)]
        for k, (qa, dv, a) in enumerate(zip(qposadr, dofadr, act)):
            tau = KP * (target[k] - data.qpos[qa]) - KV * data.qvel[dv]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = np.clip(tau, lo, hi)
        for leg, tgt in other_targets.items():
            hold_leg(model, data, leg, tgt)
        if brake_x is not None:
            brake_trolley(model, data, brake_x)
        else:
            data.qfrc_applied[model.jnt_dofadr[IDS['tjnt']]] = 0.0
        mujoco.mj_step(model, data)
        if viewer and i % sync_every == 0:
            viewer.sync()
    if tighten_steps > 0:
        pd_drive_all(model, data, active_leg, end_angles, other_targets, tighten_steps,
                     viewer, sync_every, brake_x=brake_x)


def pd_drive_smooth_until_advance_all(model, data, active_leg, start_angles, end_angles,
                                       other_targets, target_advance, max_steps,
                                       viewer=None, sync_every=10):
    L = IDS['legs'][active_leg]
    qposadr, dofadr, act = L['qposadr'], L['dofadr'], L['act']
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
        for leg, tgt in other_targets.items():
            hold_leg(model, data, leg, tgt)
        data.qfrc_applied[model.jnt_dofadr[IDS['tjnt']]] = 0.0
        mujoco.mj_step(model, data)
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


def press_leg_onto_rung(model, data, leg, target_angles, other_targets,
                         target_force=PRESS_TARGET_FORCE, viewer=None, sync_every=10,
                         max_steps=PRESS_MAX_STEPS, brake_x=None):
    L = IDS['legs'][leg]
    qposadr, dofadr, act = L['qposadr'], L['dofadr'], L['act']
    force = 0.0
    steps_taken = 0
    for i in range(max_steps):
        for k, (qa, dv, a) in enumerate(zip(qposadr, dofadr, act)):
            tau = KP * (target_angles[k] - data.qpos[qa]) - KV * data.qvel[dv]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = np.clip(tau, lo, hi)
        for other_leg, tgt in other_targets.items():
            hold_leg(model, data, other_leg, tgt)
        if brake_x is not None:
            brake_trolley(model, data, brake_x)
        else:
            data.qfrc_applied[model.jnt_dofadr[IDS['tjnt']]] = 0.0
        mujoco.mj_step(model, data)
        steps_taken = i + 1
        force = grip_force(model, data, leg)
        if viewer and i % sync_every == 0:
            viewer.sync()
        if force >= target_force:
            break
    return force, n_rung_contacts(data, leg), steps_taken


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
        print('\n\u2501\u2501 PHASE 1: INITIAL GRIP -- all 4 legs reach and press, one at a time \u2501\u2501')
        # Brake the trolley at a fixed position for ALL of Phase 1. Without
        # this, each leg's press generates real reaction force that shifts
        # the trolley -- confirmed directly this was silently breaking
        # EARLIER legs' grip (FL succeeded at 68.6N on its own turn, but
        # ended Phase 1 at 0.00N/0 contacts, dragged 0.19m off its rung by
        # the trolley shifting under later legs' presses). Re-pressing
        # broken legs after the fact was tried and made it worse (a
        # whack-a-mole loop -- fixing one leg's grip disturbs another).
        # Braking prevents the disturbance from happening at all.
        setup_brake_x = tx(data)
        current_angles = {}
        for leg in LEG_ORDER:
            target = np.array([rung_world_x(LEG_INITIAL_RUNG[leg]), LADDER_Y, LEG_GRIP_Z[leg]])
            angles = ik_leg(model, data, leg, target)
            L = IDS['legs'][leg]
            for k, qa in enumerate(L['qposadr']):
                data.qpos[qa] = angles[k]
            mujoco.mj_forward(model, data)
            others = {l: current_angles[l] for l in current_angles}
            force, ncon, steps = press_leg_onto_rung(model, data, leg, angles, others,
                                                       brake_x=setup_brake_x)
            pd_drive_all(model, data, leg, angles, others, SETTLE_STEPS, brake_x=setup_brake_x)
            current_angles[leg] = angles
            print(f'  {leg}: force={force:.2f}N ncon={ncon} steps={steps}  Trolley X={tx(data):.4f}')

        print('\n  -- re-verifying grip (braked, catches any residual small-drift misses) --')
        for leg in LEG_ORDER:
            f = grip_force(model, data, leg)
            if f < 20.0:
                idx = LEG_INITIAL_RUNG[leg]
                target = np.array([rung_world_x(idx), LADDER_Y, LEG_GRIP_Z[leg]])
                angles = ik_leg(model, data, leg, target)
                others = {l: current_angles[l] for l in LEG_ORDER if l != leg}
                force, ncon, steps = press_leg_onto_rung(model, data, leg, angles, others,
                                                           brake_x=setup_brake_x)
                pd_drive_all(model, data, leg, angles, others, SETTLE_STEPS, brake_x=setup_brake_x)
                current_angles[leg] = angles
                print(f'    re-pressed {leg}: force={force:.2f}N ncon={ncon}')

        print('\n  -- final grip check after Phase 1 --')
        for leg in LEG_ORDER:
            f, n = grip_force(model, data, leg), n_rung_contacts(data, leg)
            print(f'  {leg}: force={f:.2f}N ncon={n}')

        # COLLECTIVE SETTLE before Phase 2 begins. Phase 1's sequential
        # gripping (4 presses plus a re-verification pass) leaves residual
        # velocity/dynamics in the system that a single SETTLE_STEPS pause
        # doesn't fully damp out -- this is the same root cause that made
        # the single-leg version's early cycles noticeably worse than its
        # later ones (traced back then to an unsettled, high-variance
        # initial state bleeding into the first several cycles' dynamics).
        # All 4 legs held at their just-established grip targets, trolley
        # braked, for a longer, dedicated settle before any cycling starts.
        print('\n  -- collective settle before crawling begins --')
        settle_brake_x = tx(data)
        for i in range(3000):
            for leg in LEG_ORDER:
                hold_leg(model, data, leg, current_angles[leg])
            brake_trolley(model, data, settle_brake_x)
            mujoco.mj_step(model, data)
        for leg in LEG_ORDER:
            f, n = grip_force(model, data, leg), n_rung_contacts(data, leg)
            print(f'  after settle: {leg} force={f:.2f}N ncon={n}')

        print(f'\n\u2501\u2501 PHASE 2: CRAWL -- cycling {LEG_ORDER} \u2501\u2501')
        x0 = tx(data)
        successful_holds = len(LEG_ORDER)
        n_cycles = 24
        gait_i = 0
        for cycle in range(1, n_cycles + 1):
            active_leg = LEG_ORDER[gait_i % len(LEG_ORDER)]
            gait_i += 1
            other_targets = {l: current_angles[l] for l in LEG_ORDER if l != active_leg}

            print(f'\n  cycle {cycle}: {active_leg} active')
            current_idx = nearest_rung_for_leg(data, active_leg)
            current_target = np.array([rung_world_x(current_idx), LADDER_Y, LEG_GRIP_Z[active_leg]])
            retract_angles = ik_leg(model, data, active_leg,
                                     current_target - np.array([RSPACE * PULL_OVERSHOOT, 0, 0]))
            steps_used, stop_reason = pd_drive_smooth_until_advance_all(
                model, data, active_leg, current_angles[active_leg], retract_angles,
                other_targets, RSPACE, PULL_STEPS, viewer)
            print(f'    [A] Pull  Trolley X={tx(data):.4f}  stop_reason={stop_reason} steps={steps_used}')

            hold_x = tx(data)
            next_idx = min(current_idx + 1, RUNGS - 1)
            next_target = np.array([rung_world_x(next_idx), LADDER_Y, LEG_GRIP_Z[active_leg]])
            next_reach = ik_leg(model, data, active_leg, next_target)
            pd_drive_staged_reach(model, data, active_leg, retract_angles, next_reach,
                                   other_targets, SWING_STEPS // 3, viewer, brake_x=hold_x)

            force, ncon, steps = press_leg_onto_rung(model, data, active_leg, next_reach,
                                                        other_targets, viewer=viewer, brake_x=hold_x)
            pd_drive_all(model, data, active_leg, next_reach, other_targets, SETTLE_STEPS,
                         viewer, brake_x=hold_x)
            print(f'    [B] {active_leg} -> rung_{next_idx+1:02d}: force={force:.2f}N ncon={ncon} '
                  f'steps={steps}  Trolley X={tx(data):.4f}')
            if ncon > 0:
                successful_holds += 1

            current_angles[active_leg] = next_reach

        total = tx(data) - x0
        print(f'\n  CRAWL COMPLETE')
        print(f'    Distance: {total:.4f} m  |  cycles: {n_cycles}  |  '
              f'successful holds: {successful_holds}/{n_cycles + len(LEG_ORDER)}')

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
