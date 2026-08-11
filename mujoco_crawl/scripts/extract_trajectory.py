#!/usr/bin/env python3
"""
Runs the working, gripper-free straight-ladder MuJoCo crawl (bare FL foot,
friction hold -- 9/9 successful holds, verified) and extracts a joint-
target trajectory plan for real-hardware replay.

Simpler than the gripper version in one genuinely useful way: there's no
finger/jaws state to track at all, because there's no gripper. "Holding a
rung" here is purely the FL leg's own position (pressed against the rung)
-- which is exactly how the real Go2's bare foot would have to work too,
since it has no gripper either. That makes this version a much more
direct match for real hardware than the gripper one.

Each logged phase has: FL joint targets (hip, thigh, calf), phase
duration (from sim, in seconds), and the support-leg stance (only set
once, at the start -- unchanged after that, same as the sim).

Does NOT and cannot capture whether the real foot's friction actually
holds under real load -- that's a physical question the sim's contact
model only approximates. Replaying the joint motion gets the leg moving
through the same positions; whether it actually stays gripped on real
hardware is a separate thing to verify empirically, ideally with the
robot's estimated joint torques (LowState's tau_est) as a rough proxy
for "is this leg under load."
"""
import json
import numpy as np
import mujoco
import sys
sys.path.insert(0, '.')
import crawl as ct

N_CYCLES = 8   # keep small for a first hardware test


def main():
    model = mujoco.MjModel.from_xml_path(ct.MODEL)
    data = mujoco.MjData(model)
    ct.build_maps(model)
    mujoco.mj_resetData(model, data)
    dt = model.opt.timestep

    plan = []

    def log_phase(name, fl_angles, duration_steps, pressed, support_stance=None):
        plan.append({
            'phase': name,
            'fl_hip': float(fl_angles[0]),
            'fl_thigh': float(fl_angles[1]),
            'fl_calf': float(fl_angles[2]),
            'duration_s': round(duration_steps * dt, 4),
            'pressed': bool(pressed),   # True = this phase is meant to be
                                          # holding/pressing on a rung;
                                          # False = free swing, no load
                                          # expected
            'support_stance': support_stance,  # (hip, thigh, calf) for
                                                 # FR/RL/RR, or None to
                                                 # leave unchanged
        })

    current_idx = 0
    reach_angles = ct.ik_fl(model, data, np.array([ct.rung_world_x(0), ct.LADDER_Y, ct.GRIP_Z]))
    for k, qa in enumerate(ct.IDS['fl_qposadr']):
        data.qpos[qa] = reach_angles[k]
    mujoco.mj_forward(model, data)

    log_phase('initial_reach_rung_01', reach_angles, ct.SWING_STEPS, pressed=False,
               support_stance=list(ct.STANCE))

    force, ncon, steps = ct.press_foot_onto_rung(model, data, reach_angles)
    ct.pd_drive(model, data, reach_angles, ct.SETTLE_STEPS, brake_x=ct.tx(data))
    log_phase('initial_press_rung_01', reach_angles, steps + ct.SETTLE_STEPS, pressed=True)
    print(f'rung_01: force={force:.2f}N ncon={ncon}')

    for i in range(min(N_CYCLES, ct.RUNGS - 1)):
        next_idx = current_idx + 1
        current_target = np.array([ct.rung_world_x(current_idx), ct.LADDER_Y, ct.GRIP_Z])
        retract_angles = ct.ik_fl(model, data, current_target - np.array([ct.RSPACE * ct.PULL_OVERSHOOT, 0, 0]))
        steps_used, stop_reason = ct.pd_drive_smooth_until_advance(
            model, data, reach_angles, retract_angles,
            target_advance=ct.RSPACE, max_steps=ct.PULL_STEPS)
        log_phase(f'cycle{i}_pull', retract_angles, steps_used, pressed=True)

        hold_x = ct.tx(data)
        next_target = np.array([ct.rung_world_x(next_idx), ct.LADDER_Y, ct.GRIP_Z])
        next_reach = ct.ik_fl(model, data, next_target)
        mid_x = (current_target[0] + next_target[0]) / 2
        clearance_angles = ct.ik_fl(model, data, np.array([mid_x, ct.LADDER_Y - 0.15, ct.GRIP_Z]))
        ct.pd_drive_smooth(model, data, retract_angles, clearance_angles, ct.SWING_STEPS // 2,
                            tighten_steps=0, brake_x=hold_x)
        log_phase(f'cycle{i}_swing_clearance', clearance_angles, ct.SWING_STEPS // 2, pressed=False)

        ct.pd_drive_smooth(model, data, clearance_angles, next_reach, ct.SWING_STEPS // 2,
                            tighten_steps=300, brake_x=hold_x)
        log_phase(f'cycle{i}_swing_reach_rung_{next_idx+1:02d}', next_reach,
                   ct.SWING_STEPS // 2 + 300, pressed=False)

        force, ncon, steps = ct.press_foot_onto_rung(model, data, next_reach, brake_x=hold_x)
        ct.pd_drive(model, data, next_reach, ct.SETTLE_STEPS, brake_x=hold_x)
        log_phase(f'cycle{i}_press_rung_{next_idx+1:02d}', next_reach,
                   steps + ct.SETTLE_STEPS, pressed=True)
        print(f'cycle{i}: rung_{next_idx+1:02d} force={force:.2f}N ncon={ncon} stop_reason={stop_reason}')

        reach_angles = next_reach
        current_idx = next_idx

    with open('trajectory_plan.json', 'w') as f:
        json.dump(plan, f, indent=2)
    print(f'\nWrote trajectory_plan.json: {len(plan)} phases, '
          f'{sum(p["duration_s"] for p in plan):.1f}s total (at sim speed)')


if __name__ == '__main__':
    main()
