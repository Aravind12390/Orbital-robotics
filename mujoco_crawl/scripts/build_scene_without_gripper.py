#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import mujoco
import math

MENAGERIE_ROOT = '/home/student22/mujoco_menagerie'
MENAGERIE      = f'{MENAGERIE_ROOT}/unitree_go2'
GO2_XML        = f'{MENAGERIE}/go2.xml'
ASSETS         = f'{MENAGERIE}/assets'
OUT            = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'

# Recalibrated for the bare foot (no gripper extending the reach anymore).
# Found via IK grid search: gx=0.25, gy=0.30, gz=0.40 gives a clean,
# comfortably-within-joint-limits solution for rung_01 (previously
# LADDER_X=1.619 was calibrated for the gripper's extra ~0.15m of reach
# from the pinch site; the bare foot only reaches to about the calf's own
# kinematic length, ~0.25-0.35m from the hip).
LADDER_X = 1.25
LADDER_Y = 0.300
GRIP_Z   = 0.40   # world Z target for the foot -- see crawl.py, must match

# ============================== Ladder geometry (per spec) ===================
RUNG_LENGTH   = 0.49     # m -- physical length of each rung bar
RAIL_LENGTH   = 2.50     # m -- total length of each side rail
RUNG_CIRCUM   = 0.125    # m -- rung circumference (rung modeled as a cylinder)
N_RUNGS       = 9
RUNG_SPACING  = 0.25     # m -- center-to-center spacing along the ladder's length
RAIL_SEP      = 0.29     # m -- "rung to rung (top to bottom)": distance between
                          #      the two rails; the rung extends symmetrically
                          #      past each rail since RUNG_LENGTH > RAIL_SEP
                          #      (49cm rung, 29cm between rails -> 10cm overhang
                          #      on each end)

RUNG_RADIUS = RUNG_CIRCUM / (2 * math.pi)          # ~0.0199 m
RUNG_SPAN   = (N_RUNGS - 1) * RUNG_SPACING          # total span of the rung field
RAIL_MARGIN = (RAIL_LENGTH - RUNG_SPAN) / 2         # clearance at each rail end
RUNG_X0     = -RUNG_SPAN / 2                        # first rung's local X (ladder-centered)
RAIL_HALF   = RAIL_LENGTH / 2

print(f'Derived: rung_radius={RUNG_RADIUS:.5f}m  rung_span={RUNG_SPAN:.3f}m  '
      f'rail_margin={RAIL_MARGIN:.3f}m')

# ============================== Go2 body =====================================
tree = ET.parse(GO2_XML)
root = tree.getroot()

def get_block(el_root, tag):
    return el_root.find(tag)

default_el = get_block(root, 'default')
asset_el   = get_block(root, 'asset')
act_el     = get_block(root, 'actuator')
contact_el = get_block(root, 'contact')

worldbody = root.find('worldbody')
robot = [b for b in worldbody if b.tag == 'body'][0]
for ch in list(robot):
    if ch.tag == 'freejoint' or (ch.tag == 'joint' and ch.get('type') == 'free'):
        robot.remove(ch)
robot.set('pos',   '0 0 0.19')
robot.set('euler', '1.5708 0 0')

def find_body(el, name):
    if el.tag == 'body' and el.get('name') == name:
        return el
    for ch in el:
        r = find_body(ch, name)
        if r is not None:
            return r
    return None

fl_calf = find_body(robot, 'FL_calf')
print(f'FL_calf found: {fl_calf is not None}')

# ==================== No gripper: the bare FL foot lands on the rung =========
# The gripper (Robotiq 2F-85) is gone entirely. The robot's own foot geom
# "FL" (a 2.2cm-radius sphere, already defined in go2.xml with high tangential
# friction -- mu=0.8, condim=6, priority=1) is what contacts the rung now.
# We add a site at the exact same location purely so crawl.py has something
# to run IK against and read position from (the foot geom itself doesn't
# have a clean "center point" handle the way a site does).
fl_calf.append(ET.fromstring('<site name="foot_site" pos="-0.002 0 -0.213" size="0.005"/>'))

# The foot geom "FL" needs its collision opened up to the rung's isolated
# group (see rung contype/conaffinity below) -- same isolation technique as
# the old gripper pads: rungs only ever collide with THIS geom, nothing else
# in the scene (torso, other legs, trolley) touches them.
def add_attr_to_geom(root_el, geom_name, attrs):
    for geom in root_el.iter('geom'):
        if geom.get('name') == geom_name:
            for k, v in attrs.items():
                geom.set(k, v)
            return True
    return False

found = add_attr_to_geom(robot, 'FL', {
    'contype': '3', 'conaffinity': '3',
    # Default "foot" class friction is "0.8 0.02 0.01" (sliding, torsional,
    # rolling) -- tuned for a foot planted on flat ground, where rolling
    # resistance barely matters. Against a CURVED rung, that low rolling
    # friction (0.01) lets the sphere roll/slide around the cylinder's
    # surface under lateral load instead of staying planted. Raised
    # substantially higher here; only overrides this one geom, not the
    # other 3 feet (they never touch a rung).
    'friction': '0.9 0.05 0.5',
    # Default "foot" class also sets solimp="0.015 1 0.022" (very stiff --
    # dmin=0.015 is close to a hard contact) with priority="1", meaning
    # THIS geom's solref/solimp win the contact against the rung
    # regardless of what the rung itself specifies (tried softening the
    # rung's own solref first -- had zero effect, confirming this).
    # Softened here instead: gentler ramp-up on first contact rather than
    # resolving a deep-penetration correction in a single timestep (which
    # is what was producing the ~600N single-step spikes / hard-impact
    # feel on approach).
    'solref': '0.05 1',
    'solimp': '0.85 0.95 0.01',
})
print(f'FL foot collision group extended: {found}')

# IMU mounting site (unrelated to the gripper removal, kept from before)
robot.append(ET.fromstring('<site name="imu_site" pos="0 0 0" size="0.005"/>'))
robot_str = ET.tostring(robot, encoding='unicode')

default_str = ET.tostring(default_el, encoding='unicode')
asset_str   = ET.tostring(asset_el, encoding='unicode')

# ---- contact: go2's own self-collision excludes, unchanged. No gripper
#      subtree to exclude anymore -- the foot is already part of go2's own
#      body tree and its self-collision behavior is whatever go2.xml
#      already specifies.
contact_str = ET.tostring(contact_el, encoding='unicode') if contact_el is not None else '<contact/>'

# ---- actuators: go2's own leg motors only -- no finger actuator anymore.
act_str = ET.tostring(act_el, encoding='unicode')

# ---- rung geoms: cylinders now (matching the circumference spec), radius
#      RUNG_RADIUS, length RUNG_LENGTH, spaced RUNG_SPACING apart along X.
rung_geoms = []
for i in range(N_RUNGS):
    x = RUNG_X0 + i * RUNG_SPACING
    rung_geoms.append(
        f'<geom name="rung_{i+1:02d}" type="cylinder" size="{RUNG_RADIUS:.5f} {RUNG_LENGTH/2:.4f}" '
        f'pos="{x:.4f} 0 0.45" '   # default cylinder axis is local Z, which is
                                    # already vertical here (ladder body has no
                                    # rotation) -- exactly what's needed to span
                                    # between the two Z-separated rails
        f'rgba="0.12 0.12 0.12 1" contype="2" conaffinity="2" friction="0.9 0.02 0.002" '
        f'solref="0.05 1" solimp="0.9 0.95 0.01"/>'
    )
rung_geoms_str = '\n      '.join(rung_geoms)

# ---- side rails: two thin bars spanning the full RAIL_LENGTH, separated by
#      RAIL_SEP in Z, positioned so the rungs (centered at z=0.45, half-length
#      RUNG_LENGTH/2) extend symmetrically past both rails.
RUNG_CENTER_Z = 0.45
rail_z_lo = RUNG_CENTER_Z - RUNG_LENGTH / 2   # exactly at the rung's bottom end
rail_z_hi = RUNG_CENTER_Z + RUNG_LENGTH / 2   # exactly at the rung's top end
RAIL_THICKNESS = 0.015   # half-extent -> 3cm full thickness

# Ground support posts: 4 vertical posts (one at each rail end x rail) from
# the floor (z=0) up to the lower rail, so the ladder visually rests on the
# ground instead of floating. Positioned at the rail's two X-ends.

r = RUNG_RADIUS  # kept for readability below

scene = f"""<mujoco model="go2_real_ladder_crawl">
  <compiler angle="radian" meshdir="{ASSETS}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -0.981" cone="elliptic"
          solver="Newton" iterations="100" tolerance="1e-10" impratio="10"/>
  {default_str}
  {asset_str}
  {contact_str}
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="10 10 0.1"
          rgba="0.80 0.80 0.80 1" friction="0.005 0.001 0.001"/>

    <body name="ladder" pos="{LADDER_X} {LADDER_Y} 0">
      <geom type="box" size="{RAIL_HALF:.3f} {RAIL_THICKNESS} {RAIL_THICKNESS}" pos="0 0 {rail_z_lo:.4f}"
            rgba="0.20 0.20 0.20 1" contype="0" conaffinity="0"/>
      <geom type="box" size="{RAIL_HALF:.3f} {RAIL_THICKNESS} {RAIL_THICKNESS}" pos="0 0 {rail_z_hi:.4f}"
            rgba="0.20 0.20 0.20 1" contype="0" conaffinity="0"/>
      {rung_geoms_str}

      <!-- Ground support posts: purely visual/structural (no collision --
           they'd otherwise sit right where the trolley needs to roll
           through), so the ladder reads as resting on the floor instead
           of floating. One post at each of the 4 rail-end x rail-height
           corners, from the ground up to that rail. -->
      <geom type="cylinder" size="0.02 {rail_z_lo/2:.4f}" pos="{-RAIL_HALF:.3f} 0 {rail_z_lo/2:.4f}"
            rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.02 {rail_z_lo/2:.4f}" pos="{RAIL_HALF:.3f} 0 {rail_z_lo/2:.4f}"
            rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0"/>
    </body>

    <body name="trolley" pos="0 0 0.18">
      <!-- z=0.18, not the previous 0.12: wheel bodies sit at local z=-0.10
           with radius 0.08 (cylinder, rotated so radius extends in world Z)
           -- at 0.12 the wheel bottoms were at -0.06 (embedded 6cm below
           the floor); 0.18 puts them exactly at z=0, resting on it. -->
      <joint name="trolley_slide" type="slide" axis="1 0 0" damping="0.5" limited="false"/>
      <geom type="box" size="0.35 0.25 0.02" rgba="0.22 0.22 0.22 1" mass="5.0"/>
      <body name="wfl" pos=" 0.27  0.26 -0.10"><joint name="wfl_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.08 0.025" euler="1.5708 0 0" mass="0.5" rgba="0.10 0.10 0.10 1" friction="0.005 0.001 0.001"/></body>
      <body name="wfr" pos=" 0.27 -0.26 -0.10"><joint name="wfr_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.08 0.025" euler="1.5708 0 0" mass="0.5" rgba="0.10 0.10 0.10 1" friction="0.005 0.001 0.001"/></body>
      <body name="wrl" pos="-0.27  0.26 -0.10"><joint name="wrl_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.08 0.025" euler="1.5708 0 0" mass="0.5" rgba="0.10 0.10 0.10 1" friction="0.005 0.001 0.001"/></body>
      <body name="wrr" pos="-0.27 -0.26 -0.10"><joint name="wrr_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.08 0.025" euler="1.5708 0 0" mass="0.5" rgba="0.10 0.10 0.10 1" friction="0.005 0.001 0.001"/></body>
      {robot_str}
    </body>
  </worldbody>

  {act_str}

  <sensor>
    <jointpos name="trolley_x"   joint="trolley_slide"/>
    <framepos name="fl_calf_pos" objtype="body" objname="FL_calf"/>
    <framepos name="foot_pos"    objtype="site" objname="foot_site"/>

    <accelerometer name="imu_accel" site="imu_site"/>
    <gyro          name="imu_gyro"  site="imu_site"/>
    <framepos  name="mocap_pos"  objtype="body" objname="base"/>
    <framequat name="mocap_quat" objtype="body" objname="base"/>
    <framelinvel name="truth_vel" objtype="body" objname="base"/>
  </sensor>
</mujoco>"""

with open(OUT, 'w') as f:
    f.write(scene)
print(f'Written: {OUT}')
print(f'rung_01 world X={LADDER_X + RUNG_X0:.4f}  Y={LADDER_Y:.3f}')

try:
    m = mujoco.MjModel.from_xml_path(OUT)
    print(f'Model OK: {m.nbody} bodies {m.nu} actuators {m.neq} constraints')
except Exception as e:
    print(f'Error: {e}')
