#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import mujoco
import math

MENAGERIE_ROOT = '/home/student22/mujoco_menagerie'
MENAGERIE      = f'{MENAGERIE_ROOT}/unitree_go2'
GO2_XML        = f'{MENAGERIE}/go2.xml'
ASSETS         = f'{MENAGERIE}/assets'
OUT            = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'

# Recalibrated for bare feet (no gripper extending reach anymore), now for
# ALL FOUR legs instead of just FL. Found via IK grid search per leg pair:
# gx=0.25, gy=0.30 works for all four; gz differs by leg pair, since
# forward kinematics (verified directly, at rest pose) shows FL/RL share
# one hip height (world Z=0.4165) and FR/RR share a different, lower one
# (world Z=0.3235) -- a real ~9.3cm split baked into the robot's own
# geometry once lying on its side. GRIP_Z per pair is defined in crawl.py
# (FL/RL: 0.40, FR/RR: 0.30) -- both converge cleanly with comfortable
# joint margins at gx=0.25, gy=0.30.
LADDER_X = 1.25
LADDER_Y = 0.300

# ============================== Ladder geometry (per spec) ===================
RUNG_LENGTH   = 0.49     # m -- physical length of each rung bar
RAIL_LENGTH   = 3.00     # m -- total length of each side rail (increased from
                          #      2.50 -- N_RUNGS=11 now gives a 2.5m rung span,
                          #      which would leave zero rail margin at 2.50)
RUNG_CIRCUM   = 0.125    # m -- rung circumference (rung modeled as a cylinder)
N_RUNGS       = 11    # extended from 9: verified directly that RL/RR (back
                       # legs, hip X=-0.1934) cannot reach the old rung_01
                       # at all (0.139m IK error, calf pinned at its hard
                       # limit) -- their comfortable reach is around x=0.0,
                       # which didn't exist as a rung position before. This
                       # puts the new rung_01 exactly there, while rung_02
                       # lands at x=0.25 (the old rung_01 position, still
                       # comfortable for FL/FR).
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

def add_attr_to_geom(root_el, geom_name, attrs):
    for geom in root_el.iter('geom'):
        if geom.get('name') == geom_name:
            for k, v in attrs.items():
                geom.set(k, v)
            return True
    return False


FOOT_LOCAL_POS = '-0.002 0 -0.213'   # same for all 4 feet -- go2.xml's
                                       # "foot" default class applies this
                                       # offset identically to every leg

def add_foot_site_and_collision(robot_el, leg):
    calf = find_body(robot_el, f'{leg}_calf')
    calf.append(ET.fromstring(f'<site name="{leg.lower()}_foot_site" pos="{FOOT_LOCAL_POS}" size="0.005"/>'))
    found = add_attr_to_geom(robot_el, leg, {
        'contype': '3', 'conaffinity': '3',
        'friction': '0.9 0.05 0.5',
        'solref': '0.05 1',
        'solimp': '0.85 0.95 0.01',
    })
    return found


for leg in ['FL', 'FR', 'RL', 'RR']:
    found = add_foot_site_and_collision(robot, leg)
    print(f'{leg} foot site + collision extended: {found}')

# IMU mounting site (unrelated to the gripper removal, kept from before)
robot.append(ET.fromstring('<site name="imu_site" pos="0 0 0" size="0.005"/>'))
robot_str = ET.tostring(robot, encoding='unicode')

default_str = ET.tostring(default_el, encoding='unicode')
asset_str   = ET.tostring(asset_el, encoding='unicode')

# ---- contact: go2's own self-collision excludes, unchanged. No gripper
#      subtree to exclude anymore -- the foot is already part of go2's own
#      body tree and its self-collision behavior is whatever go2.xml
#      already specifies.
# The robot's own "collision" class (go2.xml) only sets a render group,
# not contype/conaffinity -- confirmed directly by reading it -- so every
# leg segment silently inherits MuJoCo's default collision group (1,1),
# same as the trolley chassis box below (which also doesn't set them
# explicitly). Without an exclude, a leg swinging near the chassis could
# generate real, unwanted contact forces -- exactly the kind of
# trolley-on-leg disturbance that shouldn't happen, especially now that
# all 4 legs are actively moving (not just FL, which apparently never
# swung close enough to the chassis to make this visible before).
if contact_el is None:
    contact_el = ET.Element('contact')   # go2.xml has no <contact> block at all -- confirmed directly
for leg in ['FL', 'FR', 'RL', 'RR']:
    for part in ['hip', 'thigh', 'calf']:
        ET.SubElement(contact_el, 'exclude', {'body1': 'trolley', 'body2': f'{leg}_{part}'})
ET.SubElement(contact_el, 'exclude', {'body1': 'trolley', 'body2': 'base'})

# The wheel bodies (wfl/wfr/wrl/wrr) are separate from "trolley" -- the
# excludes above don't cover them. Missed previously; matters more now
# that the wheels sit in a tighter caster cluster (radius 0.10m) closer to
# the legs' own swept region than the old wide corner layout was.
for leg in ['FL', 'FR', 'RL', 'RR']:
    for part in ['hip', 'thigh', 'calf']:
        for wheel in ['wfl', 'wfr', 'wrl', 'wrr']:
            ET.SubElement(contact_el, 'exclude', {'body1': wheel, 'body2': f'{leg}_{part}'})

contact_str = ET.tostring(contact_el, encoding='unicode')

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

# ---- Rolling-chair wheel geometry: smaller wheels, rod length recomputed
#      so they still touch the ground (world z=0), and spoke connectors
#      computed via actual trig from the rod's base to each wheel position.
TROLLEY_WORLD_Z = 0.18
WHEEL_RADIUS = 0.05                          # reduced from 0.08
WHEEL_Z = WHEEL_RADIUS - TROLLEY_WORLD_Z      # keeps wheel bottoms at world z=0
WHEEL_XY = [(0.10, 0.10), (0.10, -0.10), (-0.10, 0.10), (-0.10, -0.10)]

# Rod must span from the wheel cluster up to the hub's ACTUAL bottom
# surface, not a fixed symmetric assumption -- checked directly and found
# a real 2cm gap (hub bottom at z=0.02, old rod top at z=0.0), which is
# exactly what read as "floating." Hub is size=0.08x0.08x0.04 at pos
# z=0.06, so its bottom surface is at 0.06-0.04=0.02.
HUB_BOTTOM_Z = 0.06 - 0.04
ROD_TOP_Z = HUB_BOTTOM_Z
ROD_BOTTOM_Z = WHEEL_Z
ROD_CENTER_Z = (ROD_TOP_Z + ROD_BOTTOM_Z) / 2
ROD_HALF_LEN = (ROD_TOP_Z - ROD_BOTTOM_Z) / 2

def make_spoke(wx, wy, wz):
    length = math.hypot(wx, wy)
    mid_x, mid_y = wx / 2, wy / 2
    yaw = math.atan2(wy, wx)
    return (f'<geom type="box" size="{length/2:.4f} 0.008 0.008" '
            f'pos="{mid_x:.4f} {mid_y:.4f} {wz:.4f}" euler="0 0 {yaw:.5f}" '
            f'rgba="0.25 0.25 0.25 1" mass="0.05"/>')

wheel_spokes_str = '\n      '.join(make_spoke(wx, wy, WHEEL_Z) for wx, wy in WHEEL_XY)

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
      <!-- z=0.18: wheel bodies sit at local z=-0.10 with radius 0.08
           (cylinder, rotated so radius extends in world Z) -- wheel
           bottoms land exactly at world z=0, resting on the floor. -->
      <joint name="trolley_slide" type="slide" axis="1 0 0" damping="0.5" limited="false"/>
      <geom type="box" size="0.08 0.08 0.04" pos="0 0 0.06" rgba="0.22 0.22 0.22 1" mass="5.0"/>
      <!-- Compact central hub (0.08x0.08), centered at the same origin --
           balances the robot's mount point unchanged. Sized from a real
           sweep of all 4 legs' full joint ranges (3000 samples each):
           leg extent relative to this origin is x=[-0.41,0.41],
           y=[-0.23,0.23], z=[-0.09,0.47]. -->

      <!-- Rolling-chair layout: a single central rod drops straight down
           from the hub to a compact caster-style wheel cluster at the
           bottom, instead of wheels spread wide at the trolley's own
           corners. Rod runs from z=0 (hub) to WHEEL_Z (wheel height,
           recomputed below for the smaller wheel radius). -->
      <geom type="cylinder" size="0.035 {ROD_HALF_LEN:.4f}" pos="0 0 {ROD_CENTER_Z:.4f}" rgba="0.25 0.25 0.25 1" mass="0.4"/>

      <!-- Wheel cluster: tight radius (0.10m) around the rod's base,
           Wheels reduced from radius 0.08 to 0.05 -- smaller, and now
           explicitly connected to the rod via 4 spoke geoms (below)
           instead of just floating near it with no visible connection. -->
{wheel_spokes_str}
      <body name="wfl" pos=" 0.10  0.10 {WHEEL_Z:.4f}"><joint name="wfl_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="{WHEEL_RADIUS} 0.02" euler="1.5708 0 0" mass="0.3" rgba="0.10 0.10 0.10 1" friction="0.005 0.001 0.001"/></body>
      <body name="wfr" pos=" 0.10 -0.10 {WHEEL_Z:.4f}"><joint name="wfr_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="{WHEEL_RADIUS} 0.02" euler="1.5708 0 0" mass="0.3" rgba="0.10 0.10 0.10 1" friction="0.005 0.001 0.001"/></body>
      <body name="wrl" pos="-0.10  0.10 {WHEEL_Z:.4f}"><joint name="wrl_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="{WHEEL_RADIUS} 0.02" euler="1.5708 0 0" mass="0.3" rgba="0.10 0.10 0.10 1" friction="0.005 0.001 0.001"/></body>
      <body name="wrr" pos="-0.10 -0.10 {WHEEL_Z:.4f}"><joint name="wrr_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="{WHEEL_RADIUS} 0.02" euler="1.5708 0 0" mass="0.3" rgba="0.10 0.10 0.10 1" friction="0.005 0.001 0.001"/></body>
      {robot_str}
    </body>
  </worldbody>

  {act_str}

  <sensor>
    <jointpos name="trolley_x"   joint="trolley_slide"/>
    <framepos name="fl_foot_pos" objtype="site" objname="fl_foot_site"/>
    <framepos name="fr_foot_pos" objtype="site" objname="fr_foot_site"/>
    <framepos name="rl_foot_pos" objtype="site" objname="rl_foot_site"/>
    <framepos name="rr_foot_pos" objtype="site" objname="rr_foot_site"/>

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
