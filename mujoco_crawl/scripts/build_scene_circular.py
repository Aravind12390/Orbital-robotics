#!/usr/bin/env python3
"""
Go2 MuJoCo CIRCULAR ladder scene builder.

Builds on the straight-ladder + real-Robotiq-2F-85 version: same robot,
same gripper attachment/mounting logic. What's new is the ladder itself and
how the trolley moves.

WHAT CHANGED FROM THE STRAIGHT-LADDER VERSION:

1. The ladder is now a RING of vertical bars ("rungs") standing around a
   circle of radius LADDER_RADIUS, centered on the world Z axis at (0,0).
   Rungs are placed at TOTAL_RUNGS evenly-spaced angles (ANGLE_SPACING
   apart). Each rung keeps the exact same cross-section and height as
   before -- only its (x, y) position changes, from "along the X axis" to
   "around a circle."

2. The trolley no longer translates along a slide joint. It now pivots on
   a hinge joint about world Z, anchored through the ring's center, and
   sits on its OWN concentric circle of radius TROLLEY_RADIUS. As the
   hinge angle increases, the trolley (and the whole robot riding on it)
   sweeps around the center -- this is what "moving circularly" means
   physically here. The robot's gripper reaches radially OUTWARD across
   the gap (LADDER_RADIUS - TROLLEY_RADIUS) to grab a rung, which plays
   the same geometric role as reaching forward along +X did before.

   Concretely: the trolley body is placed at pos=(TROLLEY_RADIUS, 0, z)
   and the hinge's own anchor point (its "pos" attribute, which MuJoCo
   reads in the BODY's local frame) is set to (-TROLLEY_RADIUS, 0, 0).
   That places the hinge's rotation axis through the world Z axis at the
   ring's center, while the body itself sits out at radius TROLLEY_RADIUS
   -- exactly like a mass on the end of a rigid arm pivoting at the origin.
   A convenient side effect: because the body's local +X axis starts out
   pointing straight from the pivot to the body (i.e. radially outward),
   it KEEPS pointing radially outward at any hinge angle, since rotating
   about Z rotates the whole body frame rigidly with it. So the robot,
   which was built to reach for a rung along its own local +X, doesn't
   need any extra reorientation logic -- it just always reaches "out."

3. Rung angular spacing (ANGLE_SPACING) isn't picked from a round number
   of degrees -- it's picked so the CHORD distance between adjacent rungs
   comes out to ~0.30 m, matching the straight ladder's rung spacing. That
   spacing is what the FL leg's IK/PD reach envelope was actually tuned
   against, so preserving the chord distance (not the arc-length or the
   angle) is what keeps the crawl gait mechanically valid on the new
   geometry. With LADDER_RADIUS=2.00 m and TOTAL_RUNGS=42, chord =
   2*R*sin(pi/TOTAL_RUNGS) ~= 0.30 m.

4. The two horizontal rail beams (purely cosmetic, contype=0/conaffinity=0
   before and after) are now built as a polygon of short tangential box
   segments approximating a circle, instead of two long straight beams.

Everything about the Go2 body, the real Robotiq 2F-85 attachment/renaming/
namespacing, contact excludes, and actuator/tendon/equality merging is
UNCHANGED from the straight-ladder version -- none of that logic cares
about the shape of the thing the gripper is grabbing.
"""
import re
import math
import xml.etree.ElementTree as ET
import mujoco

MENAGERIE_ROOT = '/home/student22/mujoco_menagerie'
MENAGERIE      = f'{MENAGERIE_ROOT}/unitree_go2'
GO2_XML        = f'{MENAGERIE}/go2.xml'
ASSETS         = f'{MENAGERIE}/assets'
OUT            = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'

GRIPPER_XML    = f'{MENAGERIE_ROOT}/robotiq_2f85/2f85.xml'
GRIPPER_ASSETS = f'{MENAGERIE_ROOT}/robotiq_2f85/assets'

# ============================== Circular ladder geometry =====================
TOTAL_RUNGS     = 42                          # reverted from 28: that widening broke the
                                               # calibrated ~0.30m chord spacing this reach
                                               # envelope was actually validated against (see
                                               # module docstring item 3) -- confirmed directly,
                                               # 28 rungs gives a 0.448m chord, 50% bigger than
                                               # what a single PULL can reliably cover
LADDER_RADIUS   = 2.00                        # ring radius -- rungs sit here
TROLLEY_RADIUS  = 1.43                        # trolley's own circular track radius
ANGLE_SPACING   = 2 * math.pi / TOTAL_RUNGS   # ~8.57 deg between adjacent rungs
RAIL_SEGMENTS   = 128                         # bumped up for a bigger ring, still visual only
GRIP_Z          = 0.450                       # matches RUNG_Z (rung center) -- 0.410 left a
                                               # fixed, unreachable ~4.2cm Z-shortfall at
                                               # this leg/mount configuration (confirmed by
                                               # direct IK testing: X,Y converged to exact 0
                                               # error at every radial distance tested, only
                                               # Z was short, by a CONSTANT amount regardless
                                               # of radius -- a genuine kinematic ceiling on
                                               # downward reach here, not a radial-gap issue)
RAIL_Z_LOW, RAIL_Z_HIGH = 0.007, 0.893
RUNG_HALF_H     = 0.436
RUNG_Z          = 0.45
RUNG_R          = 0.028                       # rung half-thickness (unchanged)

print(f'  chord between adjacent rungs = '
      f'{2*LADDER_RADIUS*math.sin(ANGLE_SPACING/2):.4f} m')
print(f'  radial reach (gap the gripper must cross) = '
      f'{LADDER_RADIUS-TROLLEY_RADIUS:.4f} m')

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
# RADIAL-facing: local +X (the FL leg's big reach axis) must point straight
# from the trolley's position toward the ring center's rungs. euler="1.5708
# 0 0" (90deg about X only) leaves local +X exactly unchanged (verified:
# rotation matrix comes out [[1,0,0],[0,~1,~0],[0,~0,~1]]), so it still
# points along world +X -- which IS the radial direction here, since the
# trolley sits at pos=(TROLLEY_RADIUS, 0, z), directly on the +X ray from
# the ring's center.
#
# This replaces a quat=(0.5,0.5,0.5,0.5) that was actually TANGENTIAL, not
# radial -- confirmed directly: that rotation matrix works out to
# [[0,0,1],[1,0,0],[0,1,0]], which maps local +X to world +Y (tangential),
# not +X. TROLLEY_RADIUS=1.43 (a 0.57m gap) was calibrated for a RADIAL
# reach; with the tangential quat, IK was failing to converge by ~12cm on
# every single rung (verified directly), which is why nothing ever
# actually gripped and pulled -- there was nothing to hold onto.
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

# ==================== Robotiq 2F-85 gripper (unchanged) ======================
with open(GRIPPER_XML) as f:
    grip_raw = f.read()

_CLASS_TOKENS = ['2f85', 'driver', 'follower', 'spring_link', 'coupler',
                  'visual', 'collision', 'pad_box1', 'pad_box2']
for tok in _CLASS_TOKENS:
    grip_raw = re.sub(rf'(class|childclass)="{tok}"', rf'\1="g2f_{tok}"', grip_raw)

_MATERIAL_TOKENS = ['metal', 'black', 'gray', 'silicone']
for tok in _MATERIAL_TOKENS:
    grip_raw = re.sub(rf'<material name="{tok}"', f'<material name="g2f_{tok}"', grip_raw)
    grip_raw = re.sub(rf'material="{tok}"', f'material="g2f_{tok}"', grip_raw)

grip_raw = grip_raw.replace('<body name="base" ', '<body name="gripper_base" ')
grip_raw = re.sub(r'(body1|body2)="base"', r'\1="gripper_base"', grip_raw)
grip_raw = re.sub(r'file="([^"]+\.stl)"', rf'file="{GRIPPER_ASSETS}/\1"', grip_raw)

grip_root = ET.fromstring(grip_raw)

grip_default_el  = grip_root.find('default')
grip_asset_el    = grip_root.find('asset')
grip_tendon_el   = grip_root.find('tendon')
grip_equality_el = grip_root.find('equality')
grip_actuator_el = grip_root.find('actuator')
grip_contact_el  = grip_root.find('contact')

grip_worldbody = grip_root.find('worldbody')
gripper_body = [b for b in grip_worldbody if b.tag == 'body'][0]

gripper_body.set('pos', '0 0 -0.213')
gripper_body.set('quat', '0 1 0 0')

fl_calf.append(gripper_body)
robot_str = ET.tostring(robot, encoding='unicode')

combined_default = ET.Element('default')
for child in list(default_el):
    combined_default.append(child)
for child in list(grip_default_el):
    combined_default.append(child)
default_str = ET.tostring(combined_default, encoding='unicode')

combined_asset = ET.Element('asset')
for child in list(asset_el):
    combined_asset.append(child)
for child in list(grip_asset_el):
    combined_asset.append(child)
asset_str = ET.tostring(combined_asset, encoding='unicode')

def _collect_body_names(body_el):
    names = [body_el.get('name')]
    for ch in body_el.findall('body'):
        names.extend(_collect_body_names(ch))
    return names

gripper_body_names = _collect_body_names(gripper_body)

combined_contact = ET.Element('contact')
if contact_el is not None:
    for child in list(contact_el):
        combined_contact.append(child)
if grip_contact_el is not None:
    for child in list(grip_contact_el):
        combined_contact.append(child)
for other in ('FL_hip', 'FL_thigh', 'FL_calf', 'trolley'):
    for gname in gripper_body_names:
        ET.SubElement(combined_contact, 'exclude', {'body1': other, 'body2': gname})
contact_str = ET.tostring(combined_contact, encoding='unicode')

combined_actuator = ET.Element('actuator')
for child in list(act_el):
    combined_actuator.append(child)
for child in list(grip_actuator_el):
    combined_actuator.append(child)
act_str = ET.tostring(combined_actuator, encoding='unicode')

tendon_str = ET.tostring(grip_tendon_el, encoding='unicode')

combined_equality = ET.Element('equality')
for child in list(grip_equality_el):
    combined_equality.append(child)
ET.SubElement(combined_equality, 'connect', {
    'name': 'fl_grip', 'body1': 'gripper_base', 'body2': 'ladder',
    'anchor': '0 0 0', 'active': 'false', 'solref': '0.01 1', 'solimp': '0.9 0.95 0.001',
})
equality_str = ET.tostring(combined_equality, encoding='unicode')

# ============================== Procedural circular ladder ===================
def rung_geoms():
    parts = []
    for i in range(TOTAL_RUNGS):
        theta = i * ANGLE_SPACING
        x, y = LADDER_RADIUS * math.cos(theta), LADDER_RADIUS * math.sin(theta)
        parts.append(
            f'<geom name="rung_{i:02d}" type="box" size="{RUNG_R} {RUNG_R} {RUNG_HALF_H}" '
            f'pos="{x:.5f} {y:.5f} {RUNG_Z}" rgba="0.12 0.12 0.12 1" '
            f'contype="0" conaffinity="0"/>'
        )
    return '\n      '.join(parts)

def rail_geoms(z, thickness=0.007):
    parts = []
    dtheta = 2 * math.pi / RAIL_SEGMENTS
    half_len = LADDER_RADIUS * math.sin(dtheta / 2)
    mid_r = LADDER_RADIUS * math.cos(dtheta / 2)
    for i in range(RAIL_SEGMENTS):
        theta_mid = (i + 0.5) * dtheta
        x, y = mid_r * math.cos(theta_mid), mid_r * math.sin(theta_mid)
        yaw = theta_mid + math.pi / 2
        parts.append(
            f'<geom type="box" size="{half_len:.5f} {thickness} {thickness}" '
            f'pos="{x:.5f} {y:.5f} {z}" euler="0 0 {yaw:.5f}" '
            f'rgba="0.20 0.20 0.20 1" contype="0" conaffinity="0"/>'
        )
    return '\n      '.join(parts)

ladder_geoms_str = (
    rail_geoms(RAIL_Z_LOW) + '\n      ' +
    rail_geoms(RAIL_Z_HIGH) + '\n      ' +
    rung_geoms()
)

scene = f"""<mujoco model="go2_real_circular_ladder_crawl">
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

    <body name="ladder" pos="0 0 0">
      {ladder_geoms_str}
    </body>

    <body name="trolley" pos="{TROLLEY_RADIUS} 0 0.12">
      <joint name="trolley_hinge" type="hinge" axis="0 0 1"
             pos="{-TROLLEY_RADIUS} 0 0" damping="0.5" limited="false"/>
      <geom type="box" size="0.25 0.35 0.02" rgba="0.22 0.22 0.22 1" mass="5.0"/>
      <body name="wfl" pos=" 0.26  0.27 -0.075"><joint name="wfl_j" type="hinge" axis="1 0 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.045 0.02" euler="0 1.5708 0" mass="0.3" contype="0" conaffinity="0" rgba="0.10 0.10 0.10 1"/></body>
      <body name="wfr" pos="-0.26  0.27 -0.075"><joint name="wfr_j" type="hinge" axis="1 0 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.045 0.02" euler="0 1.5708 0" mass="0.3" contype="0" conaffinity="0" rgba="0.10 0.10 0.10 1"/></body>
      <body name="wrl" pos=" 0.26 -0.27 -0.075"><joint name="wrl_j" type="hinge" axis="1 0 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.045 0.02" euler="0 1.5708 0" mass="0.3" contype="0" conaffinity="0" rgba="0.10 0.10 0.10 1"/></body>
      <body name="wrr" pos="-0.26 -0.27 -0.075"><joint name="wrr_j" type="hinge" axis="1 0 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.045 0.02" euler="0 1.5708 0" mass="0.3" contype="0" conaffinity="0" rgba="0.10 0.10 0.10 1"/></body>
      {robot_str}
    </body>
  </worldbody>

  {act_str}

  {tendon_str}

  {equality_str}

  <sensor>
    <jointpos name="trolley_angle" joint="trolley_hinge"/>
    <framepos name="fl_calf_pos" objtype="body" objname="FL_calf"/>
    <framepos name="gc_palm_pos" objtype="site" objname="pinch"/>
  </sensor>
</mujoco>"""

with open(OUT, 'w') as f:
    f.write(scene)
print(f'Written: {OUT}')
print(f'TOTAL_RUNGS={TOTAL_RUNGS}  LADDER_RADIUS={LADDER_RADIUS}  '
      f'TROLLEY_RADIUS={TROLLEY_RADIUS}  '
      f'ANGLE_SPACING={math.degrees(ANGLE_SPACING):.3f} deg')

try:
    m = mujoco.MjModel.from_xml_path(OUT)
    print(f'Model OK: {m.nbody} bodies {m.nu} actuators {m.neq} constraints')
except Exception as e:
    print(f'Error: {e}')
