#!/usr/bin/env python3
import re
import xml.etree.ElementTree as ET
import mujoco

MENAGERIE_ROOT = '/home/student22/mujoco_menagerie'
MENAGERIE      = f'{MENAGERIE_ROOT}/unitree_go2'
GO2_XML        = f'{MENAGERIE}/go2.xml'
ASSETS         = f'{MENAGERIE}/assets'
OUT            = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'

GRIPPER_XML    = f'{MENAGERIE_ROOT}/robotiq_2f85/2f85.xml'
GRIPPER_ASSETS = f'{MENAGERIE_ROOT}/robotiq_2f85/assets'

# From diagnostic: gc_lp pivot at (0.565, 0.300, 0.410)
# rung_01 world X = LADDER_X - 1.05  ->  set = 0.569  ->  LADDER_X = 1.619
# rung_01 world Y = LADDER_Y          ->  set = 0.300
LADDER_X = 1.619
LADDER_Y = 0.300

# ============================== Go2 body =====================================
tree = ET.parse(GO2_XML)
root = tree.getroot()

def get_block(el_root, tag):
    el = el_root.find(tag)
    return el

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

# ==================== Robotiq 2F-85 gripper (real hardware model) ============
# We attach the actual Robotiq 2F-85 from mujoco_menagerie instead of a
# hand-built placeholder gripper. Its default classes, materials, and one
# body name ("base") collide with go2's own names, so everything gets
# namespaced with a "g2f_" prefix before merging. Mesh files are pointed at
# via absolute paths so we don't have to touch the student's menagerie clone.
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

# "base" collides with go2's own root body -> rename to "gripper_base"
# everywhere it's used as a body name or an exclude/equality reference.
grip_raw = grip_raw.replace('<body name="base" ', '<body name="gripper_base" ')
grip_raw = re.sub(r'(body1|body2)="base"', r'\1="gripper_base"', grip_raw)

# Meshes: point at absolute paths in the gripper's own assets/ folder so we
# don't need a second <compiler meshdir=.../> or to copy files anywhere.
grip_raw = re.sub(r'file="([^"]+\.stl)"', rf'file="{GRIPPER_ASSETS}/\1"', grip_raw)

# The 4 real collision geoms on the fingertips (right_pad1/2, left_pad1/2 --
# these already carry realistic hardware friction/solref/solimp from the
# real Robotiq model) need conaffinity extended to bit 1 (value 2) so they
# can actually contact the rungs, which live in that isolated group so
# nothing else in the scene (torso, trolley, floor) collides with them.
for pad_name in ('right_pad1', 'right_pad2', 'left_pad1', 'left_pad2'):
    grip_raw = grip_raw.replace(f'name="{pad_name}"/>', f'name="{pad_name}" contype="3" conaffinity="3"/>')

grip_root = ET.fromstring(grip_raw)

grip_default_el  = grip_root.find('default')
grip_asset_el    = grip_root.find('asset')
grip_tendon_el   = grip_root.find('tendon')
grip_equality_el = grip_root.find('equality')
grip_actuator_el = grip_root.find('actuator')
grip_contact_el  = grip_root.find('contact')

grip_worldbody = grip_root.find('worldbody')
gripper_body = [b for b in grip_worldbody if b.tag == 'body'][0]  # "base_mount"

# Attach at the FL foot location (same offset the old custom gripper used).
# Orientation found by empirical search over candidate mounting quaternions
# (identity/90/180 about each axis), checking which one lets the gripper's
# built-in "pinch" site actually reach the calibrated rung target via IK.
# quat="0 1 0 0" (180 deg about X) was the only clean solution: 0 residual
# error, all three joint angles comfortably inside their limits.
gripper_body.set('pos', '0 0 -0.213')
gripper_body.set('quat', '0 1 0 0')

fl_calf.append(gripper_body)
robot_str = ET.tostring(robot, encoding='unicode')

# ---- merge <default>: go2's is <default><default class="go2">...</default></default>,
#      2f85's is the same shape with class="g2f_2f85" -- merge the two named
#      children under one bare wrapper.
combined_default = ET.Element('default')
for child in list(default_el):
    combined_default.append(child)
for child in list(grip_default_el):
    combined_default.append(child)
default_str = ET.tostring(combined_default, encoding='unicode')

# ---- merge <asset>
combined_asset = ET.Element('asset')
for child in list(asset_el):
    combined_asset.append(child)
for child in list(grip_asset_el):
    combined_asset.append(child)
asset_str = ET.tostring(combined_asset, encoding='unicode')

# ---- merge <contact>: go2's self-collision excludes + 2f85's own +
#      new excludes so the newly-attached gripper doesn't produce "ghost"
#      contacts against the leg it's mounted on or the trolley deck as it
#      swings. Previously only gripper_base/base_mount were excluded --
#      the finger linkage itself (driver/coupler/spring_link/follower/pad,
#      left AND right) was NOT, and those parts sweep much closer to
#      FL_thigh/FL_calf and the trolley chassis during the pull/swing
#      motion, which is what was producing spurious contact forces.
def _collect_body_names(body_el):
    names = [body_el.get('name')]
    for ch in body_el.findall('body'):
        names.extend(_collect_body_names(ch))
    return names

gripper_body_names = _collect_body_names(gripper_body)  # all 14 bodies, both fingers

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

# ---- actuators: go2's own leg motors + the gripper's single tendon-driven
#      finger actuator (replaces the old hand-built gc_lp/gc_rp actuators)
combined_actuator = ET.Element('actuator')
for child in list(act_el):
    combined_actuator.append(child)
for child in list(grip_actuator_el):
    combined_actuator.append(child)
act_str = ET.tostring(combined_actuator, encoding='unicode')

# ---- tendon: only the gripper defines one (couples its two finger joints)
tendon_str = ET.tostring(grip_tendon_el, encoding='unicode')

# ---- equality: only the gripper's own finger-coupling constraints remain.
#      Grasping is now real contact + friction (see crawl.py's
#      close_until_gripped), not a constraint hack, so there's no
#      "fl_grip" connect here anymore.
combined_equality = ET.Element('equality')
for child in list(grip_equality_el):
    combined_equality.append(child)
equality_str = ET.tostring(combined_equality, encoding='unicode')

r = 0.028

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
      <geom type="box" size="1.5 0.007 0.007" pos="0 0 0.007"
            rgba="0.20 0.20 0.20 1" contype="0" conaffinity="0"/>
      <geom type="box" size="1.5 0.007 0.007" pos="0 0 0.893"
            rgba="0.20 0.20 0.20 1" contype="0" conaffinity="0"/>
      <geom name="rung_01" type="box" size="{r} {r} 0.436" pos="-1.05 0 0.45" rgba="0.12 0.12 0.12 1" contype="2" conaffinity="2" friction="0.9 0.02 0.002"/>
      <geom name="rung_02" type="box" size="{r} {r} 0.436" pos="-0.75 0 0.45" rgba="0.12 0.12 0.12 1" contype="2" conaffinity="2" friction="0.9 0.02 0.002"/>
      <geom name="rung_03" type="box" size="{r} {r} 0.436" pos="-0.45 0 0.45" rgba="0.12 0.12 0.12 1" contype="2" conaffinity="2" friction="0.9 0.02 0.002"/>
      <geom name="rung_04" type="box" size="{r} {r} 0.436" pos="-0.15 0 0.45" rgba="0.12 0.12 0.12 1" contype="2" conaffinity="2" friction="0.9 0.02 0.002"/>
      <geom name="rung_05" type="box" size="{r} {r} 0.436" pos=" 0.15 0 0.45" rgba="0.12 0.12 0.12 1" contype="2" conaffinity="2" friction="0.9 0.02 0.002"/>
      <geom name="rung_06" type="box" size="{r} {r} 0.436" pos=" 0.45 0 0.45" rgba="0.12 0.12 0.12 1" contype="2" conaffinity="2" friction="0.9 0.02 0.002"/>
      <geom name="rung_07" type="box" size="{r} {r} 0.436" pos=" 0.75 0 0.45" rgba="0.12 0.12 0.12 1" contype="2" conaffinity="2" friction="0.9 0.02 0.002"/>
      <geom name="rung_08" type="box" size="{r} {r} 0.436" pos=" 1.05 0 0.45" rgba="0.12 0.12 0.12 1" contype="2" conaffinity="2" friction="0.9 0.02 0.002"/>
    </body>

    <body name="trolley" pos="0 0 0.12">
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

  {tendon_str}

  <!-- Grasping is real contact + friction now (crawl.py's close_until_gripped),
       not a constraint hack -- this <equality> block only has the Robotiq
       2F-85's own built-in finger-coupling constraints. -->
  {equality_str}

  <sensor>
    <jointpos name="trolley_x"   joint="trolley_slide"/>
    <framepos name="fl_calf_pos" objtype="body" objname="FL_calf"/>
    <framepos name="gc_palm_pos" objtype="site" objname="pinch"/>
  </sensor>
</mujoco>"""

with open(OUT, 'w') as f:
    f.write(scene)
print(f'Written: {OUT}')
print(f'rung_01 world X={LADDER_X-1.05:.3f}  Y={LADDER_Y:.3f}')

try:
    m = mujoco.MjModel.from_xml_path(OUT)
    print(f'Model OK: {m.nbody} bodies {m.nu} actuators {m.neq} constraints')
except Exception as e:
    print(f'Error: {e}')
