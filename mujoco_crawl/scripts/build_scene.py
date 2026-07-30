#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import mujoco

MENAGERIE = '/home/student22/mujoco_menagerie/unitree_go2'
GO2_XML   = f'{MENAGERIE}/go2.xml'
ASSETS    = f'{MENAGERIE}/assets'
OUT       = '/home/student22/mujoco_crawl/models/go2_real_scene.xml'

# From diagnostic: gc_lp pivot at (0.565, 0.300, 0.410)
# rung_01 world X = LADDER_X - 1.05  ->  set = 0.569  ->  LADDER_X = 1.619
# rung_01 world Y = LADDER_Y          ->  set = 0.300
LADDER_X = 1.619
LADDER_Y = 0.300

tree = ET.parse(GO2_XML)
root = tree.getroot()

def get_block(tag):
    el = root.find(tag)
    return ET.tostring(el, encoding='unicode') if el is not None else ''

default_str = get_block('default')
asset_str   = get_block('asset')
act_str     = get_block('actuator')
contact_str = get_block('contact')

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

# IMPORTANT: contype="0" conaffinity="0" on ALL gripper geoms
# -> purely visual, zero collision, zero explosion risk
# The equality constraint handles the actual grip physics.
#
# FIX: added <site name="gc_grip_site"/> on gc_palm -- this is the point
# IK/grasp code targets and anchors to, instead of guessing an offset from
# FL_calf's own origin.
gripper_xml = (
    '<body name="gc_palm" pos="0 0 -0.213">'
    '<site name="gc_grip_site" pos="0 0 0" size="0.006" rgba="1 0 0 1"/>'
    '<geom type="box" size="0.055 0.015 0.015" pos="0 0 -0.015"'
    ' rgba="0.08 0.08 0.08 1" mass="0.001"'
    ' contype="0" conaffinity="0"/>'

    '<body name="gc_lp" pos="-0.040 0 -0.02">'
    '<joint name="gc_lp_joint" type="hinge" axis="0 1 0"'
    ' range="0 1.2217" damping="2.0" armature="0.001"/>'
    '<geom type="box" size="0.010 0.009 0.120" pos="0 0 -0.060"'
    ' rgba="0.2 0.2 0.2 1" mass="0.001"'
    ' contype="0" conaffinity="0"/>'
    '<body name="gc_ld" pos="0 0 -0.120">'
    '<geom type="box" size="0.009 0.007 0.090" pos="0 0 -0.045"'
    ' rgba="0.1 0.1 0.1 1" mass="0.001"'
    ' contype="0" conaffinity="0"/>'
    '</body></body>'

    '<body name="gc_rp" pos="0.040 0 -0.02">'
    '<joint name="gc_rp_joint" type="hinge" axis="0 -1 0"'
    ' range="0 1.2217" damping="2.0" armature="0.001"/>'
    '<geom type="box" size="0.010 0.009 0.120" pos="0 0 -0.060"'
    ' rgba="0.2 0.2 0.2 1" mass="0.001"'
    ' contype="0" conaffinity="0"/>'
    '<body name="gc_rd" pos="0 0 -0.120">'
    '<geom type="box" size="0.009 0.007 0.090" pos="0 0 -0.045"'
    ' rgba="0.1 0.1 0.1 1" mass="0.001"'
    ' contype="0" conaffinity="0"/>'
    '</body></body>'
    '</body>'
)

fl_calf.append(ET.fromstring(gripper_xml))
robot_str = ET.tostring(robot, encoding='unicode')

act_str = act_str.replace(
    '</actuator>',
    '  <position name="gc_lp" joint="gc_lp_joint" kp="30" kv="3" ctrlrange="0 1.2217"/>\n'
    '  <position name="gc_rp" joint="gc_rp_joint" kp="30" kv="3" ctrlrange="0 1.2217"/>\n'
    '</actuator>'
)

r = 0.028

scene = f"""<mujoco model="go2_real_ladder_crawl">
  <compiler angle="radian" meshdir="{ASSETS}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -0.001"
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
      <geom name="rung_01" type="box" size="{r} {r} 0.440" pos="-1.05 0 0.42" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <geom name="rung_02" type="box" size="{r} {r} 0.440" pos="-0.75 0 0.42" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <geom name="rung_03" type="box" size="{r} {r} 0.440" pos="-0.45 0 0.42" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <geom name="rung_04" type="box" size="{r} {r} 0.440" pos="-0.15 0 0.42" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <geom name="rung_05" type="box" size="{r} {r} 0.440" pos=" 0.15 0 0.42" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <geom name="rung_06" type="box" size="{r} {r} 0.440" pos=" 0.45 0 0.42" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <geom name="rung_07" type="box" size="{r} {r} 0.440" pos=" 0.75 0 0.42" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
      <geom name="rung_08" type="box" size="{r} {r} 0.440" pos=" 1.05 0 0.42" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>
    </body>

    <!-- FIX: trolley now actually has a joint! Previously this body had no
         joint element at all, which rigidly welds it to the world in MuJoCo
         -- it could never move, no matter what crawl.py did. Axis is X
         because the rungs are spaced along local/world X inside "ladder". -->
    <body name="trolley" pos="0 0 0.12">
      <joint name="trolley_slide" type="slide" axis="1 0 0" damping="0.5" limited="false"/>
      <geom type="box" size="0.35 0.25 0.02" rgba="0.22 0.22 0.22 1" mass="5.0"/>
      <body name="wfl" pos=" 0.27  0.26 -0.10"><joint name="wfl_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.08 0.025" euler="1.5708 0 0" mass="0.5" rgba="0.10 0.10 0.10 1" friction="0.001 0.001 0.001"/></body>
      <body name="wfr" pos=" 0.27 -0.26 -0.10"><joint name="wfr_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.08 0.025" euler="1.5708 0 0" mass="0.5" rgba="0.10 0.10 0.10 1" friction="0.001 0.001 0.001"/></body>
      <body name="wrl" pos="-0.27  0.26 -0.10"><joint name="wrl_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.08 0.025" euler="1.5708 0 0" mass="0.5" rgba="0.10 0.10 0.10 1" friction="0.001 0.001 0.001"/></body>
      <body name="wrr" pos="-0.27 -0.26 -0.10"><joint name="wrr_j" type="hinge" axis="0 1 0" damping="0.001" limited="false"/><geom type="cylinder" size="0.08 0.025" euler="1.5708 0 0" mass="0.5" rgba="0.10 0.10 0.10 1" friction="0.001 0.001 0.001"/></body>
      {robot_str}
    </body>
  </worldbody>

  {act_str}

  <!-- FIX: was <weld body1="FL_calf" body2="ladder" .../>. A weld/connect's
       anchor points are baked in at COMPILE time from qpos0 -- activating it
       later does NOT grab "wherever the gripper currently is", it snaps
       toward whatever relative pose existed when this file was built. That's
       why the grip never held. The real fix has to happen at runtime (see
       crawl.py's grasp()/release()), which recomputes eq_data every time it
       grabs. Also switched weld -> connect (ball joint) so the wrist/leg can
       still rotate around the grip point while the leg swings -- a weld
       would rigidly lock orientation too and fight the swing motion.
       body1 changed from FL_calf to gc_palm so the anchor is the actual
       fingertip, not an arbitrary point on the calf. -->
  <equality>
    <connect name="fl_grip" body1="gc_palm" body2="ladder"
             anchor="0 0 0" active="false" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </equality>

  <sensor>
    <jointpos name="trolley_x"   joint="trolley_slide"/>
    <framepos name="fl_calf_pos" objtype="body" objname="FL_calf"/>
    <framepos name="gc_palm_pos" objtype="site" objname="gc_grip_site"/>
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
