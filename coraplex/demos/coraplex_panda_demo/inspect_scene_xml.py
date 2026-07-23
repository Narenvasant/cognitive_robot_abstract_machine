"""
Inspect /tmp/scene.xml (the file MujocoBuilder actually rebuilds and MuJoCo
simulates) -- prints the auto-generated "home" keyframe's qpos list and
cube0's <body> block, so we can check whether qpos values look scrambled
relative to what each joint/DOF should hold.

Run directly: python3 inspect_scene_xml.py
"""

import xml.etree.ElementTree as ET

SCENE_XML = "/tmp/scene.xml"

tree = ET.parse(SCENE_XML)
root = tree.getroot()

print("=== joints, in document order (this is roughly qpos order for hinge/slide joints) ===")
for i, joint in enumerate(root.iter("joint")):
    print(f"  [{i}] name={joint.get('name')!r} type={joint.get('type', 'hinge')} "
          f"pos={joint.get('pos')} axis={joint.get('axis')}")

print()
print("=== freejoints specifically (each contributes 7 qpos values: x y z qw qx qy qz) ===")
for i, joint in enumerate(root.iter("freejoint")):
    print(f"  [{i}] name={joint.get('name')!r}")

print()
print("=== home keyframe qpos (raw) ===")
for key in root.iter("key"):
    if key.get("name") == "home":
        qpos = key.get("qpos", "").split()
        print(f"  total qpos values: {len(qpos)}")
        print(f"  qpos: {qpos}")

print()
print('=== cube0 <body> block, full content ===')
for body in root.iter("body"):
    if body.get("name") == "cube0":
        print(ET.tostring(body, encoding="unicode"))

print()
print('=== /hand <body> block, full content (for comparison) ===')
for body in root.iter("body"):
    if body.get("name") == "/hand":
        print(ET.tostring(body, encoding="unicode"))