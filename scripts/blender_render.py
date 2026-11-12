"""
[연구실] Blender 로 원본 / 생성 / SLERP BVH 렌더 (EEVEE, mp4)
=========================================================

scripts/export_bvh.py 가 만든 클립 폴더 (gt.bvh, pred.bvh, slerp.bvh, meta.json) 를 읽어
세 스켈레톤을 한 장면에 놓고 30fps mp4 로 렌더한다.

    blender --background --python scripts/blender_render.py -- <클립폴더> [옵션]

옵션 (-- 뒤에):
    --out <파일.mp4>       기본 <클립폴더>/render.mp4
    --width 1280 --height 720
    --samples 16           EEVEE 렌더 샘플 수
    --frame_end N          앞에서 N 프레임만 (빠른 확인용)
    --still N              N 번 프레임 (0부터) 한 장만 PNG 로 (<out 이름>_fNNN.png)

장면
----
- BVH import: scale 0.01 (cm → m), Y-up → Blender Z-up, 회전 모드는 BVH 순서 그대로 (NATIVE).
- 스켈레톤 모양: 관절 사이 원기둥 + 관절 구. 메시 하나에 뼈별 vertex group + Armature modifier
  (뼈에 단단히 붙어 움직임). 원본=파랑, 생성=빨강, SLERP=회색 (render_compare.py 와 같은 색).
- 각 스켈레톤 머리 위 텍스트 라벨 (원본 / 생성 / SLERP), 항상 카메라를 향함.
- 화면 왼쪽 위 구간 표시 (원본 재생 / context (입력) / transition (생성) / target), 오른쪽 위 프레임 번호.
- 바닥: 체크무늬 평면. 카메라: 고정 직교 카메라, 모든 프레임의 세 스켈레톤이 다 들어가게.
  보는 방향은 meta.json 의 진행 방향/옆 방향으로 정한다 (render_compare.py 의 비스듬한 시점과 비슷).
- 조명: 태양광 + 밝은 배경 (환경광), 그림자, AO.

한글 글꼴은 fc-match 로 Noto Sans CJK / Nanum 을 찾고, 없으면 영어 라벨을 쓴다.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

TRACKS = [  # (파일 키, 한글 라벨, 영어 라벨, sRGB 색)
    ("gt", "원본", "original", (31, 119, 180)),
    ("pred", "생성", "generated", (214, 39, 40)),
    ("slerp", "SLERP", "SLERP", (128, 128, 128)),
]
PHASE_KO = {"original": "원본 재생", "context": "context (입력)", "transition": "transition (생성)",
            "target": "target"}
PHASE_EN = {"original": "original", "context": "context (input)", "transition": "transition (generated)",
            "target": "target"}
BONE_RADIUS = 0.022          # m
JOINT_RADIUS = 0.032
HEAD_RADIUS = 0.09


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="blender --background --python scripts/blender_render.py --")
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--frame_end", type=int, default=None)
    ap.add_argument("--still", type=int, default=None)
    return ap.parse_args(argv)


def srgb_to_linear(c: tuple[int, int, int]) -> tuple[float, float, float, float]:
    def f(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return (f(c[0]), f(c[1]), f(c[2]), 1.0)


def bvh_to_blender(v) -> Vector:
    """BVH 월드 (Y 위) → Blender (Z 위). 가져오기 옵션 axis_forward='-Z', axis_up='Y' 와 같은 변환."""
    return Vector((v[0], -v[2], v[1]))


# -----------------------------------------------------------------------------
# 재질 / 글꼴
# -----------------------------------------------------------------------------
def make_material(name: str, rgba, roughness: float = 0.45, emission: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    if emission > 0:
        key = "Emission Color" if "Emission Color" in bsdf.inputs else "Emission"
        bsdf.inputs[key].default_value = rgba
        bsdf.inputs["Emission Strength"].default_value = emission
    mat.diffuse_color = rgba
    return mat


def checker_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("floor")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.9
    tex = nt.nodes.new("ShaderNodeTexChecker")
    tex.inputs["Scale"].default_value = 1.0                      # 1 m 칸 (Object 좌표 기준)
    tex.inputs["Color1"].default_value = srgb_to_linear((228, 228, 228))
    tex.inputs["Color2"].default_value = srgb_to_linear((210, 210, 210))
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def load_korean_font():
    """Noto Sans CJK / Nanum 글꼴을 찾아 불러온다. 없으면 None (영어 라벨)."""
    for pattern in ("Noto Sans CJK KR:bold", "NanumGothic:bold", "Noto Sans KR:bold"):
        try:
            path = subprocess.run(["fc-match", "-f", "%{file}", pattern], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        if path and any(k in path for k in ("CJK", "Nanum", "NotoSansKR")):
            try:
                return bpy.data.fonts.load(path)
            except RuntimeError:
                pass
    return None


def make_text(name: str, body: str, font, size: float, mat, align: str = "CENTER") -> bpy.types.Object:
    curve = bpy.data.curves.new(name, type="FONT")
    curve.body = body
    curve.size = size
    curve.align_x = align
    curve.align_y = "BOTTOM"
    if font is not None:
        curve.font = font
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# -----------------------------------------------------------------------------
# 스켈레톤
# -----------------------------------------------------------------------------
def import_bvh(path: Path, name: str) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.import_anim.bvh(filepath=str(path), global_scale=0.01, frame_start=1, use_fps_scale=False,
                            update_scene_fps=True, update_scene_duration=True, rotate_mode="NATIVE",
                            axis_forward="-Z", axis_up="Y")
    arm = bpy.context.view_layer.objects.active
    arm.name = name
    arm.show_in_front = False
    arm.hide_render = True                                       # 뼈 자체는 렌더 안 됨 (메시로 그림)
    return arm


def add_cylinder(bm, a: Vector, b: Vector, r: float, seg: int = 12) -> list:
    d = b - a
    if d.length < 1e-6:
        return []
    axis = d.normalized()
    e1 = axis.orthogonal().normalized()
    e2 = axis.cross(e1)
    ring_a, ring_b = [], []
    for k in range(seg):
        t = 2 * math.pi * k / seg
        off = (math.cos(t) * e1 + math.sin(t) * e2) * r
        ring_a.append(bm.verts.new(a + off))
        ring_b.append(bm.verts.new(b + off))
    for k in range(seg):
        k2 = (k + 1) % seg
        bm.faces.new((ring_a[k], ring_a[k2], ring_b[k2], ring_b[k]))
    return ring_a + ring_b


def add_sphere(bm, c: Vector, r: float) -> list:
    res = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=r, matrix=Matrix.Translation(c))
    return res["verts"]


def build_skeleton_mesh(arm: bpy.types.Object, mat) -> bpy.types.Object:
    """뼈 rest 위치 (아마추어 공간) 에 관절 사이 원기둥·관절 구를 만들고 뼈별 vertex group + Armature modifier 로 붙인다."""
    bones = arm.data.bones
    bm = bmesh.new()
    groups: dict[str, list] = {b.name: [] for b in bones}
    for bone in bones:
        head = bone.head_local.copy()
        if bone.parent is not None:                              # 부모 관절 → 이 관절 (부모 뼈에 붙음)
            groups[bone.parent.name] += add_cylinder(bm, bone.parent.head_local.copy(), head, BONE_RADIUS)
        if bone.name.lower() == "head" and bone.parent is not None:
            # LaFAN1 End Site 는 길이 0 이라 Blender 가 임의 방향 뼈를 만든다 → End Site 원기둥은 그리지 않고,
            # 머리 구는 목→머리 방향으로 조금 위에 둔다
            up = (head - bone.parent.head_local).normalized()
            groups[bone.name] += add_sphere(bm, head + up * HEAD_RADIUS * 0.9, HEAD_RADIUS)
        groups[bone.name] += add_sphere(bm, head, JOINT_RADIUS)
    bm.verts.ensure_lookup_table()
    index = {v: i for i, v in enumerate(bm.verts)}
    vg_idx = {name: [index[v] for v in vs] for name, vs in groups.items()}

    mesh = bpy.data.meshes.new(f"{arm.name}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    obj = bpy.data.objects.new(f"{arm.name}_body", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = arm.matrix_world.copy()
    obj.data.materials.append(mat)
    for name, ids in vg_idx.items():
        if ids:
            obj.vertex_groups.new(name=name).add(ids, 1.0, "REPLACE")
    mod = obj.modifiers.new("armature", "ARMATURE")
    mod.object = arm
    mod.use_vertex_groups = True
    mod.use_deform_preserve_volume = False
    obj.parent = arm
    obj.matrix_parent_inverse = arm.matrix_world.inverted()
    return obj


def head_bone_name(arm: bpy.types.Object) -> str:
    for b in arm.data.bones:
        if b.name.lower() == "head":
            return b.name
    return max(arm.data.bones, key=lambda b: b.head_local.z).name


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    clip = args.clip_dir.resolve()
    meta = json.loads((clip / "meta.json").read_text(encoding="utf-8"))
    out = (args.out or clip / "render.mp4").resolve()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    font = load_korean_font()
    if font is None:
        print("[알림] 한글 글꼴을 찾지 못해 영어 라벨을 씁니다 (sudo apt install fonts-noto-cjk).")

    # ----- 스켈레톤 -----
    arms = {}
    for key, ko, en, rgb in TRACKS:
        path = clip / f"{key}.bvh"
        if not path.exists():
            print(f"[알림] {path.name} 없음 → 건너뜀")
            continue
        arm = import_bvh(path, key)
        color = srgb_to_linear(rgb)
        build_skeleton_mesh(arm, make_material(f"{key}_mat", color))
        arms[key] = (arm, ko if font else en, color)

    n_frames = int(meta["frames"])
    scene.render.fps = int(meta.get("fps", 30))
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = min(n_frames, args.frame_end) if args.frame_end else n_frames

    # ----- 모든 프레임의 관절 위치 (카메라·바닥 범위용) -----
    pts = []
    for f in range(1, n_frames + 1, 3):
        scene.frame_set(f)
        for arm, _, _ in arms.values():
            mw = arm.matrix_world
            pts += [mw @ pb.head for pb in arm.pose.bones]
            pts.append(mw @ arm.pose.bones[head_bone_name(arm)].head + Vector((0, 0, 0.6)))   # 라벨 공간
    scene.frame_set(1)
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    floor_z = min(0.0, lo.z - JOINT_RADIUS)
    center = (lo + hi) / 2

    # ----- 바닥 -----
    size = max(hi.x - lo.x, hi.y - lo.y) + 200.0             # 직교 카메라라 가장자리가 안 보이게 크게
    bpy.ops.mesh.primitive_plane_add(size=size, location=(center.x, center.y, floor_z))
    floor = bpy.context.active_object
    floor.name = "floor"
    floor.data.materials.append(checker_material())

    # ----- 카메라: 진행 방향 f, 옆 방향 s (생성 = +s) 으로 비스듬히, 직교 투영 -----
    fwd = bvh_to_blender(meta["forward_dir"])
    side = bvh_to_blender(meta["side_dir"])
    fwd.z = side.z = 0.0
    fwd.normalize()
    side.normalize()
    elev = math.radians(22)
    horiz = (0.64 * fwd + 0.77 * side).normalized()
    view_from = (horiz * math.cos(elev) + Vector((0, 0, math.sin(elev)))).normalized()
    cam_data = bpy.data.cameras.new("camera")
    cam_data.type = "ORTHO"
    cam = bpy.data.objects.new("camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    rot = (-view_from).to_track_quat("-Z", "Y")
    right, up = rot @ Vector((1, 0, 0)), rot @ Vector((0, 1, 0))
    xs = [(p - center).dot(right) for p in pts]
    ys = [(p - center).dot(up) for p in pts]
    aspect = args.width / args.height
    margin = 1.12
    cam_data.ortho_scale = max(max(xs) - min(xs), (max(ys) - min(ys)) * aspect) * margin
    look = center + right * (max(xs) + min(xs)) / 2 + up * (max(ys) + min(ys)) / 2
    cam.location = look + view_from * 40.0
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = rot
    cam_data.clip_start, cam_data.clip_end = 0.1, 200.0

    # ----- 라벨 (머리 위, 카메라를 향함) -----
    label_size = cam_data.ortho_scale * 0.04
    for key, (arm, label, color) in arms.items():
        txt = make_text(f"{key}_label", label, font, label_size, make_material(f"{key}_txt", color, emission=0.6))
        c = txt.constraints.new("COPY_LOCATION")
        c.target, c.subtarget, c.head_tail = arm, head_bone_name(arm), 1.0
        c.use_offset = True
        txt.location = (0, 0, 0.18)
        r = txt.constraints.new("COPY_ROTATION")
        r.target = cam

    # ----- 화면 글자 (카메라에 붙임): 구간 / 프레임 번호 -----
    w = cam_data.ortho_scale
    h = w / aspect
    black = make_material("txt_black", srgb_to_linear((20, 20, 20)), emission=0.3)
    red = make_material("txt_red", srgb_to_linear((214, 39, 40)), emission=0.6)
    phase_txt = make_text("phase", "", font, w * 0.045, black, align="LEFT")
    frame_txt = make_text("frame", "", font, w * 0.026, black, align="RIGHT")
    info_txt = make_text("info", "", font, w * 0.019, black, align="RIGHT")
    for obj, (x, y) in ((phase_txt, (-w / 2 + w * 0.03, h / 2 - w * 0.07)),
                        (frame_txt, (w / 2 - w * 0.03, h / 2 - w * 0.05)),
                        (info_txt, (w / 2 - w * 0.03, -h / 2 + w * 0.025))):
        obj.parent = cam
        obj.location = (x, y, -1.0)
    info_txt.data.body = (f"{meta['source_bvh']}  frame {meta['source_start_frame']}–"
                          f"{meta['source_start_frame'] + n_frames - 1}  |  gpos_l2 {meta['gpos_l2']:.2f} cm "
                          f"(SLERP {meta['gpos_l2_slerp']:.2f})")
    names = PHASE_KO if font else PHASE_EN
    phases = meta["phases"]
    t0, t1 = meta["transition"]
    fps = scene.render.fps

    def on_frame(sc, *_):
        f = sc.frame_current - 1                                 # meta 는 0부터
        ph = next((p for p in phases if p[1] <= f < p[2]), phases[-1])
        phase_txt.data.body = names[ph[0]]
        phase_txt.data.materials[0] = red if t0 <= f < t1 else black
        frame_txt.data.body = f"frame {f:3d} / {n_frames - 1}   {f / fps:5.2f} s"

    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(on_frame)
    on_frame(scene)

    # ----- 조명 / 배경 -----
    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = 2.2
    sun_data.angle = math.radians(8)
    sun = bpy.data.objects.new("sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(40), 0, math.atan2(horiz.y, horiz.x) + math.radians(60))
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = srgb_to_linear((245, 245, 245))
    bg.inputs["Strength"].default_value = 0.45

    # ----- 렌더 설정 (EEVEE → mp4) -----
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in \
        bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys() else "BLENDER_EEVEE"
    ee = scene.eevee
    ee.taa_render_samples = args.samples
    for attr, val in (("use_gtao", True), ("use_soft_shadows", True), ("shadow_cube_size", "1024"),
                      ("shadow_cascade_size", "2048")):
        if hasattr(ee, attr):
            setattr(ee, attr, val)
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.use_lock_interface = True                       # 핸들러가 글자를 바꾸므로
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.use_file_extension = False
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out)

    if args.still is not None:
        png = out.with_name(f"{out.stem}_f{args.still:03d}.png")
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(png)
        scene.frame_set(args.still + 1)
        bpy.ops.render.render(write_still=True)
        print(f"저장: {png}")
        return

    print(f"렌더: {clip.name}, 프레임 {scene.frame_start}–{scene.frame_end} @ {fps}fps, "
          f"{args.width}x{args.height}, {scene.render.engine} samples {args.samples} → {out}")
    bpy.ops.render.render(animation=True)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
