bl_info = {
    "name": "Multi-Angle Camera Array",
    "author": "Architecture Dev",
    "version": (1, 6, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > MultiCam",
    "description": "Stable v1.5.2 architecture + pure Thomson repulsion + Focal Length & height-order control.",
    "category": "Camera",
}

import bpy
import math
import random
from mathutils import Vector

# ==========================================
# 1. 算法层 (Math / Algorithms)
# ==========================================
class MathService:
    @staticmethod
    def get_standard_vectors() -> list[Vector]:
        """获取标准的六个轴向正视角"""
        return [
            Vector((0, -1, 0)),  # Front
            Vector((0, 1, 0)),   # Back
            Vector((-1, 0, 0)),  # Left
            Vector((1, 0, 0)),   # Right
            Vector((0, 0, 1)),   # Top
            Vector((0, 0, -1))   # Bottom
        ]

    @staticmethod
    def get_fibonacci_sphere_vectors(samples: int) -> list[Vector]:
        """基础斐波那契分布（作为斥力算法的优秀初始均匀拓扑种子）"""
        vectors = []
        phi = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(samples):
            y = 1 - (i + 0.5) * (2.0 / samples)
            radius = math.sqrt(1 - y * y)
            theta = phi * i
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            vectors.append(Vector((x, y, z)))
        return vectors
        
    @staticmethod
    def get_platonic_vectors(samples: int) -> list[Vector]:
        """
        获取特定几何体的解析解。
        基于 1.5.2 架构，移除了所有冗余的人类直觉硬编码。
        仅保留 N=2 作为世界坐标的物理对齐基准，其余全部交由动态演算。
        """
        if samples == 2:
            return [Vector((0, -1, 0)), Vector((0, 1, 0))]
        return []

    @staticmethod
    def get_repulsion_even_vectors(samples: int, iterations: int = 500) -> list[Vector]:
        """
        纯动态计算算法 (基于 v1.5.2 基线):
        除 N=2 外，全部使用 Thomson 静电斥力模型计算最优解，避免陷入局部最优陷阱。
        """
        if samples == 0: return []
        if samples == 1: return [Vector((0, -1, 0))]
        
        # 拦截：仅拦截 N=2 的基准情况
        platonic = MathService.get_platonic_vectors(samples)
        if platonic:
            return platonic
        
        # 回退：物理斥力迭代 (Thomson Problem Numerical Solver)
        vectors = MathService.get_fibonacci_sphere_vectors(samples)
        
        learning_rate = 0.2
        cooling_rate = 0.98  
        
        for _ in range(iterations):
            forces = [Vector((0, 0, 0)) for _ in range(samples)]
            max_force = 0.0
            
            for i in range(samples):
                for j in range(i + 1, samples):
                    diff = vectors[i] - vectors[j]
                    dist_sq = diff.length_squared
                    if dist_sq > 0.00001: 
                        force = diff.normalized() / dist_sq
                        forces[i] += force
                        forces[j] -= force
                        
            for i in range(samples):
                vectors[i] += forces[i] * learning_rate
                vectors[i].normalize()
                max_force = max(max_force, forces[i].length)
                
            learning_rate *= cooling_rate
            
            # 提前终止优化
            if max_force * learning_rate < 0.0001:
                break
            
        return vectors

    @staticmethod
    def sort_vectors_by_height(master_obj: bpy.types.Object, vectors: list[Vector], descending: bool) -> list[Vector]:
        """
        按世界坐标的 Z 高度重排方向向量（计入主物体的旋转）。

        生成出来的摄像机编号本身就是「上 → 下」或「下 → 上」的顺序，
        时间轴相邻标记的视角只会小幅变化，避免播放时镜头大幅度跳动。

        Z 保持严格单调：任何高度差都会分层，绝不为缩短跳角而回退高度，
        确保「从上到下」这一语义是准确可验证的。
        同一水平高度（Z 相同，如前后左右四个赤道视角）内部再按方位角蛇形扫掠，
        逐层升降时方位角来回摆动而非绕圈回头，避免赤道层出现 180° 跳变。
        """
        rot3 = master_obj.matrix_world.to_3x3()

        def azimuth(v: Vector) -> float:
            w = rot3 @ v.normalized()
            return math.atan2(w.y, w.x)

        def azimuth_gap(a: float, b: float) -> float:
            """两个方位角的最短夹角（处理 ±π 环绕）"""
            d = abs(a - b) % (2 * math.pi)
            return min(d, 2 * math.pi - d)

        # 以 Z 为主键排序（sorted 稳定，同高度保持原生成顺序）
        ordered = sorted(vectors, key=lambda v: (rot3 @ v.normalized()).z, reverse=descending)

        result: list[Vector] = []
        layer: list[Vector] = []
        layer_z = None
        last_az = None

        def flush_layer():
            nonlocal last_az
            if not layer:
                return
            by_az = sorted(layer, key=azimuth)
            # 蛇形：选与上一层末尾方位角更接近的那一端作为本层起点
            if last_az is not None and \
                    azimuth_gap(azimuth(by_az[0]), last_az) > azimuth_gap(azimuth(by_az[-1]), last_az):
                by_az.reverse()
            result.extend(by_az)
            last_az = azimuth(by_az[-1])
            layer.clear()

        for vec in ordered:
            z = (rot3 @ vec.normalized()).z
            if layer_z is not None and abs(z - layer_z) > 1e-6:
                flush_layer()
            layer.append(vec)
            layer_z = z

        flush_layer()
        return result

    @staticmethod
    def get_random_sphere_vectors(samples: int, seed: int) -> list[Vector]:
        vectors = []
        rng = random.Random(seed)
        for _ in range(samples):
            theta = rng.uniform(0, 2 * math.pi)
            z = rng.uniform(-1, 1)
            radius = math.sqrt(1 - z * z)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            vectors.append(Vector((x, y, z)))
        return vectors


# ==========================================
# 2. 业务逻辑层 (Rig Generation Service)
# ==========================================
class RigService:
    NODE_PREFIX = "CamRigNode_"

    @classmethod
    def cleanup_existing_rig(cls, master_obj: bpy.types.Object):
        if not master_obj:
            return
        prefix = f"{cls.NODE_PREFIX}{master_obj.name}"
        nodes_to_delete = [child for child in master_obj.children if child.name.startswith(prefix)]
        for node in nodes_to_delete:
            cls._delete_hierarchy(node)

    @classmethod
    def _delete_hierarchy(cls, obj: bpy.types.Object):
        for child in obj.children:
            cls._delete_hierarchy(child)
        bpy.data.objects.remove(obj, do_unlink=True)

    @classmethod
    def build_rig(cls, context: bpy.types.Context, props: 'MultiCamProperties'):
        master_obj = props.master_object
        if not master_obj:
            return
            
        cls.cleanup_existing_rig(master_obj)

        vectors = []
        if props.include_standard:
            vectors.extend(MathService.get_standard_vectors())
            
        if props.distribution_mode == 'EVEN' and props.extra_cam_count > 0:
            vectors.extend(MathService.get_repulsion_even_vectors(props.extra_cam_count))
        elif props.distribution_mode == 'RANDOM' and props.extra_cam_count > 0:
            vectors.extend(MathService.get_random_sphere_vectors(props.extra_cam_count, props.random_seed))

        if not vectors:
            return

        # --- 可选的按高度重排（生成顺序 / 上→下 / 下→上）---
        if props.camera_order != 'GENERATION':
            vectors = MathService.sort_vectors_by_height(
                master_obj, vectors, descending=(props.camera_order == 'TOP_DOWN')
            )

        for i, vec in enumerate(vectors):
            # --- 创建独立的空物体节点 ---
            empty_name = f"{cls.NODE_PREFIX}{master_obj.name}_{i:03d}"
            node_empty = bpy.data.objects.new(empty_name, None)
            node_empty.empty_display_type = 'SPHERE'
            node_empty.empty_display_size = 0.2
            
            node_empty.parent = master_obj
            node_empty.location = (0, 0, 0) 
            context.collection.objects.link(node_empty)

            # --- 创建对应的摄像机并应用焦距 ---
            cam_data = bpy.data.cameras.new(name=f"CamData_{master_obj.name}_{i:03d}")
            
            # 【新增逻辑】：统一赋予用户在 UI 面板设置的焦距参数
            cam_data.lens = props.focal_length 
            
            cam_obj = bpy.data.objects.new(name=f"RigCam_{master_obj.name}_{i:03d}", object_data=cam_data)
            
            cam_obj.parent = node_empty
            cam_obj.location = vec.normalized() * props.distance
            context.collection.objects.link(cam_obj)

            # --- 绑定约束 ---
            constraint = cam_obj.constraints.new(type='TRACK_TO')
            constraint.target = node_empty
            constraint.track_axis = 'TRACK_NEGATIVE_Z'
            constraint.up_axis = 'UP_Y'

    @classmethod
    def bind_cameras_to_markers(cls, context: bpy.types.Context, props: 'MultiCamProperties') -> int:
        master_obj = props.master_object
        if not master_obj:
            return 0

        scene = context.scene
        node_prefix = f"{cls.NODE_PREFIX}{master_obj.name}"
        marker_prefix = f"Mark_{master_obj.name}_"
        
        rig_cameras = []
        for node in master_obj.children:
            if node.name.startswith(node_prefix):
                for child in node.children:
                    if child.type == 'CAMERA':
                        rig_cameras.append(child)
        
        if not rig_cameras:
            return 0

        # 按名称排序：名称尾部的序号即生成时确定的排列顺序
        # （选择「从上到下 / 从下到上」时，build_rig 已按高度写入了编号，这里直接沿用）
        rig_cameras.sort(key=lambda c: c.name)

        # 防弹清理逻辑
        markers_to_remove = []
        for m in scene.timeline_markers:
            if m.name.startswith("Mark_") or (m.camera in rig_cameras):
                markers_to_remove.append(m)
                
        for m in markers_to_remove:
            scene.timeline_markers.remove(m)

        start_frame = scene.frame_start
        interval = props.marker_interval

        for i, cam in enumerate(rig_cameras):
            frame = start_frame + (i * interval)
            marker_name = f"{marker_prefix}{i:03d}"
            marker = scene.timeline_markers.new(name=marker_name, frame=frame)
            marker.camera = cam

        scene.frame_end = start_frame + ((len(rig_cameras) - 1) * interval)
        
        return len(rig_cameras)


# ==========================================
# 3. 数据模型 (Properties)
# ==========================================
class MultiCamProperties(bpy.types.PropertyGroup):
    master_object: bpy.props.PointerProperty(
        name="总父级物体 (Master)",
        description="选择一个物体作为阵列的中心和总父级。更换物体不影响其他物体的阵列",
        type=bpy.types.Object
    )
    
    distance: bpy.props.FloatProperty(
        name="摄像机距离",
        description="摄像机距离中心空物体的统一半径",
        default=5.0,
        min=0.1
    )
    
    # 【新增逻辑】：避免报错，使用标准的 FloatProperty 控制焦距
    focal_length: bpy.props.FloatProperty(
        name="摄像机焦距 (mm)",
        description="统一设置生成的摄像机的焦距大小",
        default=50.0,
        min=1.0,
        max=500.0
    )
    
    include_standard: bpy.props.BoolProperty(
        name="包含6个标准正视角",
        description="是否生成前后左右上下6个标准视角的摄像机",
        default=True
    )
    
    distribution_mode: bpy.props.EnumProperty(
        name="额外填充模式",
        items=[
            ('NONE', "无", "不生成额外摄像机"),
            ('EVEN', "绝对均匀 (动态演算)", "使用物理斥力迭代计算势能最低态，实现完美均匀分布"),
            ('RANDOM', "随机分布", "随机分布摄像机"),
        ],
        default='EVEN'
    )
    
    extra_cam_count: bpy.props.IntProperty(
        name="额外摄像机数量",
        description="生成的额外摄像机密度/数量",
        default=8,
        min=1,
        max=500
    )
    
    random_seed: bpy.props.IntProperty(
        name="随机种子",
        description="控制随机分布形态的种子",
        default=42
    )

    marker_interval: bpy.props.IntProperty(
        name="标记间隔 (帧)",
        description="将摄像机绑定到时间轴标记时，每个视角的帧数间隔",
        default=1,
        min=1,
        max=500
    )

    camera_order: bpy.props.EnumProperty(
        name="摄像机排列顺序",
        description="决定摄像机编号与时间轴标记的排列方式，避免播放时视角大幅度跳动",
        items=[
            ('GENERATION', "生成顺序", "按算法生成的原始顺序排列（旧行为）"),
            ('TOP_DOWN', "从上到下 (Z 降序)", "按世界坐标 Z 高度从高到低排列，镜头沿垂直方向平滑过渡"),
            ('BOTTOM_UP', "从下到上 (Z 升序)", "按世界坐标 Z 高度从低到高排列"),
        ],
        default='GENERATION'
    )


# ==========================================
# 4. 表现层与控制器 (UI & Operators)
# ==========================================
class MULTICAM_OT_generate(bpy.types.Operator):
    """一键生成多角度摄像机阵列"""
    bl_idname = "multicam.generate"
    bl_label = "生成 / 更新摄像机阵列"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.multi_cam_props.master_object is not None

    def execute(self, context):
        props = context.scene.multi_cam_props
        RigService.build_rig(context, props)
        self.report({'INFO'}, f"摄像机阵列生成完毕: 绑定至 {props.master_object.name}")
        return {'FINISHED'}

class MULTICAM_OT_bind_markers(bpy.types.Operator):
    """一键将生成的摄像机按顺序绑定到时间轴的标记上"""
    bl_idname = "multicam.bind_markers"
    bl_label = "绑定摄像机到时间轴标记"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.multi_cam_props.master_object is not None

    def execute(self, context):
        props = context.scene.multi_cam_props
        count = RigService.bind_cameras_to_markers(context, props)
        
        if count > 0:
            self.report({'INFO'}, f"成功将 {count} 个摄像机绑定到时间轴标记 (已清理旧标记)")
        else:
            self.report({'WARNING'}, "未找到当前选定物体的摄像机阵列，请先生成！")
        return {'FINISHED'}


class MULTICAM_PT_main_panel(bpy.types.Panel):
    bl_label = "多角度摄像机生成器"
    bl_idname = "MULTICAM_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MultiCam'

    def draw(self, context):
        layout = self.layout
        props = context.scene.multi_cam_props

        # 核心设置控制区
        box = layout.box()
        box.label(text="核心设置:", icon='OBJECT_DATA')
        box.prop(props, "master_object")
        box.prop(props, "distance")
        box.prop(props, "focal_length")  # 【新增逻辑】：将焦距设置暴露在 UI 上

        # 视角分布控制区
        box = layout.box()
        box.label(text="分布策略:", icon='OUTLINER_OB_CAMERA')
        box.prop(props, "include_standard")
        box.prop(props, "distribution_mode")

        if props.distribution_mode != 'NONE':
            col = box.column(align=True)
            col.prop(props, "extra_cam_count")
            if props.distribution_mode == 'RANDOM':
                col.prop(props, "random_seed")

        # 【新增逻辑】：排列顺序（按 Z 高度排序，减少时间轴播放时的视角跳动）
        box_order = layout.box()
        box_order.label(text="排列顺序:", icon='SORT_DESC')
        box_order.prop(props, "camera_order")
        if props.camera_order != 'GENERATION':
            note = box_order.column(align=True)
            note.label(text="相邻标记沿垂直方向连续过渡", icon='INFO')
            note.label(text="需重新点击「生成」以刷新编号")

        # 生成按钮
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator(MULTICAM_OT_generate.bl_idname, icon='FILE_REFRESH')

        # 渲染辅助控制区
        layout.separator()
        box_render = layout.box()
        box_render.label(text="渲染辅助:", icon='RENDER_ANIMATION')
        box_render.prop(props, "marker_interval")
        
        row_bind = box_render.row()
        row_bind.scale_y = 1.2
        row_bind.operator(MULTICAM_OT_bind_markers.bl_idname, icon='MARKER_HLT')


# ==========================================
# 注册模块
# ==========================================
classes = (
    MultiCamProperties,
    MULTICAM_OT_generate,
    MULTICAM_OT_bind_markers,
    MULTICAM_PT_main_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.multi_cam_props = bpy.props.PointerProperty(type=MultiCamProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.multi_cam_props

if __name__ == "__main__":
    register()