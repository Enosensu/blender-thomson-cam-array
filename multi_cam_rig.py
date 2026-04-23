bl_info = {
    "name": "Multi-Angle Camera Array",
    "author": "Architecture Dev",
    "version": (1, 5, 2),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > MultiCam",
    "description": "Stable v1.5.1 architecture with pure Thomson Problem repulsion for N>2.",
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
        回归至 1.5.1 架构，但移除了所有冗余的人类直觉硬编码。
        仅保留 N=2 作为世界坐标的物理对齐基准，其余全部交由动态演算。
        """
        if samples == 2:
            return [Vector((0, -1, 0)), Vector((0, 1, 0))]
        return []

    @staticmethod
    def get_repulsion_even_vectors(samples: int, iterations: int = 500) -> list[Vector]:
        """
        纯动态计算算法 (基于 v1.5.1 基线):
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

        for i, vec in enumerate(vectors):
            empty_name = f"{cls.NODE_PREFIX}{master_obj.name}_{i:03d}"
            node_empty = bpy.data.objects.new(empty_name, None)
            node_empty.empty_display_type = 'SPHERE'
            node_empty.empty_display_size = 0.2
            
            node_empty.parent = master_obj
            node_empty.location = (0, 0, 0) 
            context.collection.objects.link(node_empty)

            cam_data = bpy.data.cameras.new(name=f"CamData_{master_obj.name}_{i:03d}")
            cam_obj = bpy.data.objects.new(name=f"RigCam_{master_obj.name}_{i:03d}", object_data=cam_data)
            
            cam_obj.parent = node_empty
            cam_obj.location = vec.normalized() * props.distance
            context.collection.objects.link(cam_obj)

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

        box = layout.box()
        box.label(text="核心设置:", icon='OBJECT_DATA')
        box.prop(props, "master_object")
        box.prop(props, "distance")

        box = layout.box()
        box.label(text="分布策略:", icon='OUTLINER_OB_CAMERA')
        box.prop(props, "include_standard")
        box.prop(props, "distribution_mode")

        if props.distribution_mode != 'NONE':
            col = box.column(align=True)
            col.prop(props, "extra_cam_count")
            if props.distribution_mode == 'RANDOM':
                col.prop(props, "random_seed")

        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator(MULTICAM_OT_generate.bl_idname, icon='FILE_REFRESH')

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