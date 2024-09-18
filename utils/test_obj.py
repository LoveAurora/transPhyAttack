from pywavefront import Wavefront

# 加载OBJ和MTL文件
scene = Wavefront('../3d_model/bmw-white-withDetach-AttachAgain.obj', collect_faces=True)

# 遍历场景中的所有对象
for mesh in scene.mesh_list:
    print(mesh.name)
    for face in mesh.faces:
        print(face)

# 遍历场景中的所有材质
