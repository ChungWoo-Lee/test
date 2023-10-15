import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pythreejs import *
# File paths
calibration_file_path = "C:/Users/leeju/OneDrive/바탕 화면/kitti_object_vis-master/kitti_object_vis-master/data/object/training/calib/000000.txt"
image_file_path = "C:/Users/leeju/OneDrive/바탕 화면/kitti_object_vis-master/kitti_object_vis-master/data/object/training/image_2/000000.png"
lidar_data_path = "C:/Users/leeju/OneDrive/바탕 화면/kitti_object_vis-master/kitti_object_vis-master/data/object/training/velodyne/000000.bin"

# Read calibration data
with open(calibration_file_path, 'r') as fid:
    lines = fid.readlines()
    #앞뒤 공백을 지우고, 띄어쓰기 기준으로 라인을 나눠 배열로 저장
    for i in range(7):
        lines[i] = lines[i].strip().split(' ')
        #첫 번째 P0: 이런 것들 다 지워준다
        del lines[i][0]
        # 각 배열의 원소들을 한 칸씩 띄워 한 문자열로 저장
        lines[i] = ' '.join(lines[i])

    # 문자열로부터 숫자 데이터를 뽑아 배열로 저장. 분류 기준은 띄어쓰기다.(문자 -> 숫자)
    P0 = np.fromstring(lines[0], dtype=float, sep=' ')
    P1 = np.fromstring(lines[1], dtype=float, sep=' ')
    P2 = np.fromstring(lines[2], dtype=float, sep=' ')
    P3 = np.fromstring(lines[3], dtype=float, sep=' ')
    R0_rect = np.fromstring(lines[4], dtype=float, sep=' ')
    Tr_velo_to_cam = np.fromstring(lines[5], dtype=float, sep=' ')
    Tr_imu_to_velo = np.fromstring(lines[6], dtype=float, sep=' ')
    Tr_cam_to_road = np.fromstring(lines[7], dtype=float, sep=' ')

#Read LiDAR data
data = np.fromfile(lidar_data_path, dtype=np.float32).reshape(-1, 4)
tr=Tr_velo_to_cam.reshape(3,4)
# Perform LiDAR-to-Image mapping
Tr= np.vstack([Tr_velo_to_cam.reshape(3,4),[0,0,0,1]])
R0=np.eye(4)
R0[:3,:3]=R0_rect.reshape(3,3)
r=R0_rect.reshape(3,3)
P2=P2.reshape(3,4)
P= np.vstack([P2,[0,0,0,1]])
XYZ=data[:,:3].T
XYZ1 = np.vstack((XYZ, np.ones(data.shape[0])))
xy_1 = np.dot(np.dot(np.dot(P, R0), Tr), XYZ1)
xy1 = np.dot(np.dot(np.dot(P2, R0), Tr), XYZ1)

# weight는 가중치, homogenuous coordinate에서 원래 좌표에서 사영 공간까지의 거리(?)
#(우리가 예전까지 s로 뒀던 것들)
# 곰곰히 생각해보면, weight값이 결국 depth값하고 1대 1 대응이 된다는 것을 알 수 있다
#(2d 사영공간으로부터 원점까지의 scale이니까)

weight = xy1[2, :]
x = xy1[0, :] /weight
y = xy1[1, :]/weight
# weight이 음수인 애들은 차 뒤편에 있다는 이야기니 제외시켜준다
k = np.where(weight > 0)[0]

img = Image.open(image_file_path)

display(img)
plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.scatter(x[k], y[k], c=weight[k], cmap='gist_earth', s=5)
plt.xlim(-100, img.width+100)
plt.ylim(-100, img.height+100)
plt.gca().invert_yaxis()
plt.show()

x=x[k].reshape(-1,1)
y=y[k].reshape(-1,1)

w=weight[k].reshape(-1,1)
xy=np.hstack((x,y))

xy=np.hstack((xy,w))


sorted_indices = np.argsort(xy[:, 0])[::1]
sorted_xy = xy[sorted_indices]
img_x_min = min(np.where(sorted_xy[:,0]>= 0)[0])
img_x_max = max(np.where(sorted_xy[:,0]<= img.width)[0])
sorted_xy=sorted_xy[img_x_min:img_x_max,:]

sorted_xy = sorted_xy[:, [1, 0,2]]
print('sorted_xy',sorted_xy)
sorted_indices = np.argsort(sorted_xy[:, 0])[::1]
sorted_xy = sorted_xy[sorted_indices]

img_y_min = min(np.where(sorted_xy[:,0]>= 0)[0])
img_y_max = max(np.where(sorted_xy[:,0]<= img.height)[0])

sorted_xy=sorted_xy[img_y_min:img_y_max,:]
sorted_xy = sorted_xy[:, [1, 0,2]]

display(img)
plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.scatter(sorted_xy[:,0], sorted_xy[:,1], c=sorted_xy[:,2], cmap='gist_earth', s=5)
plt.xlim(-100, img.width+100)
plt.ylim(-100, img.height+100)
plt.gca().invert_yaxis()
plt.show()

xy=sorted_xy.T
print("xy",xy)
xy[0,:]=xy[0,:]*xy[2,:]
xy[1,:]=xy[1,:]*xy[2,:]


valid_xy1=xy
print("vaild xy1",np.shape(valid_xy1))


#p2의 원소 구성은 다음과 같이 이루어져있다

#[fu,s,Cu,tx]
#[0,fv,Cv,ty]
#[0,0,1,tz]

#translation_cam2는 p2의 t정보 / rotation_cam2는 p2의 r정보를 의미한다
#월드좌표계(x,y,depth) = rotation_역행렬 * ((픽셀좌표계 (x,y,z)) - (tx,ty,tz))
#위 식 설명은 ppt에 좀 더 자세히 설명해서 첨부해드리겠소...

translation_cam2=P2[:,3].reshape(3,1)


valid_xy1=valid_xy1-translation_cam2

rotation_cam2=P2[:3,:3]
valid_xy1=np.dot(np.linalg.inv(rotation_cam2),valid_xy1)

#kitti dataset figure를 보면, 라이다의 (x,y,z)축이 카메라한테는 (z,-x,-y)임을 알 수 있다
#(z,-x,-y) -> (x,y,z)해주는 과정

p_cam2=valid_xy1

depth_camera02=valid_xy1

z_ero=np.zeros((1,3)).T
r_tr=np.dot(r,tr)

tt=tr[:,3].reshape(3,1)
t= r_tr[:,3].reshape(3,1)

r=np.eye(3)
r=r_tr[:,:3]
rr=tr[:,:3]

print(z_ero)

cam_position= z_ero-tt

cam_position=np.dot(np.linalg.inv(rr),cam_position)
print(np.dot(np.linalg.inv(r),z_ero-t))
print("cam_position",cam_position)
###############################
'''
tmp_x=cam_position[2,:]
tmp_y=-cam_position[0,:]
tmp_z=-cam_position[1,:]
cam_position[0,:]=tmp_x
cam_position[1,:]=tmp_y
cam_position[2,:]=tmp_z
'''

cam_position=cam_position.T

depth_camera02=depth_camera02-t

depth_camera02=np.dot(np.linalg.inv(r),depth_camera02)
depth_camera02=depth_camera02.T
print("p_cam2",p_cam2)
'''
tmp__x=p_cam2[:,2]
tmp__y=-p_cam2[:,0]
tmp__z=-p_cam2[:,1]
p_cam2[:,0]=tmp__x
p_cam2[:,1]=tmp__y
p_cam2[:,2]=tmp__z
'''
Fov= 2*math.atan(P2[1,1]/P2[0,0])*(180/pi)


a,b,c=cam_position[0,:]
camposition=[a,b,c]
#depth(여기선 x축상 거리) 값들을 정규화 시켜준 뒤, colormap에 태워준다.

print(camposition)
#############
z_coords = depth_camera02[:, 2]

# Define your custom color range
custom_color_min = np.min(z_coords)
custom_color_max = np.max(z_coords)  # Adjust as needed

# Normalize the z-coordinates to the custom color range [0, 1]
normalized_z = (z_coords - custom_color_min) / (custom_color_max - custom_color_min)
# Use the "viridis" colormap to map normalized_z to colors
cmap = plt.get_cmap('magma')
colors = cmap(normalized_z*2)
########
geometry = BufferGeometry(
     attributes={
        'position': BufferAttribute(depth_camera02, normalized=False),
        'color': BufferAttribute(colors, normalized=True)
     }
 )

material = PointsMaterial(size=0.1, vertexColors='VertexColors')
point=Points(geometry=geometry, material=material)

camera = PerspectiveCamera(up=[0,0,1], position=[-10,0,60], near=0.1, aspect=400/300)
key_light = DirectionalLight(position=[0, 10, 10])
ambient_light = AmbientLight()
scene=Scene(children=[point,camera,key_light,ambient_light], background=None)
scene.add(AxesHelper(size=3))

cam=PerspectiveCamera(position=camposition, aspect=370/1224, fov=Fov, near=min(depth_camera02[:,0]), far=max(depth_camera02[:,0]))
camera_helper = CameraHelper(cam)
cam.lookAt([30,0,0])

'''
#Lidar sensor를 표현해주기 위한 PointLightHelper
point_light = PointLight(color="#ffffff", intensity=1, distance=100)
lidar_position=(lidar_position[0],lidar_position[1],lidar_position[2])
point_light_helper = PointLightHelper(point_light, position=lidar_position, distance=0.1, sphereSize=0.1)
scene.add(point_light)
scene.add(point_light_helper)
'''

scene.add(camera_helper)
controller=OrbitControls(controlling=camera)
renderer=Renderer(camera=camera, scene=scene, controls=[controller],width=800,height=600)
display(renderer)