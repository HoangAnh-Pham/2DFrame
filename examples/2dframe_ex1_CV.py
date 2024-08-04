''' 
Tính khung theo phương pháp chuyển vị
sử dụng thư viện 2DFrame

'''
# Nạp các thư viện
import matplotlib.pyplot as plt
import Module_2DFrame as st
import numpy as np
# -----------------------------------------------------------------
# Hàm tính và vẽ biểu đồ mô men đơn vị
def MZ(model,Supp=[]):
    model['Nload'] = []
    model['Eload'] = []
    model['Supp'] = Supp
    model = st.Solve2Dframe(model)

    st.show_reaction(model)
    st.show_displacement(model)
    st.show_moment(model)
    
    return model['res'][2,2], model['res'][3,2], model['res'][3,0]
# -----------------------------------------------------------------
# Hàm tính và vẽ biểu đồ mô men do tải trọng    
def M0(model):
    model['Nload'] = Nload
    model['Eload'] = Eload
    model = st.Solve2Dframe(model)

    st.show_reaction(model)
    st.show_displacement(model)
    st.show_moment(model)
    
    return model['res'][2,2], model['res'][3,2], model['res'][3,0]
# -----------------------------------------------------------------
# Tạo sơ đồ tính
model = st.SystemModel('2Dframe')
# -----------------------------------------------------------------
# Số liệu nút
L, H = 4, 3;
model['Node'] = [
    [0, 0],
    [L, 0],
    [0, H],
    [L, H],
    ]
# -----------------------------------------------------------------
# Số liệu phần tử
model['Ele'] = [
    [1, 3, 1, 1, 1],
    [2, 4, 1, 1, 1],
    [3, 4, 2, 1, 1],
    ]
# -----------------------------------------------------------------
# Số liệu tiết diện
E = 2.1e11
A1 = 25.0e-4; I1 = 1.0e-5
A2 = 25.0e-4; I2 = 1.0e-5
model['Mat'] = [
    [E, A1, I1],
    [E, A2, I2],
    ]
# -----------------------------------------------------------------
# Liên kết
model['Bound'] = [
    [1, 1, 1, 0],
    [2, 1, 1, 1],
    [3, 0, 0, 1],
    [4, 0, 0, 1],
    [4, 1, 0, 0],
    ]
# -----------------------------------------------------------------
# Tải trọng nút
P = 100; M = 50
Nload = [[3, P, 0, -M]]
# Tải trọng phần tử
q = 10;
Eload = [[3, 0, -q]]
# -----------------------------------------------------------------
# Vẽ Hệ cơ bản
plt.close('all')
st.show_geometry(model)
# -----------------------------------------------------------------
# Giải hệ theo phương pháp chuyển vị
# -----------------------------------------------------------------
# M0
R1P, R2P, R3P = M0(model)
F0 = model['force']
# -----------------------------------------------------------------
# M1
r11, r21, r31 = MZ(model,Supp=[[3,0,0,1]])
F1 = model['force']
# -----------------------------------------------------------------
# M2
r12, r22, r32 = MZ(model,Supp=[[4,0,0,1]])
F2 = model['force']
# -----------------------------------------------------------------
# M3
r13, r23, r33 = MZ(model,Supp=[[4,1,0,0]])
F3 = model['force']
# -----------------------------------------------------------------
# Giải hệ pt tìm Z
RP = np.array([[R1P],[R2P],[R3P]])
r = np.array([[r11,r12,r13],
              [r21,r22,r23],
              [r31,r32,r33],
              ])
Z = np.linalg.inv(r).dot(-RP)
# -----------------------------------------------------------------
# Cộng tác dụng tìm nội lực
F = F1*Z[0,0] + F2*Z[1,0] + F3*Z[2,0] + F0

modelP = model
modelP['force'] = F
st.show_moment(model)