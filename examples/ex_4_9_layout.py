
import calfem.core as cfc
import calfem.utils as cfu
import calfem.vis_mpl as cfv
import Module_2DFrame as st
import numpy as np

# Initial data value

E1 = 2.1e11; E2 = 2.0789e11; E3 = 2e11
b1 = 0.12; h1 = 0.3
b2 = 0.1; h2 = 0.22
b3 = 0.1; h3 = 0.12
F0 = 400; F1 = 1000; F2 = 800; F3 = 600

def f_data():
# ----- Connectivity matrix Ele -------------------------------------
# [node_i, node_j, section_id, si, sj]
    Ele = [
    [1, 2, 1, 1, 1],
    [2, 3, 1, 1, 1],
    [3, 4, 1, 1, 1],
    [8, 9, 1, 1, 1],
    [9, 10, 1, 1, 1],
    [10, 11, 1, 1, 1], 
    [14, 15, 1, 1, 1],
    [15, 16, 1, 1, 1],
    [16, 17, 1, 1, 1],
    [21, 22, 1, 1, 1],
    [22, 23, 1, 1, 1],
    [23, 24, 1, 1, 1],
    [2, 5, 2, 1, 1],
    [5, 9, 2, 1, 1],
    [15, 18, 2, 1, 1],
    [18, 22, 2, 1, 1],
    [3, 6, 2, 1, 1],
    [6, 10, 2, 1, 1],
    [10, 12, 2, 1, 1],
    [12, 16, 2, 1, 1],
    [16, 19, 2, 1, 1],
    [19, 23, 2, 1, 1],
    [4, 7, 2, 1, 1],
    [7, 11, 2, 1, 1],
    [11, 13, 2, 1, 1],
    [13, 17, 2, 1, 1],
    [17, 20, 2, 1, 1],
    [20, 24, 2, 1, 1],
    [8, 16, 3, 0, 0],
    [14, 10, 3, 0, 0],
    
    ]
# ----- Node coordinate Coord -------------------------------------
# [x, y]
    A, B = 4, 3;
    Node = [
    [0, 0],
    [0, B],
    [0, 2*B],
    [0, 3*B],
    [A, B],
    [A, 2*B],
    [A, 3*B],
    [2*A, 0],
    [2*A, B],
    [2*A, 2*B],
    [2*A, 3*B],
    [3*A, 2*B],
    [3*A, 3*B],
    [4*A, 0],
    [4*A, B],
    [4*A, 2*B],
    [4*A, 3*B],
    [5*A, B],
    [5*A, 2*B],
    [5*A, 3*B],
    [6*A, 0],
    [6*A, B],
    [6*A, 2*B],
    [6*A, 3*B],
    ]
# ----- Node restraint Bc -------------------------------------
# [node_id, xr, yr, fi] # 1: restrained; 0: free
    Bound = [
    [1, 1, 1, 1],
    [8, 1, 1, 1],
    [14, 1, 1, 1],
    [21, 1, 1, 1],
    ]
# ----- Element properties ---------------------------------------
    # E1 = 2.1e11; E2 = 2.0789e11; E3 = 2e11
    # b1 = 0.12; h1 = 0.3
    # b2 = 0.1; h2 = 0.22
    # b3 = 0.1; h3 = 0.12
    A1 = b1*h1; I1 = b1*h1**3/12
    A2 = b2*h2; I2 = b2*h2**3/12
    A3 = b3*h3; I3 = b3*h3**3/12
# Section properties
    Mat = [
    [E1, A1, I1],
    [E2, A2, I2],
    [E3, A3, I3],
    ]
# ----- Node load ---------------------------------------
# [node_id, Fx, Fy, M]
    # F0 = 400; F1 = 1000; F2 = 800; F3 = 600
    Nload = [
    [2, F3, -F0, 0],
    [3, F2, -F0, 0],
    [4, F1, -F0, 0],
    [5, 0, -F0, 0],
    [6, 0, -F0, 0],
    [7, 0, -F0, 0],
    [9, 0, -F0, 0],
    [10, 0, -F0, 0],
    [11, 0, -F0, 0],
    [12, 0, -F0, 0],
    [13, 0, -F0, 0],
    [15, 0, -F0, 0],
    [16, 0, -F0, 0],
    [17, 0, -F0, 0],
    [18, 0, -F0, 0],
    [19, 0, -F0, 0],
    [20, 0, -F0, 0],
    [22, 0, -F0, 0],
    [23, 0, -F0, 0],
    [24, 0, -F0, 0],
    ]
# ----- Element load (local) ---------------------------------------
# [ele_id, qx, qy]
    Eload = [
    ]
# ----- Support displacement ---------------------------------------
# [node_id, Dx, Dy, Fi]
    Supp = [
    ]    
    
    Spring = [
    ]
    return '2Dframe', Node, Ele, Mat, Bound, Eload, Nload, Supp, Spring

# -----------------------------------------------------------------
model = st.init_model(f_data())
model = st.Solve2Dframe(model)
# -----------------------------------------------------------------
cfv.closeAll()
cfu.disp_h1("Analysis of a plane frame")

st.ele_name = 'k'   
st.dscale = 0.5

# Plot results
st.show_geometry(model,'S',show_ele=True)
Node = model['Node']
st.draw_dim([Node[20][0],Node[20][1],0],[Node[23][0],Node[23][1],0],-2,text='3x3m')
st.draw_dim([Node[0][0],Node[0][1],0],[Node[20][0],Node[20][1],0],-2,text='6x4m')

st.show_FEM(model)

st.disp_ndisp(model)
st.show_moment(model)
st.show_shear(model)
st.show_axial(model)
st.show_reaction(model)
st.show_displacement(model)
