# ----------------------------------------------------------------
# PURPOSE
#    Analysis of a plane frame.
# ----------------------------------------------------------------

import calfem.utils as cfu
import calfem.vis_mpl as cfv
import Module_2DFrame as st

cfu.disp_h1("Analysis of a 2D beam")

def f_data():
# ----- Connectivity matrix Ele -------------------------------------
    Ele = [
    [1, 2, 1, 1, 1],
    [2, 3, 1, 1, 1],
    [3, 4, 1, 1, 1],
    [5, 2, 2, 0, 0],
    [5, 3, 2, 0, 0],
    ]
# ----- Node coordinate Coord -------------------------------------
    Node = [
    [0, H],
    [L, H],
    [2*L, H],
    [3*L, H],
    [0, 0],
    ]
# ----- Element properties ---------------------------------------
    A1 = 4e-3; I1 = 5.4e-5;
    A2 = 1e-3; I2 = 5.4e-5;
    E = 2e8

    # Section properties
    Mat = [
    [E, A1, I1],
    [E, A2, I2],
    ]
# ----- Node restraint Bc -------------------------------------
    # [node_id, xr, yr, fi] # 1: restrained; 0: free
    Bound = [
    [1, 1, 1, 1],
    [5, 1, 1, 1],
    ]
    # Supp = [node_id, dx, dy, dfi]    
    Supp = []
    # Spring = [node_id, kx, ky, kfi] 
    Spring = []
    
# ----- Node load ---------------------------------------
    # [node_id, Fx, Fy, M]
    q = 10
    Nload = []
    
    Eload = [
        [2, 0, -q],
        [3, 0, -q],
        ]
    
    return '2Dframe', Node, Ele, Mat, Bound, Eload, Nload, Supp, Spring

# -----------------------------------------------------------------
cfv.closeAll()
L, H = 2, 2

# Create Model
model = st.init_model(f_data())

# Solve
model = st.Solve2Dframe(model)

# Show reaction
cfu.disp_h2("Reaction")
cfu.disp_array(model['res'], ["Rx", "Ry", "M"])

# Show displacement
cfu.disp_h2("Node displacement")
st.disp_ndisp(model)

# Show element force
cfu.disp_h2("Element force")
st.disp_eforce(model)

# Plot results
st.f_unit='kN'
st.u_unit='kN/m'
st.m_unit='kNm'
st.dim_unit = 'm'
st.text_size = 12
st.dscale = 1
st.ele_name = 'k'

st.show_geometry(model,'S',show_ele=True)
st.draw_dim([0,0,0],[L,0,0],-1.,text=None)
st.draw_dim([L,0,0],[2*L,0,0],-1.,text=None)
st.draw_dim([2*L,0,0],[3*L,0,0],-1.,text=None)
st.draw_dim([3*L,0,0],[3*L,H,0],-1.,text=None)

st.show_FEM(model)

st.show_reaction(model,'Autto')
st.show_displacement(model, 'Auto')
st.show_moment(model, 'Auto')
st.show_shear(model, 'Auto')
