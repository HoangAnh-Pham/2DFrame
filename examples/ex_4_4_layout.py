# ----------------------------------------------------------------
# PURPOSE
#    Analysis of a beam.
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
    ]
# ----- Node coordinate Coord -------------------------------------
    Node = [
    [0, 0],
    [3, 0],
    [9, 0],
    ]
# ----- Element properties ---------------------------------------
    A = 1; I = 2510e-8;
    E = 210e11

    # Section properties
    Mat = [
    [E, A, I],
    ]
# ----- Node restraint Bc -------------------------------------
    # [node_id, xr, yr, fi] # 1: restrained; 0: free
    Bound = [
    [1, 1, 1, 0],
    [3, 0, 1, 0],
    ]
    # Supp = [node_id, dx, dy, dfi]    
    Supp = []
    # Spring = [node_id, kx, ky, kfi] 
    Spring = []
    
# ----- Node load ---------------------------------------
    # [node_id, Fx, Fy, M]
    P = 10000;
    Nload = [
    [2, 0, -P, 0],
    ]
    
    Eload = []
    
    return '2Dframe', Node, Ele, Mat, Bound, Eload, Nload, Supp, Spring

# -----------------------------------------------------------------
cfv.closeAll()

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
st.dim_unit = 'm'
st.text_size = 12
st.dscale = 1
st.ele_name = 'k'

st.show_geometry(model,'S',show_ele=True)
st.draw_dim([0,0,0],[3,0,0],-1.5,text=None)
st.draw_dim([3,0,0],[9,0,0],-1.5,text=None)

st.show_FEM(model)

st.show_reaction(model,'Autto')
st.show_displacement(model, 'Auto')
st.show_moment(model, 'Auto')
st.show_shear(model, 'Auto')

