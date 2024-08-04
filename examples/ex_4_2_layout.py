# ----------------------------------------------------------------
# PURPOSE
#    Analysis of a plane truss.
# ----------------------------------------------------------------

import calfem.utils as cfu
import calfem.vis_mpl as cfv
import Module_2DFrame as st

cfu.disp_h1("Analysis of a plane truss")

def f_data():
# ----- Connectivity matrix Ele -------------------------------------
    Ele = [
    [2, 3, 1],
    [1, 3, 2],
    ]
# ----- Node coordinate Coord -------------------------------------
    a, b = 3, 4;
    Node = [
    [0, 0],
    [0, b],
    [a, b],
    ]
# ----- Element properties ---------------------------------------
    S1 = 0.2*0.2;
    S2 = 0.2*0.3
    E = 2.e7
    I = 0
    # Section properties
    Mat = [
    [E, S1, I],
    [E, S2, I],
    ]
# ----- Node restraint Bc -------------------------------------
    # [node_id, xr, yr, fi] # 1: restrained; 0: free
    Bound = [
    [1, 1, 1, 0],
    [2, 1, 1, 0],
    ]
    # Supp = [node_id, dx, dy, dfi]    
    Supp = []
    
    Spring = []
# ----- Node load ---------------------------------------
    # [node_id, Fx, Fy, M]
    Py=50;
    Nload = [
    [3, 0, -Py, 0],
    ]
    
    Eload = []
    
    return '2Dtruss', Node, Ele, Mat, Bound, Eload, Nload, Supp, Spring

# -----------------------------------------------------------------
cfv.closeAll()

# Create Model
model = st.init_model(f_data())

# Solve
model = st.Solve2Dtruss(model)

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
st.dscale = 2
st.ele_name = 'k'

st.show_geometry(model,'S',show_ele=True)
st.draw_dim([0,0,0],[3,0,0],-1.0,text=None)
st.draw_dim([3,0,0],[3,4,0],-1.0,text=None)

st.show_FEM(model)

st.show_reaction(model,'Autto')
st.show_displacement(model, 'Auto')
st.show_axial(model, 'Auto')

