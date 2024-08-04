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
    [1, 2, 1],
    [2, 3, 1],
    [1, 4, 1],
    [4, 6, 1],
    [4, 2, 1],
    [2, 6, 1],
    [2, 5, 1],
    [5, 3, 1],
    [6, 5, 1],
    ]
# ----- Node coordinate Coord -------------------------------------
    Node = [
    [0, 0],
    [2*a, 0],
    [4*a, 0],
    [a, b],
    [3*a, b],
    [2*a, 2*b],
    ]
# ----- Element properties ---------------------------------------
    S1 = 0.2*0.2;
    E = 2.e7
    k = 1e4;
    # Section properties
    Mat = [
    [E, S1, 0],
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
    Spring = [[2, 0, k, 0],
              [3, 0, 0, 0]]
    
# ----- Node load ---------------------------------------
    # [node_id, Fx, Fy, M]
    Px=50; Py=50;
    Nload = [
    [4, 0, -Py, 0],
    [5, 0, -Py, 0],
    [6, Px, -Py, 0],
    ]
    
    Eload = []
    
    return '2Dtruss', Node, Ele, Mat, Bound, Eload, Nload, Supp, Spring

# -----------------------------------------------------------------
cfv.closeAll()
a, b = 3, 2;

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
st.text_size = 15
st.dscale = 0.9
st.ele_name = 'k'

st.show_geometry(model,'S',show_ele=True)
st.draw_dim([0,0,0],[a,0,0],-1.5,text=None)
st.draw_dim([a,0,0],[2*a,0,0],-1.5,text=None)
st.draw_dim([2*a,0,0],[3*a,0,0],-1.5,text=None)
st.draw_dim([3*a,0,0],[4*a,0,0],-1.5,text=None)

st.draw_dim([4*a,0,0],[4*a,b,0],-1.5,text=None)
st.draw_dim([4*a,b,0],[4*a,2*b,0],-1.5,text=None)

st.show_FEM(model)

st.show_reaction(model,'Autto')
st.show_displacement(model, 'Auto')
st.show_axial(model, 'Auto')

