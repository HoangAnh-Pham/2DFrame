# ----------------------------------------------------------------
# PURPOSE
#    Analysis of a plane truss.
# ----------------------------------------------------------------

import matplotlib.pyplot as plt
import Module_2DFrame as st

def f_data():
# ----- Connectivity matrix Ele -------------------------------------
    Ele = [
    [1, 3, 1],
    [2, 4, 1],
    [3, 5, 1],
    [4, 6, 1],
    [4, 3, 1],
    [6, 5, 1],
    [2, 3, 1],
    [4, 5, 1],
    [1, 4, 1],
    [3, 6, 1],
    ]
# ----- Node coordinate Coord -------------------------------------
    a = 2;
    Node = [
    [0, a],
    [0, 0],
    [a, a],
    [a, 0],
    [2*a, a],
    [2*a, 0],
    ]
# ----- Element properties ---------------------------------------
    A = 25.0e-4
    E = 2.1e8
    I = 0
    # Section properties
    Mat = [
    [E, A, I],
    ]
# ----- Node restraint Bc -------------------------------------
    # [node_id, xr, yr, fi] # 1: restrained; 0: free
    Bound = [
    [1, 1, 1, 0],
    [2, 1, 1, 0],
    ]
    # Supp = [node_id, dx, dy, dfi]    
    Supp = [[1, 0.0, 0.0, 0]]
    
    Spring = []
# ----- Node load ---------------------------------------
    # [node_id, Fx, Fy, M]
    Px = 0; Py=100;
    Nload = [
    [5, Px, -Py, 0],
    ]
    
    Eload = []
    
    return '2Dtruss', Node, Ele, Mat, Bound, Eload, Nload, Supp, Spring

# -----------------------------------------------------------------
plt.close('all')
st.f_unit='kN'

# Create Model
model = st.init_model(f_data())

# Solve
model = st.Solve2Dtruss(model)

# Show reaction
st.disp_react(model)

# Show displacement
st.disp_ndisp(model)

# Show element force
st.disp_eforce(model)

# Plot results
st.show_geometry(model,'E',show_ele=False)
st.show_FEM(model)
st.show_reaction(model,'Autto')
st.show_displacement(model, 'Auto')
st.show_axial(model, 'Auto')

