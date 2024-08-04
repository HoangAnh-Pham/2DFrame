# ----------------------------------------------------------------
# PURPOSE
#    Analysis of a plane frame.
# ----------------------------------------------------------------

import matplotlib.pyplot as plt
import Module_2DFrame as st

# Create Model

model = st.SystemModel('2Dframe')

# ----- Node coordinate Coord -------------------------------------
L, H = 4, 3;
# [x, y]
model['Node'] = [
    [0, 0],
    [L, 0],
    [0, H],
    [L, H],
    ]
# ----- Connectivity matrix Ele -----------------------------------
# [node_i, node_j, section_id, si, sj]
model['Ele'] = [
    [1, 3, 1, 1, 1],
    [2, 4, 1, 1, 1],
    [3, 4, 2, 1, 1],
    ]
# ----- Node restraint Bc -----------------------------------------
# [node_id, xr, yr, fi] # 1: restrained; 0: free
model['Bound'] = [
    [1, 1, 1, 0],
    [2, 1, 1, 1],
    ]
# ----- Element properties ----------------------------------------
E = 2.1e11
A1 = 25.0e-4; I1 = 1.0e-5
A2 = 25.0e-4; I2 = 1.0e-5
# Section properties
model['Mat'] = [
    [E, A1, I1],
    [E, A2, I2],
    ]
# ----- Node load ---------------------------------------
# [node_id, Fx, Fy, M]
P = 100; M = 50
model['Nload'] = [
    [3, P, 0, -M],
    ]
# ----- Element load (local) ---------------------------------------
# [ele_id, qx, qy]
q = 10;
model['Eload'] = [
    [3, 0, -q],
    ]
# -----------------------------------------------------------------
# Solve
model = st.Solve2Dframe(model)
# Show reaction
st.disp_react(model)
# Show displacement
st.disp_ndisp(model,[3,4])
# Show element force
st.disp_eforce(model)
# -----------------------------------------------------------------
# Plot results
plt.close('all')

st.dim_unit='m'
st.f_unit='kN'
st.m_unit='kNm'
st.u_unit='kN/m'

st.show_geometry(model,etype='S-',show_ele=True)
st.draw_dim([0,0,0], [L,0,0], offset=-0.5)
st.draw_dim([L,0,0], [L,H,0], offset=-0.5)

st.show_FEM(model)
st.show_reaction(model)
st.show_displacement(model)
st.show_moment(model)
st.show_shear(model)
st.show_axial(model)