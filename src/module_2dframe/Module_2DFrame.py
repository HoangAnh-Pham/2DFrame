"""
MODULE <Module_2DFrame>
Phan tich he thanh 2D bang CALFEM
Pham Hoang Anh (2024-2025)

"""

import numpy as np
import calfem.core as cfc
import calfem.utils as cfu
import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from scipy.sparse import csr_matrix

# ===== Analysis function ==========================================
# ------------------------------------------------------------------
# Initial model
def init_model(str_data_file):
    
    str_type, Node, Ele, Mat, Bound, Eload, Nload, Supp, Spring = str_data_file
    
    model = SystemModel(str_type)
    model['Node'] = Node
    model['Ele'] = Ele
    model['Mat'] = Mat
    model['Bound'] = Bound
    model['Eload'] = Eload
    model['Nload'] = Nload
    model['Supp'] = Supp
    model['Spring'] = Spring
    
    return model

# Set system model
def SystemModel(str_type='2Dframe'):
    ndof = 3
    if str_type=='2Dtruss':
        ndof = 2
        nsec = 2
    if str_type=='3Dtruss':
        ndof = 3
        nsec = 2
    if str_type=='2Dframe':
        ndof = 3
        nsec = 5

    model = {'type': str_type}
    model['ndof'] = ndof
    model['nsec'] = nsec
    model['Node'] = []
    model['Ele'] = []
    model['Mat'] = []
    model['Bound'] = []
    model['Eload'] = []
    model['Nload'] = []
    model['Supp'] = []
    model['Spring'] = []

    return model

# ------------------------------------------------------------------
def semi_beam2e(ex, ey, ep, eq=None, s1=1,s2=1):
   
    E, A, I = ep
    DEA = E*A
    DEI = E*I
  
    qX = 0.
    qY = 0.
    if not eq is None:
        qX, qY = eq

    x1, x2 = ex
    y1, y2 = ey
    dx = x2-x1
    dy = y2-y1
    L = np.sqrt(dx*dx+dy*dy)
   
    Kle = np.array([
        [ DEA/L,           0.,          0., -DEA/L,           0.,          0.],
        [    0.,  12*DEI/L**3,  6*DEI/L**2,     0., -12*DEI/L**3,  6*DEI/L**2],
        [    0.,   6*DEI/L**2,     4*DEI/L,     0.,  -6*DEI/L**2,     2*DEI/L],
        [-DEA/L,           0.,          0.,  DEA/L,           0.,          0.],
        [    0., -12*DEI/L**3, -6*DEI/L**2,     0.,  12*DEI/L**3, -6*DEI/L**2],
        [    0.,   6*DEI/L**2,     2*DEI/L,     0.,  -6*DEI/L**2,     4*DEI/L]
    ])

    fle = L*np.array([qX/2, qY/2, qY*L/12, qX/2, qY/2, -qY*L/12]).reshape(6,1)

    T1 = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, -((2 * (-1 + s1) * (2 + s2)) / (L * (-4 + s1 * s2))), (s1 * (-4 + s2)) / (-4 + s1 * s2), 0, (2 * (-1 + s1) * (2 + s2)) / (L * (-4 + s1 * s2)), -((2 * (-1 + s1) * s2) / (-4 + s1 * s2))],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, -((2 * (2 + s1) * (-1 + s2)) / (L * (-4 + s1 * s2))), -((2 * s1 * (-1 + s2)) / (-4 + s1 * s2)), 0, (2 * (2 + s1) * (-1 + s2)) / (L * (-4 + s1 * s2)), ((-4 + s1) * s2) / (-4 + s1 * s2)]
    ])
    
    D11 = E * I
    k1 = (3 * D11 * s1) / (L * (1 - s1 - 1e-10))
    k2 = (3 * D11 * s2) / (L * (1 - s2 - 1e-10))
    
    k = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, k1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, k2]
    ])
    
    Kle = T1.T @ Kle @ T1
    Kle = Kle + (np.eye(6) - T1).T @ k @ (np.eye(6) - T1)
    fle = T1.T @ fle

    nxX = dx/L
    nyX = dy/L
    nxY = -dy/L
    nyY = dx/L
    G = np.array([
        [nxX, nyX,   0,   0,   0,   0],
        [nxY, nyY,   0,   0,   0,   0],
        [  0,   0,   1,   0,   0,   0],
        [  0,   0,   0, nxX, nyX,   0],
        [  0,   0,   0, nxY, nyY,   0],
        [  0,   0,   0,   0,   0,   1]
    ])

    Ke = G.T @ Kle @ G
    fe = G.T @ fle

    if eq is None:
        return Ke
    else:
        return Ke, fe

def semi_beam2s(ex, ey, ep, ed, eq=None, nep=None, s1=1,s2=1):

    E, A, I = ep
    DEA = E*A
    DEI = E*I
  
    qX = 0.
    qY = 0.
    if not eq is None:
        qX, qY = eq

    ne=2
    if nep != None: 
       ne=nep
    
    x1, x2 = ex
    y1, y2 = ey
    dx = x2-x1
    dy = y2-y1
    L = np.sqrt(dx*dx+dy*dy)

    nxX = dx/L
    nyX = dy/L
    nxY = -dy/L
    nyY = dx/L
    G = np.array([
        [nxX, nyX,   0,   0,   0,   0],
        [nxY, nyY,   0,   0,   0,   0],
        [  0,   0,   1,   0,   0,   0],
        [  0,   0,   0, nxX, nyX,   0],
        [  0,   0,   0, nxY, nyY,   0],
        [  0,   0,   0,   0,   0,   1]
    ])

    edl = G @ ed.reshape(6,1)
    
    T1 = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, -((2 * (-1 + s1) * (2 + s2)) / (L * (-4 + s1 * s2))), (s1 * (-4 + s2)) / (-4 + s1 * s2), 0, (2 * (-1 + s1) * (2 + s2)) / (L * (-4 + s1 * s2)), -((2 * (-1 + s1) * s2) / (-4 + s1 * s2))],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, -((2 * (2 + s1) * (-1 + s2)) / (L * (-4 + s1 * s2))), -((2 * s1 * (-1 + s2)) / (-4 + s1 * s2)), 0, (2 * (2 + s1) * (-1 + s2)) / (L * (-4 + s1 * s2)), ((-4 + s1) * s2) / (-4 + s1 * s2)]
    ])

    edl = T1 @ edl

    a1 = np.array([
        edl[0],
        edl[3]
    ])
    C1 = np.array([
        [1.,      0.],
        [-1/L,   1/L]
    ]) 
    C1a = C1 @ a1

    a2 = np.array([
        edl[1],
        edl[2],
        edl[4],
        edl[5]
    ])
    C2 = np.array([
        [1.,      0.,    0.,     0.],
        [0.,      1.,    0.,     0.],
        [-3/L**2, -2/L,  3/L**2, -1/L],
        [2/L**3,  1/L**2, -2/L**3, 1/L**2]
    ]) 
    C2a = C2 @ a2
    
    X = np.arange(0., L+L/(ne-1), L/(ne-1)).reshape(ne,1) 
    zero = np.zeros(ne).reshape(ne,1)    
    one = np.ones(ne).reshape(ne,1)
  
    u = np.concatenate((one,  X), 1) @ C1a
    du = np.concatenate((zero,  one), 1) @ C1a
    if DEA != 0:
       u = u -(X**2-L*X)*qX/(2*DEA)
       du = du -(2*X-L)*qX/(2*DEA)

    v = np.concatenate((one,  X, X**2, X**3), 1) @ C2a
#   dv = np.concatenate((zero,  one, 2*X, 3*X**2), 1) @ C2a
    d2v=np.concatenate((zero, zero, 2*one, 6*X), 1) @ C2a
    d3v = np.concatenate((zero, zero, zero, 6*one), 1) @ C2a
    if DEI != 0:
       v = v+(X**4 - 2*L*X**3 + L**2*X**2)*qY/(24*DEI)
#      dv = dv+(2*X**3 - 3*L*X**2 + L**2*X)*qY/(12*DEI)
       d2v = d2v+(6*X**2 - 6*L*X + L**2*one)*qY/(12*DEI)
       d3v = d3v+(2*X - L*one)*qY/(2*DEI)
 
    N = DEA*du
    M = DEI*d2v
    V = -DEI*d3v 
    es = np.concatenate((N, V, M), 1)
    edi = np.concatenate((u, v), 1)
    eci = X

    if nep is None:
        return es
    else:
        return es, edi, eci    
# ------------------------------------------------------------------

# Solve 2D frame  
def Solve2Dframe(model):
    
    # Nhận dữ liệu từ model
    ndof = model['ndof']
    nsec = model['nsec']
    Coord = np.array(model['Node'])
    Ele = np.array(model['Ele'])
    Mat = np.array(model['Mat'])
    Bc = np.array(model['Bound'])
    Eload = np.array(model['Eload'])
    Nload = np.array(model['Nload'])
    Supp = np.array(model['Supp'])
    Spring = np.array(model['Spring'])
    
    # Xác định số nút và số phần tử của mô hình
    nnode = Coord.shape[0]
    nel = Ele.shape[0]
    
    # Kết nối phần tử
    sdof = ndof*nnode
    dof, edof = setdof(nnode, ndof, Ele[:,0:2].astype(int));
    ex, ey = cfc.coordxtr(edof, Coord, dof, 2)
    
    edof0 = edof.copy()
    
    model['ex'] = ex
    model['ey'] = ey
    
    # Thêm dof do giải phóng mô men đầu thanh
    end_i = Ele[:,3]
    end_j = Ele[:,4]
    ni = sum(end_i<1); #print(ni)
    nj = sum(end_j<1); #print(nj)

    edof[end_i<1,2] = np.array([i+sdof+1 for i in range(ni)])
    edof[end_j<1,5] = np.array([i+sdof+ni+1 for i in range(nj)])
    sdof = sdof + ni + nj
    
    model['dof'] = dof
    model['edof'] = edof
    
    # Tạo danh sách các dof bị ngăn cản
    bc_prescr = bcond(ndof, dof, Bc)
    
    # Tạo danh sách các chuyển vị cưỡng bức 
    bcVal = bsupp(ndof, dof, bc_prescr, Supp)
    
    # Tạo danh sách các dof có spring
    bc_spr = bcond(ndof, dof, Spring)
    
    # Tạo danh sách các spring 
    sprVal = bsupp(ndof, dof, bc_spr, Spring)
    
    # FEM solution
    K = np.zeros([sdof,sdof])
    f = np.zeros([sdof,1])
    
    # Lập tải trọng trên phần tử
    elf = eload(nel, Eload, Ele, Mat, Coord) 
    
    # Stiffness matrix K and force vector f
    i = 0; 
    for elx, ely, eltopo in zip(ex, ey, edof):
        ep = Mat[int(Ele[i,2])-1,0:3]
        eq = elf[i,:]
        Ke, fe = cfc.beam2e(elx, ely, ep, eq)
        #Ke, fe = semi_beam2e(elx, ely, ep, eq, s1=Ele[i,3], s2=Ele[i,4])
        cfc.assem(eltopo, K, Ke, f, fe)
        
        # semi-rigid connection
        s1,s2 = end_i[i], end_j[i]
        D11 = ep[0] * ep[2]
        
        x1, x2 = elx
        y1, y2 = ely
        dx = x2-x1
        dy = y2-y1
        L = np.sqrt(dx*dx+dy*dy)
              
        if s1<1:
            k1 = (3 * D11 * s1) / (L * (1 - s1))
            Ke = cfc.spring1e(k1)
            edof1 = np.array([edof0[i,2],eltopo[2]]) 
            cfc.assem(edof1,K,Ke)
        if s2<1:
            k2 = (3 * D11 * s2) / (L * (1 - s2))
            Ke = cfc.spring1e(k2)
            edof2 = np.array([edof0[i,5],eltopo[5]]) 
            cfc.assem(edof2,K,Ke)
                   
        i+=1
    
    # Thêm tải trọng tại nút 
    f = nload(ndof, dof, Nload, f) 
    
    # Thêm spring
    for i in range(len(bc_spr)): 
        cfc.assem(np.array([bc_spr[i]]), K, sprVal[i])

    # Solve equation system
    #a, r = cfc.solveq(K, f, bc_prescr, bcVal)
    
    # Chuyển sang ma trận thưa
    K = csr_matrix(K)
    # f = csr_matrix(f)
    
    a, r = cfc.spsolveq(K, f, bc_prescr, bcVal)

    # Reaction force
    res = np.array(r[0:ndof*nnode].reshape(nnode,ndof).copy())
    res = res[Bc[:,0]-1,:]
    
    # Node displacement
    ndis = np.array(a[0:ndof*nnode].reshape(nnode,ndof).copy())
    
    # Element Analysis
    # nsec = 11
    ed= cfc.extract_eldisp(edof, a)
    es = np.zeros([edof.shape[0], nsec, 3])
    edi = np.zeros([edof.shape[0], nsec, 2])
    ec = np.zeros([edof.shape[0], nsec, 1])
    i = 0
    for elx, ely, eld in zip(ex, ey, ed):
        ep = Mat[int(Ele[i,2])-1,0:3]
        eq = elf[i,:]
        es[i], edi[i], ec[i] = cfc.beam2s(elx, ely, ep, eld, eq, nsec)    
        #es[i], edi[i], ec[i] = semi_beam2s(elx, ely, ep, eld, eq, nsec, s1=Ele[i,3], s2=Ele[i,4])        
        # Change sign of shear
        es[i,:,1] = -es[i,:,1]
        
        i+=1
    
    model['disp'] = ndis
    model['force'] = es
    model['edisp'] = edi
    model['esect'] = ec
    model['res'] = res
    
    return model

# ------------------------------------------------------------------
# Solve 2D truss 
def Solve2Dtruss(model):
    
    # Nhận dữ liệu từ model
    ndof = model['ndof']
    nsec = model['nsec']
    Coord = np.array(model['Node'])
    Ele = np.array(model['Ele'])
    Mat = np.array(model['Mat'])
    Bc = np.array(model['Bound'])
    # Eload = np.array(model['Eload'])
    Nload = np.array(model['Nload'])
    Supp = np.array(model['Supp'])
    Spring = np.array(model['Spring'])
    
    # Xác định số nút và số phần tử của mô hình
    nnode = Coord.shape[0]
    nel = Ele.shape[0]
    
    # Kết nối phần tử
    sdof = ndof*nnode
    dof, edof = setdof(nnode, ndof, Ele[:,0:2]);
    ex, ey = cfc.coordxtr(edof, Coord[:,0:2], dof, 2)
    
    model['dof'] = dof
    model['edof'] = edof
    model['ex'] = ex
    model['ey'] = ey
    
    # Tạo danh sách các dof bị ngăn cản
    bc_prescr = bcond(ndof, dof, Bc)
    
    # Tạo danh sách các chuyển vị cưỡng bức 
    bcVal = bsupp(ndof, dof, bc_prescr, Supp)
    
    # Tạo danh sách các dof có spring
    bc_spr = bcond(ndof, dof, Spring)
    
    # Tạo danh sách các spring 
    sprVal = bsupp(ndof, dof, bc_spr, Spring)
    
    # FEM solution
    K = np.zeros([sdof,sdof])
    f = np.zeros([sdof,1])
    
    # Lập tải trọng trên phần tử
    # elf = eload(nel, Eload) 
    
    # Stiffness matrix K and force vector f
    i = 0;
    for elx, ely, eltopo in zip(ex, ey, edof):
        ep = Mat[Ele[i,2]-1]
        # eq = elf[i,:]
        Ke = cfc.bar2e(elx, ely, ep[0:2])
        cfc.assem(eltopo, K, Ke)
        
        i+=1

    # Thêm tải trọng tại nút 
    f = nload(ndof, dof, Nload, f) 
    
    # Thêm spring
    for i in range(len(bc_spr)): 
        cfc.assem(np.array([bc_spr[i]]), K, sprVal[i])

    # Solve equation system
    #a, r = cfc.solveq(K, f, bc_prescr, bcVal)
    
    # Chuyển sang ma trận thưa
    K = csr_matrix(K)
    # f = csr_matrix(f)
    
    a, r = cfc.spsolveq(K, f, bc_prescr, bcVal)
    
    # Reaction force
    res = np.array(r[0:ndof*nnode].reshape(nnode,ndof).copy())
    res = res[Bc[:,0]-1,:]
    
    # Node displacement
    ndis = np.array(a.reshape(nnode,ndof).copy())
    
    # Element Analysis
    nsec = 2
    ed= cfc.extract_eldisp(edof, a)
    es = np.zeros([edof.shape[0], nsec, 1])
    edi = np.zeros([edof.shape[0], nsec, 1])
    ec = np.zeros([edof.shape[0], nsec, 1])
    i = 0
    for elx, ely, eld in zip(ex, ey, ed):
        ep = Mat[Ele[i,2]-1]
        #eq = elf[i,:]
        es[i], edi[i], ec[i] = cfc.bar2s(elx, ely, ep[0:2], eld, nep=nsec)      
        i+=1
    
    model['disp'] = ndis
    model['force'] = es
    model['edisp'] = edi
    model['esect'] = ec
    model['res'] = res
    
    return model

# ------------------------------------------------------------------
# Solve 3D truss 
def Solve3Dtruss(model):
    
    # Nhận dữ liệu từ model
    ndof = model['ndof']
    nsec = model['nsec']
    Coord = np.array(model['Node'])
    Ele = np.array(model['Ele'])
    Mat = np.array(model['Mat'])
    Bc = np.array(model['Bound'])
    # Eload = np.array(model['Eload'])
    Nload = np.array(model['Nload'])
    Supp = np.array(model['Supp'])
    
    # Xác định số nút và số phần tử của mô hình
    nnode = Coord.shape[0]
    nel = Ele.shape[0]
    
    # Kết nối phần tử
    sdof = ndof*nnode
    dof, edof = setdof(nnode, ndof, Ele[:,0:2]);
    ex, ey, ez = cfc.coordxtr(edof, Coord, dof, 2)
    
    model['ex'] = ex
    model['ey'] = ey
    model['ez'] = ez
    
    # Tạo danh sách các dof bị ngăn cản
    bc_prescr = bcond(ndof, dof, Bc)
    
    # Tạo danh sách các chuyển vị cưỡng bức 
    bcVal = bsupp(ndof, dof, bc_prescr, Supp)
    
    # FEM solution
    K = np.zeros([sdof,sdof])
    f = np.zeros([sdof,1])
    
    # Lập tải trọng trên phần tử
    # elf = eload(nel, Eload) 
    
    # Stiffness matrix K and force vector f
    i = 0;
    for elx, ely, elz, eltopo in zip(ex, ey, ez, edof):
        ep = Mat[Ele[i,2]-1]
        # eq = elf[i,:]
        Ke = cfc.bar3e(elx, ely, elz, ep[0:2])
        cfc.assem(eltopo, K, Ke)
        
        i+=1

    # Thêm tải trọng tại nút 
    f = nload(ndof, dof, Nload, f) 
    
    # Solve equation system
    #a, r = cfc.solveq(K, f, bc_prescr, bcVal)
    
    # Chuyển sang ma trận thưa
    K = csr_matrix(K)
    #f = csr_matrix(f)
    
    a, r = cfc.spsolveq(K, f, bc_prescr, bcVal)
    
    # Reaction force
    res = np.array(r[0:ndof*nnode].reshape(nnode,ndof).copy())
    res = res[Bc[:,0]-1,:]
    
    # Node displacement
    ndis = np.array(a.reshape(nnode,ndof).copy())
    
    # Element Analysis
    nsec = 2
    ed= cfc.extract_eldisp(edof, a)
    es = np.zeros([edof.shape[0], nsec, 1])
    edi = np.zeros([edof.shape[0], nsec, 1])
    ec = np.zeros([edof.shape[0], nsec, 1])
    i = 0
    for elx, ely, elz, eld in zip(ex, ey, ez, ed):
        ep = Mat[Ele[i,2]-1]
        #eq = elf[i,:]
        es[i], edi[i], ec[i] = cfc.bar3s(elx, ely, elz, ep[0:2], eld, nep=nsec)      
        i+=1
    
    model['disp'] = ndis
    model['force'] = es
    model['edisp'] = edi
    model['esect'] = ec
    model['res'] = res
    
    return model

# ------------------------------------------------------------------
# Set dof, edof
def setdof(nnode, ndof, ENODE):
    # nnode - số nút
    # ndof - số bậc tự do 1 nút
    # ENODE - ma trận kết nối phần tử
    
    nel, nnel = ENODE.shape  # nel: số phần tử, nnel: số nút phần tử

    # DOF = np.zeros((nnode, ndof), dtype=int)
    # id = 1
    # for i1 in range(nnode):
    #     for j1 in range(ndof):
    #         DOF[i1, j1] = id + j1
    #     id += ndof
    
    # Vector hóa tạo DOF
    DOF = np.arange(1, nnode * ndof + 1).reshape(nnode, ndof)
    
    # # EDOF = np.zeros((nel, nnel*ndof+1), dtype=int)
    # EDOF = np.zeros((nel, nnel*ndof), dtype=int)
    # for i2 in range(nel):
    #     ide = DOF[ENODE[i2, :].astype(int) - 1, :]
    #     ide = ide.flatten()
    #     # EDOF[i2, :] = np.concatenate(([i2+1], id))
    #     EDOF[i2, :] = ide
    
    # Vector hóa tạo EDOF
    ENODE = ENODE.astype(int) - 1  # Chuyển về 0-based indexing
    nodes = ENODE[:, :nnel].flatten()
    node_indices = np.repeat(nodes, ndof)
    dof_indices = np.tile(np.arange(ndof), len(nodes))
    EDOF = DOF[node_indices, dof_indices].reshape(nel, nnel * ndof)

    return DOF, EDOF

# Set bound condition
def bcond(ndof, DOF, BC):
    # ndof - số bậc tự do 1 nút
    # BC - ma trận điều kiện liên kết
    
    # n = len(BC)

    # BCON = []
    # for i in range(n):
    #     node = int(BC[i,0])
    #     for j in range(ndof):
    #         if BC[i,j+1] != 0:
    #             dof = DOF[node-1,j]
    #             BCON.append(dof)
    # BCON = np.array(BCON)
    
    if len(BC) == 0:
        return np.array([])
    BC = BC.astype(int)
    nodes = BC[:, 0] - 1  # 0-based indexing
    flags = BC[:, 1:] != 0
    valid_nodes, valid_dofs = np.where(flags)
    BCON = DOF[nodes[valid_nodes], valid_dofs] if valid_nodes.size > 0 else np.array([])
    
    # print(BCON)
    return BCON

# Set bound movement value
def bsupp(ndof, DOF, BDOF, SUPP):
    # ndof - số bậc tự do 1 nút
    # BDOF - danh sách các dof bị ngăn cản
    
    n = len(SUPP)

    BSUP = 0.0*BDOF
    for i in range(n):
        node = int(SUPP[i,0])
        for j in range(ndof):
            if SUPP[i,j+1] != 0:
                dof = DOF[node-1,j]
                BSUP[BDOF==dof] = SUPP[i,j+1]

    return BSUP

# Include nodal load
def nload(ndof, DOF, NLOAD, F):
    # ndof - số bậc tự do 1 nút

    n = len(NLOAD)
    
    for i in range(n):
        node = int(NLOAD[i,0])
        for j in range(ndof):
            if NLOAD[i,j+1] != 0:
                dof = DOF[node-1,j]
                load = NLOAD[i,j+1]
                F[dof-1,0] = F[dof-1,0] + load
    return F

# Set element loads
def eload(nel, ELOAD, Ele=[], Mat=[], Coord=[]):
    # nel - số bậc phần tử
    n = len(ELOAD)
    
    ELD = np.zeros((nel, 2), dtype=float)
    for i in range(n):
        ele = int(ELOAD[i,0])
        ELD[ele-1,:] = ELOAD[i,1:3]
        
    # self weight
    noe = len(Ele)
    for i in range(noe):
        N1 = int(Ele[i,0])
        N2 = int(Ele[i,1])
        x1,y1 = Coord[N1-1,0:2]
        x2,y2 = Coord[N2-1,0:2]
        dx,dy = x2-x1, y2-y1
        Li = (dx**2+dy**2)**0.5
        if len(Mat[int(Ele[i,2])-1]) > 3:
            ro = Mat[int(Ele[i,2])-1,3]
            Ai = Mat[int(Ele[i,2])-1,1]
            q0 = ro*Ai
            qx = -q0*dy/Li
            qy = -q0*dx/Li
        
            ELD[i,0] = ELD[i,0] + qx
            ELD[i,1] = ELD[i,1] + qy
        
    return ELD
    
# ------------------------------------------------------------------
def disp_ematrix(model,elist=None,coord='Global'):
    str_type = model['type']
    Ele = np.array(model['Ele'])
    Mat = np.array(model['Mat'])
    Eload = np.array(model['Eload'])
    ex = model['ex']
    ey = model['ey']
    
    elf = eload(len(Ele), Eload) 
    
    if elist == None:
        elist = np.arange(len(Ele)) + 1
        
    for eid in elist:
        print('Element',eid)
        ep = Mat[int(Ele[eid-1,2])-1,0:3]
        eq = elf[eid-1]
        elx = ex[eid-1]
        ely = ey[eid-1]
        
        le = ((elx[1]-elx[0])**2 + (ely[1]-ely[0])**2)**0.5
        if coord=='Local':
            elx = [0.,le]
            ely = [0.,0.]
        
        if str_type=='2Dframe':
            Ke, fe = cfc.beam2e(elx, ely, ep, eq)
            cfu.disp_array(Ke)
            cfu.disp_array(fe)
        if str_type=='2Dtruss':
            Ke = cfc.bar2e(elx, ely, ep[0:2])
            cfu.disp_array(Ke)
    
def disp_react(model):
    res = model['res']
    bc = np.array(model['Bound'])
    nlist = bc[:,0]

    str_type = model['type']
    if str_type == '2Dframe':
        d_name = ["Rx", "Ry", "M"]
    else:
        d_name = ["Rx", "Ry", "Rz"]
        
    for i in range(len(nlist)):
        print('Node',nlist[i])   
        cfu.disp_array([res[i]], d_name)
    
# Show nodal displacement
def disp_ndisp(model,nlist=None):
    ndis = model['disp']
    str_type = model['type']
    if str_type == '2Dframe':
        d_name = ["dx", "dy", "fi"]
    else:
        d_name = ["dx", "dy", "dz"]
    if nlist == None:
        nlist = np.arange(len(ndis)) + 1
    for nid in nlist:
        print('Node',nid)
        
        cfu.disp_array([ndis[nid-1]], d_name)

# Show element force
def disp_eforce(model,elist=None):
    es = model['force']
    if elist == None:
        elist = np.arange(len(es)) + 1
    for eid in elist:
        print('Element',eid)
        nsec = len(es[eid-1])
        cfu.disp_array(es[eid-1,[0,nsec-1]], ["N", "Q", "M"])

# ===== Graphical function =========================================        
# Plot option

ele_color = 'b'
sup_color = 'c'
dia_color = 'r'
p_load_color = 'm'
u_load_color = 'g'
ele_name = 'k'
node_name = 'k'
text_size = 12
text_color = 'r'
dscale = 1

f_unit = 'kN'
u_unit = 'kN/m'
m_unit = 'kNm'
dim_unit = 'm'
nota_p = None
nota_u = None
nota_m = None

# Draw supports
def dbound(model):
    Coord = np.array(model['Node'])
    Bc = np.array(model['Bound'])
    Sp = np.array(model['Spring'])
    
    nb, nc = Bc.shape
    max_size_x = np.max(Coord[:,0])-np.min(Coord[:,0])
    max_size_y = np.max(Coord[:,1])-np.min(Coord[:,1])
    max_size = dscale*max(max_size_x,max_size_y)
    
    support_size = max_size/20
    for i in range(nb):
        X1 = Coord[Bc[i, 0]-1, 0]
        Y1 = Coord[Bc[i, 0]-1, 1]
        #txt = cfv.text(f"{Bc[i,1:nc]}", [X1, Y1])
        #txt.set_color("k")
        if f"{Bc[i,1:nc]}" == '[1 1 1]':
            draw_fix([X1, Y1],support_size)
        if f"{Bc[i,1:nc]}" == '[0 1 1]':
            draw_fix([X1, Y1],support_size)
            draw_roller([X1, Y1+support_size/2],support_size)
        if f"{Bc[i,1:nc]}" == '[1 0 1]':
            draw_fix([X1, Y1],support_size)
            draw_roller([X1-support_size/2, Y1],support_size,'hor')
        if f"{Bc[i,1:nc]}" == '[1 1 0]':
            draw_hinge([X1, Y1],support_size)
        if f"{Bc[i,1:nc]}" == '[0 1 0]':
            draw_hinge([X1, Y1],support_size)
            draw_roller([X1, Y1],support_size)
        if f"{Bc[i,1:nc]}" == '[1 0 0]':
            draw_hinge([X1, Y1],support_size,'hor')
            draw_roller([X1, Y1],support_size,'hor')
        if f"{Bc[i,1:nc]}" == '[0 0 1]':
            draw_moment([X1, Y1],support_size)
            
    nb = len(Sp)   
    for i in range(nb):
        X1 = Coord[int(Sp[i, 0])-1, 0]
        Y1 = Coord[int(Sp[i, 0])-1, 1]
        
        if Sp[i,2] != 0:
            draw_spring([X1, Y1],[X1, Y1-support_size-support_size/3],
                        color=sup_color)
            draw_roller([X1, Y1],support_size)
        if Sp[i,1] != 0:
            draw_spring([X1, Y1],[X1+support_size+support_size/3, Y1],
                        color=sup_color)
            draw_roller([X1, Y1],support_size,'hor')

def draw_fix(node,size):
    x = node[0]-size/2; y = node[1]-size/2
    rect = plt.Polygon([[x,y],[x+size,y],[x+size,y+size],[x,y+size]], color=sup_color)
    plt.gca().add_patch(rect)
    
def draw_moment(node,size):
    x = node[0]-size/2; y = node[1]-size/2
    rect = plt.Polygon([[x,y],[x+size,y],[x+size,y+size],[x,y+size]], 
                       edgecolor=sup_color, facecolor='w',zorder=2)
    plt.gca().add_patch(rect)
    
def draw_hinge(node,size,d=None):
    x = node[0]; y = node[1]
    
    xn = [x, x-size/2, x+size/2]
    yn = [y, y-size, y-size]
    
    if d=='hor':
        xn = [x, x+size, x+size]
        yn = [y, y-size/2, y+size/2]
    
    t2 = plt.Polygon([[xn[0],yn[0]],[xn[1],yn[1]],[xn[2],yn[2]]], color=sup_color)
    plt.gca().add_patch(t2)
    
def draw_roller(node,size,d=None):
    # draw_hinge(node,size,d)
    x = node[0]; y = node[1]
    
    xn = [x-size/2, x+size/2]
    yn = [y-size-size/3, y-size-size/3]
    
    if d=='hor':
        xn = [x+size+size/3, x+size+size/3]
        yn = [y-size/2, y+size/2]
        
    plt.plot(xn, yn, color=sup_color,linewidth=2)

def draw_spring(node1, node2, num_coils=3, size=None, color='k',k=None):

    x1,y1 = node1
    x2,y2 = node2
    
    dx = x2-x1
    dy = y2-y1
    L = (dx**2+dy**2)**(0.5)
    cx = dx/L;
    cy = dy/L
    
    x = np.linspace(0, num_coils * 2 * np.pi, 100)  
    
    if size == None:
        size = L/5;
    y = size*np.sin(x)
    
    x = x/max(x)*L

    xg = []
    yg = []
    for xi, yi in zip(x, y):
        xgi = xi*cx - yi*cy
        ygi = xi*cy + yi*cx
        xg.append(xgi)
        yg.append(ygi)
        
    x = np.array(xg) + x1
    y = np.array(yg) + y1

    plt.plot(x, y, color=color, linewidth=2)  
    
    if k != None:
        txt = cfv.text(k, [(x1+x2)/2+size, (y1+y2)/2+size], 
                         size=text_size,rotation=0,rotation_mode='anchor')
        txt.set_color(ele_name)
        
    #plt.scatter([x1,x2], [y1,y2],
    #            edgecolor=color, c=(0.8,0.8,0.8), marker='o',
    #            )
    #plt.axis('equal')  
    #plt.axis('off')
# ------------------------------------------------------------------    
# Draw name of nodes
def dnode(model, ID=None):
    
    Coord = np.array(model['Node'])
    ncoord = np.size(Coord,1)
    
    if ID is None:
        nlist = np.arange(len(Coord[:, 0])) + 1
    else:
        nlist = ID    
    for i in nlist:
        X1 = Coord[i-1, 0]
        Y1 = Coord[i-1, 1]
        if ncoord > 2:
            Z1 = Coord[i-1, 2]
        else:
            Z1 = 0
        
        txt = cfv.text(f"{i}", [X1, Y1], size=text_size) 
        txt.set_color(node_name)
    
# Draw name of elements 
def dename(model, ntype, ID=None):
    
    Coord = np.array(model['Node'])
    Ele0 = np.array(model['Ele'])
    Ele = Ele0[:,0:3].astype(int)
    if ID is None:
        elist = np.arange(len(Ele[:, 0])) + 1
    else:
        elist = ID
    
    for i in elist:
        X1 = Coord[Ele[i-1, 0]-1, 0]
        Y1 = Coord[Ele[i-1, 0]-1, 1]
        #Z1 = Node[Ele[i-1, 0]-1, 2]
        X2 = Coord[Ele[i-1, 1]-1, 0]
        Y2 = Coord[Ele[i-1, 1]-1, 1]
        #Z2 = Node[Ele[i-1, 1]-1, 2]
        
        dx = X2-X1; dy = Y2-Y1; di = (dx**2+dy**2)**0.5
        cs = dx/di; sn = dy/di
        angle = math.atan2(sn, cs)  # ALWAYS USE THIS
        angle *= 180 / math.pi
        if angle < 0: angle += 360
        
        if ntype == 'E':
            txt = cfv.text(f'({i})', [(X1+X2)/2, (Y1+Y2)/2], 
                           size=text_size, rotation=angle,rotation_mode='anchor')
            txt.set_color(ele_name)
        else:
            txt = cfv.text(f'{ntype}{Ele[i-1,2]}', [(X1+X2)/2, (Y1+Y2)/2], 
                           size=text_size, rotation=angle,rotation_mode='anchor')
            txt.set_color(ele_name)
            
        # if Ele[i-1, 3] == 0:  
        #     ring = patches.Wedge([X1,Y1], 0.1, -60, 60, width=0)
        #     p = PatchCollection(
        #         [ring], 
        #         edgecolor = 'k', 
        #     )
        #     plt.gca().add_collection(p)
        
# ------------------------------------------------------------------
# Draw arrow
def d_arrow(N1, N2, size, thk, cl):
    x1 = N1[0]; y1 = N1[1]
    x2 = N2[0]; y2 = N2[1]
    dx = -x1 + x2
    dy = -y1 + y2
    
    plt.arrow(x1,y1,dx,dy,width=thk,head_width=size/2,head_length=size,
              length_includes_head=True,color=cl,zorder=2)
    
# Draw point load    
def d_pload(node, load, scale, arrow_size, unit, pos=None, size=True, ha='center', nota=nota_p):
    
    if pos is None:
        fac = scale
    else:
        fac = scale*pos
    
    if size == True:    
        Fx = load[0] * fac
        Fy = load[1] * fac
    else:
        Fx = fac
        Fy = fac
    
    X0 = node[0]
    Y0 = node[1]
    X = [X0 - Fx, X0, X0]
    Y = [Y0, Y0 - Fy, Y0]
 
    for i in range(2):
        if pos is None:
            xv2 = X0; yv2 = Y0
            xv1 = X[i]; yv1 = Y[i]
            xv3 = xv1; yv3 = yv1
        else:
            xv1 = X0; yv1 = Y0
            xv2 = X[i]; yv2 = Y[i]
            xv3 = xv2; yv3 = yv2
            
        if load[i] != 0:
            d_arrow([xv1,yv1],[xv2,yv2],arrow_size,0,p_load_color)
            value = round(abs(load[i]),2)
            txt = f"{value}{unit}"
            #plt.text(xv1, yv1, f"{load[i]}{unit}", fontsize=10, ha='center') 
            if nota != None:
                txt = nota
            plt.text(xv3, yv3, 
                     txt, ha=ha, color=text_color,
                     fontsize=text_size, zorder=2)
            
# Draw distributed load
def d_uload(N1, N2, load, scale, size, unit, nota=nota_u, coord='local'): 
    qx, qy = load
    
    X1, Y1 = N1[0], N1[1]
    X2, Y2 = N2[0], N2[1]
    
    dX = X2 - X1
    dY = Y2 - Y1
    L = np.sqrt(dX**2 + dY**2)
    
    if qy != 0:
        v1 = qy * scale
        v2 = qy * scale
        
        xv1 = X1 + v1 / L * dY
        yv1 = Y1 - v1 / L * dX
        xv2 = X2 + v2 / L * dY
        yv2 = Y2 - v2 / L * dX
        
        if coord=='global':
            xv1 = X1
            xv2 = X2
        
        xv3 = xv1 + dX/3; yv3 = yv1 + dY/3
        xv4 = xv3 + dX/3; yv4 = yv3 + dY/3
        
        d_arrow([xv1, yv1], [X1, Y1], size, 0, u_load_color)
        d_arrow([xv2, yv2], [X2, Y2], size, 0, u_load_color)
        d_arrow([xv3, yv3], [X1+dX/3, Y1+dY/3], size, 0, u_load_color)
        d_arrow([xv4, yv4], [X1+dX/3*2, Y1+dY/3*2], size, 0, u_load_color)
        
        #plt.text(xv1, yv1, f'{qy}{unit}', color='r', ha='center')
        #plt.text(xv2, yv2, f'{qy}{unit}', color='r', ha='center')
        value = abs(qy)
        txt = f"{value}{unit}"
        if nota != None:
            txt = nota
        plt.text((xv1+xv2)/2, (yv1+yv2)/2, 
                 txt, color=text_color, ha='center', 
                 size=text_size,zorder=2)
        
        plt.plot([xv1, xv2], [yv1, yv2], u_load_color)           

# Draw moment load  
def d_mload(node, load, scale, size, unit, nota=nota_m): 
    
    if load !=0:
        Fz=0.5*abs(load)/load*scale;
    
    center =(node[0],node[1])
    radius = Fz

    # Add the ring
    rwidth = 0.0
    ring = patches.Wedge(node, abs(radius), 0, 180, width=rwidth)
    
    # Triangle edges
    offset = size/4
    xcent  = center[0] - radius + (rwidth/2)
    left   = [xcent - offset, center[1]]
    right  = [xcent + offset, center[1]]
    bottom = [(left[0]+right[0])/2., center[1]-size]
    arrow  = plt.Polygon([left, right, bottom, left])    

    # Add the arrow to the plot
    p = PatchCollection(
        [ring, arrow], 
        edgecolor = p_load_color, 
        facecolor = p_load_color
    )
    plt.gca().add_collection(p)
    value = round(abs(load),2)
    txt = f"{value}{unit}"
    if nota != None:
        txt = nota
    plt.text(bottom[0],bottom[1], txt, 
             ha='center', va='top', color=text_color, size=text_size, zorder=2)
    
# Draw nodal loads
def dnload(model,f_unit=f_unit,m_unit=m_unit,pos=None):
    Coord = np.array(model['Node'])
    Nload = np.array(model['Nload'])

    max_size_x = np.max(Coord[:,0])-np.min(Coord[:,0])
    max_size_y = np.max(Coord[:,1])-np.min(Coord[:,1])
    max_size = dscale*max(max_size_x,max_size_y)

    max_Nload = 0;
    if Nload is not None and len(Nload) > 0:
        max_Nload = np.max(abs(Nload[:, 1:3]))
        scale = 0.20*max_size/max_Nload;
        arrow_size = max_size/30
    
    for i in range(Nload.shape[0]):
        node = Coord[int(Nload[i,0])-1,:]
        load = Nload[i,1:4]
        d_pload(node, load, scale, arrow_size, f_unit, pos)
        if len(load) > 2 and abs(load[2]) > 1e-9:
            d_mload(node, load[2], max_size/10, arrow_size, m_unit)
        
# Draw element loads
def deload(model,unit=u_unit):
    Coord = np.array(model['Node'])
    Eload = np.array(model['Eload'])
    Ele = np.array(model['Ele'])
    
    max_size_x = np.max(Coord[:,0])-np.min(Coord[:,0])
    max_size_y = np.max(Coord[:,1])-np.min(Coord[:,1])
    max_size = dscale*max(max_size_x,max_size_y)

    max_Eload = 0;
    if Eload is not None and len(Eload) > 0:
        max_Eload = np.max(abs(Eload[:, 1:3]))        
        scale = 0.10*max_size/max_Eload;
        arrow_size = max_size/40 
    
    for i in range(Eload.shape[0]):
        ele = int(Eload[i,0]);
        node1 = Coord[int(Ele[ele-1,0])-1,:];
        node2 = Coord[int(Ele[ele-1,1])-1,:];
        load = Eload[i,1:3]
        d_uload(node1, node2, load, scale, arrow_size, unit)

# ------------------------------------------------------------------
import math
def annotate_dim(ax,xyfrom,xyto,text=None):

    dx = xyto[0]-xyfrom[0]
    dy = xyto[1]-xyfrom[1]
    di = np.sqrt( dx**2 + dy**2 )
    di = round(di,3)
    
    if text is None:
        #text = str(di)
        text = f'{di}{dim_unit}'
    
    cs = dx/di
    sn = dy/di
    angle = math.atan2(sn, cs)  # ALWAYS USE THIS
    angle *= 180 / math.pi
    if angle < 0: angle += 360

    ax.annotate("",xyfrom,xyto,arrowprops=dict(arrowstyle='<->'))
    ax.text((xyto[0]+xyfrom[0])/2,(xyto[1]+xyfrom[1])/2,text,
            fontsize=text_size,ha='center',rotation=angle,rotation_mode='anchor')
   
def draw_dim(N1, N2, offset=0, text=None):
    X1, Y1 = N1[0], N1[1]
    X2, Y2 = N2[0], N2[1]
    
    if len(N1)>2:
        Z1 = N1[2]
        Z2 = N2[2]
    else:
        Z1 = 0
        Z2 = 0
    
    dX, dY, dZ = X2 - X1, Y2 - Y1, Z2 - Z1
    L = (dX**2 + dY**2 + dZ**2)**0.5
    v1, v2 = offset, offset
    
    xv1, yv1 = X1 - v1/L*dY, Y1 + v1/L*dX + v1/L*dZ
    xv2, yv2 = X2 - v2/L*dY, Y2 + v2/L*dX + v2/L*dZ
    
    # Draw dimension line
    plt.plot([X1, xv1], [Y1, yv1], [Z1, Z1], color='k', linewidth=0.5)
    plt.plot([X2, xv2], [Y2, yv2], [Z2, Z2], color='k', linewidth=0.5)
    
    annotate_dim(plt.gca(),[xv1, yv1],[xv2, yv2],text)
# ------------------------------------------------------------------
# Draw element
def dframe2(model,ID=None):
    str_type = model['type']
    ndof = model['ndof']
    Coord = np.array(model['Node'])
    Ele = np.array(model['Ele'])
    
    if ID is None:
        elist = np.arange(len(Ele[:, 0])) + 1
    else:
        elist = ID

    # Xác định số nút 
    nnode = Coord.shape[0]
    
    # Kết nối phần tử
    dof, edof = setdof(nnode, ndof, Ele[:,0:2]);
    ex, ey = cfc.coordxtr(edof, Coord, dof, 2)
    
    cfv.figure()
    for i in elist:
        plt.plot(ex[i-1], ey[i-1], color=ele_color, linewidth=2)
        
        # draw hinge
        if model['type'] == '2Dframe':
            if Ele[i-1, 3] < 1:
                dx = ex[i-1][1]-ex[i-1][0]
                dy = ey[i-1][1]-ey[i-1][0]
                le = (dx**2+dy**2)**0.5
                di = le/30
                dxi = dx/le*di
                dyi = dy/le*di
                if Ele[i-1, 3] == 0:
                    color = 'w'
                else:
                    color = sup_color
                plt.scatter(ex[i-1][0]+dxi, ey[i-1][0]+dyi, s=50,
                            edgecolor=ele_color, color=color,
                            zorder=2)
            if Ele[i-1, 4] < 1:
                dx = ex[i-1][1]-ex[i-1][0]
                dy = ey[i-1][1]-ey[i-1][0]
                le = (dx**2+dy**2)**0.5
                di = 29*le/30
                dxi = dx/le*di
                dyi = dy/le*di
                if Ele[i-1, 4] == 0:
                    color = 'w'
                else:
                    color = sup_color
                plt.scatter(ex[i-1][0]+dxi, ey[i-1][0]+dyi, s=50,
                            edgecolor=ele_color, color=color,
                            zorder=2)
        
    if model['type'] == '2Dtruss':
        plt.scatter(ex, ey, s=50,
                    edgecolor=node_name, color='w',
                    zorder=2)
        
# ------------------------------------------------------------------
# Draw 2D frame geometry
def show_geometry(model,etype='E',ID=None,show_node=False,show_ele=False,pos=None,title=None):
        
    #plotpar = [1, 2, 0]; cfv.eldraw2(ex, ey, plotpar)
    #if str_type == '2Dtruss':
    #    cfv.draw_node_circles(ex, ey, color='k', filled=True, marker_type='o')
    
    dframe2(model,ID)
    if model['Bound'] != []:
        dbound(model)
    
    if show_node == True:
        dnode(model)
    if show_ele == True:
        dename(model, etype, ID)
        
    dnload(model, f_unit, m_unit, pos=pos)
    #dnload(model, f_unit, m_unit)
    deload(model, u_unit)
    
    plt.axis("equal")
    if title==None:
        plt.title("Geometry")
    else:
        plt.title(title)
    plt.axis('off')

# Draw 3D frame geometry
def show_geometry3D(model,etype='E', ID=None, node=None, ax=None):
    ndof = model['ndof']
    Coord = np.array(model['Node'])
    Ele = np.array(model['Ele'])
    
    if ID is None:
        elist = np.arange(len(Ele[:, 0])) + 1
    else:
        elist = ID

    # Xác định số nút 
    nnode = Coord.shape[0]
    
    # Kết nối phần tử
    dof, edof = setdof(nnode, ndof, Ele[:,0:2]);
    ex, ey, ez = cfc.coordxtr(edof, Coord, dof, 2)
    
    # Vẽ thanh    
    if not ax:
        ax = plt.gca()
    else:
        ax = cfv.figure().add_subplot(projection='3d')
    
    for i in elist:
        if etype=='E':
            ename=i
        if etype=='M':
            ename=Ele[i-1,2]
        ax.plot(ex[i-1], ey[i-1], ez[i-1], color=ele_color, linewidth=2)
        if etype != None:
            ax.text(np.mean(ex[i-1]), np.mean(ey[i-1]), np.mean(ez[i-1]), 
                str(ename), color=ele_name)
    
    # Ghi tên nút
    if node !=None:
        for i in range(nnode):
            X1 = Coord[i, 0]; Y1 = Coord[i, 1]; Z1 = Coord[i, 2]
            ax.text(X1, Y1, Z1, str(i+1), verticalalignment='top', color=node_name)
    
    # dbound3D(model)
    # dnload(model,f_unit)
    
    plt.axis("equal")
    plt.title("Geometry")
    plt.axis('off')
# ------------------------------------------------------------------
# Draw FE model
def show_FEM(model,scale='Auto',show_node=False,beam=False):
    
    dframe2(model)
    dename(model,'E')
    if show_node == True:
        dnode(model)
    
    ex = model['ex']
    ey = model['ey']
    plt.scatter(ex, ey, s=50,
                #edgecolor=node_name, 
                color=ele_color,
                zorder=2)
    
    Coord = np.array(model['Node'])
    Ele = np.array(model['Ele'])
    ndof = model['ndof']
    nnode = len(Coord)
    dof, edof = setdof(nnode, ndof, Ele[:,0:2]);
    
    max_size_x = np.max(Coord[:,0])-np.min(Coord[:,0])
    max_size_y = np.max(Coord[:,1])-np.min(Coord[:,1])
    max_size = dscale*max(max_size_x,max_size_y)

    max_Nload = 1;
    scale = 0.20*max_size/max_Nload;
    arrow_size = max_size/30
    
    for i in range(nnode):
        node = Coord[i,:]
        load = dof[i,:]
        
        if beam==True:
            load = np.array([0,i*2+1,i*2+2])
        d_pload(node, load, scale, arrow_size, '', pos=-0.5, size=False, ha='left')
        if len(load)==3:
            d_mload(node, load[2], max_size/10, arrow_size, '')
    
    plt.axis("equal")
    plt.title("FE model")
    plt.axis('off')
# ------------------------------------------------------------------
# Draw reaction
def show_reaction(model,scale='Auto', title=None):
    res = model['res']
    Bc = np.array(model['Bound'])
    
    n, nr = res.shape
    b_res = np.zeros([n,nr+1])
    b_res[:,1:nr+1] = res
    b_res[:,0] = Bc[:,0]
    
    model1 = model.copy();
    model1['Nload'] = b_res   
    
    dframe2(model)
    # if model['Bound'] != []:
    #     dbound(model)

    dnload(model1)
    
    plt.axis("equal")
    if title==None:
        plt.title("Reaction")
    else:
        plt.title(title)
    plt.axis('off')
    plt.axis('off')

# ------------------------------------------------------------------    
# Draw deformed frame
def show_displacement(model,scale='Auto',title=None):
    str_type = model['type']
    ex = model['ex']
    ey = model['ey']
    ed = model['edisp']
    nel = len(ex)
    
    plotpar0 = [2, 2, 0]
    plotpar1 = [1, 4, 0]
    sfac = cfv.scalfact2(ex, ey, ed, 0.1)
    
    if scale !='Auto':
        sfac = scale
    
    ndof = model['ndof']
    Ele = np.array(model['Ele'])
    Coord = np.array(model['Node'])
    disp = model['disp']
    Coordn = Coord + sfac*disp[:,0:2]
    nnode = Coord.shape[0]
    dof, edof = setdof(nnode, ndof, Ele[:,0:2]);
    exn, eyn = cfc.coordxtr(edof, Coordn, dof, 2)

    cfv.figure()
    cfv.eldraw2(ex, ey, plotpar0)
    
    if str_type == '2Dtruss':
        cfv.eldraw2(exn, eyn, [1, 4, 1])
    if str_type == '2Dframe':
        for i in range(nel):
            cfv.dispbeam2(ex[i], ey[i], ed[i], plotpar1, sfac)
    if title==None:
        cfv.title("Displacement")
    else:
        cfv.title(title)    
    plt.axis('off')

# ------------------------------------------------------------------
# Draw element internal force
def secforce2(ex, ey, es, plotpar=[2, 1], sfac=None, eci=None):

    if ex.shape != ey.shape:
        raise ValueError("Check size of ex, ey dimensions.")
    
    es0 = es.copy()
    c = len(es)
    Nbr = c

    x1, x2 = ex
    y1, y2 = ey
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx * dx + dy * dy)
    nxX = dx / L
    nyX = dy / L
    n = np.array([nxX, nyX])

    if sfac is None:
        sfac = (0.2 * L) / max(abs(es))

    if eci is None:
        eci = np.arange(0.0, L + L / (Nbr - 1), L / (Nbr - 1)).reshape(Nbr, 1)

    eci = eci.flatten()  # ensure 1D for indexing
    
    p1 = plotpar[0]
    if p1 == 1:
        line_color = (0, 0, 0)
    elif p1 == 2:
        line_color = (0, 0, 1)
    elif p1 == 3:
        line_color = (1, 0, 1)
    elif p1 == 4:
        line_color = (1, 0, 0)
    else:
        raise ValueError("Invalid value for plotpar[1].")
    line_style = "solid"

    p2 = plotpar[1]
    if p2 == 1:
        line_color1 = (0, 0, 0)
    elif p2 == 2:
        line_color1 = (0, 0, 1)
    elif p2 == 3:
        line_color1 = (1, 0, 1)
    elif p2 == 4:
        line_color1 = (1, 0, 0)
    else:
        raise ValueError("Invalid value for plotpar[1].")

    a = len(eci)
    if a != c:
        raise ValueError("Check size of eci dimension.")

    es = es * sfac

    # From local x-coordinates to global coordinates of the element
    A = np.zeros(2 * Nbr).reshape(Nbr, 2)
    A[0, 0] = ex[0]
    A[0, 1] = ey[0]
    for i in range(Nbr):
        A[i, 0] = A[0, 0] + eci[i] * n[0]
        A[i, 1] = A[0, 1] + eci[i] * n[1]
    B = np.array(A)

    # Plot diagram
    for i in range(0, Nbr):
        A[i, 0] = A[i, 0] + es[i] * n[1]
        A[i, 1] = A[i, 1] - es[i] * n[0]

    xc = np.array(A[:, 0])
    yc = np.array(A[:, 1])

    plt.plot(xc, yc, color=line_color, linewidth=1)
    
    value1 = round(es0[0],2)
    value2 = round(es0[Nbr-1],2)
    
    if abs(value1)>1e-6:
        plt.text(xc[0],yc[0], f"{value1}", 
             ha='center', color='k', fontsize=text_size)
    if abs(value2)>1e-6:
        plt.text(xc[Nbr-1],yc[Nbr-1], f"{value2}", 
             ha='center', color='k', fontsize=text_size)
    
    # Plot stripes in diagram
    xs = np.zeros(2)
    ys = np.zeros(2)
    for i in range(Nbr):
        xs[0] = B[i, 0]
        xs[1] = A[i, 0]
        ys[0] = B[i, 1]
        ys[1] = A[i, 1]
        plt.plot(xs, ys, color=line_color, linewidth=1)

    # Plot element
    plt.plot(ex, ey, color=line_color1, linewidth=2)
    
# Draw Axial force diagram
def show_axial(model,scale='Auto',title=None):
    ex = model['ex']
    ey = model['ey']
    es = model['force']
    nel = len(ex)
    
    plotpar = [4, 2]
    sfac = cfv.scalfact2(ex, ey, es[:,:, 0], 0.2)
    
    if scale !='Auto':
        sfac = scale

    cfv.figure()
    plt.axis("equal")
    for i in range(nel):
        secforce2(ex[i], ey[i], es[i,:, 0], plotpar, -sfac)
    if title==None:
        cfv.title("Axial force")
    else:
        cfv.title(title)
    plt.axis('off')

# Draw shear force diagram
def show_shear(model,scale='Auto',title=None):
    ex = model['ex']
    ey = model['ey']
    es = model['force']
    nel = len(ex)
    
    plotpar = [4, 2]
    sfac = cfv.scalfact2(ex, ey, es[:,:, 1], 0.2)
    
    if scale !='Auto':
        sfac = scale

    cfv.figure()
    plt.axis("equal")
    for i in range(nel):
        secforce2(ex[i], ey[i], es[i,:, 1], plotpar, -sfac)
    if title==None:
        cfv.title("Shear force")
    else:
        cfv.title(title)
    plt.axis('off')
        
# Draw moment diagram
def show_moment(model,scale='Auto',title=None):
    ex = model['ex']
    ey = model['ey']
    es = model['force']
    nel = len(ex)
    
    plotpar = [4, 2]
    sfac = cfv.scalfact2(ex, ey, es[:,:, 2], 0.2)
    
    if scale !='Auto':
        sfac = scale

    cfv.figure()
    plt.axis("equal")
    for i in range(nel):
        secforce2(ex[i], ey[i], es[i,:, 2], plotpar, sfac)
    if title==None:
        cfv.title("Bending moment")
    else:
        cfv.title(title)
    plt.axis('off')
