"""
MODULE <Module_2DFrame>
Phan tich he thanh 2D bang CALFEM
Pham Hoang Anh (2024)

"""

import numpy as np
import calfem.core as cfc
import calfem.utils as cfu
import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

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
        nsec = 11

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
    dof, edof = setdof(nnode, ndof, Ele[:,0:2]);
    ex, ey = cfc.coordxtr(edof, Coord, dof, 2)
    
    model['ex'] = ex
    model['ey'] = ey
    
    # Thêm dof do giải phóng mô men đầu thanh
    end_i = Ele[:,3]
    end_j = Ele[:,4]
    ni = sum(end_i==0); #print(ni)
    nj = sum(end_j==0); #print(nj)

    edof[end_i==0,2] = np.array([i+sdof+1 for i in range(ni)])
    edof[end_j==0,5] = np.array([i+sdof+ni+1 for i in range(nj)])
    sdof = sdof + ni + nj
    
    model['dof'] = dof
    model['edof'] = edof
    #print(edof)
    
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
    elf = eload(nel, Eload) 
    
    # Stiffness matrix K and force vector f
    i = 0; 
    for elx, ely, eltopo in zip(ex, ey, edof):
        ep = Mat[Ele[i,2]-1]
        eq = elf[i,:]
        Ke, fe = cfc.beam2e(elx, ely, ep, eq)
        cfc.assem(eltopo, K, Ke, f, fe)
        
        i+=1
    
    # Thêm tải trọng tại nút 
    f = nload(ndof, dof, Nload, f) 
    
    # Thêm spring
    for i in range(len(bc_spr)): 
        cfc.assem(np.array([bc_spr[i]]), K, sprVal[i])

    # Solve equation system
    a, r = cfc.solveq(K, f, bc_prescr, bcVal)

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
        ep = Mat[Ele[i,2]-1]
        eq = elf[i,:]
        es[i], edi[i], ec[i] = cfc.beam2s(elx, ely, ep, eld, eq, nsec)        
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
    ex, ey = cfc.coordxtr(edof, Coord, dof, 2)
    
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
    a, r = cfc.solveq(K, f, bc_prescr, bcVal)
    
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
    a, r = cfc.solveq(K, f, bc_prescr, bcVal)
    
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

    DOF = np.zeros((nnode, ndof), dtype=int)
    id = 1
    for i1 in range(nnode):
        for j1 in range(ndof):
            DOF[i1, j1] = id + j1
        id += ndof

    # EDOF = np.zeros((nel, nnel*ndof+1), dtype=int)
    EDOF = np.zeros((nel, nnel*ndof), dtype=int)
    for i2 in range(nel):
        id = DOF[ENODE[i2, :] - 1, :]
        id = id.flatten()
        # EDOF[i2, :] = np.concatenate(([i2+1], id))
        EDOF[i2, :] = id

    return DOF, EDOF

# Set bound condition
def bcond(ndof, DOF, BC):
    # ndof - số bậc tự do 1 nút
    # BC - ma trận điều kiện liên kết
    
    n = len(BC)

    BCON = []
    for i in range(n):
        node = int(BC[i,0])
        for j in range(ndof):
            if BC[i,j+1] != 0:
                dof = DOF[node-1,j]
                BCON.append(dof)
    BCON = np.array(BCON)
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
def eload(nel, ELOAD):
    # nel - số bậc phần tử
    n = len(ELOAD)
    
    ELD = np.zeros((nel, 2), dtype=float)
    for i in range(n):
        ele = int(ELOAD[i,0])
        ELD[ele-1,:] = ELOAD[i,1:3]
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
        ep = Mat[Ele[eid-1,2]-1]
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
dscale = 1

f_unit = ''
u_unit = ''
m_unit = ''
dim_unit = ''

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
        
    plt.plot(xn, yn, color=sup_color)

def draw_spring(node1, node2, num_coils=3, size=None, color='k'):

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

    plt.plot(x, y, color=color)  
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
    Ele = np.array(model['Ele'])
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
            txt = cfv.text(f'{i}', [(X1+X2)/2, (Y1+Y2)/2], 
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
def d_pload(node, load, scale, arrow_size, unit, pos=None, size=True, ha='center'):
    
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
            #plt.text(xv1, yv1, f"{load[i]}{unit}", fontsize=10, ha='center') 
            plt.text(xv3, yv3, 
                     f"{value}{unit}", ha=ha, color='r',
                     fontsize=text_size,zorder=2)
            
# Draw distributed load
def d_uload(N1, N2, load, scale, size, unit): 
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
        
        xv3 = xv1 + dX/3; yv3 = yv1 + dY/3
        xv4 = xv3 + dX/3; yv4 = yv3 + dY/3
        
        d_arrow([xv1, yv1], [X1, Y1], size, 0, u_load_color)
        d_arrow([xv2, yv2], [X2, Y2], size, 0, u_load_color)
        d_arrow([xv3, yv3], [X1+dX/3, Y1+dY/3], size, 0, u_load_color)
        d_arrow([xv4, yv4], [X1+dX/3*2, Y1+dY/3*2], size, 0, u_load_color)
        
        #plt.text(xv1, yv1, f'{qy}{unit}', color='r', ha='center')
        #plt.text(xv2, yv2, f'{qy}{unit}', color='r', ha='center')
        value = abs(qy)
        plt.text((xv1+xv2)/2, (yv1+yv2)/2, 
                 f'{value}{unit}', color='r', ha='center', 
                 size=text_size,zorder=2)
        
        plt.plot([xv1, xv2], [yv1, yv2], u_load_color)           

# Draw moment load  
def d_mload(node, load, scale, size, unit): 
    
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
    plt.text(bottom[0],bottom[1], f"{value}{unit}", 
             ha='center', va='top', color='r', size=text_size, zorder=2)
    
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
    X1, Y1, Z1 = N1
    X2, Y2, Z2 = N2
    
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
            if Ele[i-1, 3] == 0:
                dx = ex[i-1][1]-ex[i-1][0]
                dy = ey[i-1][1]-ey[i-1][0]
                le = (dx**2+dy**2)**0.5
                di = le/30
                dxi = dx/le*di
                dyi = dy/le*di
                plt.scatter(ex[i-1][0]+dxi, ey[i-1][0]+dyi, s=50,
                            edgecolor=node_name, color='w',
                            zorder=2)
            if Ele[i-1, 4] == 0:
                dx = ex[i-1][1]-ex[i-1][0]
                dy = ey[i-1][1]-ey[i-1][0]
                le = (dx**2+dy**2)**0.5
                di = 29*le/30
                dxi = dx/le*di
                dyi = dy/le*di
                plt.scatter(ex[i-1][0]+dxi, ey[i-1][0]+dyi, s=50,
                            edgecolor=node_name, color='w',
                            zorder=2)
        
    if model['type'] == '2Dtruss':
        plt.scatter(ex, ey, s=50,
                    edgecolor=node_name, color='w',
                    zorder=2)
        
# ------------------------------------------------------------------
# Draw 2D frame geometry
def show_geometry(model,etype='E',ID=None,show_node=False,show_ele=False):
        
    #plotpar = [1, 2, 0]; cfv.eldraw2(ex, ey, plotpar)
    #if str_type == '2Dtruss':
    #    cfv.draw_node_circles(ex, ey, color='k', filled=True, marker_type='o')
    
    dframe2(model,ID)
    dbound(model)
    
    if show_node == True:
        dnode(model)
    if show_ele == True:
        dename(model, etype, ID)
        
    #dnload(model, f_unit,pos=-0.5)
    dnload(model, f_unit, m_unit)
    deload(model, u_unit)
    
    plt.axis("equal")
    plt.title("Geometry")
    plt.axis('off')

# Draw 3D frame geometry
def show_geometry3D(model,etype='E',ID=None):
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
    ax = cfv.figure().add_subplot(projection='3d')
    for i in elist:
        if etype=='E':
            ename=i+1
        if etype=='M':
            ename=Ele[i-1,2]
        ax.plot(ex[i-1], ey[i-1], ez[i-1], color=ele_color, linewidth=2)
        ax.text(np.mean(ex[i-1]), np.mean(ey[i-1]), np.mean(ez[i-1]), 
                str(ename), color=ele_name)
    
    # Ghi tên nút
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
def show_FEM(model,scale='Auto',show_node=False):
    
    dframe2(model)
    dename(model,'E')
    if show_node == True:
        dnode(model)
    
    ex = model['ex']
    ey = model['ey']
    plt.scatter(ex, ey, s=50,
                #edgecolor=node_name, color='w',
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
        d_pload(node, load, scale, arrow_size, '', pos=-0.5, size=False, ha='left')
        if len(load)==3:
            d_mload(node, load[2], max_size/10, arrow_size, '')
    
    plt.axis("equal")
    plt.title("FE model")
    plt.axis('off')
# ------------------------------------------------------------------
# Draw reaction
def show_reaction(model,scale='Auto'):
    res = model['res']
    Bc = np.array(model['Bound'])
    Ele = np.array(model['Ele'])
    
    n, nr = res.shape
    b_res = np.zeros([n,nr+1])
    b_res[:,1:nr+1] = res
    b_res[:,0] = Bc[:,0]
    
    model1 = model.copy();
    model1['Nload'] = b_res
    
    ex = model1['ex']
    ey = model1['ey']    
    
    cfv.figure()
    plt.axis("equal")
    for i in range(np.size(Ele,0)):
        plt.plot(ex[i], ey[i], color=ele_color, linewidth=2)
    #plotpar = [1, 2, 0]
    #cfv.eldraw2(ex, ey, plotpar)
    dnload(model1)
    
    cfv.title("Reaction")
    plt.axis('off')

# ------------------------------------------------------------------    
# Draw deformed frame
def show_displacement(model,scale='Auto'):
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
        
    cfv.title("Displacement")
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
    plt.text(xc[0],yc[0], f"{value1}", 
             ha='center', color='k')
    plt.text(xc[Nbr-1],yc[Nbr-1], f"{value2}", 
             ha='center', color='k')
    
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
def show_axial(model,scale='Auto'):
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
    cfv.title("Axial force")
    plt.axis('off')

# Draw shear force diagram
def show_shear(model,scale='Auto'):
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
    cfv.title("Shear force")
    plt.axis('off')
        
# Draw moment diagram
def show_moment(model,scale='Auto'):
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
    cfv.title("Bending moment")
    plt.axis('off')
