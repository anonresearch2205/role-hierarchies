NAME 
ROWS
 N  OBJ
 E  c0      
 E  c1      
 E  c2      
 E  c3      
 E  e_0_0_u_0_p_0
 E  e_0_1_u_0_p_1
 E  u_0_p_0 
 E  u_0_p_1 
 E  e_1_0_u_1_p_0
 E  e_1_1_u_1_p_1
 E  u_1_p_1 
 E  no_u_1_p_0
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    bc_0      c0        -1
    bc_0      c1        -1
    bc_1      c2        -1
    bc_1      c3        -1
    u_0       OBJ       0
    u_1       OBJ       0
    p_0       OBJ       0
    p_1       OBJ       0
    e_0_0     c3        1
    e_0_0     e_0_0_u_0_p_0  1
    e_0_1     c0        1
    e_0_1     c2        1
    e_0_1     e_0_1_u_0_p_1  1
    e_1_0     OBJ       0
    e_1_0     e_1_0_u_1_p_0  1
    e_1_1     c1        1
    e_1_1     e_1_1_u_1_p_1  1
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      u_0_p_0   1
    RHS1      u_0_p_1   1
    RHS1      u_1_p_1   1
BOUNDS
 BV BND1      bc_0    
 BV BND1      bc_1    
 BV BND1      u_0     
 BV BND1      u_1     
 BV BND1      p_0     
 BV BND1      p_1     
 BV BND1      e_0_0   
 BV BND1      e_0_1   
 BV BND1      e_1_0   
 BV BND1      e_1_1   
QCMATRIX   e_0_0_u_0_p_0
    u_0       p_0       -0.5
    p_0       u_0       -0.5
QCMATRIX   e_0_1_u_0_p_1
    u_0       p_1       -0.5
    p_1       u_0       -0.5
QCMATRIX   u_0_p_0 
    u_0       p_0       0.5
    p_0       u_0       0.5
QCMATRIX   u_0_p_1 
    u_0       p_1       0.5
    p_1       u_0       0.5
QCMATRIX   e_1_0_u_1_p_0
    u_1       p_0       -0.5
    p_0       u_1       -0.5
QCMATRIX   e_1_1_u_1_p_1
    u_1       p_1       -0.5
    p_1       u_1       -0.5
QCMATRIX   u_1_p_1 
    u_1       p_1       0.5
    p_1       u_1       0.5
QCMATRIX   no_u_1_p_0
    u_1       p_0       0.5
    p_0       u_1       0.5
ENDATA
