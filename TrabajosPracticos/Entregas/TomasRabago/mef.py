import numpy as np

def solve(K, r, Fr, s, Us):

    N = np.shape(K)[1]
    F = np.zeros([N, 1])
    U = np.zeros([N, 1])
    U[s] = Us
    F[r] = Fr
    Kr = K[np.ix_(r, r)]
    Kv = K[np.ix_(r, s)]
    U[r] = np.linalg.solve(Kr, F[r]-Kv.dot(U[s]))
    F[s] = K[s, :].dot(U)
    return F, U


def Kelemental(MN, MC, Ee, Ae, e):

    Lx = MN[MC[e, 1], 0]-MN[MC[e, 0], 0]
    Ly = MN[MC[e, 1], 1]-MN[MC[e, 0], 1]
    L = np.sqrt(Lx**2+Ly**2)
    phi = np.arctan2(Ly, Lx)
    cos = np.cos(phi)
    sin = np.sin(phi)
    Ke = (Ee*Ae/L)*np.array([[cos**2, cos*sin, -cos**2, -cos*sin],
                             [cos*sin, sin**2, -cos*sin, -sin**2],
                             [-cos**2, -cos*sin, cos**2, cos*sin],
                             [-cos*sin, -sin**2, cos*sin, sin**2]])
    Ke[np.abs(Ke/Ke.max()) < 1E-15] = 0
    return Ke


def Kglobal(MN, MC, E, A, glxn):

    Nn = MN.shape[0] # cantidad de nodos
    Ne, Nnxe = MC.shape # "Ne", numero de filas de MC. "Nnxe", numero de columnas de MC
    Kg = np.zeros([glxn*Nn, glxn*Nn])
    for e in range(Ne):
        Ee = E[e]
        Ae = A[e]
        Ke = Kelemental(MN, MC, Ee, Ae, e)
        for i in range(Nnxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, Nnxe).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, Nnxe).astype(int)
            for j in range(Nnxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, Nnxe).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, Nnxe).astype(int)
                Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(rangoi, rangoj)]
    return Kg