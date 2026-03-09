
import numpy as np

class AmoebaFrameConverter:
    def __init__(self, coords):
        self.coords = np.array(coords)

    def get_rotation_matrix(self, i, j, k, l=None, mode="Z-then-X"):
        ri = self.coords[i]
        rj = self.coords[j]
        rk = self.coords[k]
        
        v_ij = rj - ri
        v_ij /= np.linalg.norm(v_ij)
        
        if mode == "Z-then-X" or mode == 0:
            z = v_ij
            v_ik = rk - ri
            v_ik /= np.linalg.norm(v_ik)
            x = v_ik - np.dot(v_ik, z) * z
            if np.linalg.norm(x) < 1e-6: # 共线退化处理
                # 寻找一个不共线的辅助向量
                for aux in [np.array([1,0,0]), np.array([0,1,0])]:
                    x = aux - np.dot(aux, z) * z
                    if np.linalg.norm(x) > 1e-6: break
            x /= np.linalg.norm(x)
            y = np.cross(z, x)

        elif mode == "Bisector" or mode == 1:
            v_ik = rk - ri
            v_ik /= np.linalg.norm(v_ik)
            z = v_ij + v_ik
            z /= np.linalg.norm(z)
            x = v_ij - np.dot(v_ij, z) * z
            x /= np.linalg.norm(x)
            y = np.cross(z, x)

        elif mode == "Z-Bisect" or mode == 2:
            z = v_ij
            v_ik = rk - ri
            v_ik /= np.linalg.norm(v_ik)
            v_il = self.coords[l] - ri
            v_il /= np.linalg.norm(v_il)
            bisect = v_ik + v_il
            if np.linalg.norm(bisect) < 1e-6: # 平分线退化
                bisect = np.cross(v_ik, v_ij) # 垂直于平面的方向
            bisect /= np.linalg.norm(bisect)
            x = bisect - np.dot(bisect, z) * z
            if np.linalg.norm(x) < 1e-6:
                # 再次退化处理
                x = v_ik - np.dot(v_ik, z) * z
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        R = np.vstack([x, y, z]).T
        return R.T

    def rotate_multipoles(self, i, j, k, l=None, mode="Z-then-X", dipole=None, quadrupole=None):
        R_inv = self.get_rotation_matrix(i, j, k, l, mode)
        d_local = R_inv @ np.array(dipole) if dipole is not None else None
        q_local = R_inv @ np.array(quadrupole) @ R_inv.T if quadrupole is not None else None
        return d_local, q_local

def cartesian_to_tinker_quadrupole(Q_matrix):
    # 转换为 Trace-less
    trace = np.trace(Q_matrix)
    Q = Q_matrix - (1.0/3.0) * trace * np.eye(3)
    return {'qXX': Q[0,0], 'qXY': Q[0,1], 'qYY': Q[1,1], 'qXZ': Q[0,2], 'qYZ': Q[1,2], 'qZZ': Q[2,2]}
