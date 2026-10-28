from . import si_overconditioning
import numpy as np
from . import OptimalTransport
from . import si_utils
from scipy.optimize import linprog
from . import si_adapter
def find_valid_basis(H, x_nonzero_indices):
    """Tìm một tập chỉ số sao cho ma trận H[:, basis] là khả nghịch (Full Rank)"""
    num_constraints = H.shape[0]
    basis = list(x_nonzero_indices)
    
    # Sử dụng khử Gauss (hoặc QR) để tìm các cột độc lập tuyến tính bổ sung
    # Ở đây dùng phương pháp đơn giản: thử từng cột
    for i in range(H.shape[1]):
        if len(basis) == num_constraints:
            break
        if i not in basis:
            test_basis = basis + [i]
            if np.linalg.matrix_rank(H[:, test_basis]) == len(test_basis):
                basis.append(i)
    return sorted(basis)

def solve_quadratic_roots(a, b, c, current_z, z_max):
    """Tìm nghiệm thực nhỏ nhất của az^2 + bz + c = 0 trong khoảng (current_z, z_max]"""
    eps = 1e-7
    roots = []
    
    if abs(a) < 1e-8: # Tuyến tính: bz + c = 0
        if abs(b) > 1e-8:
            roots.append(-c / b)
    else: # Bậc hai
        delta = b**2 - 4*a*c
        if delta >= 0:
            sqrt_delta = np.sqrt(delta)
            roots.append((-b + sqrt_delta) / (2*a))
            roots.append((-b - sqrt_delta) / (2*a))
            
    valid_roots = [r for r in roots if r > current_z + eps and r <= z_max + eps]
    return min(valid_roots) if valid_roots else None

def parametric_simplex_quadratic(ns, nt, H, h, u, v, w, z_min, z_max):
    # print("Starting parametric simplex with quadratic cost...")
    H = np.array(H, dtype=float)
    h = np.array(h, dtype=float)
    u, v, w = np.array(u), np.array(v), np.array(w)
    # print("Shape:", H.shape, h.shape, u.shape, v.shape, w.shape)
    h = h.flatten()
    u, v, w = u.flatten(), v.flatten(), w.flatten()
    num_constraints, num_vars = H.shape
    current_z = z_min
    results = []

    # 1. Tìm lời giải ban đầu tại z_min
    
    c_start = u + v * z_min + w * (z_min**2)
    res = linprog(c_start, A_ub = - np.identity(ns * nt), b_ub = np.zeros((ns * nt, 1)), A_eq=H, b_eq=h, method='simplex',options={'maxiter': 1000000})



    if not res.success:
        return [] # Trả về rỗng nếu không có lời giải khả thi

    # Xác định các biến cơ sở ban đầu (đảm bảo ma trận vuông và độc lập tuyến tính)
    # nonzero_x = np.where(res.x > 1e-8)[0]
    basis_indices = res.basis #find_valid_basis(H, nonzero_x)

    while current_z <= z_max + 1e-9:
        B = H[:, basis_indices]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            # Nếu xảy ra lỗi ma trận suy biến, cố gắng tìm basis khác từ x hiện tại
            break

        # 2. Tính hệ số Reduced Cost: RC(z) = A*z^2 + B*z + C
        cb_u, cb_v, cb_w = u[basis_indices], v[basis_indices], w[basis_indices]
        
        # Công thức: rc = c - cB * B^-1 * H
        pi_u = cb_u @ B_inv
        pi_v = cb_v @ B_inv
        pi_w = cb_w @ B_inv

        rc_u = u - pi_u @ H
        rc_v = v - pi_v @ H
        rc_w = w - pi_w @ H

        # 3. Tìm z_next (điểm mà một RC biến ngoài cơ sở trở thành < 0)
        next_z = z_max
        entering_var = None
        
        for i in range(num_vars):
            if i in basis_indices:
                continue
            
            # Giải phương trình RC_i(z) = 0
            root = solve_quadratic_roots(rc_w[i], rc_v[i], rc_u[i], current_z, z_max)
            if root is not None and root < next_z:
                next_z = root
                entering_var = i

        # Lưu lại khoảng ổn định
        results.append((current_z, next_z, np.sort(basis_indices)))

        if entering_var is None or next_z >= z_max:
            break

        # 4. Thực hiện Pivot (Rời khỏi cơ sở)
        # Tính hướng di chuyển d = B^-1 * H_entering
        d = B_inv @ H[:, entering_var]
        x_B = B_inv @ h
        
        # Tỷ số Min-ratio để tìm biến rời khỏi cơ sở
        min_ratio = float('inf')
        leaving_idx_in_basis = -1
        for i in range(len(d)):
            if d[i] > 1e-9:
                ratio = x_B[i] / d[i]
                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving_idx_in_basis = i
        
        if leaving_idx_in_basis == -1: # Không tìm được biến rời đi (unbounded)
            break
            
        basis_indices[leaving_idx_in_basis] = entering_var
        current_z = next_z

    return results

def compute_DA_intervals(ns, nt, a, b, c_, S, h, m_min=-20, m_max=20):
    OMEGA = OptimalTransport.constructOMEGA(ns, nt)

    Omega_a = OMEGA.dot(a)
    Omega_b = OMEGA.dot(b)

    w_tilde = c_ + Omega_a * Omega_a
    r_tilde = 2*Omega_a * Omega_b
    o_tilde = Omega_b * Omega_b

    intervals = parametric_simplex_quadratic(ns, nt,
        S, h,
        w_tilde, r_tilde, o_tilde,
        z_min=m_min,
        z_max=m_max
    )
    return intervals

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------



def parametric(ns, nt, a, b, X, S_, h_, SELECTION_F, fraction, config, z_min=-20, z_max=20):
    TD = []
    
    # --- PRE-COMPUTE DA BASIS INTERVALS ---
    OMEGA = OptimalTransport.constructOMEGA(ns, nt)
    c_ = np.zeros((ns * nt, 1))
    for i in range(X.shape[1]):
        c_ += (OMEGA.dot(X[:, [i]])) * (OMEGA.dot(X[:, [i]]))
    
    # Returns list of (ml, mr, B) where B is the constant OT basis
    list_da_itv = compute_DA_intervals(ns, nt, a, b, c_, S_, h_, m_min=z_min, m_max=z_max)
    
    for ml, mr, B in list_da_itv:
        z = ml
        while z < mr:
            Ydeltaz = a + b * z
            cost = c_ + OMEGA.dot(Ydeltaz)*OMEGA.dot(Ydeltaz)
            
            cost0 = c_ + OMEGA.dot(a)*OMEGA.dot(a)
            cost1 = 2*OMEGA.dot(a)*OMEGA.dot(b)
            cost2 = OMEGA.dot(b)*OMEGA.dot(b)
            
            adapter_z = si_adapter.SelectionAdapter(fraction=fraction)
            selection, top_indices = adapter_z.select_with_known_basis(ns, S_, cost, B)
            adapter_z.S_ = S_
            
            # Compute itvFS using the projection of 'a' and 'b' onto the current GAMMA
            itvFS = si_overconditioning.sort_conditioning(z, adapter_z, B, cost0, cost1, cost2, top_indices, config)
            
            # Intersect itvFS with the current DA interval [ml, mr]
            itvDA = [(ml, mr)]
            intervalinloop = si_utils.interval_intersection(itvFS, itvDA)
            
            # Update TD if selection matches target
            if sorted(selection) == sorted(SELECTION_F):
                TD = si_utils.interval_union(TD, intervalinloop)
            
            # Jump logic
            if not intervalinloop:
                z = mr # Move to next DA basis if no FS intersection found
            else:
                z = intervalinloop[-1][1] + 0.0001
                
    return TD