
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

# ---------- Core dataclasses ----------

@dataclass
class RebelParams:
    v_R: float = 1.0    # payoff if rebels win
    l_R: float = 0.0    # payoff if rebels lose
    k_R: float = 0.10   # fixed cost of fighting
    @property
    def Delta_R(self) -> float:
        return self.v_R - self.l_R

@dataclass
class GovLossParams:
    L_I: float = 0.20   # gov't loss if war & alliance intervenes
    L_U: float = 0.50   # gov't loss if war & alliance stays out

@dataclass
class TypeParams:
    # Alliance primitives
    pi: float           # P(A intervenes | attack, t)
    d: float            # drop in rebel win prob if A intervenes
    delta_qU: float     # capacity-building shift in q_U (usually <= 0)
    delta_s: float      # shift in status-quo payoff s (conditionality, etc.)
    # Domestic policy technology (effects on q_U and s)
    phi_r: float        # how repression lowers q_U (>=0)
    phi_y: float        # how concessions lower q_U (allow negative if emboldenment)
    psi_r: float        # how repression lowers s (>=0)
    psi_y: float        # how concessions raise s (>=0)
    # Domestic policy costs (quadratic)
    c_r: float          # marginal cost of repression
    c_y: float          # marginal cost of concessions

# ---------- Baseline helpers ----------

def deterrence_gap_D0(tp: TypeParams, q0: float, s0: float, rp: RebelParams) -> Tuple[float, float, float, float]:
    """
    Returns (D0, qU0_t, s0_t, hatq0) where D0 = [q_U - pi*d] - hat{q} evaluated at r=y=0.
    """
    qU0_t = np.clip(q0 + tp.delta_qU, 0.0, 1.0)
    s0_t  = s0 + tp.delta_s
    hatq0 = (s0_t - rp.l_R + rp.k_R) / rp.Delta_R
    D0    = (qU0_t - tp.pi*tp.d) - hatq0
    return D0, qU0_t, s0_t, hatq0

def A_coeffs(tp: TypeParams, rp: RebelParams) -> Tuple[float, float]:
    """
    Attack occurs if D(r,y) >= 0, where
    D(r,y) = D0 - (phi_r - psi_r/Delta_R) r - (phi_y + psi_y/Delta_R) y.
    Let A_r = phi_r - psi_r/Delta_R, A_y = phi_y + psi_y/Delta_R.
    Larger A's mean the instrument is more effective at reducing D.
    """
    A_r = tp.phi_r - tp.psi_r / rp.Delta_R
    A_y = tp.phi_y + tp.psi_y / rp.Delta_R
    return A_r, A_y

def minimal_deterrence_cost_and_policy(D0: float, A_r: float, A_y: float, c_r: float, c_y: float
                                       ) -> Tuple[float, float, float, bool, bool]:
    """
    Given D0>0, find min 0.5(c_r r^2 + c_y y^2) s.t. A_r r + A_y y >= D0, r,y>=0.
    If both A_r and A_y <= 0, policy cannot deter: return (inf,0,0,False,False).
    If D0 <= 0, already deterred: return (0,0,0,True,False).
    Returns (min_cost, r*, y*, feasible, computed).
    """
    if D0 <= 0:
        return 0.0, 0.0, 0.0, True, False

    feasible = (A_r > 1e-12) or (A_y > 1e-12)
    if not feasible:
        return float('inf'), 0.0, 0.0, False, False

    a = (A_r**2 / c_r) if (A_r > 1e-12) else 0.0
    b = (A_y**2 / c_y) if (A_y > 1e-12) else 0.0
    denom = a + b
    if denom <= 1e-12:
        return float('inf'), 0.0, 0.0, False, False

    lam = D0 / denom
    r = lam * A_r / c_r if (A_r > 1e-12) else 0.0
    y = lam * A_y / c_y if (A_y > 1e-12) else 0.0
    cost = 0.5 * (c_r * r * r + c_y * y * y)
    return float(cost), float(r), float(y), True, True

# ---------- Simulation ----------

def simulate_one_type(N: int, seed: int,
                      rp: RebelParams, gp: GovLossParams, tp: TypeParams,
                      q0_min: float=0.20, q0_max: float=0.60,
                      s0_min: float=0.00, s0_max: float=0.20):
    rng = np.random.default_rng(seed)
    attacks_no_policy = 0
    attacks_with_policy = 0
    deter_count = 0
    impossible_count = 0
    r_list_det, y_list_det, cost_list = [], [], []
    rows_micro = []

    for _ in range(N):
        q0 = rng.uniform(q0_min, q0_max)
        s0 = rng.uniform(s0_min, s0_max)
        D0, qU0_t, s0_t, hatq0 = deterrence_gap_D0(tp, q0, s0, rp)
        A_r, A_y = A_coeffs(tp, rp)
        L = tp.pi * gp.L_I + (1.0 - tp.pi) * gp.L_U

        attack_no_policy = D0 >= 0.0
        attacks_no_policy += int(attack_no_policy)

        cost, r_star, y_star, feasible, computed = minimal_deterrence_cost_and_policy(
            D0, A_r, A_y, tp.c_r, tp.c_y
        )
        if (not feasible) and (D0 > 0.0):
            impossible_count += 1

        implement = (cost <= L)
        if implement and computed:
            deter_count += 1
            attack_with_policy = False
            r_list_det.append(r_star)
            y_list_det.append(y_star)
            cost_list.append(cost)
        else:
            r_star = 0.0
            y_star = 0.0
            attack_with_policy = attack_no_policy

        attacks_with_policy += int(attack_with_policy)

        rows_micro.append({
            "q0": q0, "s0": s0, "qU0_t": qU0_t, "s0_t": s0_t, "hatq0": hatq0,
            "pi": tp.pi, "d": tp.d, "A_r": A_r, "A_y": A_y, "D0": D0, "L": L,
            "attack_no_policy": int(attack_no_policy),
            "attack_with_policy": int(attack_with_policy),
            "r_star": r_star, "y_star": y_star,
            "policy_cost": cost if (implement and computed) else 0.0,
            "policy_implemented": int(implement and computed),
            "deterrence_feasible": int(feasible)
        })

    summary = {
        "attack_rate_no_policy": attacks_no_policy / N,
        "attack_rate_with_policy": attacks_with_policy / N,
        "share_governments_choosing_to_deter": deter_count / N,
        "share_impossible_to_deter_via_policy": impossible_count / N,
        "avg_r_if_policy_applied": float(np.mean(r_list_det)) if r_list_det else 0.0,
        "avg_y_if_policy_applied": float(np.mean(y_list_det)) if y_list_det else 0.0,
        "avg_policy_cost_when_applied": float(np.mean(cost_list)) if cost_list else 0.0,
    }
    return pd.DataFrame([summary]), pd.DataFrame(rows_micro)

def simulate_many(types: Dict[str, TypeParams], N: int, seed: int,
                  rp: RebelParams, gp: GovLossParams,
                  q0_min: float=0.20, q0_max: float=0.60,
                  s0_min: float=0.00, s0_max: float=0.20):
    all_sum = []
    all_micro = []
    for name, tp in types.items():
        df_s, df_m = simulate_one_type(N, seed, rp, gp, tp, q0_min, q0_max, s0_min, s0_max)
        df_s.insert(0, "type", name)
        df_m.insert(0, "type", name)
        all_sum.append(df_s)
        all_micro.append(df_m)
    return pd.concat(all_sum, ignore_index=True), pd.concat(all_micro, ignore_index=True)

# ---------- Presets ----------

def default_presets() -> Dict[str, TypeParams]:
    return {
        "None": TypeParams(
            pi=0.00, d=0.00, delta_qU=0.00, delta_s=0.00,
            phi_r=0.15, phi_y=0.05, psi_r=0.10, psi_y=0.10,
            c_r=1.00, c_y=1.00
        ),
        "Defense pact": TypeParams(
            pi=0.60, d=0.20, delta_qU=-0.05, delta_s=0.00,
            phi_r=0.20, phi_y=0.06, psi_r=0.10, psi_y=0.10,
            c_r=0.85, c_y=1.10
        ),
        "Forward basing": TypeParams(
            pi=0.80, d=0.25, delta_qU=-0.10, delta_s=0.00,
            phi_r=0.25, phi_y=0.06, psi_r=0.10, psi_y=0.10,
            c_r=0.75, c_y=1.00
        ),
        "Conditional defense": TypeParams(
            pi=0.70, d=0.22, delta_qU=-0.08, delta_s=0.05,
            phi_r=0.20, phi_y=0.08, psi_r=0.12, psi_y=0.15,
            c_r=1.20, c_y=0.80
        ),
    }
