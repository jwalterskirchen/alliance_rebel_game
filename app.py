
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import (
    RebelParams, GovLossParams, TypeParams,
    simulate_many, default_presets
)

st.set_page_config(page_title="Alliance–Rebel Deterrence Simulator", layout="wide")

st.title("Alliance–Rebel Deterrence Simulator")
st.markdown("""
Adjust alliance parameters and domestic policy technologies with sliders,
then run a Monte Carlo to see **attack rates**, **policy choices** (repression vs concessions),
and how different alliance **types** compare.
""")

# -------------------- Sidebar: global controls --------------------
with st.sidebar:
    st.header("Simulation controls")
    N = st.slider("Draws (N)", min_value=500, max_value=20000, value=4000, step=500)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.markdown("---")
    st.subheader("Baselines for draws")
    q0_min, q0_max = st.slider("Baseline rebel success q0 range", 0.0, 1.0, (0.20, 0.60), 0.01)
    s0_min, s0_max = st.slider("Baseline status quo s0 range", 0.0, 0.5, (0.00, 0.20), 0.01)

    st.markdown("---")
    st.subheader("Rebel & Gov loss parameters")
    v_R = st.number_input("Rebel payoff if win (v_R)", value=1.0, step=0.05, format="%.2f")
    l_R = st.number_input("Rebel payoff if lose (ℓ_R)", value=0.0, step=0.05, format="%.2f")
    k_R = st.number_input("Fixed cost of fighting (k_R)", value=0.10, step=0.01, format="%.2f")

    L_I = st.number_input("Gov loss if attacked & intervention (L_I)", value=0.20, step=0.05, format="%.2f")
    L_U = st.number_input("Gov loss if attacked & no intervention (L_U)", value=0.50, step=0.05, format="%.2f")

    rp = RebelParams(v_R=v_R, l_R=l_R, k_R=k_R)
    gp = GovLossParams(L_I=L_I, L_U=L_U)

# -------------------- Main pane: type presets & custom --------------------
presets = default_presets()

st.subheader("Alliance types to simulate")
left, right = st.columns([1.2, 1.0])

with left:
    preset_names = list(presets.keys())
    chosen = st.multiselect("Select preset types", preset_names, default=preset_names)

with right:
    st.markdown("**Create/override a custom type:**")
    cname = st.text_input("Custom type name", value="Custom")
    st.caption("Adjust sliders; if you include this name in the list on the left, the custom values are used.")

    # Sliders for custom type
    st.markdown("**Alliance credibility & effectiveness**")
    pi = st.slider("π (intervention probability)", 0.0, 1.0, 0.70, 0.01)
    d  = st.slider("d (effectiveness: q drops if intervene)", 0.0, 1.0, 0.22, 0.01)

    st.markdown("**Alliance externalities**")
    delta_qU = st.slider("Δ q_U (capacity shift if A stays out)", -0.30, 0.30, -0.08, 0.01)
    delta_s  = st.slider("Δ s (status-quo shift via conditionality)", -0.20, 0.20, 0.05, 0.01)

    st.markdown("**Domestic policy technology**")
    phi_r = st.slider("φ_r (repression lowers q_U)", 0.0, 0.5, 0.20, 0.01)
    phi_y = st.slider("φ_y (concessions lower q_U; allow negative if embolden)", -0.30, 0.30, 0.08, 0.01)
    psi_r = st.slider("ψ_r (repression lowers s)", 0.0, 0.5, 0.12, 0.01)
    psi_y = st.slider("ψ_y (concessions raise s)", 0.0, 0.5, 0.15, 0.01)

    st.markdown("**Government costs (quadratic)**")
    c_r = st.slider("c_r (cost of repression)", 0.10, 2.00, 1.20, 0.05)
    c_y = st.slider("c_y (cost of concessions)", 0.10, 2.00, 0.80, 0.05)

# Override / add custom type object if selected
custom_type = TypeParams(
    pi=float(pi), d=float(d), delta_qU=float(delta_qU), delta_s=float(delta_s),
    phi_r=float(phi_r), phi_y=float(phi_y), psi_r=float(psi_r), psi_y=float(psi_y),
    c_r=float(c_r), c_y=float(c_y)
)

# Build dictionary of selected types (use custom values if name collides)
selected_types = {name: presets[name] for name in chosen if name in presets}
if cname in chosen:
    selected_types[cname] = custom_type

# If nothing selected, default to Custom
if not selected_types:
    chosen = [cname]
    selected_types[cname] = custom_type

st.markdown("---")

# -------------------- Run simulation --------------------
run = st.button("Run simulation")

if run:
    with st.spinner("Simulating..."):
        df_summary, df_micro = simulate_many(
            selected_types, N=N, seed=int(seed),
            rp=rp, gp=gp, q0_min=q0_min, q0_max=q0_max, s0_min=s0_min, s0_max=s0_max
        )

    st.success("Done.")
    st.subheader("Summary")
    st.dataframe(df_summary, use_container_width=True)

    # ------------- Charts (matplotlib only, one per figure, no explicit colors) -------------
    # Chart 1: Attack rates
    fig1 = plt.figure()
    x = np.arange(len(df_summary))
    width = 0.35
    plt.bar(x - width/2, df_summary["attack_rate_no_policy"], width, label="No policy")
    plt.bar(x + width/2, df_summary["attack_rate_with_policy"], width, label="With optimal policy")
    plt.xticks(x, df_summary["type"], rotation=0)
    plt.ylabel("Attack rate")
    plt.title("Rebel attack rate by alliance type")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig1)

    # Chart 2: Policy levels among deterrers
    fig2 = plt.figure()
    width2 = 0.35
    plt.bar(x - width2/2, df_summary["avg_r_if_policy_applied"], width2, label="Repression r* (among deterrers)")
    plt.bar(x + width2/2, df_summary["avg_y_if_policy_applied"], width2, label="Concessions y* (among deterrers)")
    plt.xticks(x, df_summary["type"], rotation=0)
    plt.ylabel("Average policy level")
    plt.title("Domestic policy used to deter (if any)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    # Downloads
    st.subheader("Download results")
    csv_sum = df_summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download summary CSV", data=csv_sum, file_name="summary_by_type.csv", mime="text/csv")

    # micro can be large, so gate it
    if st.checkbox("Include micro-level panel (can be large)"):
        csv_micro = df_micro.to_csv(index=False).encode("utf-8")
        st.download_button("Download micro CSV", data=csv_micro, file_name="micro_panel.csv", mime="text/csv")

else:
    st.info("Choose types, adjust sliders, then click **Run simulation**.")

st.markdown("---")
with st.expander("Model notes"):
    st.markdown(r"""
**Rebel attack condition.** Rebels attack iff
\(
D(r,y;t) \equiv \big(q_U^0(t)-\pi(t)d(t)\big) - \frac{s^0(t)-\ell_R+k_R}{\Delta_R}
- (\phi_r-\psi_r/\Delta_R)r - (\phi_y+\psi_y/\Delta_R)y \ge 0.
\)

If \(D_0 = D(0,0;t)>0\), the lowest-cost deterring policy solves
\(\min \tfrac12(c_r r^2 + c_y y^2)\) s.t. \(A_r r + A_y y \ge D_0\),
with \(A_r=\phi_r-\psi_r/\Delta_R\), \(A_y=\phi_y+\psi_y/\Delta_R\).
Optimal \((r^*,y^*)\) scales with these \(A\)'s and the policy cost parameters.
""")
