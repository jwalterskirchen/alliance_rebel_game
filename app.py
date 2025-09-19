
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import (
    RebelParams, GovLossParams, TypeParams,
    simulate_many, default_presets, sweep_2d
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
if cname in chosen or not selected_types:
    selected_types[cname] = custom_type

st.markdown("---")

# -------------------- Tabs: Simulation vs Tipping Points --------------------
tab_sim, tab_tip = st.tabs(["Simulation", "Tipping points (2D sweeps)"])

with tab_sim:
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
    else:
        st.info("Choose types, adjust sliders, then click **Run simulation**.")

with tab_tip:
    st.markdown("### Tipping-points explorer")
    st.caption("Visualize attack/no-attack boundaries and the government’s policy-deterrence boundary in 2D parameter space.")

    # Choose a single type to analyze
    type_to_analyze = st.selectbox("Type to analyze", list(selected_types.keys()))
    tp = selected_types[type_to_analyze]

    # ---- Presets ----
    st.markdown("#### One-click bifurcation presets")
    PRESETS = {
        "Credibility vs Effectiveness (π–d)": dict(
            x_name="pi", x_range=(0.0, 1.0), y_name="d", y_range=(0.0, 0.6),
            q0=0.40, s0=0.10, nx=120, ny=120
        ),
        "Conditionality vs Repression cost (Δs–c_r)": dict(
            x_name="delta_s", x_range=(-0.10, 0.20), y_name="c_r", y_range=(0.10, 2.00),
            q0=0.40, s0=0.10, nx=120, ny=120
        ),
        "Repression efficiency vs grievance (φ_r–ψ_r)": dict(
            x_name="phi_r", x_range=(0.0, 0.5), y_name="psi_r", y_range=(0.0, 0.5),
            q0=0.40, s0=0.10, nx=120, ny=120
        ),
        "Concessions capacity vs benefit (φ_y–ψ_y)": dict(
            x_name="phi_y", x_range=(-0.30, 0.30), y_name="psi_y", y_range=(0.0, 0.5),
            q0=0.40, s0=0.10, nx=120, ny=120
        ),
        "Baseline rebel strength vs credibility (q0–π)": dict(
            x_name="q0", x_range=(0.10, 0.80), y_name="pi", y_range=(0.0, 1.0),
            q0=0.40, s0=0.10, nx=120, ny=120
        ),
        "Status-quo vs conditionality (s0–Δs)": dict(
            x_name="s0", x_range=(0.00, 0.30), y_name="delta_s", y_range=(-0.10, 0.20),
            q0=0.40, s0=0.10, nx=120, ny=120
        ),
    }

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0, key="preset_choice")

    # Initialize session_state keys if missing
    for key, default in [
        ("tip_x_name", "pi"),
        ("tip_y_name", "d"),
        ("tip_x_range", (0.0, 1.0)),
        ("tip_y_range", (0.0, 0.6)),
        ("tip_q0_base", 0.40),
        ("tip_s0_base", 0.10),
        ("tip_nx", 100),
        ("tip_ny", 100),
    ]:
        st.session_state.setdefault(key, default)

    colp1, colp2, colp3 = st.columns([1,1,1])
    with colp1:
        if st.button("Apply preset"):
            p = PRESETS[st.session_state["preset_choice"]]
            st.session_state["tip_x_name"] = p["x_name"]
            st.session_state["tip_y_name"] = p["y_name"]
            st.session_state["tip_x_range"] = p["x_range"]
            st.session_state["tip_y_range"] = p["y_range"]
            st.session_state["tip_q0_base"] = p["q0"]
            st.session_state["tip_s0_base"] = p["s0"]
            st.session_state["tip_nx"] = p["nx"]
            st.session_state["tip_ny"] = p["ny"]
            st.experimental_rerun()
    with colp2:
        st.write("")  # spacer
    with colp3:
        st.write("")

    # ---- Manual controls bound to session_state ----
    st.markdown("#### Manual controls (editable)")
    param_options = ["pi","d","delta_qU","delta_s","phi_r","phi_y","psi_r","psi_y","c_r","c_y","q0","s0"]
    colx, coly = st.columns(2)
    with colx:
        x_name = st.selectbox("X-axis parameter", param_options, index=param_options.index(st.session_state["tip_x_name"]), key="tip_x_name")
        x_min, x_max = st.slider("X range", -0.5, 1.5, st.session_state["tip_x_range"], 0.01, key="tip_x_range")
    with coly:
        y_name = st.selectbox("Y-axis parameter", param_options, index=param_options.index(st.session_state["tip_y_name"]), key="tip_y_name")
        y_min, y_max = st.slider("Y range", -0.5, 1.5, st.session_state["tip_y_range"], 0.01, key="tip_y_range")

    q0_base = st.slider("Baseline q0 (center)", 0.0, 1.0, st.session_state["tip_q0_base"], 0.01, key="tip_q0_base")
    s0_base = st.slider("Baseline s0 (center)", 0.0, 0.5, st.session_state["tip_s0_base"], 0.01, key="tip_s0_base")

    nx = st.slider("Grid steps (X)", 20, 200, st.session_state["tip_nx"], 10, key="tip_nx")
    ny = st.slider("Grid steps (Y)", 20, 200, st.session_state["tip_ny"], 10, key="tip_ny")

    overlay = st.checkbox("Overlay contours for D0=0 and C*=L", value=True)

    run_sweep = st.button("Run 2D sweep")

    if run_sweep:
        with st.spinner("Sweeping..."):
            out = sweep_2d(
                tp, rp, gp,
                st.session_state["tip_x_name"], st.session_state["tip_x_range"][0], st.session_state["tip_x_range"][1], int(st.session_state["tip_nx"]),
                st.session_state["tip_y_name"], st.session_state["tip_y_range"][0], st.session_state["tip_y_range"][1], int(st.session_state["tip_ny"]),
                st.session_state["tip_q0_base"], st.session_state["tip_s0_base"]
            )

        X, Y = out["X"], out["Y"]
        D0 = out["D0"]
        Cstar = out["Cstar"]
        L = out["L"]
        class_np = out["class_no_policy"]
        class_wp = out["class_with_policy"]
        mix_share_y = out["mix_share_y"]

        # --- Figure 1: No policy (attack vs no attack) ---
        fig_np = plt.figure()
        plt.imshow(class_np, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()], aspect="auto")
        plt.xlabel(st.session_state["tip_x_name"])
        plt.ylabel(st.session_state["tip_y_name"])
        plt.title("No policy: 0 = no attack, 1 = attack")
        if overlay:
            try:
                CS = plt.contour(X, Y, D0, levels=[0.0])
                plt.clabel(CS, inline=True, fontsize=8)
            except Exception:
                pass
        plt.tight_layout()
        st.pyplot(fig_np)

        # --- Figure 2: With optimal policy (three regimes) ---
        fig_wp = plt.figure()
        plt.imshow(class_wp, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()], aspect="auto")
        plt.xlabel(st.session_state["tip_x_name"])
        plt.ylabel(st.session_state["tip_y_name"])
        plt.title("With optimal policy: 0=no attack baseline, 1=attack persists, 2=policy deters")
        if overlay:
            try:
                CS1 = plt.contour(X, Y, D0, levels=[0.0])
                plt.clabel(CS1, inline=True, fontsize=8)
            except Exception:
                pass
            try:
                diff = Cstar - L
                CS2 = plt.contour(X, Y, diff, levels=[0.0])
                plt.clabel(CS2, inline=True, fontsize=8)
            except Exception:
                pass
        plt.tight_layout()
        st.pyplot(fig_wp)

        # --- Figure 3: Policy mix among deterrers (share y*) ---
        fig_mix = plt.figure()
        plt.imshow(mix_share_y, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()], aspect="auto")
        plt.xlabel(st.session_state["tip_x_name"])
        plt.ylabel(st.session_state["tip_y_name"])
        plt.title("Policy mix when policy deters: y* / (r* + y*)")
        plt.tight_layout()
        st.pyplot(fig_mix)

        # Quick legend
        st.markdown("""
**Legend**  
- **Figure 1:** Dark/light regions split at the **D0=0** contour: crossing it is the *tipping point* for attack without domestic policy.  
- **Figure 2:** The additional **C\* = L** contour is the *tipping point* for whether the government finds it optimal to deter.  
  Regions: 0 (no attack even at r=y=0), 1 (attack persists; too costly or infeasible to deter), 2 (gov deters with optimal policy).  
- **Figure 3:** Where region=2, color shows the share of **concessions** in the optimal mix; closer to 1 means concessions dominate.
""")
    else:
        st.info("Pick a preset (optional), review the controls, then click **Run 2D sweep**.")
