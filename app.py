import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Physics / Math helpers
# -----------------------------
def reciprocal_vectors(a1, a2):
    A = np.column_stack([a1, a2])          # 2x2
    B = 2*np.pi * np.linalg.inv(A).T       # columns are b1, b2
    return B[:, 0], B[:, 1]

def nn_vectors(a):
    # Nearest-neighbor vectors (A -> B) consistent with a1,a2 definition below
    return np.array([
        [ a/np.sqrt(3), 0.0 ],
        [-a/(2*np.sqrt(3)),  a/2 ],
        [-a/(2*np.sqrt(3)), -a/2 ]
    ])

def f_k(kx, ky, deltas):
    k = np.array([kx, ky])
    return np.sum(np.exp(1j * (deltas @ k)))

def bands_graphene(kx, ky, t, Delta, deltas):
    fk = f_k(kx, ky, deltas)
    E = np.sqrt(Delta**2 + (t**2) * (np.abs(fk)**2))
    return +E, -E

def K_point(b1, b2):
    # One inequivalent K corner (others are symmetry-related)
    return (b1 + 2*b2) / 3.0

def estimate_gap_near_K(t, Delta, Kvec, deltas, qmax=0.15, nq=121):
    # Numerical estimate: Eg ≈ 2 * min_{|q|<=qmax} Eplus(K+q)
    q = np.linspace(-qmax, qmax, nq)
    Emin = np.inf

    for qx in q:
        for qy in q:
            if qx*qx + qy*qy > qmax*qmax:
                continue
            kp = Kvec + np.array([qx, qy])
            Eplus, _ = bands_graphene(kp[0], kp[1], t, Delta, deltas)
            if Eplus < Emin:
                Emin = Eplus

    return 2.0 * Emin

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Graphene Gap App (Tight-Binding)", layout="centered")
st.title("Graphene Gap (Tight-Binding) — Web App")
st.write("این اپ با شکستن تقارن زیرشبکه‌ها (±Δ) گاف انرژی را **عددياً** نزدیک نقطه K استخراج می‌کند و با رابطه نظری **Eg = 2|Δ|** مقایسه می‌کند.")

with st.sidebar:
    st.header("ورودی‌ها")

    t = st.slider("t (eV) — hopping", min_value=0.5, max_value=5.0, value=2.7, step=0.1)

    a = st.slider("a (مقیاس طول) — معمولاً 1", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    st.subheader("Sweep روی Δ")
    Delta_min = st.number_input("Δ_min (eV)", value=0.0, step=0.05, format="%.3f")
    Delta_max = st.number_input("Δ_max (eV)", value=0.8, step=0.05, format="%.3f")
    nDelta = st.slider("تعداد نقاط Δ", min_value=3, max_value=51, value=9, step=2)

    st.subheader("تنظیمات عددی اطراف K")
    qmax = st.slider("qmax (بازه زوم اطراف K)", min_value=0.02, max_value=0.40, value=0.15, step=0.01)
    nq = st.slider("nq (رزولوشن شبکه نمونه‌برداری)", min_value=31, max_value=301, value=121, step=10)

    st.caption("نکته: nq بزرگ‌تر → دقت بیشتر ولی کندتر.")

# Guard
if Delta_max < Delta_min:
    st.error("Δ_max باید بزرگ‌تر یا مساوی Δ_min باشد.")
    st.stop()

# Build lattice and K
a1 = a * np.array([0.5,  np.sqrt(3)/2])
a2 = a * np.array([0.5, -np.sqrt(3)/2])
b1, b2 = reciprocal_vectors(a1, a2)
K = K_point(b1, b2)
deltas = nn_vectors(a)

# Compute sweep
Delta_vals = np.linspace(float(Delta_min), float(Delta_max), int(nDelta))
Eg_num = np.zeros_like(Delta_vals)
Eg_th = 2.0 * np.abs(Delta_vals)

with st.spinner("در حال محاسبه گاف عددی..."):
    for i, D in enumerate(Delta_vals):
        Eg_num[i] = estimate_gap_near_K(t=t, Delta=float(D), Kvec=K, deltas=deltas, qmax=float(qmax), nq=int(nq))

# Report key metrics
st.subheader("خروجی‌ها")
col1, col2, col3 = st.columns(3)
col1.metric("K-point (kx, ky)", f"({K[0]:.3f}, {K[1]:.3f})")
col2.metric("t (eV)", f"{t:.2f}")
col3.metric("qmax / nq", f"{qmax:.2f} / {nq}")

# Plot: Eg vs Delta
fig1 = plt.figure()
plt.plot(Delta_vals, Eg_num, marker="o", label="Eg عددی (اسکن نزدیک K)")
plt.plot(Delta_vals, Eg_th, linestyle="--", label=r"نظری: $E_g=2|\Delta|$")
plt.xlabel(r"$\Delta$ (eV)")
plt.ylabel(r"$E_g$ (eV)")
plt.title("Gap opening in graphene by sublattice symmetry breaking")
plt.grid(True)
plt.legend()
st.pyplot(fig1)

# Table
st.write("جدول مقایسه:")
table = np.column_stack([Delta_vals, Eg_num, Eg_th, Eg_num - Eg_th])
st.dataframe(
    {
        "Delta (eV)": table[:, 0],
        "Eg_num (eV)": table[:, 1],
        "Eg_theory (eV)": table[:, 2],
        "diff (num - theory)": table[:, 3],
    }
)

# Optional: one-point diagnostic around a chosen Delta
st.subheader("تست تک‌نقطه‌ای (اختیاری)")
Delta_test = st.slider("Δ_test (eV)", min_value=float(Delta_min), max_value=float(Delta_max), value=float(Delta_vals[len(Delta_vals)//2]), step=0.01)
Eg_test = estimate_gap_near_K(t=t, Delta=float(Delta_test), Kvec=K, deltas=deltas, qmax=float(qmax), nq=int(nq))
st.write(f"برای Δ = **{Delta_test:.3f} eV** → گاف عددی: **Eg ≈ {Eg_test:.4f} eV** ، نظری: **{2*abs(Delta_test):.4f} eV**")
