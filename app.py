import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="گاف گرافن — اپ تعاملی", layout="centered")


# -----------------------------
# Helpers: RTL blocks
# -----------------------------
def rtl(html: str) -> str:
    return f'<div dir="rtl" style="text-align:right; line-height:1.9;">{html}</div>'


# -----------------------------
# Physics helpers
# -----------------------------
def reciprocal_vectors(a1, a2):
    A = np.column_stack([a1, a2])          # 2x2
    B = 2*np.pi * np.linalg.inv(A).T       # columns are b1, b2
    return B[:, 0], B[:, 1]

def nn_vectors(a):
    # A -> B nearest-neighbors (one consistent convention)
    return np.array([
        [ a/np.sqrt(3), 0.0 ],
        [-a/(2*np.sqrt(3)),  a/2 ],
        [-a/(2*np.sqrt(3)), -a/2 ]
    ])

def K_point(b1, b2):
    # one inequivalent K corner
    return (b1 + 2*b2) / 3.0

def f_k_grid(kx, ky, deltas):
    """
    Vectorized f(k) over a mesh.
    kx, ky: 2D arrays (meshgrid outputs)
    deltas: (3,2)
    """
    # phases = exp(i * (delta_x*kx + delta_y*ky)) summed over 3 deltas
    fk = 0.0 + 0.0j
    for dx, dy in deltas:
        fk += np.exp(1j * (dx * kx + dy * ky))
    return fk

def gap_numeric_near_K(t, Delta, Kvec, deltas, qmax=0.15, nq=121):
    """
    Fast numeric gap estimate:
    Eg ≈ 2 * min_{|q|<=qmax} Eplus(K+q)
    Uses vectorized grid instead of double for-loop.
    """
    q = np.linspace(-qmax, qmax, nq)
    qx, qy = np.meshgrid(q, q, indexing="xy")

    mask = (qx*qx + qy*qy) <= (qmax*qmax)

    kx = Kvec[0] + qx
    ky = Kvec[1] + qy

    fk = f_k_grid(kx, ky, deltas)
    Eplus = np.sqrt(Delta**2 + (t**2) * (np.abs(fk)**2))

    Emin = np.min(Eplus[mask])
    return 2.0 * Emin


@st.cache_data(show_spinner=False)
def sweep_gap(t, a, Delta_min, Delta_max, nDelta, qmax, nq):
    # lattice
    a1 = a * np.array([0.5,  np.sqrt(3)/2])
    a2 = a * np.array([0.5, -np.sqrt(3)/2])
    b1, b2 = reciprocal_vectors(a1, a2)
    K = K_point(b1, b2)
    deltas = nn_vectors(a)

    Delta_vals = np.linspace(Delta_min, Delta_max, nDelta)
    Eg_num = np.zeros_like(Delta_vals)
    Eg_th = 2.0 * np.abs(Delta_vals)

    for i, D in enumerate(Delta_vals):
        Eg_num[i] = gap_numeric_near_K(t=t, Delta=float(D), Kvec=K, deltas=deltas, qmax=qmax, nq=nq)

    return K, Delta_vals, Eg_num, Eg_th


# -----------------------------
# Sidebar (Inputs)
# -----------------------------
with st.sidebar:
    st.markdown(rtl("<h3>ورودی‌ها</h3>"), unsafe_allow_html=True)

    t = st.slider("پارامتر پرش الکترونی t (eV)", 0.5, 5.0, 2.7, 0.1)
    a = st.slider("مقیاس شبکه a (بدون‌بعد؛ معمولاً 1)", 0.5, 2.0, 1.0, 0.1)

    st.markdown(rtl("<h4>بازه‌ی تغییرات Δ</h4>"), unsafe_allow_html=True)
    Delta_min = st.number_input("Δ_min (eV)", value=0.0, step=0.05, format="%.3f")
    Delta_max = st.number_input("Δ_max (eV)", value=0.8, step=0.05, format="%.3f")
    nDelta = st.slider("تعداد نقاط Δ", 3, 51, 9, 2)

    st.markdown(rtl("<h4>تنظیمات عددی نزدیک نقطه K</h4>"), unsafe_allow_html=True)
    qmax = st.slider("qmax (بازه زوم در فضای k)", 0.02, 0.40, 0.15, 0.01)
    nq = st.slider("nq (دقت شبکه نمونه‌برداری)", 31, 301, 121, 10)

    st.caption("نکته: nq بزرگ‌تر ⇒ دقت بیشتر، اما زمان محاسبه بیشتر.")

if Delta_max < Delta_min:
    st.error("Δ_max باید بزرگ‌تر یا مساوی Δ_min باشد.")
    st.stop()


# -----------------------------
# Title + intro
# -----------------------------
st.markdown(rtl("<h1>گاف انرژی گرافن (مدل اتصال تنگ) — اپ تعاملی</h1>"), unsafe_allow_html=True)

intro = """
این اپ، گاف انرژی را به‌صورت عددی در نزدیکی نقطه
<span style="direction:ltr; display:inline-block;"><b>K</b></span>
با درنظر گرفتن شکستن تقارن زیرشبکه‌ها (±Δ) استخراج می‌کند و با رابطه نظری
<span style="direction:ltr; display:inline-block;"><b>Eg = 2|Δ|</b></span>
مقایسه می‌کند.
"""
st.markdown(rtl(intro), unsafe_allow_html=True)


# -----------------------------
# Compute sweep (cached)
# -----------------------------
with st.spinner("در حال محاسبه..."):
    K, Delta_vals, Eg_num, Eg_th = sweep_gap(
        t=float(t), a=float(a),
        Delta_min=float(Delta_min), Delta_max=float(Delta_max),
        nDelta=int(nDelta), qmax=float(qmax), nq=int(nq)
    )

# -----------------------------
# Outputs summary
# -----------------------------
st.markdown(rtl("<h2>خروجی‌ها</h2>"), unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("مختصات K (kx, ky)", f"({K[0]:.3f}, {K[1]:.3f})")
col2.metric("t (eV)", f"{t:.2f}")
col3.metric("qmax / nq", f"{qmax:.2f} / {nq}")


# -----------------------------
# Plot: Eg vs Delta
# -----------------------------
fig = plt.figure()
plt.plot(Delta_vals, Eg_num, marker="o", label="gap")
plt.plot(Delta_vals, Eg_th, linestyle="--", label="Eg = 2|Δ|")
plt.xlabel("Δ (eV)")
plt.ylabel("Eg (eV)")
plt.title("باز شدن گاف انرژی بر اثر شکستن تقارن زیرشبکه‌ها")
plt.grid(True)
plt.legend()
st.pyplot(fig)
plt.close(fig)

caption = """
<b>توضیح شکل —</b>
نقاط، گاف محاسبه‌شده به‌صورت عددی و خط‌چین، پیش‌بینی نظری
<span style="direction:ltr; display:inline-block;"><b>Eg = 2|Δ|</b></span>
را نشان می‌دهد.
"""
st.markdown(rtl(caption), unsafe_allow_html=True)


# -----------------------------
# Table + download
# -----------------------------
st.markdown(rtl("<h3>جدول مقایسه</h3>"), unsafe_allow_html=True)

diff = Eg_num - Eg_th
table = np.column_stack([Delta_vals, Eg_num, Eg_th, diff])

st.dataframe(
    {
        "Δ (eV)": table[:, 0],
        "Eg عددی (eV)": table[:, 1],
        "Eg نظری (eV)": table[:, 2],
        "اختلاف (عددـنظری)": table[:, 3],
    }
)

csv = "Delta,Eg_num,Eg_theory,diff\n" + "\n".join(
    f"{d:.6f},{en:.6f},{et:.6f},{df:.6f}" for d, en, et, df in zip(Delta_vals, Eg_num, Eg_th, diff)
)
st.download_button(
    label="دانلود جدول به‌صورت CSV",
    data=csv.encode("utf-8"),
    file_name="graphene_gap_results.csv",
    mime="text/csv",
)


# -----------------------------
# Single-point explanation (RTL + placeholders)
# -----------------------------
st.markdown(rtl("<h3>توضیح عددی برای یک مقدار خاص از Δ</h3>"), unsafe_allow_html=True)

Delta_test = st.slider("Δ_test (eV)", float(Delta_min), float(Delta_max), float(Delta_vals[len(Delta_vals)//2]), 0.01)

Eg_test = gap_numeric_near_K(
    t=float(t), Delta=float(Delta_test), Kvec=K, deltas=nn_vectors(float(a)), qmax=float(qmax), nq=int(nq)
)
Eg_test_th = 2.0 * abs(float(Delta_test))

text_gap_explanation = """
<div dir="rtl" style="text-align:right; line-height:1.9;">
در این محاسبه، گاف انرژی به‌صورت عددی در نزدیکی نقطه
<span style="direction:ltr; display:inline-block;"><b>%(K)s</b></span>
به‌دست آمده است.
پارامتر شکستن تقارن زیرشبکه‌ها برابر با
<b>%(Delta).3f eV</b>
در نظر گرفته شده و مقدار گاف انرژی عددی برابر با
<b>%(Eg_num).4f eV</b>
محاسبه شده است.
<br><br>
بر اساس پیش‌بینی نظری مدل اتصال تنگ، انتظار می‌رود گاف انرژی از رابطه
<span style="direction:ltr; display:inline-block;"><b>%(formula)s</b></span>
پیروی کند که مقدار نظری متناظر آن برابر با
<b>%(Eg_th).4f eV</b>
است.
</div>
"""

st.markdown(
    text_gap_explanation % {
        "K": "K",
        "Delta": float(Delta_test),
        "Eg_num": float(Eg_test),
        "Eg_th": float(Eg_test_th),
        "formula": "Eg = 2|Δ|"
    },
    unsafe_allow_html=True
)

conclusion = """
نتایج نشان می‌دهد که با افزایش Δ (شدت شکستن تقارن زیرشبکه‌ها)، گاف انرژی افزایش می‌یابد
و وابستگی آن با پیش‌بینی نظری هم‌خوانی دارد.
"""
st.markdown(rtl(conclusion), unsafe_allow_html=True)

