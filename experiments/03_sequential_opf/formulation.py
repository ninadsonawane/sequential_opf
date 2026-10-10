"""Source-grounded adapted equations; raster/SVG math works offline."""

import opf_bootstrap
from pathlib import Path
import html
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = opf_bootstrap.HERE / "results"
EQUATIONS = [
    (
        "Power identity",
        r"S_i=V_i\,\overline{(YV)_i}=P_i+jQ_i",
        "Balanced positive-sequence phasors. The system has 19 HVAC buses, 3 LFAC buses and 3 DC buses. It is multi-frequency, not an unbalanced three-phase model.",
    ),
    (
        "Rectangular active balance",
        r"P_i=e_i(G_i e-B_i f)+f_i(B_i e+G_i f)",
        "V=e+jf; G and B are the real and imaginary parts of Ybus. G_i and B_i denote row i; products inside parentheses are row-vector products.",
    ),
    (
        "Rectangular reactive balance",
        r"Q_i=f_i(G_i e-B_i f)-e_i(B_i e+G_i f)",
        "These identities give the state equations corresponding to P1 Eqs. 7 and 8. Here f is the imaginary voltage component; F below denotes operating frequency in Hz.",
    ),
    (
        "AC nodal constraints",
        r"P_i=P_i^g-P_i^d+P_i^s,\qquad Q_i=Q_i^g-Q_i^d+Q_i^s",
        "Our converter powers are positive INTO the AC network. Keep this sign convention consistent; do not copy signs from P1 without translating its withdrawn-power convention. There are no controllable shunt banks in this reconstruction.",
    ),
    (
        "Voltage and current bases",
        r"Z_B=V_{\rm rated}^2/S_B,\qquad I_B=S_B/(\sqrt{3}V_{\rm rated})",
        "S_B = 100 MVA. V_rated is line-to-line kV, distinct from bus voltage magnitude in per unit. The AC current base uses three-phase apparent power.",
    ),
    (
        "LFAC line parameters",
        r"r=R(F)\ell/Z_B,\qquad x=2\pi FL\ell/Z_B,\qquad b=2\pi FC\ell Z_B",
        "Two 1000 km lines; L = 1.0007 mH/km, C = 11.5407 nF/km. Nominal-pi series impedance r+jx and charging jb/2 at each end. R(F) is a disclosed piecewise-linear interpolation through the DC, 10 Hz and 60 Hz values, not a rerun of the authors' line-constants model.",
    ),
    (
        "Relation to P1 scaling",
        r"F_r=F/F_0,\quad \nu=V_{\rm rated}/V_0,\quad x=x_0F_r/\nu^2,\quad b=b_0F_r\nu^2",
        "Our F0 = 10 Hz, V0 = 500 kV preserve the P3 reconstruction reference point. P1 uses a 60 Hz frequency base and defines its voltage ratio as the squared voltage ratio. With constant physical resistance, r=r0/nu²; our primary run additionally updates R(F).",
    ),
    (
        "Converter transformer current",
        r"I_s=\overline{S_s/V_s},\qquad V_f=V_s+Z_T I_s",
        "Transformer KVL with converter-to-AC current positive. The capacitive-filter interpretation from the reconstruction is retained.",
    ),
    (
        "Converter filter KCL and reactor KVL",
        r"I_c=I_s+jB_fV_f,\qquad V_c=V_f+Z_cI_c,\qquad S_c=V_c\overline{I_c}",
        "The printed P3 Eq9 and Table3 filter notation are not used literally; those interpretations were audited in the previous reconstruction.",
    ),
    (
        "LF station parameter referral",
        r"r_T=r_{T0}/\nu^2,\quad x_T=x_{T0}F_r/\nu^2,\quad B_f=B_{f0}F_r\nu^2",
        "The reactor follows the same R/X referral. Physical R, L, C are held fixed as a declared benchmark extrapolation. HVAC station parameters stay at their 60 Hz reference. Candidate converter voltage capability scales with candidate design voltage; no OEM design equivalence is claimed.",
    ),
    (
        "Converter losses and DC withdrawal",
        r"P_{\rm sw}=a_0+a_1 I_{\rm ref}+a_2I_{\rm ref}^2,\quad P_{\rm dc}=\Re(S_c)+P_{\rm sw}",
        "For LF stations Iref=|Ic|/nu, referred to the 500 kV reference current. a0=.011033, a1=.00346; a2=.0044 (rectifier) or .0066 (inverter). Transformer/reactor Joule loss is Re(ZT)|Is|²+Re(Zc)|Ic|²; it is already included in Re(Sc)-Ps and is not counted twice. Values of a and powers use the 100 MVA per-unit base.",
    ),
    (
        "DC and B2B power balance",
        r"v_d\odot(G_dv_d)=-P_{\rm dc},\quad P_{{\rm dc},4}+P_{{\rm dc},6}=0,\quad P_{{\rm dc},5}+P_{{\rm dc},7}=0",
        "Positive Pdc denotes withdrawal from a DC link. The 3-terminal DC grid keeps v24=1.0 p.u.; its unknown slack withdrawal determines VSC2 active power. VSC4 and VSC7 active powers are dependent B2B balance outputs.",
    ),
    (
        "Physical terminal line currents",
        r"I_{ij}=(V_i-V_j)/z_{ij}+j(b_{ij}/2)V_i",
        "Check BOTH line-end currents, including charging. Convert AC currents to amperes with the applicable line-voltage current base. The 500 kV HVAC tie current base differs from the 345 kV internal branch base.",
    ),
    (
        "Generator bounds",
        r"0\leq P_g\leq P_g^{\max},\quad |Q_g|\leq Q_g^{\max},\quad P_g^2+Q_g^2\leq (S_g^{\max})^2",
        "All ratings are declared synthetic assumptions. Non-slack Pg is controlled; slack Pg and voltage-controlled Qg are PF outputs. They remain subject to the same bounds.",
    ),
    (
        "Voltage and converter limits",
        r"0.94\leq |V_i|\leq1.06,\quad I_{{\rm ref},c}\leq2.1,\quad0.85\leq|V_c|\leq1.20",
        "DC voltages are bounded 0.95–1.05 p.u. The 2.1 p.u. reference current bound is 1.05 × an assumed 200 MVA station rating divided by 100 MVA. Seven generator voltage targets are controlled within 0.96–1.04 p.u.; this extends P1's fixed generator Vsch. Both-end line-current and active-power limits also apply; ratings are listed in the assumptions table.",
    ),
    (
        "Fixed transfer obligations",
        r"P_{{\rm VSC}1}=-200\ {\rm MW},\quad P_{{\rm VSC}3}=100\ {\rm MW},\quad P_{g,20}=200\ {\rm MW}",
        "The optimizer cannot reduce the required LF generation/DC transfers to zero to obtain an artificial loss saving. VSC5 allocates the LF delivery between its two receiving connections; dependent flows are constrained to the intended directions.",
    ),
    (
        "Total losses",
        r"L=P_{{\rm HVAC}\,lines}+P_{{\rm LFAC}\,lines}+P_{{\rm DC}\,lines}+\sum_c(P_{{\rm sw},c}+P_{J,c})",
        "Independent conservation check: total generation minus fixed demand equals L. Fixed demand makes minimization of generation equivalent to minimization of total losses.",
    ),
    (
        "Planning objective, P1-style",
        r"\min_{u,V_{\rm rated},F}\ L(u,V_{\rm rated},F)+\lambda V_{\rm rated}",
        "Compute separate minimum-load and peak-load solutions for lambda = 0, .1, .2, .3, .4 MW/kV. This corresponds to P1's generation-plus-voltage weighted objective up to a demand-dependent constant. These weights are not economic capital costs. Unlike P1 planning, this study retains line operating limits.",
    ),
    (
        "Planning ranges",
        r"69\leq V_{\rm rated}\leq500\ {\rm kV},\qquad1\leq F\leq60\ {\rm Hz}",
        "The range matches P1's numerical study. It assumes equipment appropriate to the studied frequency; V/f magnetic, transient, protection and OEM constraints have not been modeled. Choose the next declared design voltage above the peak lambda=.1 solution, then verify all operation points.",
    ),
    (
        "Operation objectives",
        r"J_B=\min_{u}\ L(u,V^*,10\ {\rm Hz}),\qquad J_C=\min_{u,F}\ L(u,V^*,F)",
        "A evaluates given feasible schedules at 10 Hz; B optimizes dispatch at 10 Hz; C also optimizes frequency. All use the same selected voltage rating, demand and transfer obligations. No capacitor-switching term exists because this case has no dispatchable capacitor banks.",
    ),
    (
        "Optional operational multi-objective extension",
        r"J_{\beta}=L+\beta\sum_i(|V_i|-1)^2",
        "Separate peak-hour tradeoff for beta = 0, 1000, 5000 MW/p.u.². This objective is our extension, not P1's capacitor-switching objective; it does not replace the main B/C loss comparison.",
    ),
    (
        "PF-constrained reduced optimization",
        r"R(s,u)=0,\qquad\min_u J(s(u),u),\qquad c(s(u),u)\geq0",
        "The inner sequential PF determines dependent state s. The outer SciPy SLSQP optimizer operates on bounded controls, scaled to a unit box; shared central finite differences supply objective/constraint derivatives. This is a continuous local NLP, not P1's custom mixed-integer PCPDIPM solver.",
    ),
    (
        "In-place Gauss-Seidel bus update",
        r"\widetilde V_i^{k+1}=\left[\overline{S_i/V_i^k}-\sum_{j<i}Y_{ij}\widetilde V_j^{k+1}-\sum_{j>i}Y_{ij}V_j^k\right]/Y_{ii}",
        "Earlier bus values in the sweep are already updated; later bus values are from the previous sweep. At a PV bus, Q is recovered from the current nodal injection before the update, then its specified magnitude is restored. Slack buses are skipped. This is why arbitrary parallelization of bus updates would change the method.",
    ),
    (
        "GS sweep and Anderson acceleration",
        r"V^{k+1}=V^k+g^k-(\Delta V+\Delta g)\gamma^k,\qquad \gamma^k=\arg\min_\gamma\|g^k-\Delta g\gamma\|_2",
        "g is the in-place GS fixed-point residual. History depth 8, coefficient threshold 1000, PV magnitude projection, P/Q mismatch tolerance 1e-10 p.u., DC/B2B interface tolerance 1e-9 p.u. Same Python source and Numba compilation; no fastmath, no machine learning inference. Threads operate across independent scenarios, not GS bus updates.",
    ),
]


def build_formulation():
    target = ROOT / "equations"
    target.mkdir(exist_ok=True)
    fragments = []
    for i, (title, latex, prose) in enumerate(EQUATIONS, 1):
        fig = plt.figure(figsize=(13, 1.0))
        fig.patch.set_facecolor("white")
        fig.text(0.02, 0.48, "$" + latex + "$", fontsize=17, va="center")
        fig.savefig(target / f"eq{i:02}.svg", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        fragments.append(
            f'<section class="equation"><h3>E{i}. {html.escape(title)}</h3><img src="equations/eq{i:02}.svg" alt="{html.escape(latex)}"><p>{html.escape(prose)}</p></section>'
        )
    text = "".join(
        f"E{i}. {title}\n{latex}\n{prose}\n\n"
        for i, (title, latex, prose) in enumerate(EQUATIONS, 1)
    )
    (ROOT / "formulation.txt").write_text(text, encoding="utf-8")
    return "\n".join(fragments)


if __name__ == "__main__":
    build_formulation()
