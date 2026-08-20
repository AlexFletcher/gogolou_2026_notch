#!/usr/bin/env python3
"""
ens_pipeline.py  —  end-to-end modelling pipeline for Gogolou et al.
====================================================================
Fits the four differentiation scenarios (CC/CV/VC/VV) under the contaminant
model structure with NUTS (NumPyro), then regenerates every data panel for
Figure 5 and supplementary figures S4 and S5, and reports the posterior summary
statistics (credible intervals, posterior contraction, cross-condition
contrasts) supporting the quantitative claims in the Results text.

DMSO parameters are inferred freely; DAPT is fitted with P0 FIXED to the 
DMSO-inferred value (the cultures are identical at D9, before treatment diverges).

Model (cell numbers; T = P+N+G+U):
    dP/dt = (alpha - k(t)) P
    dN/dt = k(t) rho(t) P
    dG/dt = k(t) (1-rho(t)) P          (glia: no self-renewal)
    dU/dt = alpha_U U                  (decoupled co-emerging population)
    IC at D9: P=P0, U=1-P0, N=G=0.
    k(t)   = k0            or  k0 + k1 (t-9)
    rho(t) = rho0          or  clip[rho0 + rho1 (t-9), 0, 1]
Observation model: beta-binomial (neuron field) + Dirichlet-multinomial
(SOX10/S100b field), conditioning on field totals.

Usage:
    python ens_pipeline.py            # inference (if chains absent), then figures + statistics
    python ens_pipeline.py --infer    # force re-run inference, then figures + statistics
    python ens_pipeline.py --figures  # only regenerate figures (chains must exist)
    python ens_pipeline.py --stats    # only report posterior statistics (chains must exist)
Requires: numpyro, jax (inference); matplotlib, numpy (figures).
"""
import os, csv, glob, re, time, argparse
import numpy as np
 
# ============================ CONFIG =================================
HUCD_CSV = "/mnt/user-data/uploads/cellmarker_hucd.csv"
PG_CSV   = "/mnt/user-data/uploads/cellmarker_sox10_s100b.csv"
CHAIN    = "nuts_{cond}_{model}.npz"          # posterior chain filename pattern
OUTDIR   = "/mnt/user-data/outputs"
NDRAW    = 50                                  # posterior draws shown in S3/S4 A-D
WARMUP, SAMPLES, CHAINS, TARGET = 1200, 1000, 4, 0.92
rng = np.random.default_rng(0)
 
MODELS  = {"CC": (False, False), "CV": (False, True),
           "VC": (True, False),  "VV": (True, True)}     # (rate varying?, bias varying?)
ORDER4  = ["CC", "VC", "CV", "VV"]                        # display / panel order
DAYS    = [15, 22]
T0, DT, NSTEP = 9.0, 0.25, int(round((22 - 9.0) / 0.25))  # forward-solve grid (inference)
IDX = {15: 23, 22: 51}                                    # trajectory indices for D15/D22
 
# lazy jax/numpyro (so --figures works without them installed)
try:
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import numpyro, numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro.diagnostics import summary as npsummary
    numpyro.set_host_device_count(CHAINS)
    _HAVE_JAX = True
except Exception:
    _HAVE_JAX = False
 
# ============================ DATA ==================================
def _read(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
 
def load_recs():
    """Aggregate field counts to per-biological-replicate counts (for inference)."""
    recs = {}
    for r in _read(HUCD_CSV):
        k = (r["condition"].strip().lower(), int(r["day"]), int(r["bio_rep"])); d = recs.setdefault(k, {})
        d["M_N"] = d.get("M_N", 0) + int(float(r["total_cells"])); d["y"] = d.get("y", 0) + int(float(r["total_hucd"]))
    for r in _read(PG_CSV):
        k = (r["condition"].strip().lower(), int(r["day"]), int(r["bio_rep"])); d = recs.setdefault(k, {})
        d["M_PG"] = d.get("M_PG", 0) + int(float(r["total_cells"]))
        d["nP"] = d.get("nP", 0) + int(float(r["sox10-s100b"])); d["nG"] = d.get("nG", 0) + int(float(r["total_s100b"]))
    for d in recs.values():
        if "M_PG" in d: d["nR"] = max(d["M_PG"] - d.get("nP", 0) - d.get("nG", 0), 0)
    return {k: v for k, v in recs.items() if "M_N" in v and "M_PG" in v}
 
def _fields(rows, valcol):
    o = {}
    for r in rows:
        c = r["condition"].strip().lower(); d = int(r["day"]); tot = float(r["total_cells"])
        if tot > 0: o.setdefault((c, d), []).append((int(r["bio_rep"]), float(r[valcol]) / tot))
    return o
 
_DATA = None
def _data():
    global _DATA
    if _DATA is None:
        H, P = _read(HUCD_CSV), _read(PG_CSV)
        _DATA = {"P": _fields(P, "sox10-s100b"), "G": _fields(P, "total_s100b"), "N": _fields(H, "total_hucd")}
    return _DATA
def tech(cell, cond, day): return np.array([f for b, f in _data()[cell].get((cond, day), [])])
def biol(cell, cond, day):
    d = {}
    for b, f in _data()[cell].get((cond, day), []): d.setdefault(b, []).append(f)
    return np.array([np.mean(v) for v in d.values()])
 
# ===================== FORWARD MODEL (JAX, inference) ===============
def jax_forward(P0, k0, k1, r0, r1, tvk, tvr, alpha, aU):
    def kf(t): return k0 + (k1 * (t - T0) if tvk else 0.0)
    def rf(t): return r0 + (r1 * (t - T0) if tvr else 0.0)
    def der(P, N, G, U, t):
        k = jnp.maximum(kf(t), 1e-6); rho = jnp.clip(rf(t), 0.0, 1.0)
        return ((alpha - k) * P, k * rho * P, k * (1 - rho) * P, aU * U)
    def step(c, i):
        P, N, G, U = c; t = T0 + i * DT
        a1, b1, c1, d1 = der(P, N, G, U, t)
        a2, b2, c2, d2 = der(P + .5*DT*a1, N + .5*DT*b1, G + .5*DT*c1, U + .5*DT*d1, t + .5*DT)
        a3, b3, c3, d3 = der(P + .5*DT*a2, N + .5*DT*b2, G + .5*DT*c2, U + .5*DT*d2, t + .5*DT)
        a4, b4, c4, d4 = der(P + DT*a3, N + DT*b3, G + DT*c3, U + DT*d3, t + DT)
        P = jnp.maximum(P + DT/6*(a1+2*a2+2*a3+a4), 1e-12); N = jnp.maximum(N + DT/6*(b1+2*b2+2*b3+b4), 1e-12)
        G = jnp.maximum(G + DT/6*(c1+2*c2+2*c3+c4), 1e-12); U = jnp.maximum(U + DT/6*(d1+2*d2+2*d3+d4), 1e-12)
        return (P, N, G, U), (P, N, G, U)
    _, tr = jax.lax.scan(step, (P0, 1e-12, 1e-12, 1.0 - P0), jnp.arange(NSTEP))
    Ps, Ns, Gs, Us = tr
    def fr(ix):
        P, N, G, U = Ps[ix], Ns[ix], Gs[ix], Us[ix]; T = P + N + G + U
        return jnp.array([P / T, N / T, G / T, U / T])
    return jnp.stack([fr(IDX[15]), fr(IDX[22])])          # (2,4): rows D15,D22; cols P,N,G,U
 
# ===================== FORWARD MODEL (numpy, figures) ==============
def sim_traj(P0, k0, k1, r0, r1, alpha, aU, tvk, tvr, ts, dt=0.05):
    P, N, G, U = P0, 1e-12, 1e-12, 1 - P0; t = 9.0; i = 0; out = {"P": [], "N": [], "G": []}; tg = list(ts)
    while i < len(tg):
        if t >= tg[i] - 1e-9:
            T = P + N + G + U; out["P"].append(P/T); out["N"].append(N/T); out["G"].append(G/T); i += 1; continue
        k = max(k0 + (k1*(t-9) if tvk else 0), 1e-6); rho = min(max(r0 + (r1*(t-9) if tvr else 0), 0), 1)
        P = max(P + dt*(alpha-k)*P, 1e-12); N = max(N + dt*k*rho*P, 1e-12)
        G = max(G + dt*k*(1-rho)*P, 1e-12); U = max(U + dt*aU*U, 1e-12); t += dt
    return {c: np.array(v) for c, v in out.items()}
 
def chain(cond, m): return np.load(CHAIN.format(cond=cond, model=m))
def col(ch, key):   return ch[key].reshape(-1) if key in ch.files else None
def med(cond, m, key):
    ch = chain(cond, m); return float(np.median(ch[key])) if key in ch.files else 0.0
def sim_median(cond, m, ts):
    tvk, tvr = MODELS[m]
    return sim_traj(med(cond, m, "P0"), med(cond, m, "k0"), med(cond, m, "k1"),
                    med(cond, m, "rho0"), med(cond, m, "rho1"), med(cond, m, "alpha"), med(cond, m, "aU"),
                    tvk, tvr, ts)
 
# ========================= NUTS INFERENCE ===========================
def make_model(tvk, tvr, fixed_P0=None):
    def model(nb_didx, nb_M, nb_y, dm_didx, dm_M, dm_cnt):
        P0 = fixed_P0 if fixed_P0 is not None else numpyro.sample("P0", dist.Uniform(0.1, 0.9))
        phiN = jnp.exp(numpyro.sample("lphiN", dist.Uniform(jnp.log(1.), jnp.log(2000.))))
        phiPG = jnp.exp(numpyro.sample("lphiPG", dist.Uniform(jnp.log(1.), jnp.log(2000.))))
        k0 = numpyro.sample("k0", dist.Uniform(0.01, 1.0))
        k1 = numpyro.sample("k1", dist.Normal(0.0, 0.05)) if tvk else 0.0
        rho0 = numpyro.sample("rho0", dist.Uniform(0.0, 1.0))
        rho1 = numpyro.sample("rho1", dist.Normal(0.0, 0.05)) if tvr else 0.0
        alpha = numpyro.sample("alpha", dist.Uniform(-0.4, 0.4))
        aU = numpyro.sample("aU", dist.Uniform(-0.4, 0.4))
        numpyro.deterministic("phiN", phiN); numpyro.deterministic("phiPG", phiPG)
        fr = jax_forward(P0, k0, k1, rho0, rho1, tvk, tvr, alpha, aU)
        piN = jnp.clip(fr[:, 1], 1e-9, 1 - 1e-9)
        pn = piN[nb_didx]; a = phiN * pn; b = phiN * (1 - pn)
        with numpyro.plate("nb", nb_y.shape[0]):
            numpyro.sample("y_obs", dist.BetaBinomial(a, b, total_count=nb_M), obs=nb_y)
        piP = jnp.clip(fr[:, 0], 1e-9, 1.); piG = jnp.clip(fr[:, 2], 1e-9, 1.); piR = jnp.clip(fr[:, 1] + fr[:, 3], 1e-9, 1.)
        vv = jnp.stack([piP, piG, piR], axis=1); vv = vv / vv.sum(axis=1, keepdims=True)
        conc = (phiPG * vv)[dm_didx]
        with numpyro.plate("dm", dm_cnt.shape[0]):
            numpyro.sample("pg_obs", dist.DirichletMultinomial(conc, total_count=dm_M), obs=dm_cnt)
    return model
 
def cond_arrays(recs, cond):
    nb_didx = []; nb_M = []; nb_y = []; dm_didx = []; dm_M = []; dm_cnt = []; dmap = {15: 0, 22: 1}
    for d in DAYS:
        for r in range(1, 6):
            v = recs.get((cond, d, r))
            if v is None: continue
            nb_didx.append(dmap[d]); nb_M.append(v["M_N"]); nb_y.append(v["y"])
            dm_didx.append(dmap[d]); dm_M.append(v["M_PG"]); dm_cnt.append([v["nP"], v["nG"], v["nR"]])
    return (jnp.array(nb_didx), jnp.array(nb_M), jnp.array(nb_y, dtype=jnp.int32),
            jnp.array(dm_didx), jnp.array(dm_M), jnp.array(dm_cnt, dtype=jnp.int32))
 
def fit(recs, cond, m, fixed_P0=None, seed=0):
    tvk, tvr = MODELS[m]; args = cond_arrays(recs, cond)
    kernel = NUTS(make_model(tvk, tvr, fixed_P0), target_accept_prob=TARGET)
    mcmc = MCMC(kernel, num_warmup=WARMUP, num_samples=SAMPLES, num_chains=CHAINS,
                chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), *args)
    post = mcmc.get_samples(group_by_chain=True); return post, npsummary(post)
 
def run_inference():
    if not _HAVE_JAX:
        raise SystemExit("NumPyro/JAX not available — install them to run inference (figures still work with existing chains).")
    recs = load_recs(); REPORT = ["k0", "k1", "rho0", "rho1", "alpha", "aU", "phiN", "phiPG", "P0"]
    lines = ["=== NUTS diagnostics (4 chains). DMSO: P0 inferred. DAPT: P0 fixed to the "
             "DMSO median (shared initial progenitor fraction). Normal(0,0.05) priors on k1, rho1. ===\n"]
    # --- DMSO (free P0) ---
    dmso_P0 = {}
    for m in ORDER4:
        t = time.time(); post, S = fit(recs, "dmso", m)
        np.savez(CHAIN.format(cond="dmso", model=m), **{k: np.array(v) for k, v in post.items()})
        dmso_P0[m] = float(np.median(np.array(post["P0"])))
        wr = max(float(S[p]["r_hat"]) for p in S if p not in ("y_obs", "pg_obs"))
        print(f"dmso {m}: P0={dmso_P0[m]:.3f} worstRhat={wr:.4f} t={time.time()-t:.0f}s", flush=True)
        lines += _diag_block("DMSO", m, S, REPORT)
    # --- DAPT (P0 fixed to DMSO median, per model) ---
    for m in ORDER4:
        t = time.time(); post, S = fit(recs, "dapt", m, fixed_P0=dmso_P0[m])
        out = {k: np.array(v) for k, v in post.items()}
        out["P0"] = np.full(np.array(post["k0"]).shape, dmso_P0[m])   # record fixed P0
        np.savez(CHAIN.format(cond="dapt", model=m), **out)
        wr = max(float(S[p]["r_hat"]) for p in S if p not in ("y_obs", "pg_obs"))
        print(f"dapt {m}: P0={dmso_P0[m]:.3f} (fixed) worstRhat={wr:.4f} t={time.time()-t:.0f}s", flush=True)
        lines += _diag_block("DAPT [P0 fixed]", m, S, REPORT, fixed_P0=dmso_P0[m])
    open(f"{OUTDIR}/nuts_full_diagnostics.txt", "w").write("\n".join(lines))
    print("inference complete; chains + diagnostics written")
 
def _diag_block(tag, m, S, REPORT, fixed_P0=None):
    wr = max(float(S[p]["r_hat"]) for p in S if p not in ("y_obs", "pg_obs"))
    me = min(float(S[p]["n_eff"]) for p in S if p not in ("y_obs", "pg_obs"))
    out = [f"-- {tag} {m} --  (worst Rhat={wr:.4f}, min n_eff={me:.0f})"]
    for p in REPORT:
        if p in S:
            r = float(S[p]["r_hat"]); n = float(S[p]["n_eff"]); fl = "  <--" if (r > 1.01 or n < 400) else ""
            out.append(f"   {p:6s}: {float(S[p]['median']):.3f} [{float(S[p]['5.0%']):.3f}, {float(S[p]['95.0%']):.3f}]   Rhat={r:.4f}  n_eff={n:.0f}{fl}")
        elif p == "P0" and fixed_P0 is not None:
            out.append(f"   {p:6s}: {fixed_P0:.3f}  [fixed to DMSO value]")
    return out + [""]
 
# ============================ FIGURES ===============================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
 
# exact manuscript palette (light = DMSO, dark = DAPT)
CELL = {"dmso": {"P": "#603C90", "N": "#D89C18", "G": "#1878B4"},
        "dapt": {"P": "#301848", "N": "#84600C", "G": "#0C3C60"}}
RATE = {"dmso": "#603C90", "dapt": "#301848"}
BIAS = {"dmso": "#D89C18", "dapt": "#84600C"}
POP  = {"dmso": "#5B8C85", "dapt": "#2E4A46"}
LINE = {"CC": ("#1A1A1A", "-"), "CV": ("#1A1A1A", "--"), "VC": ("#CC2222", "-"), "VV": ("#CC2222", "--")}
LBL, CELLTITLE = {"P": "Progenitor", "N": "Neuron", "G": "Glia"}, {"P": "Progenitors", "N": "Neurons", "G": "Glia"}
VRATE, VBIAS = ["VC", "VV"], ["CV", "VV"]
RATElab = {"CC": "C", "VC": "V", "CV": "C", "VV": "V"}; BIASlab = {"CC": "C", "VC": "C", "CV": "V", "VV": "V"}
MTITLE = {"CC": "Constant rate, constant bias", "VC": "Time-varying rate, constant bias",
          "CV": "Constant rate, time-varying bias", "VV": "Time-varying rate, time-varying bias"}
 
def save(fig, stem):
    fig.savefig(f"{OUTDIR}/{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/{stem}.svg", bbox_inches="tight"); plt.close(fig)
 
def fix_svg_fonts():
    pat = re.compile(r"font-family:[^;\"}]*"); repl = "font-family:Arial, 'Liberation Sans', Helvetica, sans-serif"
    for f in glob.glob(f"{OUTDIR}/fig5*.svg") + glob.glob(f"{OUTDIR}/figS[3-6]*.svg"):
        s = open(f, encoding="utf-8").read(); open(f, "w", encoding="utf-8").write(pat.sub(repl, s))
 
# ---- fit panels (5D, 5J) ----
def _fit_axis(ax, cond, cell, tsim):
    c = CELL[cond][cell]
    for m in ORDER4:
        lc, ls = LINE[m]; ax.plot(tsim, sim_median(cond, m, tsim)[cell], color=lc, ls=ls, lw=1.1, alpha=0.9, zorder=2)
    for day in DAYS:
        tv = tech(cell, cond, day)
        if len(tv):
            ax.boxplot([tv], positions=[day], widths=1.6, patch_artist=True, showfliers=False,
                       medianprops=dict(color="white", lw=1.3), zorder=3,
                       boxprops=dict(facecolor=c, edgecolor=c, alpha=0.55, lw=0),
                       whiskerprops=dict(color=c, lw=1), capprops=dict(color=c, lw=1))
            ax.scatter(np.full_like(tv, day) + rng.uniform(-0.5, 0.5, len(tv)), tv, s=7, color=c, alpha=0.5, ec="none", zorder=4)
            bv = biol(cell, cond, day)
            ax.scatter(np.full_like(bv, day) + rng.uniform(-0.35, 0.35, len(bv)), bv, s=26, facecolors="none", edgecolors=c, lw=1.2, zorder=5)
    ax.set_xlim(8, 23); ax.set_xticks([9, 15, 22]); ax.set_ylim(0, 1.0)
    ax.set_title(CELLTITLE[cell], color=c, fontsize=10, fontweight="bold", pad=3)
    ax.set_xlabel("Time (days)", fontsize=8.5); ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
 
def draw_fit(fig, gs_row, cond, rowlabel=""):
    tsim = np.linspace(9, 22, 80); first = None
    for j, cell in enumerate(["P", "N", "G"]):
        ax = fig.add_subplot(gs_row[j])
        if j == 0: first = ax; ax.set_ylabel("Cell fraction", fontsize=9)
        else: ax.set_yticklabels([])
        _fit_axis(ax, cond, cell, tsim)
    if rowlabel:
        first.text(-0.55, 0.5, rowlabel, transform=first.transAxes, rotation=90, ha="center", va="center",
                   fontsize=11, fontweight="bold", color="#333")
 
def fit_legend(fig):
    h = [Line2D([0], [0], color=LINE[m][0], ls=LINE[m][1], lw=1.2, label=f"{m[0]}/{m[1]}") for m in ORDER4]
    fig.legend(handles=h, title="Rate/Bias", loc="upper right", fontsize=6.5, title_fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.995, 0.99))
 
# ---- violin panels (5E-H, 5K-N) ----
def _violins(ax, datasets, positions, colors, width=0.75):
    parts = ax.violinplot(datasets, positions=positions, widths=width, showextrema=False)
    for pc, c in zip(parts["bodies"], colors): pc.set_facecolor(c); pc.set_edgecolor("none"); pc.set_alpha(1.0)
    for d, x in zip(datasets, positions): ax.hlines(np.median(d), x - width/2, x + width/2, color="white", lw=1.6, zorder=4)
 
def _annot(ax, positions, models, conds=None):
    tr = ax.get_xaxis_transform(); y0 = -0.10
    ax.text(-0.02, y0, "Rate:", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#333")
    ax.text(-0.02, y0 - 0.09, "Bias:", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#333")
    if conds is not None:
        ax.text(-0.02, y0 - 0.18, "NOTCHi:", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#333")
    for i, (x, m) in enumerate(zip(positions, models)):
        ax.text(x, y0, RATElab[m], transform=tr, ha="center", va="top", fontsize=7, color="#333")
        ax.text(x, y0 - 0.09, BIASlab[m], transform=tr, ha="center", va="top", fontsize=7, color="#333")
        if conds is not None:
            ax.text(x, y0 - 0.18, "\u2212" if conds[i] == "dmso" else "+", transform=tr, ha="center", va="top", fontsize=8, color="#333")
 
def _vstyle(ax, ylabel, title, hline=None):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    if hline is not None: ax.axhline(hline, ls=":", color="#aaa", lw=0.9, zorder=0)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=9.5); ax.tick_params(labelsize=8)
 
def violin_single(ax, cond, param, models, ylabel, title, colmap, hline=None):
    data = [col(chain(cond, m), param) for m in models]; pos = list(range(len(models)))
    _violins(ax, data, pos, [colmap[cond]] * len(models)); _annot(ax, pos, models)
    _vstyle(ax, ylabel, title, hline); ax.set_xlim(-0.7, len(models) - 0.3)
 
def violin_combined(ax, param, models, ylabel, title, colmap, hline=None):
    data = []; pos = []; cols = []; mods = []; conds = []; x = 0.0
    for cond in ["dmso", "dapt"]:
        for m in models:
            data.append(col(chain(cond, m), param)); pos.append(x); cols.append(colmap[cond]); mods.append(m); conds.append(cond); x += 1.0
        x += 0.8
    _violins(ax, data, pos, cols); _annot(ax, pos, mods, conds)
    _vstyle(ax, ylabel, title, hline); ax.set_xlim(-0.7, pos[-1] + 0.7)
 
# ---- S3/S4 PPC panels (A-D) ----
def draw_ppc(ax, cond, m, ndraw=NDRAW):
    tsim = np.linspace(9, 22, 60); ch = chain(cond, m); tvk, tvr = MODELS[m]
    n = len(ch["k0"].reshape(-1)); idx = rng.choice(n, ndraw, replace=False)
    g = lambda k: (ch[k].reshape(-1) if k in ch.files else np.zeros(n))
    P0, al, aU, k0, r0, k1, r1 = g("P0"), g("alpha"), g("aU"), g("k0"), g("rho0"), g("k1"), g("rho1")
    for cell in ["P", "N", "G"]:
        c = CELL[cond][cell]
        for j in idx:
            tr = sim_traj(P0[j], k0[j], k1[j], r0[j], r1[j], al[j], aU[j], tvk, tvr, tsim)
            ax.plot(tsim, tr[cell] * 100, color=c, lw=0.4, alpha=0.16, zorder=1)
    for cell in ["P", "N", "G"]:
        c = CELL[cond][cell]
        for day in DAYS:
            tv = tech(cell, cond, day) * 100
            if len(tv):
                ax.boxplot([tv], positions=[day], widths=1.5, patch_artist=True, showfliers=False,
                           medianprops=dict(color="white", lw=1.2), zorder=3,
                           boxprops=dict(facecolor=c, edgecolor=c, alpha=0.55, lw=0),
                           whiskerprops=dict(color=c, lw=0.9), capprops=dict(color=c, lw=0.9))
                bv = biol(cell, cond, day) * 100
                ax.scatter(np.full_like(bv, day) + rng.uniform(-0.3, 0.3, len(bv)), bv, s=16, facecolors="none", edgecolors=c, lw=1.0, zorder=5)
    ax.set_xlim(8, 23); ax.set_xticks([9, 15, 22]); ax.set_ylim(0, 100)
    ax.set_xlabel("Time (days)", fontsize=8.5); ax.set_ylabel("% of cells", fontsize=8.5)
    ax.set_title(MTITLE[m], fontsize=9); ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], color=CELL[cond][c], lw=2, label=LBL[c]) for c in ["P", "N", "G"]],
              fontsize=6.5, frameon=False, loc="upper right")
 
# ---- S3/S4 parameter marginals (E) ----
EPARAMS = [("P0", "$P_0$", "ALL", "POP"), ("alpha", r"$\alpha$", "ALL", "POP"), ("aU", r"$\alpha_U$", "ALL", "POP"),
           ("k0", "$k_0$", "ALL", "RATE"), ("k1", "$k_1$", "VRATE", "RATE"), ("rho0", r"$\rho^0$", "ALL", "BIAS"),
           ("rho1", r"$\rho^1$", "VBIAS", "BIAS")]
MSET = {"ALL": ORDER4, "VRATE": VRATE, "VBIAS": VBIAS}; GROUP = {"RATE": RATE, "BIAS": BIAS, "POP": POP}
 
def draw_params(fig, gs_area, cond):
    inner = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_area, hspace=0.75, wspace=0.55)
    for idx, (key, lab, mkey, grp) in enumerate(EPARAMS):
        ax = fig.add_subplot(inner[idx // 4, idx % 4]); models = MSET[mkey]; c = GROUP[grp][cond]
        data = [col(chain(cond, m), key) for m in models]
        parts = ax.violinplot(data, positions=range(len(models)), widths=0.8, showextrema=False)
        for pc in parts["bodies"]: pc.set_facecolor(c); pc.set_edgecolor("none"); pc.set_alpha(1.0)
        for d, x in zip(data, range(len(models))): ax.hlines(np.median(d), x - 0.4, x + 0.4, color="white", lw=1.3, zorder=4)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=6.5)
        ax.set_title(lab, fontsize=10); ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.add_subplot(inner[1, 3]).axis("off")
 
# ---- builders ----
def build_fig5_fits():
    for cond, tag in [("dmso", "fig5D_fit_dmso"), ("dapt", "fig5J_fit_dapt")]:
        fig = plt.figure(figsize=(6.6, 2.5))
        gs = gridspec.GridSpec(1, 3, wspace=0.12, left=0.09, right=0.99, top=0.82, bottom=0.22)
        draw_fit(fig, [gs[0, 0], gs[0, 1], gs[0, 2]], cond); fit_legend(fig); save(fig, tag)
    fig = plt.figure(figsize=(6.8, 5.2))
    gs = gridspec.GridSpec(2, 3, wspace=0.12, hspace=0.55, left=0.09, right=0.99, top=0.9, bottom=0.09)
    draw_fit(fig, [gs[0, 0], gs[0, 1], gs[0, 2]], "dmso", "DMSO")
    draw_fit(fig, [gs[1, 0], gs[1, 1], gs[1, 2]], "dapt", "DAPT")
    fig.suptitle("Figure 5 panels D & J \u2014 posterior-predictive fits\n"
                 "lines = 4 models (black=const rate / red=varying rate; solid=const bias / dashed=varying bias); "
                 "boxes/dots = data (open = biological-replicate means)", fontsize=8.5, y=0.99)
    save(fig, "fig5DJ_fits_overview")
 
def build_fig5_violins():
    RL = "initial rate, k$^0$ (day$^{-1}$)"; GL = "gradient, k$^1$ (day$^{-2}$)"
    BL = r"neuronal bias, $\rho^0$"; BG = r"gradient, $\rho^1$ (day$^{-1}$)"
    for stem, p, mods, yl, ti, cm, hl in [
        ("fig5E_rate_dmso", "k0", ORDER4, RL, "Differentiation rate (DMSO)", RATE, None),
        ("fig5F_grad_dmso", "k1", VRATE, GL, "Rate gradient (DMSO)", RATE, 0),
        ("fig5G_bias_dmso", "rho0", ORDER4, BL, "Neuronal bias (DMSO)", BIAS, 0.5),
        ("fig5H_biasgrad_dmso", "rho1", VBIAS, BG, "Bias gradient (DMSO)", BIAS, 0)]:
        size = (2.4, 3.0) if p in ("k1", "rho1") else (3.0, 3.0)
        fig, ax = plt.subplots(figsize=size); violin_single(ax, "dmso", p, mods, yl, ti, cm, hl); fig.tight_layout(); save(fig, stem)
    for stem, p, mods, yl, ti, cm, hl, size in [
        ("fig5K_rate_dapt", "k0", ORDER4, RL, "Differentiation rate", RATE, None, (4.2, 3.0)),
        ("fig5L_grad_dapt", "k1", VRATE, GL, "Rate gradient", RATE, 0, (3.0, 3.0)),
        ("fig5M_bias_dapt", "rho0", ORDER4, BL, "Neuronal bias", BIAS, 0.5, (4.2, 3.0)),
        ("fig5N_biasgrad_dapt", "rho1", VBIAS, BG, "Bias gradient", BIAS, 0, (3.0, 3.0))]:
        fig, ax = plt.subplots(figsize=size); violin_combined(ax, p, mods, yl, ti, cm, hl); fig.tight_layout(); save(fig, stem)
    fig = plt.figure(figsize=(12, 9)); gs = gridspec.GridSpec(2, 4, hspace=0.5, wspace=0.42)
    violin_single(fig.add_subplot(gs[0, 0]), "dmso", "k0", ORDER4, RL, "Differentiation rate (DMSO)", RATE)
    violin_single(fig.add_subplot(gs[0, 1]), "dmso", "k1", VRATE, GL, "Rate gradient (DMSO)", RATE, 0)
    violin_single(fig.add_subplot(gs[0, 2]), "dmso", "rho0", ORDER4, BL, "Neuronal bias (DMSO)", BIAS, 0.5)
    violin_single(fig.add_subplot(gs[0, 3]), "dmso", "rho1", VBIAS, BG, "Bias gradient (DMSO)", BIAS, 0)
    violin_combined(fig.add_subplot(gs[1, 0]), "k0", ORDER4, RL, "Differentiation rate", RATE)
    violin_combined(fig.add_subplot(gs[1, 1]), "k1", VRATE, GL, "Rate gradient", RATE, 0)
    violin_combined(fig.add_subplot(gs[1, 2]), "rho0", ORDER4, BL, "Neuronal bias", BIAS, 0.5)
    violin_combined(fig.add_subplot(gs[1, 3]), "rho1", VBIAS, BG, "Bias gradient", BIAS, 0)
    fig.suptitle("Figure 5 marginal-posterior panels \u2014 violin plots\n"
                 "white line = median; purple = rate, gold = neuronal bias; lighter = DMSO, darker = DAPT", fontsize=10, y=1.0)
    save(fig, "fig5_violin_panels_overview")
 
def build_supp(cond, tag):
    for letter, m in zip("ABCD", ORDER4):
        fig, ax = plt.subplots(figsize=(3.4, 3.0)); draw_ppc(ax, cond, m); fig.tight_layout(); save(fig, f"fig{tag}{letter}_ppc_{m}_{cond}")
    fig = plt.figure(figsize=(8.5, 4.2)); draw_params(fig, gridspec.GridSpec(1, 1)[0, 0], cond); save(fig, f"fig{tag}E_params_{cond}")
    fig = plt.figure(figsize=(12, 8)); gs = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.35)
    for k, (letter, m) in enumerate(zip("ABCD", ORDER4)): draw_ppc(fig.add_subplot(gs[0, k]), cond, m)
    draw_params(fig, gs[1, :], cond)
    cn = "DMSO" if cond == "dmso" else "DAPT"
    fig.suptitle(f"Figure {tag} ({cn}) \u2014 posterior predictive checks (A\u2013D, {NDRAW} draws) and parameter marginals (E)", fontsize=10, y=0.98)
    save(fig, f"fig{tag}_overview")
 
# ---- Figure S6: ENS-compartment-normalised proportions ----
# The reviewer-requested view: proportions taken relative to the ENS compartment
# (P+N+G) rather than to all cells. Because U is dynamically uncoupled from the
# ENS lineages and the (P,N,G) subsystem is homogeneous linear, these normalised
# proportions are independent of both alpha_U and P0 -- no re-fitting is needed,
# the same posteriors are simply re-expressed.
ENSC = {"dmso": "#555555", "dapt": "#222222"}
ENSTITLE = {"E": "ENS fraction of all cells", "P": "Progenitors (ENS-normalised)",
            "N": "Neurons (ENS-normalised)", "G": "Glia (ENS-normalised)"}
ENSKEYS = ["E", "P", "N", "G"]
 
 
def biol_map(cell, cond, day):
    """Per-biological-replicate mean fraction of all cells, keyed by replicate id."""
    d = {}
    for b, f in _data()[cell].get((cond, day), []):
        d.setdefault(b, []).append(f)
    return {b: float(np.mean(v)) for b, v in d.items()}
 
 
def ens_biol(cond, day):
    """Empirical ENS-normalised proportions, per biological replicate.
 
    The neuron (HuC/D) and progenitor/glia (SOX10/S100b) counts come from
    SEPARATE stain fields, so the ENS denominator must be formed by combining
    the two fields for the SAME biological replicate -- matched on replicate id,
    not on position in the file. This assumes, as the observation model already
    does, that both fields sample the same underlying population.
    """
    mP, mN, mG = (biol_map(c, cond, day) for c in ("P", "N", "G"))
    reps = sorted(set(mP) & set(mN) & set(mG))
    P = np.array([mP[b] for b in reps])
    N = np.array([mN[b] for b in reps])
    G = np.array([mG[b] for b in reps])
    E = P + N + G
    return {"E": E, "P": P / E, "N": N / E, "G": G / E}
 
 
def ens_model(cond, m, ts):
    """Model trajectories re-expressed relative to the ENS compartment."""
    tr = sim_median(cond, m, ts)
    E = tr["P"] + tr["N"] + tr["G"]
    return {"E": E, "P": tr["P"] / E, "N": tr["N"] / E, "G": tr["G"] / E}
 
 
def _ens_axis(ax, cond, key, tsim):
    c = ENSC[cond] if key == "E" else CELL[cond][key]
    for m in ORDER4:
        lc, ls = LINE[m]
        ax.plot(tsim, ens_model(cond, m, tsim)[key], color=lc, ls=ls, lw=1.1, alpha=0.9, zorder=2)
    for day in DAYS:
        v = ens_biol(cond, day)[key]
        if len(v):
            ax.scatter(np.full_like(v, float(day)) + rng.uniform(-0.4, 0.4, len(v)), v,
                       s=30, facecolors="none", edgecolors=c, lw=1.3, zorder=5)
            ax.hlines(np.mean(v), day - 0.9, day + 0.9, color=c, lw=1.8, zorder=6)
    ax.set_xlim(8, 23); ax.set_xticks([9, 15, 22]); ax.set_ylim(0, 1.0)
    ax.set_title(ENSTITLE[key], color=c, fontsize=9, fontweight="bold", pad=3)
    ax.set_xlabel("Time (days)", fontsize=8.5); ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
 
 
def _ens_legend(fig, y=-0.01):
    """Legend below the axes: the upper-right position used elsewhere collides
    with the title of the rightmost panel in this four-column layout."""
    h = [Line2D([0], [0], color=LINE[m][0], ls=LINE[m][1], lw=1.2, label=f"{m[0]}/{m[1]}")
         for m in ORDER4]
    fig.legend(handles=h, title="Rate/Bias", loc="lower center", ncol=4, fontsize=7,
               title_fontsize=7, frameon=False, bbox_to_anchor=(0.5, y))
 
 
def build_figS6():
    tsim = np.linspace(9, 22, 80)
    for cond, tag in [("dmso", "S6A"), ("dapt", "S6B")]:
        fig = plt.figure(figsize=(8.8, 2.6))
        gs = gridspec.GridSpec(1, 4, wspace=0.30, left=0.07, right=0.99, top=0.80, bottom=0.22)
        for j, k in enumerate(ENSKEYS):
            ax = fig.add_subplot(gs[0, j])
            if j == 0:
                ax.set_ylabel("Proportion", fontsize=9)
            _ens_axis(ax, cond, k, tsim)
        _ens_legend(fig, y=-0.06)
        save(fig, f"fig{tag}_ens_normalised_{cond}")
    fig = plt.figure(figsize=(9.4, 5.6))
    gs = gridspec.GridSpec(2, 4, wspace=0.32, hspace=0.60, left=0.09, right=0.99, top=0.86, bottom=0.09)
    for i, (cond, lab) in enumerate([("dmso", "DMSO"), ("dapt", "DAPT")]):
        for j, k in enumerate(ENSKEYS):
            ax = fig.add_subplot(gs[i, j])
            if j == 0:
                ax.set_ylabel("Proportion", fontsize=9)
                ax.text(-0.42, 0.5, lab, transform=ax.transAxes, rotation=90, ha="center",
                        va="center", fontsize=11, fontweight="bold", color="#333")
            _ens_axis(ax, cond, k, tsim)
    _ens_legend(fig, y=-0.01)
    fig.suptitle("Figure S6 \u2014 proportions normalised to the ENS compartment (P+N+G)\n"
                 "open circles = biological replicates, bar = mean; lines = 4 models "
                 "(black=const rate / red=varying rate; solid=const bias / dashed=varying bias)",
                 fontsize=8.5, y=0.99)
    save(fig, "figS6_ens_normalised_overview")
 
 
def ens_table(path=None):
    """Numbers underlying Fig. S6, for the Results text and figure legend."""
    L = ["ENS-compartment-normalised proportions (biological-replicate means)",
         "E = ENS fraction of all cells; P/N/G normalised by E", ""]
    L.append(f"{'cond':6s} {'day':>4s} {'n_rep':>6s} {'ENS/total':>10s} "
             f"{'P/E':>7s} {'N/E':>7s} {'G/E':>7s}")
    for cond in ("dmso", "dapt"):
        for day in DAYS:
            v = ens_biol(cond, day)
            L.append(f"{cond:6s} {day:4d} {len(v['E']):6d} {np.mean(v['E']):10.3f} "
                     f"{np.mean(v['P']):7.3f} {np.mean(v['N']):7.3f} {np.mean(v['G']):7.3f}")
    L.append("")
    L.append("raw (fraction of all cells) vs ENS-normalised, D15 -> D22:")
    for cond in ("dmso", "dapt"):
        a, b = ens_biol(cond, 15), ens_biol(cond, 22)
        for k in ("P", "N", "G"):
            raw15 = np.mean(a[k] * a["E"]); raw22 = np.mean(b[k] * b["E"])
            n15, n22 = np.mean(a[k]), np.mean(b[k])
            L.append(f"  {cond} {k}: raw {raw15:.3f}->{raw22:.3f} ({raw22/raw15:.2f}x)   "
                     f"ENS-normalised {n15:.3f}->{n22:.3f} ({n22/n15:.2f}x)")
    txt = "\n".join(L)
    print(txt)
    if path:
        open(path, "w").write(txt + "\n")
 
 
 
    build_fig5_fits(); build_fig5_violins(); build_supp("dmso", "S3"); build_supp("dapt", "S4")
    build_figS6(); fix_svg_fonts()
    ens_table(f"{OUTDIR}/ens_normalised_table.txt")
    print("all figure panels written to", OUTDIR)
 
# ========================= STATISTICS ===============================
# Posterior summaries supporting the quantitative claims in the Results text:
#   (i)   level parameters (k0, rho0) are comparable across model specifications;
#   (ii)  gradient parameters (k1, rho1) have credible intervals spanning zero,
#         with posterior contraction quantifying how far the data move them off prior;
#   (iii) the implied change in rate over the observation window;
#   (iv)  cross-condition (DAPT vs DMSO) contrasts, which ARE well defined because
#         the two conditions are separate datasets fitted in separate MCMC runs.
#
# NOTE on (i): the four models are fitted to the SAME data, so a "posterior
# difference" between them is not a well-defined contrast. Credible-interval
# overlap is reported descriptively, as robustness across specifications --
# it is NOT a test of equality. Only the cross-condition contrast is a contrast.
 
CI_LEVEL = 95.0                 # equal-tailed credible interval (%)
NPAIR    = 200_000              # draws for cross-condition contrasts
PRIOR_SD = {"k1": 0.05, "rho1": 0.05}    # Normal(0, sd) priors on the gradients
STATS_SEED = 0
TSPAN = 22.0 - T0               # observation window, days (D9 -> D22)
CONDS = ["dmso", "dapt"]
CONDLBL = {"dmso": "DMSO (control)", "dapt": "DAPT (Notch-inhibited)"}
 
 
def _flat(cond, m, key):
    """Flattened posterior draws, or None if the parameter is absent."""
    return col(chain(cond, m), key)
 
 
def _is_fixed(x):
    """True if a stored 'parameter' is actually a constant (e.g. DAPT P0).
 
    Uses an exact range test: np.std on a repeated non-representable float
    returns ~1e-17 rather than 0, which would otherwise be summarised as a
    degenerate chain with n_eff ~ 4.
    """
    return np.ptp(np.asarray(x)) == 0.0
 
 
def _ci(x, level=CI_LEVEL):
    lo = (100.0 - level) / 2.0
    return np.percentile(x, [lo, 100.0 - lo])
 
 
def _pair(a, b, rng_):
    """Pair independent draws from two separate MCMC runs."""
    return a[rng_.choice(len(a), NPAIR)], b[rng_.choice(len(b), NPAIR)]
 
 
def stat_level(cond, param, models, lines):
    """Median and CrI of a level parameter across model specifications."""
    lines.append(f"  {param}: median [{CI_LEVEL:.0f}% CrI] by model")
    vals = {}
    for m in models:
        x = _flat(cond, m, param)
        if x is None:
            continue
        vals[m] = x
        lo, hi = _ci(x)
        lines.append(f"     {m}:  {np.median(x):.3f}  [{lo:.3f}, {hi:.3f}]   width={hi-lo:.3f}")
    if len(vals) > 1:
        meds = [np.median(v) for v in vals.values()]
        spread = max(meds) - min(meds)
        lines.append(f"     range of medians: {min(meds):.3f} to {max(meds):.3f} "
                     f"(spread {spread:.3f} = {100*spread/np.mean(meds):.0f}% of mean)")
    return vals
 
 
def stat_overlap(vals, lines):
    """Pairwise CrI overlap -- descriptive robustness, not a test (see header)."""
    if len(vals) < 2:
        return
    import itertools
    lines.append(f"  pairwise {CI_LEVEL:.0f}% CrI overlap (same data, different specifications)")
    allov = True
    for a, b in itertools.combinations(vals, 2):
        la, ha = _ci(vals[a]); lb, hb = _ci(vals[b])
        ov = min(ha, hb) - max(la, lb)
        if ov <= 0:
            allov = False
            lines.append(f"     {a} vs {b}:  NO OVERLAP  (gap {-ov:.3f})")
        else:
            frac = 100 * ov / min(ha - la, hb - lb)
            lines.append(f"     {a} vs {b}:  overlap {ov:+.3f}  ({frac:.0f}% of narrower CrI)")
    lines.append(f"     ALL PAIRS OVERLAP: {allov}")
 
 
def stat_gradient(cond, param, models, lines):
    """CrI, posterior contraction against the prior, and sign probability."""
    psd = PRIOR_SD.get(param)
    lines.append(f"  {param}: gradient summaries (prior Normal(0, {psd}))")
    for m in models:
        x = _flat(cond, m, param)
        if x is None:
            continue
        lo, hi = _ci(x)
        contraction = 1.0 - x.var() / psd**2 if psd else float("nan")
        lines.append(f"     {m}:  median {np.median(x):+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        lines.append(f"           posterior SD {x.std():.4f} vs prior SD {psd:.4f}   "
                     f"contraction {100*contraction:.1f}%")
        lines.append(f"           P({param} < 0 | data) = {float((x < 0).mean()):.3f}   "
                     f"CrI contains zero: {bool(lo < 0 < hi)}")
 
 
def stat_implied_rate(cond, models, lines):
    """Propagate the gradient: what change in k does the posterior actually imply?"""
    lines.append(f"  implied change in differentiation rate over D{T0:.0f}-D22 "
                 f"(k0 + {TSPAN:.0f}*k1, floored at zero as in the forward model)")
    for m in models:
        k0 = _flat(cond, m, "k0"); k1 = _flat(cond, m, "k1")
        if k0 is None or k1 is None:
            continue
        ratio = np.maximum(k0 + TSPAN * k1, 0.0) / k0
        lo, hi = _ci(ratio)
        lines.append(f"     {m}:  k(D22)/k(D9) median {np.median(ratio):.2f}  [{lo:.2f}, {hi:.2f}]")
        lines.append(f"           P(rate falls by >50%)            = {float((ratio < 0.5).mean()):.3f}")
        lines.append(f"           P(rate within +/-25% of initial) = "
                     f"{float(((ratio > 0.75) & (ratio < 1.25)).mean()):.3f}")
 
 
def stat_contrast(param, models, lines, ratio=True):
    """DAPT vs DMSO. A genuine contrast: separate datasets, separate chains."""
    rng_ = np.random.default_rng(STATS_SEED)
    lines.append(f"  {param}: DAPT vs DMSO ({NPAIR:,} paired draws)")
    pooled_d = []
    pooled_r = []
    for m in models:
        a = _flat("dapt", m, param); b = _flat("dmso", m, param)
        if a is None or b is None:
            continue
        A, B = _pair(a, b, rng_)
        d = A - B
        dl, dh = _ci(d)
        pooled_d.append(d)
        lines.append(f"     {m}:  difference median {np.median(d):+.3f}  [{dl:+.3f}, {dh:+.3f}]")
        lines.append(f"           P(DAPT > DMSO) = {float((d > 0).mean()):.3f}   "
                     f"CrI contains zero: {bool(dl < 0 < dh)}")
        if ratio:
            r = A / B
            rl, rh = _ci(r)
            pooled_r.append(r)
            lines.append(f"           ratio median {np.median(r):.2f}  [{rl:.2f}, {rh:.2f}]   "
                         f"P(ratio in [1.5, 2.5]) = {float(((r > 1.5) & (r < 2.5)).mean()):.2f}")
    if pooled_d:
        d = np.concatenate(pooled_d)
        lines.append(f"     pooled across specifications: P(DAPT > DMSO) = {float((d > 0).mean()):.3f}   "
                     f"min across models = "
                     f"{min(float((x > 0).mean()) for x in pooled_d):.3f}")
        if pooled_r:
            r = np.concatenate(pooled_r)
            rl, rh = _ci(r)
            lines.append(f"     pooled ratio median {np.median(r):.2f}  [{rl:.2f}, {rh:.2f}]")
 
 
def stat_convergence(lines):
    """Rhat / n_eff recomputed from the stored chains."""
    if not _HAVE_JAX:
        lines.append("  (numpyro unavailable -- skipping Rhat/n_eff)")
        return
    for cond in CONDS:
        for m in ORDER4:
            ch = chain(cond, m)
            d = {k: ch[k] for k in ("k0", "k1", "rho0", "rho1", "alpha", "aU", "P0")
                 if k in ch.files and ch[k].ndim == 2 and not _is_fixed(ch[k])}
            if not d:
                continue
            S = npsummary(d)
            worst = max(float(S[p]["r_hat"]) for p in S)
            mineff = min(float(S[p]["n_eff"]) for p in S)
            det = "  ".join(f"{p}: Rhat={float(S[p]['r_hat']):.4f}/n_eff={float(S[p]['n_eff']):.0f}"
                            for p in ("k0", "k1") if p in S)
            flag = "  <-- CHECK" if (worst > 1.01 or mineff < 400) else ""
            lines.append(f"     {cond:5s} {m}:  worst Rhat={worst:.4f}  min n_eff={mineff:.0f}{flag}")
            if det:
                lines.append(f"              {det}")
 
 
def stats_table_csv(path):
    """Machine-readable version of the main per-model summaries."""
    rows = [("condition", "model", "parameter", "median",
             f"ci{CI_LEVEL:.0f}_lo", f"ci{CI_LEVEL:.0f}_hi", "post_sd",
             "prior_sd", "contraction", "p_negative")]
    for cond in CONDS:
        for m in ORDER4:
            for p in ("P0", "k0", "k1", "rho0", "rho1", "alpha", "aU"):
                x = _flat(cond, m, p)
                if x is None or _is_fixed(x):
                    continue
                lo, hi = _ci(x)
                psd = PRIOR_SD.get(p)
                contr = 1.0 - x.var() / psd**2 if psd else ""
                rows.append((cond, m, p, f"{np.median(x):.5f}", f"{lo:.5f}", f"{hi:.5f}",
                             f"{x.std():.5f}", f"{psd:.5f}" if psd else "",
                             f"{contr:.4f}" if psd else "", f"{float((x < 0).mean()):.4f}"))
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
 
 
def report_statistics():
    """Build, print and write the full statistical report."""
    missing = [(c, m) for c in CONDS for m in ORDER4
               if not os.path.exists(CHAIN.format(cond=c, model=m))]
    if missing:
        raise SystemExit("missing chains: " + ", ".join(f"{c}/{m}" for c, m in missing)
                         + "  -- run with --infer first")
    L = [f"=================== POSTERIOR STATISTICS ===================",
         f"Equal-tailed {CI_LEVEL:.0f}% credible intervals from the stored NUTS chains.",
         f"Cross-condition contrasts use {NPAIR:,} paired draws (seed {STATS_SEED}).",
         "",
         "Cross-MODEL comparisons within a condition are descriptive only: the four",
         "models are fitted to the same data, so interval overlap indicates robustness",
         "across specifications, not a test of equality. Cross-CONDITION comparisons",
         "are genuine contrasts (separate datasets, separate chains), subject to the",
         "caveat that DAPT P0 is fixed to the DMSO median by design.",
         ""]
 
    fixed = []
    for m in ORDER4:
        x = _flat("dapt", m, "P0")
        if x is not None and _is_fixed(x):
            fixed.append(f"{m}={float(x[0]):.3f}")
    if fixed:
        L += ["DAPT P0 fixed to the DMSO median per model: " + ", ".join(fixed) +
              " (excluded from the summaries below, being constant).", ""]
 
    for cond in CONDS:
        L += [f"---------------- {CONDLBL[cond]} ----------------", ""]
        L.append(" [differentiation rate]")
        vals = stat_level(cond, "k0", ORDER4, L)
        stat_overlap(vals, L)
        stat_gradient(cond, "k1", VRATE, L)
        stat_implied_rate(cond, VRATE, L)
        L.append("")
        L.append(" [neuronal bias]")
        vals = stat_level(cond, "rho0", ORDER4, L)
        stat_overlap(vals, L)
        stat_gradient(cond, "rho1", VBIAS, L)
        L.append("")
 
    L += ["---------------- CROSS-CONDITION CONTRASTS (DAPT vs DMSO) ----------------", ""]
    L.append(" [differentiation rate]")
    stat_contrast("k0", ORDER4, L, ratio=True)
    L.append("")
    L.append(" [rate gradient]")
    stat_contrast("k1", VRATE, L, ratio=False)
    L.append("")
    L.append(" [neuronal bias]")
    stat_contrast("rho0", ORDER4, L, ratio=False)
    L.append("")
    L.append(" [bias gradient]")
    stat_contrast("rho1", VBIAS, L, ratio=False)
    L.append("")
 
    L += ["---------------- CONVERGENCE ----------------", ""]
    stat_convergence(L)
    L.append("")
 
    txt = "\n".join(L)
    print(txt)
    os.makedirs(OUTDIR, exist_ok=True)
    open(f"{OUTDIR}/posterior_statistics.txt", "w").write(txt + "\n")
    stats_table_csv(f"{OUTDIR}/posterior_statistics.csv")
    print(f"\nwritten: {OUTDIR}/posterior_statistics.txt")
    print(f"written: {OUTDIR}/posterior_statistics.csv")
 
 
# ============================= MAIN =================================
def main():
    ap = argparse.ArgumentParser(description="Gogolou et al. modelling pipeline")
    ap.add_argument("--infer", action="store_true", help="force re-run NUTS inference")
    ap.add_argument("--figures", action="store_true", help="only regenerate figures (chains must exist)")
    ap.add_argument("--stats", action="store_true", help="only report posterior statistics (chains must exist)")
    a = ap.parse_args()
    chains_exist = all(os.path.exists(CHAIN.format(cond=c, model=m)) for c in ("dmso", "dapt") for m in ORDER4)
    if a.stats:
        report_statistics(); return
    if a.figures:
        make_all_figures(); return
    if a.infer or not chains_exist:
        run_inference()
    make_all_figures()
    report_statistics()
 
if __name__ == "__main__":
    main()
 
