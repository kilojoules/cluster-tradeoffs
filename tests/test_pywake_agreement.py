"""Cross-validate pixwake's Gaussian deficit models against PyWake.

The adversarial review of 2026-08-13 found that pixwake's Gaussian models default
to ``use_radius_mask=True``, which zeroes the deficit outside |cw| = 2*sigma. At
that boundary the Gaussian is still exp(-2) = 13.5% of its centreline value, so
the truncation is a step, not a taper, and it puts pixwake measurably off the
reference implementation. Nothing in the test suite asserted otherwise.

These tests pin both facts:
  * with the mask disabled, pixwake reproduces PyWake closely;
  * with the mask enabled (the production default) it does not, and the
    disagreement is concentrated at partial lateral overlap.

If the default changes, ``test_default_matches_pywake_when_mask_disabled`` should
start passing without the explicit flag and the xfail below should be revisited.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from pixwake import WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.deficit.gaussian import TurboGaussianDeficit
from pixwake.superposition import SquaredSum

D = 240.0
WS = 9.0


def _pixwake_pair_power(deficit, dw_D, cw_D, ti=None):
    """Power of a downstream turbine at (dw, cw) rotor diameters, wind from west."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_regret_cross_section as cs

    sim = WakeSimulation(cs.create_dei_turbine(), deficit)
    x = jnp.array([0.0, dw_D * D])
    y = jnp.array([0.0, cw_D * D])
    res = sim(x, y, ws_amb=jnp.array([WS]), wd_amb=jnp.array([270.0]),
              ti_amb=ti)
    return float(res.power()[0, 1])


def _pywake_pair_power(model_name, dw_D, cw_D, ti=0.06):
    from py_wake.site import UniformSite
    from py_wake.wind_turbines import WindTurbine
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_regret_cross_section as cs

    t = cs.create_dei_turbine()
    u = np.asarray(t.power_curve.ws)
    p = np.asarray(t.power_curve.values)
    ct = np.interp(u, np.asarray(t.ct_curve.ws), np.asarray(t.ct_curve.values))
    wt = WindTurbine(name="dei", diameter=float(t.rotor_diameter),
                     hub_height=float(t.hub_height),
                     powerCtFunction=PowerCtTabular(u, p, "w", ct))
    site = UniformSite(p_wd=[1.0], ti=ti)

    if model_name == "bastankhah":
        from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit as PWBG
        from py_wake.wind_farm_models import PropagateDownwind
        from py_wake.superposition_models import SquaredSum as PWSq
        wfm = PropagateDownwind(site, wt, wake_deficitModel=PWBG(k=0.04),
                                superpositionModel=PWSq())
    else:
        from py_wake.deficit_models.gaussian import TurboGaussianDeficit as PWTG
        from py_wake.wind_farm_models import PropagateDownwind
        from py_wake.superposition_models import SquaredSum as PWSq
        wfm = PropagateDownwind(site, wt, wake_deficitModel=PWTG(A=0.04),
                                superpositionModel=PWSq())

    sim = wfm([0.0, dw_D * D], [0.0, cw_D * D], wd=[270.0], ws=[WS])
    return float(np.asarray(sim.Power)[1, 0, 0])


# (dw, cw) in rotor diameters; cw spans centreline to well outside 2*sigma
GEOMETRIES = [(4, 0.0), (4, 0.5), (4, 1.0), (8, 0.0), (8, 0.8), (8, 1.5),
              (15, 0.0), (15, 1.0), (15, 2.0), (30, 1.0), (30, 2.0)]


@pytest.mark.parametrize("dw_D,cw_D", GEOMETRIES)
@pytest.mark.parametrize("model", ["bastankhah", "turbopark"])
def test_matches_pywake_when_mask_disabled(model, dw_D, cw_D):
    """Without the radial mask, pixwake must track the reference implementation."""
    if model == "bastankhah":
        deficit = BastankhahGaussianDeficit(k=0.04, superposition=SquaredSum(),
                                            use_radius_mask=False)
        ti = None
    else:
        deficit = TurboGaussianDeficit(A=0.04, superposition=SquaredSum(),
                                       use_radius_mask=False)
        ti = 0.06
    got = _pixwake_pair_power(deficit, dw_D, cw_D, ti=ti)
    want = _pywake_pair_power(model, dw_D, cw_D)
    assert got == pytest.approx(want, rel=2e-3), (
        f"{model} at dw={dw_D}D cw={cw_D}D: pixwake {got:.1f} W vs PyWake {want:.1f} W")


@pytest.mark.parametrize("model", ["bastankhah", "turbopark"])
def test_radius_mask_is_the_source_of_disagreement(model):
    """The production default disagrees with PyWake at partial lateral overlap.

    This is an characterisation test, not an endorsement: it documents the size of
    the deviation the default introduces so that a change in behaviour is visible.
    """
    dw_D = 8
    kw = dict(superposition=SquaredSum())
    if model == "bastankhah":
        masked = BastankhahGaussianDeficit(k=0.04, use_radius_mask=True, **kw)
        clean = BastankhahGaussianDeficit(k=0.04, use_radius_mask=False, **kw)
        ti = None
    else:
        masked = TurboGaussianDeficit(A=0.04, use_radius_mask=True, **kw)
        clean = TurboGaussianDeficit(A=0.04, use_radius_mask=False, **kw)
        ti = 0.06

    # The mask only engages beyond |cw| = 2*sigma, and sigma differs by model, so
    # scan outward rather than assuming a geometry. Record the largest deviation
    # anywhere the reference model still predicts a non-trivial deficit.
    free = _pixwake_pair_power(clean, dw_D, 50.0, ti=ti)
    worst = 0.0
    worst_cw = None
    for cw_D in np.arange(0.2, 2.61, 0.1):
        p_clean = _pixwake_pair_power(clean, dw_D, float(cw_D), ti=ti)
        if (free - p_clean) / free < 0.01:
            continue  # reference deficit already negligible here
        p_masked = _pixwake_pair_power(masked, dw_D, float(cw_D), ti=ti)
        assert p_masked >= p_clean - 1e-6, "mask must not add deficit"
        rel = (p_masked - p_clean) / p_clean
        if rel > worst:
            worst, worst_cw = rel, float(cw_D)

    assert worst_cw is not None and worst > 1e-3, (
        f"{model}: expected the radial mask to discard visible deficit somewhere "
        f"in cw in [0.2, 2.6]D at dw={dw_D}D; largest deviation seen was {worst:.2e}")
    print(f"\n{model}: radial mask over-predicts downstream power by "
          f"{100*worst:.1f}% at dw={dw_D}D, cw={worst_cw:.1f}D")


def test_mask_effect_on_farm_scale_loss_is_bounded():
    """Farm-integrated effect of the mask, so the magnitude is on record.

    Uses a compact grid rather than the production layout to keep the test fast.
    The assertion is deliberately loose: it guards against the effect silently
    growing by an order of magnitude, not against small numerical drift.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_regret_cross_section as cs

    xs, ys = np.meshgrid(np.arange(4) * 5 * D, np.arange(4) * 5 * D)
    x = jnp.array(xs.ravel().astype(float))
    y = jnp.array(ys.ravel().astype(float))
    wd = jnp.linspace(0.0, 345.0, 24)
    ws = jnp.full_like(wd, WS)
    w = jnp.full_like(wd, 1.0 / 24)

    out = {}
    for mask in (True, False):
        deficit = BastankhahGaussianDeficit(k=0.04, superposition=SquaredSum(),
                                            use_radius_mask=mask)
        sim = WakeSimulation(cs.create_dei_turbine(), deficit)
        out[mask] = float(cs.compute_aep(sim, x, y, ws, wd, w, None))

    rel = abs(out[True] - out[False]) / out[False]
    assert rel < 0.05, f"radial mask shifts farm AEP by {100*rel:.2f}% (was <5%)"
