"""Limb-darkened transit models (batman) and circular-orbit geometry for injection tests."""
import numpy as np

G_CGS = 6.674e-8
R_SUN_CM = 6.957e10
SOLAR_LOGG = 4.438  # log10(G M_sun / R_sun^2), cgs
SECONDS_PER_DAY = 86400.0
LIMB_DARKENING = (0.4, 0.25)


def import_batman():
    """batman imports distutils, which Python 3.12+ only provides through setuptools' shim."""
    try:
        import batman
    except ModuleNotFoundError as exc:
        if exc.name != "distutils":
            raise
        import setuptools  # noqa: F401
        import batman
    return batman


def a_over_rstar(period_days: float, srad: float = 1.0, logg: float = SOLAR_LOGG) -> float:
    """Kepler's third law with M* = g R*^2 / G, i.e. a/R* = (g P^2 / (4 pi^2 R*))^(1/3)."""
    g = 10.0 ** logg
    radius = srad * R_SUN_CM
    period = period_days * SECONDS_PER_DAY
    return float((g * period ** 2 / (4 * np.pi ** 2 * radius)) ** (1.0 / 3.0))


def t14_days(period_days: float, a_rs: float, rp: float, b: float) -> float:
    """First-to-fourth contact duration for a circular orbit; 0.0 if there is no transit."""
    inc = np.arccos(b / a_rs)
    chord = (1.0 + rp) ** 2 - b ** 2
    if chord <= 0:
        return 0.0
    arg = np.sqrt(chord) / (a_rs * np.sin(inc))
    return float(period_days / np.pi * np.arcsin(min(arg, 1.0)))


def transit_model(
    time: np.ndarray,
    period_days: float,
    t0: float,
    rp: float,
    a_rs: float,
    b: float,
    u: tuple[float, float] = LIMB_DARKENING,
) -> np.ndarray:
    """Relative flux (1 out of transit) at each time; non-finite times get 1."""
    batman = import_batman()
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period_days
    params.rp = rp
    params.a = a_rs
    params.inc = float(np.degrees(np.arccos(b / a_rs)))
    params.ecc = 0.0
    params.w = 90.0
    params.u = list(u)
    params.limb_dark = "quadratic"
    time = np.asarray(time, dtype=np.float64)
    model = np.ones_like(time)
    finite = np.isfinite(time)
    if finite.any():
        model[finite] = batman.TransitModel(params, time[finite]).light_curve(params)
    return model


def robust_sigma(flux: np.ndarray) -> float:
    """1.4826 x median absolute deviation."""
    flux = np.asarray(flux, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(flux - np.median(flux))))
