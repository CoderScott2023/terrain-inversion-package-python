import numpy as np

def lambert(mu0, mu, phase, albedo=1.0):
    return albedo * mu0


def lommel_seeliger(mu0, mu, phase, albedo=1.0):
    return albedo * 2.0 * mu0 / np.where(mu0 + mu > 0.0, mu0 + mu, np.inf)


def mcewen_L(phase):
    g = np.degrees(phase)
    return np.clip(1.0 - 0.019*g + 2.42e-4*g**2 - 1.46e-6*g**3, 0.0, 1.0)


def lunar_lambert(mu0, mu, phase, albedo=1.0, L=None):
    if L is None:
        L = mcewen_L(phase)
    return albedo * (L * lommel_seeliger(mu0, mu, phase) + (1.0 - L) * mu0)


MODELS = {
    'lambert': lambert,
    'lommel_seeliger': lommel_seeliger,
    'lunar_lambert': lunar_lambert,
}