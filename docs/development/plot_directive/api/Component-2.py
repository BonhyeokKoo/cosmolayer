from importlib.resources import files
from cosmolayer.sac import Component
from matplotlib import pyplot as plt
path = files("cosmolayer.data") / "C=C(N)O.cosmo"
component = Component(path)
fig, ax = plt.subplots(figsize=(8, 4))
log_p = component.get_log_probabilities()
_ = ax.bar(range(len(log_p)), np.exp(log_p))
_ = ax.set_xlabel("Segment type index")
_ = ax.set_ylabel("Probability")
fig.tight_layout()
