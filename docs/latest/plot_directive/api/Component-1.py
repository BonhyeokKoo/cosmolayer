from importlib.resources import files
from cosmolayer.cosmosac import Component
from cosmolayer.cosmosac.segment_groups import SEGMENT_GROUPS
from matplotlib import pyplot as plt
path = files("cosmolayer.data") / "C=C(N)O.cosmo"
component = Component(path.read_text(), merge_profiles=False)
fig, ax = plt.subplots(figsize=(8, 4))
grid = component.sigma_grid
for i, label in enumerate(SEGMENT_GROUPS):
    _ = ax.plot(grid, component.sigma_profile[i], label=label)
_ = ax.set_xlabel("Charge density (e/Å²)")
_ = ax.set_ylabel("Surface area contribution (Å²)")
_ = ax.legend()
fig.tight_layout()
