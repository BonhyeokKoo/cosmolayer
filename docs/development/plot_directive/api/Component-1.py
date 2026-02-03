from importlib.resources import files
from cosmolayer.cosmosac import Component
from matplotlib import pyplot as plt
path = files("cosmolayer.data") / "C=C(N)O.cosmo"
component = Component(path)
fig, ax = plt.subplots(figsize=(8, 4))
grid = component.get_sigma_grid()
for s in ["NHB", "OH", "OT"]:
    _ = ax.plot(grid, component.get_sigma_profile(s), label=s)
_ = ax.plot(grid, component.get_sigma_profile(), label="Overall")
_ = ax.set_xlabel("Charge density (e/Å²)")
_ = ax.set_ylabel("Surface area contribution (Å²)")
_ = ax.legend()
fig.tight_layout()
