from cosmolayer.cosmosac import create_cosmo_sac_2010_matrices
from matplotlib import pyplot as plt
delta_w_a, delta_w_b = create_cosmo_sac_2010_matrices(298.15)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(delta_w_a + delta_w_b, cmap="Spectral")
_ = fig.colorbar(im, ax=ax, label="ΔW/(RT)")
fig.tight_layout()
