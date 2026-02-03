from cosmolayer.cosmosac import create_cosmo_sac_2002_matrix
from matplotlib import pyplot as plt
matrix = create_cosmo_sac_2002_matrix(298.15)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(matrix, cmap="Spectral")
_ = fig.colorbar(im, ax=ax, label="ΔW/(RT)")
fig.tight_layout()
