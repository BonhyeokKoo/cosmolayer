from cosmolayer.cosmosac import CosmoSac2002Model
from matplotlib import pyplot as plt
matrices = CosmoSac2002Model.create_interaction_matrices(298.15)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(matrices[0], cmap="Spectral")
_ = fig.colorbar(im, ax=ax, label="ΔW/(RT)")
fig.tight_layout()
