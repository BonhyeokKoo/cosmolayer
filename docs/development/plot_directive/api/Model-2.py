from cosmolayer.cosmosac import CosmoSac2010Model
from matplotlib import pyplot as plt
matrices = CosmoSac2010Model.create_interaction_matrices(298.15)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(sum(matrices), cmap="Spectral")
_ = fig.colorbar(im, ax=ax, label="ΔW/(RT)")
fig.tight_layout()
