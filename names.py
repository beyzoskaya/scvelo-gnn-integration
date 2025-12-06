import scvelo as scv
adata = scv.datasets.dentategyrus()
print(adata.obs['clusters'].unique())