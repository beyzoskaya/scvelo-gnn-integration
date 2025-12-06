import torch
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(model, data, adata):
    print("\n" + "="*40)
    print("--- 4. Running In-Silico Simulation ---")
    print("="*40)
    
    model.eval()
    
    # A. Pre-compute Vector Field
    with torch.no_grad():
        velocity_field = model(data.x, data.edge_index)

    # B. DYNAMIC START POINT SELECTION
    dataset_name = adata.uns.get('dataset_name', 'pancreas')
    
    # Map dataset to its starting stem cell type
    start_cluster_map = {
        'pancreas': 'Ductal',
        'dentate': 'Radial Glia-like'
    }
    
    target_cluster = start_cluster_map.get(dataset_name, 'Ductal')
    
    # Find the label key
    label_key = None
    for key in ['clusters', 'cell_type', 'louvain']:
        if key in adata.obs:
            label_key = key
            break

    if label_key and target_cluster in adata.obs[label_key].values:
        indices = np.where(adata.obs[label_key] == target_cluster)[0]
        # Pick random start points
        count = min(len(indices), 15)
        start_indices = np.random.choice(indices, count, replace=False)
        print(f"   -> Simulation Start: {count} '{target_cluster}' cells.")
    else:
        print(f"   -> Warning: Start cluster '{target_cluster}' not found. Using random.")
        start_indices = np.random.choice(data.num_nodes, 15)

    # C. Simulation Loop
    step_size = 0.8
    steps = 50
    trajectories = []
    
    X_tensor = data.x.cpu()
    velocity_field = velocity_field.cpu()
    
    print(f"   -> Simulating trajectories ({steps} steps)...")
    
    for start_idx in start_indices:
        path = []
        current_pos = X_tensor[start_idx].unsqueeze(0)
        path.append(current_pos.numpy()[0, :2]) 
        
        for _ in range(steps):
            dists = torch.cdist(current_pos, X_tensor)
            nearest_idx = torch.argmin(dists, dim=1)
            
            direction = velocity_field[nearest_idx]
            current_pos = current_pos + (direction * step_size)
            path.append(current_pos.numpy()[0, :2])
        
        trajectories.append(np.array(path))

    # D. Plotting
    print("   -> Generating Simulation Plot...")
    X = data.x.numpy()
    plt.figure(figsize=(10, 8))
    
    plt.scatter(X[:,0], X[:,1], c='lightgrey', s=15, alpha=0.4, label='Background')
    
    for i, path in enumerate(trajectories):
        plt.plot(path[:,0], path[:,1], linewidth=2, color='black', alpha=0.8)
        
        if i == 0:
            plt.scatter(path[0,0], path[0,1], c='lime', edgecolors='k', s=60, zorder=5, label=f'Start ({target_cluster})')
            plt.scatter(path[-1,0], path[-1,1], c='red', edgecolors='k', s=60, zorder=5, label='Predicted Fate')
        else:
            plt.scatter(path[0,0], path[0,1], c='lime', edgecolors='k', s=60, zorder=5)
            plt.scatter(path[-1,0], path[-1,1], c='red', edgecolors='k', s=60, zorder=5)

    plt.title(f"Generative AI Simulation: {dataset_name.capitalize()}\nTraversing the Manifold")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visual_4_simulation.png", dpi=200)
    print(f"   -> Saved 'visual_4_simulation.png'")