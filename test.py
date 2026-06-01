import h5py

with h5py.File('data/processed/kepler_trainset.h5', 'r') as f:
    print(f"N examples: {f.attrs['n_examples']}")
    print(f"Positives:  {f.attrs['n_positives']}")
    print(f"Negatives:  {f.attrs['n_negatives']}")
    print(f"Class ratio: 1 positive per {f.attrs['n_negatives'] / f.attrs['n_positives']:.2f} negatives")
    print(f"Global view shape: {f['global_view'].shape}")
    print(f"Local view shape:  {f['local_view'].shape}")
    print(f"Labels unique: {sorted(set(f['label'][:].tolist()))}")
    print(f"Kepids unique: {len(set(f['kepid'][:].tolist()))}")