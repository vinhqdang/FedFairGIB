import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_results(filepath='results/experiment_results.json'):
    with open(filepath, 'r') as f:
        flat_results = json.load(f)
        
    nested_results = {}
    for key, val in flat_results.items():
        parts = key.split('/')
        if len(parts) == 3:
            split, dataset, method = parts
            if split not in nested_results:
                nested_results[split] = {}
            if dataset not in nested_results[split]:
                nested_results[split][dataset] = {}
            nested_results[split][dataset][method] = val
            
    return nested_results

def plot_grouped_bar(results, split_mode, metric, ylabel, filename):
    sns.set_theme(style="whitegrid")
    datasets = list(results[split_mode].keys())
    methods = list(results[split_mode][datasets[0]].keys())
    
    # Exclude FairGNN if it failed on some datasets, but we should have all now.
    
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, method in enumerate(methods):
        values = []
        for ds in datasets:
            val = results[split_mode][ds].get(method, {})
            # Map metric names to JSON keys
            key = 'dp_gap' if metric == 'D_DP' else \
                  'eo_gap' if metric == 'D_EO' else \
                  'accuracy' if metric == 'Accuracy' else metric.lower()
            
            values.append(val.get(key, 0.0))
            
        offset = (i - len(methods) / 2) * width + width / 2
        ax.bar(x + offset, values, width, label=method)
    
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'{metric} across Datasets ({split_mode.upper()} Split)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets], fontsize=12)
    ax.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig(f'results/figures/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pareto_frontier(results, split_mode, dataset, filename):
    sns.set_theme(style="whitegrid")
    methods = list(results[split_mode][dataset].keys())
    
    accs = []
    dps = []
    labels = []
    
    for method in methods:
        val = results[split_mode][dataset].get(method, {})
        if 'accuracy' in val and 'dp_gap' in val:
            accs.append(val['accuracy'])
            dps.append(val['dp_gap'])
            labels.append(method)
            
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot points
    for i, label in enumerate(labels):
        # Highlight our method
        color = 'red' if label == 'FedFairGIB' else 'blue'
        marker = '*' if label == 'FedFairGIB' else 'o'
        size = 200 if label == 'FedFairGIB' else 100
        
        ax.scatter(dps[i], accs[i], c=color, marker=marker, s=size, label=label if i==0 or label=='FedFairGIB' else "")
        ax.annotate(label, (dps[i], accs[i]), xytext=(5, 5), textcoords='offset points', fontsize=10)
        
    ax.set_xlabel(r'Demographic Parity Gap ($\Delta_{DP}$) $\downarrow$', fontsize=14)
    ax.set_ylabel(r'Accuracy $\uparrow$', fontsize=14)
    ax.set_title(f'Accuracy-Fairness Trade-off ({dataset.capitalize()}, {split_mode.upper()})', fontsize=16)
    
    # Invert x-axis so better fairness (lower DP) is to the right if desired, or keep as is.
    # Usually we keep 0 on left, so top-left is best.
    
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig(f'results/figures/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists('results/experiment_results.json'):
        print("Results file not found. Please run experiments first.")
        return
        
    results = load_results()
    
    for split in ['iid', 'noniid']:
        if split not in results:
            continue
            
        print(f"Generating grouped bar plots for {split}...")
        plot_grouped_bar(results, split, 'Accuracy', 'Accuracy', f'bar_accuracy_{split}.png')
        plot_grouped_bar(results, split, 'D_DP', 'Demographic Parity Gap', f'bar_ddp_{split}.png')
        plot_grouped_bar(results, split, 'D_EO', 'Equal Opportunity Gap', f'bar_deo_{split}.png')
        
        # Datasets
        datasets = list(results[split].keys())
        for ds in datasets:
            print(f"Generating Pareto plot for {split} - {ds}...")
            plot_pareto_frontier(results, split, ds, f'pareto_{ds}_{split}.png')
            
    print("Figures successfully generated in results/figures/ directory.")

if __name__ == "__main__":
    main()
