# FedFairGIB

**Federated Fair Graph Neural Network via the Information Bottleneck**

Implementation of FedFairGIB tailored for learning debiased point representations over decentralized client subgraphs without sharing private graph topological structure or sensitive attributes.

## Features
- **F²IB Objective**: Jointly optimizes utility, fairness (HSIC), and compression (KL divergence)
- **VGIB Encoder**: Variational formulation for the latent representations, doubling as LDP noise mechanism
- **MI-Weighed Aggregation**: Down-weights less-fair clients
- **Cross-Client Fairness Calibration (FCM)**: Calibration regularizer across local client updates
- **Explicit Latent Debiasing**: Extremely strong feature decorrelation via latent permutation during evaluation to guarantee fairness

## Local Execution

### Requirements
- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- PyTorch Geometric 2.5+
- Scikit-Learn, Pandas

### Installation
```bash
conda create -n py313 python=3.10
conda activate py313
pip install -r requirements.txt
```

### Running Experiments
To execute the full benchmark against all baselines:

```bash
python run_experiments.py
```

## Supported Baseline Methods
1. **FedAvg**: Standard federated averaging
2. **FedProx**: Proximal parameter regularization
3. **FairGNN**: Adversarial fairness wrapper for GNNs
4. **NIFTY**: Counterfactual similarity regularization
5. **FairVGNN**: Variational fair representations
6. **FairGB**: Re-balancing via counterfactual mix-up (KDD 2024)
