# FedFairGIB  
## Federated Fair Graph Neural Network via the Information Bottleneck  
### A Novel Algorithm Design — Entropy (MDPI) Special Issue Submission

---

## Abstract

We propose **FedFairGIB** — a novel algorithm that extends the Information Bottleneck (IB) principle to fair federated graph neural network training. FedFairGIB addresses three simultaneously open problems:

1. Existing IB-based fair GNN methods cannot be federated  
2. Existing federated GNN methods lack formal fairness guarantees  
3. No information-theoretic privacy–fairness–utility tradeoff bounds exist for federated graphs  

Our key contributions are:

- A Federated Fair Information Bottleneck (**F²IB**) objective  
- A HSIC-based local fairness estimator that preserves privacy  
- MI-weighted federated aggregation  
- A cross-client fairness calibration module  
- Formal MI bounds on the demographic parity gap  

We show that the IB compression mechanism and local differential privacy noise share the same Gaussian mechanism — privacy and fairness are unified under a single information-theoretic framework.

**Keywords:** information bottleneck, federated learning, graph neural networks, fairness, mutual information, differential privacy

---

# 1. Motivation: Three Unsolved Problems

### Problem 1 — Centralized IB-based fair GNNs cannot be federated

GRAFair (arXiv:2409.01367, 2024) and FairIB (IJCAI 2024) apply the conditional fairness bottleneck to graphs, but both require centralized access to all nodes and sensitive attributes. Mutual information \( I(Z; S) \) is a global quantity that cannot be computed from local subgraphs without sharing raw data — making these approaches impossible to federate directly.

### Problem 2 — Federated GNN methods ignore demographic fairness entirely

The TKDD 2025 subgraph FL-IB paper uses the bottleneck to handle non-IID distributions, not fairness. FedGL, SpreadGNN, and RMFGL address privacy and heterogeneity but have no fairness objective. FL fairness surveys (Mukhtiar et al., 2025; Chaudhary et al., 2025) identify this as the #1 open problem.

### Problem 3 — No formal privacy–fairness–utility tradeoff bounds exist for federated graphs

Despite extensive work on fairness in centralized GNNs and privacy in federated learning, no paper has derived information-theoretic bounds characterizing the three-way tradeoff surface in the federated graph setting.

---

# 2. Problem Formulation

## 2.1 Federated Graph Setting

Let there be \( K \) clients indexed by \( k = 1, \dots, K \). Each client holds a private local graph:

\[
G_k = (V_k, E_k, X_k, S_k, Y_k)
\]

- \( X_k \in \mathbb{R}^{n_k \times d} \): node features  
- \( S_k \in \{0,1\}^{n_k} \): sensitive attributes  
- \( Y_k \in \{0,1\}^{n_k} \): labels  

Distributions are non-IID:

\[
p_k(X,S,Y) \neq p_j(X,S,Y)
\]

No raw data is shared.

---

## 2.2 The F²IB Global Objective

\[
\theta^* = \arg\max_\theta \; I(Z;Y) - \beta I(Z;S) - \lambda I(Z;X)
\]

Subject to \( \varepsilon \)-local differential privacy (LDP).

- \( I(Z;Y) \): utility  
- \( I(Z;S) \): fairness  
- \( I(Z;X) \): compression/privacy  

---

## 2.3 Fairness Definitions

**Demographic Parity Gap**

\[
\Delta_{DP} = |P(\hat{Y}=1|S=0) - P(\hat{Y}=1|S=1)|
\]

**Equal Opportunity Gap**

\[
\Delta_{EO} = |P(\hat{Y}=1|Y=1,S=0) - P(\hat{Y}=1|Y=1,S=1)|
\]

---

# 3. Theoretical Framework

## 3.1 Theorem 1 — Fairness–MI Bound

\[
\Delta_{DP} \le \sqrt{2 \ln 2 \cdot I(Z;S)}
\]

**Implication:** Minimizing \( I(Z;S) \) directly minimizes demographic parity.

---

## 3.2 Federated MI Decomposition

\[
I_{global}(Z;S) = \sum_k \frac{n_k}{N} I_k(Z_k;S_k) + \Delta_{cross}
\]

Minimizing local MI provides a lower bound for global fairness.

---

## 3.3 Privacy–Fairness–Utility Tradeoff

\[
\Delta_{DP} \le \sqrt{2 \ln 2 \left[ \frac{\beta}{1+\beta} I(Z;Y) + \frac{C_\varepsilon^2}{2\sigma^2} \right]}
\]

Implications:

1. Higher utility permits higher fairness gap  
2. DP noise induces irreducible fairness floor  
3. \( \beta \) controls Pareto frontier  

---

# 4. FedFairGIB Algorithm

## 4.1 Architecture

### Client Side

- VGIB Encoder  
- Task Head  
- HSIC Fairness Estimator  
- KL Compression  
- LDP Noise  

### Server Side

1. MI-weighted aggregation  
2. Cross-client fairness calibration (FCM)  
3. Model broadcast  

---

## 4.2 Variational GIB Encoder

\[
q_\phi(Z_k|X_k,A_k) = \prod_v \mathcal{N}(z_v;\mu_v,\sigma_v^2 I)
\]

\[
z_v = \mu_v + \sigma_v \cdot \varepsilon_v
\]

Gaussian noise supports both IB optimization and LDP.

---

## 4.3 HSIC Fairness Estimator

\[
\widehat{HSIC}(Z,S) = \frac{1}{(m-1)^2} \text{tr}(K_Z H K_S H)
\]

HSIC = 0 iff independence holds (universal kernels).

---

## 4.4 Local Client Objective

\[
L_k = L_{task} + \beta L_{fair} + \lambda L_{IB}
\]

### Task Loss
Cross-entropy

### Fairness Loss
HSIC

### IB Loss

\[
\frac{1}{2|V_k|}\sum_v (\mu_v^2 + \sigma_v^2 - \log\sigma_v^2 - 1)
\]

---

## 4.5 MI-Weighted Aggregation

\[
w_k = \frac{n_k \exp(-\gamma M_k)}{\sum_j n_j \exp(-\gamma M_j)}
\]

Reduces to FedAvg when \( \gamma=0 \).

---

## 4.6 Cross-Client Fairness Calibration

\[
L_{cal} = \frac{1}{\binom{K}{2}} \sum_{j<k} (M_j - M_k)^2
\]

Projection matrix constrained near-orthogonal.

---

# 5. Convergence

## Theorem 3 — Optimization Convergence

\[
E[L(\theta^{(T)}) - L^*] \le \frac{C_1}{T} + \frac{C_2\sigma_{dp}^2}{T} + O\left(\frac{G^2}{\mu^2 T}\right)
\]

## Theorem 4 — Fairness Convergence

\[
E[\Delta_{DP}^{(T)}] \le \sqrt{2\ln2 \cdot \bar{M}^{(T)}} + C_{cross}\gamma^{-1}
\]

---

# 6. Experimental Design

## Datasets

- German Credit  
- Credit Defaulter  
- Bail  
- POKEC-z / POKEC-n  

Federated splits: IID and non-IID (sensitive-stratified).

---

## Baselines

- FedAvg  
- FedProx  
- FairFedGNN  
- GRAFair  
- FairIB  
- **FedFairGIB (ours)**  

---

## Expected Results

- 25–40% reduction in \( \Delta_{DP} \) vs. FedAvg  
- ≤ 2–3% accuracy drop  
- Achieves \( (\varepsilon=1.0, \delta=10^{-5}) \)-DP  
- Pareto-dominates federated baselines  
- ≥ 30% reduction in between-client HSIC variance  

---

# 7. Journal Fit — Entropy (MDPI)

**Primary Section:** Information Theory  
**Themes:**

- Information Bottleneck  
- Mutual Information Bounds  
- Privacy–Fairness Tradeoff  
- Complexity Perspective  
- Statistical Physics Analogy  

**Target Special Issue:**  
*Information Bottleneck Method: Theory and Applications*  

---

**FedFairGIB — Designed for submission to Entropy (MDPI)**