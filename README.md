# Chapter 227: Evasion Attacks in Trading

## 1. Introduction

Evasion attacks represent one of the most practical and dangerous classes of adversarial machine learning threats facing deployed trading systems. Unlike poisoning attacks that corrupt training data or model extraction attacks that steal intellectual property, evasion attacks operate at inference time: they modify inputs presented to an already-deployed model in order to induce incorrect predictions, classifications, or trading decisions.

In financial markets, machine learning models are increasingly used for critical functions: market making, fraud detection, order routing, price prediction, and risk assessment. Each of these models accepts market data as input and produces actionable outputs. An adversary who understands how to craft inputs that exploit a model's decision boundaries can manipulate the model into making systematically wrong decisions -- buying when it should sell, classifying fraudulent orders as legitimate, or mispricing risk.

The threat is particularly acute in trading because:

- **Real-time constraints**: Trading models must make decisions in microseconds, leaving no time for human review of suspicious inputs.
- **Financial incentive**: Unlike academic adversarial examples on image classifiers, successful evasion attacks on trading models yield direct monetary profit for the attacker.
- **Data opacity**: Market data is inherently noisy, making adversarial perturbations difficult to distinguish from natural market fluctuations.
- **Interconnected systems**: A successful evasion attack on one participant's model can cascade through markets via feedback loops.

This chapter provides a comprehensive treatment of evasion attacks in the trading context: the mathematical foundations, the distinction between white-box and black-box attack scenarios, trading-specific attack vectors, the critical phenomenon of transferability, and practical implementation in Rust with real Bybit market data.

## 2. Mathematical Foundation

### 2.1 The Evasion Optimization Problem

At its core, an evasion attack is an optimization problem. Given a trained model `f(x)` that maps input features `x` to a prediction `y`, the attacker seeks a perturbation `delta` such that:

```
x_adv = x + delta
f(x_adv) != f(x)    (untargeted attack)
f(x_adv) = y_target  (targeted attack)
```

Subject to the constraint that `||delta|| <= epsilon`, where `epsilon` bounds the perturbation magnitude to keep it "realistic" or undetectable.

The formal optimization for an untargeted attack is:

```
maximize    L(f(x + delta), y_true)
subject to  ||delta||_p <= epsilon
```

Where `L` is the model's loss function and `||.||_p` denotes an Lp norm (typically L-infinity, L2, or L1).

For targeted attacks, the formulation becomes:

```
minimize    L(f(x + delta), y_target)
subject to  ||delta||_p <= epsilon
```

### 2.2 Gradient-Based Attacks

When the attacker has access to the model's gradients (white-box setting), several efficient attack methods exist:

**Fast Gradient Sign Method (FGSM)**: The simplest and most foundational gradient-based attack. It computes a single-step perturbation in the direction of the gradient's sign:

```
delta = epsilon * sign(grad_x L(f(x), y_true))
```

FGSM is fast (single forward-backward pass) but often produces suboptimal adversarial examples because it uses only the gradient's direction, not its magnitude.

**Projected Gradient Descent (PGD)**: An iterative refinement of FGSM that takes multiple smaller steps and projects back onto the epsilon-ball after each step:

```
x_0 = x + uniform_noise(-epsilon, epsilon)
x_{t+1} = Proj_{B(x,epsilon)}(x_t + alpha * sign(grad_x L(f(x_t), y_true)))
```

PGD is considered a "universal first-order adversary" -- if a model is robust to PGD attacks, it is likely robust to all first-order attacks.

**Carlini & Wagner (C&W)**: A more sophisticated optimization-based attack that reformulates the constraint as a penalty term:

```
minimize ||delta||_2 + c * max(max_{i!=t} Z(x+delta)_i - Z(x+delta)_t, -kappa)
```

Where `Z` represents the model's logits and `kappa` is a confidence parameter. C&W attacks are slower but often find smaller perturbations.

### 2.3 Gradient-Free Attacks

When gradients are unavailable (black-box setting), attackers must resort to alternative strategies:

- **Random search**: Sample perturbations from a distribution and keep those that cause misclassification.
- **Coordinate-wise perturbation**: Perturb one feature at a time, keeping changes that move the output in the desired direction.
- **Finite-difference gradient estimation**: Approximate gradients by querying the model at nearby points: `grad_i ~ (f(x + h*e_i) - f(x)) / h`.
- **Evolutionary strategies**: Use population-based optimization to evolve effective perturbations.
- **Transfer attacks**: Craft adversarial examples on a surrogate model and apply them to the target (see Section 5).

### 2.4 Transferability

A remarkable property of adversarial examples is that they often transfer between models. An adversarial example crafted to fool Model A will frequently also fool Model B, even if the models have different architectures, were trained on different data, or use different algorithms. This transferability is believed to arise because models trained on similar data learn similar decision boundaries, and adversarial perturbations exploit features of the data distribution rather than model-specific quirks.

Transferability rates depend on:
- Similarity of training data
- Similarity of model architectures
- Attack strength (stronger attacks transfer better)
- Ensemble diversity (attacks crafted against ensembles transfer better)

## 3. White-Box vs Black-Box Evasion

### 3.1 White-Box Attacks

In a white-box scenario, the attacker has full access to the target model: its architecture, parameters, gradients, and training data. This is the strongest threat model and produces the most effective attacks.

In trading, white-box access might occur when:
- An insider at a trading firm has access to proprietary models
- A model is deployed as open-source or published in a research paper
- An attacker successfully performs model extraction before launching evasion attacks

White-box attacks are important for defensive purposes: they establish an upper bound on model vulnerability. If a model withstands white-box PGD attacks, it provides strong evidence of robustness.

### 3.2 Black-Box Attacks

Black-box attacks are more realistic in trading scenarios. The attacker can only observe the model's outputs (and possibly confidence scores) for chosen inputs. Two sub-categories exist:

**Score-based**: The attacker observes the model's output probabilities or confidence scores. This enables gradient estimation via finite differences and more efficient search.

**Decision-based**: The attacker only observes the final decision (buy/sell/hold, accept/reject). This is the most restricted setting and requires the most queries.

In practice, a trading adversary often has partial information: they can observe a market maker's quotes (which reveal something about the model's state) or track an algorithmic trader's order flow (which reveals its decisions). This intermediate setting falls between pure black-box and white-box.

### 3.3 Query Efficiency

A critical practical consideration is query efficiency. Each query to the target model in a trading context corresponds to a market interaction that may cost money, leave an audit trail, or alert the target. Efficient black-box attacks minimize the number of queries needed:

- Random search: O(d/epsilon^2) queries where d is dimension
- Coordinate-wise: O(d) queries per iteration
- Finite-difference gradients: O(d) queries per gradient estimate
- Transfer attacks: 0 queries on the target (but requires a surrogate)

## 4. Trading-Specific Evasion Attacks

### 4.1 Spoofing Order Books to Fool ML Market Makers

Many modern market makers use ML models that process order book snapshots to set quotes. An attacker can place and quickly cancel orders (spoofing) to create order book states that cause the market maker's model to misprice:

- **Layering**: Place large orders at multiple price levels on one side of the book, creating the illusion of supply/demand imbalance. The ML market maker shifts its quotes, and the attacker trades on the other side.
- **Phantom liquidity**: Show deep liquidity at a price level to attract the market maker's inventory management algorithm, then pull the orders when the market maker commits.

These attacks are illegal under most jurisdictions (market manipulation), but understanding them is critical for building robust trading systems.

### 4.2 Injecting Fake Signals

ML models that consume alternative data (news sentiment, social media, satellite imagery) can be attacked by injecting fake signals:

- **Sentiment manipulation**: Post coordinated messages to move a sentiment indicator that feeds into a trading model.
- **Feature pollution**: If a model uses derived features (technical indicators), trade in patterns that produce misleading indicator values.

### 4.3 Adversarial Trade Sequences

Rather than perturbing a single input, an attacker can execute a sequence of trades designed to create a market microstructure state that fools the target model. This is a temporal evasion attack where the adversarial "perturbation" is spread across time.

### 4.4 Financial Constraints on Perturbations

Unlike image classifiers where perturbations are bounded by Lp norms, trading evasion attacks face financial constraints:

- **Cost**: Every spoofed order or adversarial trade has a cost (fees, slippage, risk of execution).
- **Market impact**: Large perturbations move the market, potentially undermining the attack.
- **Regulatory constraints**: Certain patterns trigger surveillance alerts.
- **Temporal constraints**: Market data has time-series structure; perturbations must be temporally consistent.

These constraints make the optimization problem harder but also make attacks more realistic when they succeed.

## 5. Transferability in Trading Models

### 5.1 Why Transferability Matters

Transferability is the most dangerous property of evasion attacks for trading systems. It means an attacker does not need access to the target model -- they can:

1. Build their own surrogate model trained on similar market data
2. Craft adversarial examples against their surrogate
3. Deploy those examples against the target with reasonable success probability

Since market data is largely public (prices, volumes, order books on exchanges), any adversary can build a surrogate model using the same features as the target.

### 5.2 Measuring Transferability

Transferability rate is measured as the fraction of adversarial examples crafted on a source model that also fool a target model:

```
T(source, target) = |{x_adv : f_source(x_adv) != y_true AND f_target(x_adv) != y_true}| / |{x_adv : f_source(x_adv) != y_true}|
```

In our implementation, we test transferability across three different model architectures: linear models, decision boundaries based on thresholds, and neural-network-style models.

### 5.3 Ensemble Diversity as Defense

One effective defense against transfer attacks is to use an ensemble of diverse models. If the models in the ensemble make errors on different inputs, an adversarial example that fools one model is less likely to fool the entire ensemble. Key strategies include:

- **Architecture diversity**: Combine linear models, tree-based models, and neural networks.
- **Training data diversity**: Train models on different subsets or time periods.
- **Feature diversity**: Use different feature sets for different models.
- **Adversarial training**: Include adversarial examples in training to harden decision boundaries.

## 6. Implementation Walkthrough (Rust)

Our Rust implementation provides a complete framework for studying evasion attacks on trading models. The key components are:

### 6.1 Model Architectures

We implement three model types for transfer testing:
- **LinearModel**: A simple linear classifier with weights and bias, supporting gradient computation for white-box attacks.
- **ThresholdModel**: A non-linear model using threshold-based feature transformations.
- **EnsembleModel**: Combines multiple models with weighted voting for improved robustness.

### 6.2 White-Box Attacks

The `WhiteBoxAttack` module implements:
- **FGSM**: Single-step gradient attack using `epsilon * sign(gradient)`.
- **PGD**: Multi-step iterative attack with projection back onto the epsilon-ball, typically run for 10-40 iterations with step size `alpha = epsilon / 4`.

Both attacks compute gradients analytically from the linear model's weights and use them to maximize the loss function.

### 6.3 Black-Box Attacks

The `BlackBoxAttack` module implements:
- **Random search**: Samples perturbations uniformly and keeps the best.
- **Coordinate-wise perturbation**: Iterates through features, testing positive and negative perturbations for each.
- **Query-based gradient estimation**: Uses finite differences to approximate the gradient, then applies FGSM-style updates.

### 6.4 Financial Constraints

The `FinancialConstraints` struct enforces market-realistic perturbations:
- Maximum percentage change per feature (e.g., price cannot change more than 2%)
- Feature-specific bounds (volume has different natural ranges than price)
- Temporal consistency checks

### 6.5 Attack Success Metrics

We compute comprehensive metrics including:
- Attack success rate (fraction of inputs where the prediction changed)
- Average perturbation magnitude (L2 norm of delta)
- Confidence reduction (how much the model's confidence drops)
- Transferability rates between all model pairs

## 7. Bybit Data Integration

The implementation includes a Bybit API client that fetches real market data for BTCUSDT:

```rust
pub async fn fetch_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Kline>>
```

This function calls the Bybit v5 API endpoint `/v5/market/kline` and returns candlestick data including open, high, low, close prices and volume. The data is then transformed into feature vectors suitable for model training and attack testing.

Features extracted from raw market data include:
- Price returns (log returns over various horizons)
- Volatility (rolling standard deviation of returns)
- Volume ratios (current volume relative to moving average)
- Price momentum (rate of change indicators)
- Order book imbalance (if available)

Using real market data ensures that our adversarial perturbations operate within realistic parameter ranges and that our financial constraints accurately reflect market conditions.

## 8. Key Takeaways

1. **Evasion attacks are the most practical adversarial ML threat to trading systems** because they operate at inference time on deployed models, require no access to training pipelines, and yield direct financial profit.

2. **Gradient-based attacks (FGSM, PGD) are powerful but require white-box access**. In trading, this access level is rare but not impossible (insider threats, model extraction, published research).

3. **Black-box attacks are realistic and effective**. Query-based methods can approximate gradients, and transfer attacks require zero queries on the target model. Since market data is public, building surrogate models is feasible.

4. **Transferability is the key threat multiplier**. Adversarial examples transfer between models trained on similar data, enabling attacks without any access to the target. This is especially dangerous in trading where all participants observe the same market data.

5. **Financial constraints make trading evasion attacks harder but not impossible**. Perturbations must be market-realistic (no impossible prices), cost-aware (spoofing has transaction costs), and temporally consistent. These constraints reduce the attacker's search space but also make successful attacks more dangerous because they are harder to detect.

6. **Ensemble diversity is the most effective defense**. Using multiple models with different architectures, training data, and feature sets reduces transferability. Combining this with adversarial training, input validation, and anomaly detection creates a robust defense-in-depth approach.

7. **Detection complements prevention**. While perfect robustness is provably impossible for most model classes, monitoring for statistical anomalies in input distributions, tracking model confidence over time, and maintaining audit trails of model decisions can help detect and mitigate evasion attacks even when prevention fails.

8. **Regulatory implications are significant**. Many trading-specific evasion attacks (spoofing, layering, market manipulation) are illegal. Understanding them is necessary for compliance and surveillance systems, but implementing them against live markets carries severe legal consequences.

The arms race between adversarial attacks and defenses in trading will intensify as ML adoption grows. Practitioners must build adversarial robustness into their model development lifecycle from the start, treating it as a first-class requirement alongside accuracy, latency, and fairness.
