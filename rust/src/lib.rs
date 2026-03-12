use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ============================================================================
// Bybit API Data Fetching
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline/candlestick data from Bybit v5 API
pub async fn fetch_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Kline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut klines = Vec::new();
    for row in &resp.result.list {
        if row.len() >= 6 {
            klines.push(Kline {
                timestamp: row[0].parse()?,
                open: row[1].parse()?,
                high: row[2].parse()?,
                low: row[3].parse()?,
                close: row[4].parse()?,
                volume: row[5].parse()?,
            });
        }
    }

    // Bybit returns newest first, reverse for chronological order
    klines.reverse();
    Ok(klines)
}

/// Extract feature vectors from kline data for model training
pub fn extract_features(klines: &[Kline], window: usize) -> (Array2<f64>, Array1<f64>) {
    let n = klines.len();
    if n <= window + 1 {
        return (
            Array2::zeros((0, 5)),
            Array1::zeros(0),
        );
    }

    let num_samples = n - window;
    let num_features = 5;
    let mut features = Array2::zeros((num_samples, num_features));
    let mut labels = Array1::zeros(num_samples);

    for i in 0..num_samples {
        let slice = &klines[i..i + window];
        let current = &klines[i + window - 1];

        // Feature 1: log return over window
        let log_return = (current.close / slice[0].close).ln();
        features[[i, 0]] = log_return;

        // Feature 2: volatility (std of returns within window)
        let returns: Vec<f64> = slice
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();
        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / returns.len() as f64;
        features[[i, 1]] = var.sqrt();

        // Feature 3: volume ratio (current vs average)
        let avg_vol = slice.iter().map(|k| k.volume).sum::<f64>() / slice.len() as f64;
        features[[i, 2]] = if avg_vol > 0.0 {
            current.volume / avg_vol
        } else {
            1.0
        };

        // Feature 4: high-low range normalized
        features[[i, 3]] = if current.close > 0.0 {
            (current.high - current.low) / current.close
        } else {
            0.0
        };

        // Feature 5: momentum (close vs open ratio)
        features[[i, 4]] = if current.open > 0.0 {
            current.close / current.open - 1.0
        } else {
            0.0
        };

        // Label: 1 if next period return is positive, 0 otherwise
        if i + window < n {
            let next = &klines[i + window];
            labels[i] = if next.close > current.close { 1.0 } else { 0.0 };
        }
    }

    (features, labels)
}

// ============================================================================
// Model Trait and Implementations
// ============================================================================

/// Trait for predictive models that can be attacked
pub trait TradingModel {
    fn predict(&self, x: &Array1<f64>) -> f64;
    fn predict_class(&self, x: &Array1<f64>) -> i32 {
        if self.predict(x) > 0.5 { 1 } else { 0 }
    }
    fn name(&self) -> &str;
}

/// Trait for models that expose gradients (white-box)
pub trait GradientModel: TradingModel {
    fn gradient(&self, x: &Array1<f64>) -> Array1<f64>;
}

// ---- Linear Model ----

#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Array1<f64>,
    pub bias: f64,
}

impl LinearModel {
    pub fn new(num_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array1::from_vec(
            (0..num_features).map(|_| rng.gen_range(-0.5..0.5)).collect(),
        );
        Self {
            weights,
            bias: rng.gen_range(-0.1..0.1),
        }
    }

    /// Train on data using simple gradient descent
    pub fn train(&mut self, features: &Array2<f64>, labels: &Array1<f64>, lr: f64, epochs: usize) {
        let n = features.nrows();
        if n == 0 {
            return;
        }
        for _ in 0..epochs {
            let mut grad_w: Array1<f64> = Array1::zeros(self.weights.len());
            let mut grad_b = 0.0;

            for i in 0..n {
                let x = features.row(i).to_owned();
                let pred = sigmoid(x.dot(&self.weights) + self.bias);
                let error = pred - labels[i];
                grad_w = grad_w + &x * error;
                grad_b += error;
            }

            self.weights = &self.weights - &(&grad_w * (lr / n as f64));
            self.bias -= grad_b * lr / n as f64;
        }
    }
}

impl TradingModel for LinearModel {
    fn predict(&self, x: &Array1<f64>) -> f64 {
        sigmoid(x.dot(&self.weights) + self.bias)
    }

    fn name(&self) -> &str {
        "LinearModel"
    }
}

impl GradientModel for LinearModel {
    fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
        let z = x.dot(&self.weights) + self.bias;
        let s = sigmoid(z);
        let dsig = s * (1.0 - s);
        &self.weights * dsig
    }
}

// ---- Threshold Model ----

#[derive(Debug, Clone)]
pub struct ThresholdModel {
    pub thresholds: Array1<f64>,
    pub weights: Array1<f64>,
    pub bias: f64,
}

impl ThresholdModel {
    pub fn new(num_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            thresholds: Array1::from_vec(
                (0..num_features).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            ),
            weights: Array1::from_vec(
                (0..num_features).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            ),
            bias: rng.gen_range(-0.1..0.1),
        }
    }

    /// Train by finding optimal thresholds from data statistics
    pub fn train(&mut self, features: &Array2<f64>, labels: &Array1<f64>, lr: f64, epochs: usize) {
        let n = features.nrows();
        if n == 0 {
            return;
        }

        // Set thresholds to feature medians
        for j in 0..features.ncols() {
            let mut col: Vec<f64> = (0..n).map(|i| features[[i, j]]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.thresholds[j] = col[n / 2];
        }

        // Train weights on thresholded features
        for _ in 0..epochs {
            let mut grad_w: Array1<f64> = Array1::zeros(self.weights.len());
            let mut grad_b = 0.0;

            for i in 0..n {
                let x = features.row(i).to_owned();
                let transformed = self.apply_thresholds(&x);
                let pred = sigmoid(transformed.dot(&self.weights) + self.bias);
                let error = pred - labels[i];
                grad_w = grad_w + &transformed * error;
                grad_b += error;
            }

            self.weights = &self.weights - &(&grad_w * (lr / n as f64));
            self.bias -= grad_b * lr / n as f64;
        }
    }

    fn apply_thresholds(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.len());
        for i in 0..x.len() {
            result[i] = if x[i] > self.thresholds[i] { 1.0 } else { -1.0 };
        }
        result
    }
}

impl TradingModel for ThresholdModel {
    fn predict(&self, x: &Array1<f64>) -> f64 {
        let transformed = self.apply_thresholds(x);
        sigmoid(transformed.dot(&self.weights) + self.bias)
    }

    fn name(&self) -> &str {
        "ThresholdModel"
    }
}

// ---- Ensemble Model ----

#[derive(Debug, Clone)]
pub struct EnsembleModel {
    pub linear: LinearModel,
    pub threshold: ThresholdModel,
    pub linear_weight: f64,
    pub threshold_weight: f64,
}

impl EnsembleModel {
    pub fn new(num_features: usize) -> Self {
        Self {
            linear: LinearModel::new(num_features),
            threshold: ThresholdModel::new(num_features),
            linear_weight: 0.5,
            threshold_weight: 0.5,
        }
    }

    pub fn train(&mut self, features: &Array2<f64>, labels: &Array1<f64>, lr: f64, epochs: usize) {
        self.linear.train(features, labels, lr, epochs);
        self.threshold.train(features, labels, lr, epochs);
    }
}

impl TradingModel for EnsembleModel {
    fn predict(&self, x: &Array1<f64>) -> f64 {
        let p1 = self.linear.predict(x);
        let p2 = self.threshold.predict(x);
        self.linear_weight * p1 + self.threshold_weight * p2
    }

    fn name(&self) -> &str {
        "EnsembleModel"
    }
}

// ============================================================================
// Financial Constraints
// ============================================================================

#[derive(Debug, Clone)]
pub struct FinancialConstraints {
    /// Maximum percentage change per feature
    pub max_pct_change: Vec<f64>,
    /// Absolute bounds for perturbations
    pub abs_bounds: Vec<(f64, f64)>,
}

impl FinancialConstraints {
    /// Create default constraints for 5-feature trading model
    pub fn default_trading() -> Self {
        Self {
            // return, volatility, volume_ratio, range, momentum
            max_pct_change: vec![0.02, 0.5, 0.3, 0.5, 0.02],
            abs_bounds: vec![
                (-0.05, 0.05),   // log return
                (0.0, 0.2),      // volatility (non-negative)
                (0.1, 5.0),      // volume ratio
                (0.0, 0.1),      // high-low range
                (-0.05, 0.05),   // momentum
            ],
        }
    }

    /// Apply constraints to a perturbed sample
    pub fn apply(&self, original: &Array1<f64>, perturbed: &Array1<f64>) -> Array1<f64> {
        let mut result = perturbed.clone();
        for i in 0..result.len().min(self.max_pct_change.len()) {
            // Enforce max percentage change
            let max_delta = original[i].abs() * self.max_pct_change[i];
            let delta = result[i] - original[i];
            if delta.abs() > max_delta && max_delta > 0.0 {
                result[i] = original[i] + delta.signum() * max_delta;
            }
            // Enforce absolute bounds
            if i < self.abs_bounds.len() {
                result[i] = result[i].clamp(self.abs_bounds[i].0, self.abs_bounds[i].1);
            }
        }
        result
    }
}

// ============================================================================
// White-Box Attacks
// ============================================================================

pub struct WhiteBoxAttack;

impl WhiteBoxAttack {
    /// Fast Gradient Sign Method (FGSM)
    /// Single-step attack: x_adv = x + epsilon * sign(gradient)
    pub fn fgsm(
        model: &dyn GradientModel,
        x: &Array1<f64>,
        epsilon: f64,
        constraints: Option<&FinancialConstraints>,
    ) -> Array1<f64> {
        let grad = model.gradient(x);
        let perturbation = grad.mapv(|g| epsilon * g.signum());
        let x_adv = x + &perturbation;
        match constraints {
            Some(c) => c.apply(x, &x_adv),
            None => x_adv,
        }
    }

    /// Projected Gradient Descent (PGD)
    /// Multi-step iterative attack with projection onto epsilon-ball
    pub fn pgd(
        model: &dyn GradientModel,
        x: &Array1<f64>,
        epsilon: f64,
        alpha: f64,
        num_steps: usize,
        constraints: Option<&FinancialConstraints>,
    ) -> Array1<f64> {
        let mut rng = rand::thread_rng();

        // Initialize with random perturbation within epsilon-ball
        let mut x_adv = x.mapv(|xi| xi + rng.gen_range(-epsilon..epsilon));

        for _ in 0..num_steps {
            let grad = model.gradient(&x_adv);
            // Step in gradient direction (maximize loss for untargeted attack)
            let step = grad.mapv(|g| alpha * g.signum());
            x_adv = &x_adv + &step;

            // Project back onto epsilon-ball centered at x
            for i in 0..x_adv.len() {
                let delta = x_adv[i] - x[i];
                x_adv[i] = x[i] + delta.clamp(-epsilon, epsilon);
            }
        }

        match constraints {
            Some(c) => c.apply(x, &x_adv),
            None => x_adv,
        }
    }
}

// ============================================================================
// Black-Box Attacks
// ============================================================================

pub struct BlackBoxAttack;

impl BlackBoxAttack {
    /// Random search: sample perturbations and keep the best
    pub fn random_search(
        model: &dyn TradingModel,
        x: &Array1<f64>,
        epsilon: f64,
        num_samples: usize,
        constraints: Option<&FinancialConstraints>,
    ) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let original_pred = model.predict(x);
        let original_class = if original_pred > 0.5 { 1 } else { 0 };

        let mut best_adv = x.clone();
        let mut best_score = 0.0_f64; // how far prediction moved from original

        for _ in 0..num_samples {
            let perturbation = Array1::from_vec(
                (0..x.len()).map(|_| rng.gen_range(-epsilon..epsilon)).collect(),
            );
            let candidate = x + &perturbation;
            let candidate = match constraints {
                Some(c) => c.apply(x, &candidate),
                None => candidate,
            };

            let pred = model.predict(&candidate);
            let new_class = if pred > 0.5 { 1 } else { 0 };

            // Score: how much the prediction shifted (and did the class flip?)
            let score = (pred - original_pred).abs();
            if new_class != original_class && score > best_score {
                best_score = score;
                best_adv = candidate;
            }
        }

        best_adv
    }

    /// Coordinate-wise perturbation: perturb one feature at a time
    pub fn coordinate_wise(
        model: &dyn TradingModel,
        x: &Array1<f64>,
        epsilon: f64,
        constraints: Option<&FinancialConstraints>,
    ) -> Array1<f64> {
        let original_pred = model.predict(x);
        let original_class = if original_pred > 0.5 { 1 } else { 0 };
        let target_direction = if original_class == 1 { -1.0 } else { 1.0 };

        let mut x_adv = x.clone();

        for i in 0..x.len() {
            // Try positive perturbation
            let mut x_pos = x_adv.clone();
            x_pos[i] += epsilon;
            if let Some(c) = constraints {
                x_pos = c.apply(x, &x_pos);
            }
            let pred_pos = model.predict(&x_pos);

            // Try negative perturbation
            let mut x_neg = x_adv.clone();
            x_neg[i] -= epsilon;
            if let Some(c) = constraints {
                x_neg = c.apply(x, &x_neg);
            }
            let pred_neg = model.predict(&x_neg);

            // Keep the perturbation that moves prediction toward target
            let delta_pos = (pred_pos - original_pred) * target_direction;
            let delta_neg = (pred_neg - original_pred) * target_direction;

            if delta_pos > delta_neg && delta_pos > 0.0 {
                x_adv = x_pos;
            } else if delta_neg > 0.0 {
                x_adv = x_neg;
            }
        }

        x_adv
    }

    /// Query-based gradient estimation using finite differences
    pub fn query_based(
        model: &dyn TradingModel,
        x: &Array1<f64>,
        epsilon: f64,
        h: f64,
        constraints: Option<&FinancialConstraints>,
    ) -> Array1<f64> {
        let base_pred = model.predict(x);
        let mut estimated_grad = Array1::zeros(x.len());

        // Estimate gradient via finite differences
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            x_plus[i] += h;
            let pred_plus = model.predict(&x_plus);
            estimated_grad[i] = (pred_plus - base_pred) / h;
        }

        // Apply FGSM-style update using estimated gradient
        let original_class = if base_pred > 0.5 { 1 } else { 0 };
        let direction = if original_class == 1 { -1.0 } else { 1.0 };

        let perturbation = estimated_grad.mapv(|g| epsilon * (g * direction).signum());
        let x_adv = x + &perturbation;

        match constraints {
            Some(c) => c.apply(x, &x_adv),
            None => x_adv,
        }
    }
}

// ============================================================================
// Attack Success Metrics
// ============================================================================

#[derive(Debug, Clone)]
pub struct AttackMetrics {
    pub success_rate: f64,
    pub avg_perturbation_l2: f64,
    pub avg_confidence_change: f64,
    pub num_samples: usize,
}

impl std::fmt::Display for AttackMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Success rate: {:.2}% | Avg L2 perturbation: {:.6} | Avg confidence change: {:.4} | Samples: {}",
            self.success_rate * 100.0,
            self.avg_perturbation_l2,
            self.avg_confidence_change,
            self.num_samples
        )
    }
}

/// Compute attack success metrics across a batch of samples
pub fn compute_attack_metrics(
    model: &dyn TradingModel,
    originals: &[Array1<f64>],
    adversarials: &[Array1<f64>],
) -> AttackMetrics {
    let n = originals.len();
    if n == 0 {
        return AttackMetrics {
            success_rate: 0.0,
            avg_perturbation_l2: 0.0,
            avg_confidence_change: 0.0,
            num_samples: 0,
        };
    }

    let mut successes = 0;
    let mut total_l2 = 0.0;
    let mut total_conf_change = 0.0;

    for i in 0..n {
        let orig_pred = model.predict(&originals[i]);
        let adv_pred = model.predict(&adversarials[i]);

        let orig_class = if orig_pred > 0.5 { 1 } else { 0 };
        let adv_class = if adv_pred > 0.5 { 1 } else { 0 };

        if orig_class != adv_class {
            successes += 1;
        }

        let diff = &adversarials[i] - &originals[i];
        total_l2 += diff.mapv(|d| d * d).sum().sqrt();
        total_conf_change += (adv_pred - orig_pred).abs();
    }

    AttackMetrics {
        success_rate: successes as f64 / n as f64,
        avg_perturbation_l2: total_l2 / n as f64,
        avg_confidence_change: total_conf_change / n as f64,
        num_samples: n,
    }
}

// ============================================================================
// Transferability Testing
// ============================================================================

#[derive(Debug, Clone)]
pub struct TransferResult {
    pub source_model: String,
    pub target_model: String,
    pub transfer_rate: f64,
    pub num_tested: usize,
}

impl std::fmt::Display for TransferResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {}: transfer rate {:.2}% ({} samples)",
            self.source_model,
            self.target_model,
            self.transfer_rate * 100.0,
            self.num_tested
        )
    }
}

/// Test transferability: adversarial examples crafted on source_model tested on target_model
pub fn test_transferability(
    source_model: &dyn TradingModel,
    target_model: &dyn TradingModel,
    originals: &[Array1<f64>],
    adversarials: &[Array1<f64>],
) -> TransferResult {
    let n = originals.len();
    let mut source_successes = 0;
    let mut transfer_successes = 0;

    for i in 0..n {
        let orig_class_source = source_model.predict_class(&originals[i]);
        let adv_class_source = source_model.predict_class(&adversarials[i]);

        // Only count samples where attack succeeded on source
        if orig_class_source != adv_class_source {
            source_successes += 1;

            let orig_class_target = target_model.predict_class(&originals[i]);
            let adv_class_target = target_model.predict_class(&adversarials[i]);

            if orig_class_target != adv_class_target {
                transfer_successes += 1;
            }
        }
    }

    TransferResult {
        source_model: source_model.name().to_string(),
        target_model: target_model.name().to_string(),
        transfer_rate: if source_successes > 0 {
            transfer_successes as f64 / source_successes as f64
        } else {
            0.0
        },
        num_tested: source_successes,
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Generate synthetic trading data for testing
pub fn generate_synthetic_data(num_samples: usize, num_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();
    let mut features = Array2::zeros((num_samples, num_features));
    let mut labels = Array1::zeros(num_samples);

    for i in 0..num_samples {
        for j in 0..num_features {
            features[[i, j]] = rng.gen_range(-1.0..1.0);
        }
        // Simple labeling: positive if weighted sum > 0
        let sum: f64 = (0..num_features).map(|j| features[[i, j]] * (j as f64 + 1.0)).sum();
        labels[i] = if sum > 0.0 { 1.0 } else { 0.0 };
    }

    (features, labels)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trained_linear() -> (LinearModel, Array2<f64>, Array1<f64>) {
        let (features, labels) = generate_synthetic_data(200, 5);
        let mut model = LinearModel::new(5);
        model.train(&features, &labels, 0.1, 100);
        (model, features, labels)
    }

    #[test]
    fn test_linear_model_predict() {
        let model = LinearModel::new(5);
        let x = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5]);
        let pred = model.predict(&x);
        assert!(pred >= 0.0 && pred <= 1.0, "Prediction must be in [0,1]");
    }

    #[test]
    fn test_threshold_model_predict() {
        let model = ThresholdModel::new(5);
        let x = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5]);
        let pred = model.predict(&x);
        assert!(pred >= 0.0 && pred <= 1.0, "Prediction must be in [0,1]");
    }

    #[test]
    fn test_ensemble_model_predict() {
        let model = EnsembleModel::new(5);
        let x = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5]);
        let pred = model.predict(&x);
        assert!(pred >= 0.0 && pred <= 1.0, "Prediction must be in [0,1]");
    }

    #[test]
    fn test_linear_model_training() {
        let (model, features, labels) = make_trained_linear();
        let mut correct = 0;
        for i in 0..features.nrows() {
            let x = features.row(i).to_owned();
            let pred_class = model.predict_class(&x);
            if pred_class as f64 == labels[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / features.nrows() as f64;
        assert!(accuracy > 0.5, "Trained model should beat random: {:.2}", accuracy);
    }

    #[test]
    fn test_fgsm_changes_prediction() {
        let (model, features, _labels) = make_trained_linear();
        let x = features.row(0).to_owned();
        let original_pred = model.predict(&x);
        let x_adv = WhiteBoxAttack::fgsm(&model, &x, 0.5, None);
        let adv_pred = model.predict(&x_adv);
        // FGSM should change the prediction at least somewhat
        assert!(
            (adv_pred - original_pred).abs() > 1e-6,
            "FGSM should change prediction"
        );
    }

    #[test]
    fn test_pgd_stronger_than_fgsm() {
        let (model, features, _labels) = make_trained_linear();
        let x = features.row(0).to_owned();
        let original_pred = model.predict(&x);

        let fgsm_adv = WhiteBoxAttack::fgsm(&model, &x, 0.3, None);
        let pgd_adv = WhiteBoxAttack::pgd(&model, &x, 0.3, 0.075, 20, None);

        let fgsm_change = (model.predict(&fgsm_adv) - original_pred).abs();
        let pgd_change = (model.predict(&pgd_adv) - original_pred).abs();

        // PGD should generally produce at least as strong an attack
        assert!(
            pgd_change >= fgsm_change * 0.5,
            "PGD should be comparable or stronger: PGD={:.4}, FGSM={:.4}",
            pgd_change,
            fgsm_change
        );
    }

    #[test]
    fn test_random_search_black_box() {
        let (model, features, _labels) = make_trained_linear();
        let x = features.row(0).to_owned();
        let x_adv = BlackBoxAttack::random_search(&model, &x, 0.5, 500, None);
        // Should produce a valid perturbation
        let diff = &x_adv - &x;
        let l2 = diff.mapv(|d| d * d).sum().sqrt();
        assert!(l2 < 0.5 * (x.len() as f64).sqrt() + 1.0, "Perturbation should be bounded");
    }

    #[test]
    fn test_coordinate_wise_black_box() {
        let (model, features, _labels) = make_trained_linear();
        let x = features.row(0).to_owned();
        let original_pred = model.predict(&x);
        let x_adv = BlackBoxAttack::coordinate_wise(&model, &x, 0.3, None);
        let adv_pred = model.predict(&x_adv);
        assert!(
            (adv_pred - original_pred).abs() > 1e-6,
            "Coordinate-wise should change prediction"
        );
    }

    #[test]
    fn test_query_based_black_box() {
        let (model, features, _labels) = make_trained_linear();
        let x = features.row(0).to_owned();
        let original_pred = model.predict(&x);
        let x_adv = BlackBoxAttack::query_based(&model, &x, 0.3, 0.01, None);
        let adv_pred = model.predict(&x_adv);
        assert!(
            (adv_pred - original_pred).abs() > 1e-6,
            "Query-based should change prediction"
        );
    }

    #[test]
    fn test_financial_constraints() {
        let constraints = FinancialConstraints::default_trading();
        let original = Array1::from_vec(vec![0.01, 0.05, 1.0, 0.02, 0.005]);
        let perturbed = Array1::from_vec(vec![1.0, -1.0, 100.0, -5.0, 10.0]);
        let constrained = constraints.apply(&original, &perturbed);

        // Check absolute bounds
        assert!(constrained[1] >= 0.0, "Volatility must be non-negative");
        assert!(constrained[2] <= 5.0, "Volume ratio must be bounded");
        assert!(constrained[3] >= 0.0, "Range must be non-negative");
    }

    #[test]
    fn test_attack_metrics() {
        let model = LinearModel::new(5);
        let originals: Vec<Array1<f64>> = (0..10)
            .map(|_| {
                let mut rng = rand::thread_rng();
                Array1::from_vec((0..5).map(|_| rng.gen_range(-1.0..1.0)).collect())
            })
            .collect();
        let adversarials: Vec<Array1<f64>> = originals
            .iter()
            .map(|x| x.mapv(|xi| xi + 0.5))
            .collect();

        let metrics = compute_attack_metrics(&model, &originals, &adversarials);
        assert_eq!(metrics.num_samples, 10);
        assert!(metrics.avg_perturbation_l2 > 0.0);
    }

    #[test]
    fn test_transfer_between_models() {
        let (features, labels) = generate_synthetic_data(200, 5);
        let mut model_a = LinearModel::new(5);
        let mut model_b = ThresholdModel::new(5);
        model_a.train(&features, &labels, 0.1, 100);
        model_b.train(&features, &labels, 0.1, 100);

        let originals: Vec<Array1<f64>> = (0..50)
            .map(|i| features.row(i).to_owned())
            .collect();
        let adversarials: Vec<Array1<f64>> = originals
            .iter()
            .map(|x| WhiteBoxAttack::fgsm(&model_a, x, 0.5, None))
            .collect();

        let result = super::test_transferability(&model_a, &model_b, &originals, &adversarials);
        assert_eq!(result.source_model, "LinearModel");
        assert_eq!(result.target_model, "ThresholdModel");
        // Transfer rate should be a valid probability
        assert!(result.transfer_rate >= 0.0 && result.transfer_rate <= 1.0);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let (features, labels) = generate_synthetic_data(100, 5);
        assert_eq!(features.nrows(), 100);
        assert_eq!(features.ncols(), 5);
        assert_eq!(labels.len(), 100);
        // Labels should be binary
        for &l in labels.iter() {
            assert!(l == 0.0 || l == 1.0);
        }
    }

    #[test]
    fn test_gradient_computation() {
        let model = LinearModel::new(5);
        let x = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5]);
        let grad = model.gradient(&x);
        assert_eq!(grad.len(), 5);
        // Gradient should not be all zeros for a non-zero input
        let grad_norm: f64 = grad.mapv(|g| g * g).sum().sqrt();
        assert!(grad_norm > 1e-10, "Gradient should be non-zero");
    }

    #[test]
    fn test_feature_extraction() {
        let klines: Vec<Kline> = (0..20)
            .map(|i| Kline {
                timestamp: 1000 + i as u64 * 60000,
                open: 50000.0 + i as f64 * 10.0,
                high: 50050.0 + i as f64 * 10.0,
                low: 49950.0 + i as f64 * 10.0,
                close: 50010.0 + i as f64 * 10.0,
                volume: 100.0 + i as f64,
            })
            .collect();

        let (features, labels) = extract_features(&klines, 5);
        assert!(features.nrows() > 0);
        assert_eq!(features.ncols(), 5);
        assert_eq!(features.nrows(), labels.len());
    }

    #[test]
    fn test_fgsm_with_constraints() {
        let (model, features, _labels) = make_trained_linear();
        let constraints = FinancialConstraints::default_trading();
        let x = features.row(0).to_owned();
        let x_adv = WhiteBoxAttack::fgsm(&model, &x, 0.5, Some(&constraints));

        // Verify constraints are respected
        for i in 0..x_adv.len().min(constraints.abs_bounds.len()) {
            assert!(
                x_adv[i] >= constraints.abs_bounds[i].0 && x_adv[i] <= constraints.abs_bounds[i].1,
                "Feature {} out of bounds: {} not in [{}, {}]",
                i,
                x_adv[i],
                constraints.abs_bounds[i].0,
                constraints.abs_bounds[i].1
            );
        }
    }

    #[test]
    fn test_pgd_with_constraints() {
        let (model, features, _labels) = make_trained_linear();
        let constraints = FinancialConstraints::default_trading();
        let x = features.row(0).to_owned();
        let x_adv = WhiteBoxAttack::pgd(&model, &x, 0.3, 0.075, 20, Some(&constraints));

        for i in 0..x_adv.len().min(constraints.abs_bounds.len()) {
            assert!(
                x_adv[i] >= constraints.abs_bounds[i].0 && x_adv[i] <= constraints.abs_bounds[i].1,
                "PGD Feature {} out of bounds",
                i
            );
        }
    }
}
