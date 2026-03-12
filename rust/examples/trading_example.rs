use anyhow::Result;
use evasion_attacks_trading::*;
use ndarray::Array1;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Chapter 227: Evasion Attacks in Trading ===\n");

    // ---- Step 1: Fetch data from Bybit ----
    println!("[1] Fetching BTCUSDT kline data from Bybit...");
    let (features, labels) = match fetch_klines("BTCUSDT", "15", 200).await {
        Ok(klines) => {
            println!("    Fetched {} klines from Bybit", klines.len());
            let (f, l) = extract_features(&klines, 10);
            println!("    Extracted {} samples with {} features", f.nrows(), f.ncols());
            if f.nrows() < 20 {
                println!("    Too few samples from API, supplementing with synthetic data");
                generate_synthetic_data(200, 5)
            } else {
                (f, l)
            }
        }
        Err(e) => {
            println!("    Bybit API unavailable ({}), using synthetic data", e);
            generate_synthetic_data(200, 5)
        }
    };

    let num_features = features.ncols();
    println!(
        "    Dataset: {} samples, {} features\n",
        features.nrows(),
        num_features
    );

    // ---- Step 2: Train 3 different models ----
    println!("[2] Training 3 different model architectures...");

    let mut linear_model = LinearModel::new(num_features);
    linear_model.train(&features, &labels, 0.1, 200);
    println!("    Linear model trained");

    let mut threshold_model = ThresholdModel::new(num_features);
    threshold_model.train(&features, &labels, 0.1, 200);
    println!("    Threshold model trained");

    let mut ensemble_model = EnsembleModel::new(num_features);
    ensemble_model.train(&features, &labels, 0.1, 200);
    println!("    Ensemble model trained");

    // Evaluate baseline accuracy
    for (name, model) in [
        ("Linear", &linear_model as &dyn TradingModel),
        ("Threshold", &threshold_model as &dyn TradingModel),
        ("Ensemble", &ensemble_model as &dyn TradingModel),
    ] {
        let correct: usize = (0..features.nrows())
            .filter(|&i| {
                let x = features.row(i).to_owned();
                model.predict_class(&x) as f64 == labels[i]
            })
            .count();
        println!(
            "    {} accuracy: {:.1}%",
            name,
            100.0 * correct as f64 / features.nrows() as f64
        );
    }
    println!();

    // ---- Step 3: Prepare test samples ----
    let test_size = 50.min(features.nrows());
    let test_samples: Vec<Array1<f64>> = (0..test_size)
        .map(|i| features.row(i).to_owned())
        .collect();

    let constraints = FinancialConstraints::default_trading();

    // ---- Step 4: White-box attacks ----
    println!("[3] Running white-box attacks on Linear model...");

    // FGSM attack
    let fgsm_adversarials: Vec<Array1<f64>> = test_samples
        .iter()
        .map(|x| WhiteBoxAttack::fgsm(&linear_model, x, 0.3, Some(&constraints)))
        .collect();
    let fgsm_metrics = compute_attack_metrics(&linear_model, &test_samples, &fgsm_adversarials);
    println!("    FGSM:  {}", fgsm_metrics);

    // PGD attack
    let pgd_adversarials: Vec<Array1<f64>> = test_samples
        .iter()
        .map(|x| WhiteBoxAttack::pgd(&linear_model, x, 0.3, 0.075, 20, Some(&constraints)))
        .collect();
    let pgd_metrics = compute_attack_metrics(&linear_model, &test_samples, &pgd_adversarials);
    println!("    PGD:   {}", pgd_metrics);
    println!();

    // ---- Step 5: Black-box attacks ----
    println!("[4] Running black-box attacks on Linear model...");

    // Random search
    let random_adversarials: Vec<Array1<f64>> = test_samples
        .iter()
        .map(|x| BlackBoxAttack::random_search(&linear_model, x, 0.3, 200, Some(&constraints)))
        .collect();
    let random_metrics =
        compute_attack_metrics(&linear_model, &test_samples, &random_adversarials);
    println!("    Random search:   {}", random_metrics);

    // Coordinate-wise
    let coord_adversarials: Vec<Array1<f64>> = test_samples
        .iter()
        .map(|x| BlackBoxAttack::coordinate_wise(&linear_model, x, 0.3, Some(&constraints)))
        .collect();
    let coord_metrics = compute_attack_metrics(&linear_model, &test_samples, &coord_adversarials);
    println!("    Coordinate-wise: {}", coord_metrics);

    // Query-based
    let query_adversarials: Vec<Array1<f64>> = test_samples
        .iter()
        .map(|x| BlackBoxAttack::query_based(&linear_model, x, 0.3, 0.01, Some(&constraints)))
        .collect();
    let query_metrics = compute_attack_metrics(&linear_model, &test_samples, &query_adversarials);
    println!("    Query-based:     {}", query_metrics);
    println!();

    // ---- Step 6: Transferability testing ----
    println!("[5] Testing attack transferability...");
    println!("    (Attacks crafted on source model, tested on target model)");
    println!();

    let models: Vec<(&str, &dyn TradingModel)> = vec![
        ("Linear", &linear_model),
        ("Threshold", &threshold_model),
        ("Ensemble", &ensemble_model),
    ];

    // Use PGD adversarials (crafted on linear model) for transfer testing
    println!("    PGD attacks (crafted on Linear model):");
    for (name, model) in &models {
        let result = test_transferability(&linear_model, *model, &test_samples, &pgd_adversarials);
        println!("      -> {}: {}", name, result);
    }
    println!();

    // Also craft attacks on threshold model using black-box (since it has no gradient)
    println!("    Black-box attacks (crafted on Threshold model via coordinate-wise):");
    let threshold_adversarials: Vec<Array1<f64>> = test_samples
        .iter()
        .map(|x| {
            BlackBoxAttack::coordinate_wise(&threshold_model, x, 0.3, Some(&constraints))
        })
        .collect();
    for (name, model) in &models {
        let result = test_transferability(
            &threshold_model,
            *model,
            &test_samples,
            &threshold_adversarials,
        );
        println!("      -> {}: {}", name, result);
    }
    println!();

    // ---- Step 7: Defense strategies comparison ----
    println!("[6] Defense strategies comparison...");
    println!();

    // Defense 1: No defense (baseline)
    let baseline_metrics = compute_attack_metrics(&linear_model, &test_samples, &pgd_adversarials);
    println!(
        "    No defense (Linear):    success rate = {:.1}%",
        baseline_metrics.success_rate * 100.0
    );

    // Defense 2: Ensemble defense
    let ensemble_metrics =
        compute_attack_metrics(&ensemble_model, &test_samples, &pgd_adversarials);
    println!(
        "    Ensemble defense:       success rate = {:.1}%",
        ensemble_metrics.success_rate * 100.0
    );

    // Defense 3: Input constraint filtering
    let constrained_adversarials: Vec<Array1<f64>> = pgd_adversarials
        .iter()
        .enumerate()
        .map(|(i, x_adv)| constraints.apply(&test_samples[i], x_adv))
        .collect();
    let constrained_metrics =
        compute_attack_metrics(&linear_model, &test_samples, &constrained_adversarials);
    println!(
        "    Constraint filtering:   success rate = {:.1}%",
        constrained_metrics.success_rate * 100.0
    );

    // Defense 4: Ensemble + constraints
    let ensemble_constrained_metrics =
        compute_attack_metrics(&ensemble_model, &test_samples, &constrained_adversarials);
    println!(
        "    Ensemble + constraints: success rate = {:.1}%",
        ensemble_constrained_metrics.success_rate * 100.0
    );

    println!();
    println!("=== Evasion Attack Analysis Complete ===");
    println!();
    println!("Key findings:");
    println!("  - White-box attacks (PGD) are stronger than black-box attacks");
    println!("  - Adversarial examples transfer between different model architectures");
    println!("  - Ensemble models reduce attack transferability");
    println!("  - Financial constraints limit but do not eliminate evasion attacks");
    println!("  - Defense-in-depth (ensemble + constraints) provides the best protection");

    Ok(())
}
