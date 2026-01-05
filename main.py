import argparse
import sys
from pathlib import Path

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import DataPipeline
from src.model import FlightPriceModel
from src.train import ModelTrainer

# Define base path relative to this script
BASE_PATH = Path(__file__).parent


def run_pipeline(data_path: str, model_type: str = 'xgb', tune: bool = False):
    """Run the complete ML pipeline."""
    print("=" * 70)
    print("FLIGHT PRICE PREDICTOR - ML PIPELINE")
    print("=" * 70)
    
    # 1. DATA PIPELINE
    print("\n" + "=" * 70)
    print("STEP 1: DATA PIPELINE")
    print("=" * 70)
    
    pipeline = DataPipeline()
    df = pipeline.run_pipeline(data_path)
    
    # Validate data
    pipeline.validate_data(df)
    
    # Save processed data
    processed_path = str(BASE_PATH / "data" / "processed" / "flight_data_cleaned.pkl")
    Path(BASE_PATH / "data" / "processed").mkdir(parents=True, exist_ok=True)
    pipeline.save_processed_data(df, processed_path)
    
    X, y = pipeline.get_feature_target_split(df)
    
    # 2. MODEL INITIALIZATION
    print("\n" + "=" * 70)
    print("STEP 2: MODEL INITIALIZATION")
    print("=" * 70)
    
    model = FlightPriceModel(model_type=model_type, use_tuned=True)
    model.build_pipeline()
    print(f"Model type: {model_type}")
    print(f"Hyperparameters: {model.hyperparameters}")
    
    # 3. TRAINING
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING")
    print("=" * 70)
    
    trainer = ModelTrainer(model)
    trainer.split_data(X, y)
    trainer.preprocess()
    
    if tune and model_type in ['rf', 'xgb']:
        print("\nPerforming hyperparameter tuning...")
        if model_type == 'rf':
            param_grid = {
                'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]
            }
        else:
            param_grid = {
                'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2], 'subsample': [0.8, 0.9, 1.0]
            }
        trainer.tune_hyperparameters(param_grid, n_iter=20)
    
    trainer.train()
    
    # 4. EVALUATION
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATION")
    print("=" * 70)
    
    trainer.evaluate()
    trainer.print_metrics()
    
    cv_results = trainer.cross_validate()
    print(f"\nCross-Validation R²: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']*2:.4f})")
    
    # Feature importance
    print("\nTop 10 Important Features:")
    importance = model.get_feature_importance()
    for i, row in importance.head(10).iterrows():
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    # 5. SAVE ARTIFACTS
    print("\n" + "=" * 70)
    print("STEP 5: SAVING ARTIFACTS")
    print("=" * 70)
    
    Path(BASE_PATH / "models").mkdir(exist_ok=True)
    
    trainer.save_model(
        pipeline_path=str(BASE_PATH / "models" / "flight_price_pipeline.joblib")
    )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    
    summary = trainer.get_training_summary()
    print(f"""
Model: {summary['model_type'].upper()}
Train Samples: {summary['train_samples']:,}
Test Samples: {summary['test_samples']:,}
Features: {summary['n_features']}

Performance:
  Test R²: {summary['metrics']['test_r2']:.4f}
  Test RMSE: ₹{summary['metrics']['test_rmse']:,.0f}
  Test MAE: ₹{summary['metrics']['test_mae']:,.0f}
  Test MAPE: {summary['metrics']['test_mape']:.2f}%

Saved Files:
  - models/flight_price_pipeline.joblib
  - data/processed/flight_data_cleaned.pkl
    """)
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Flight Price Predictor ML Pipeline")
    parser.add_argument("--data", "-d", type=str, 
                        default=str(BASE_PATH / "data" / "raw" / "Data_Train.xlsx"),
                        help="Path to raw data file")
    parser.add_argument("--model", "-m", type=str, default="xgb", 
                        choices=["linear", "ridge", "rf", "gb", "xgb"],
                        help="Model type")
    parser.add_argument("--tune", "-t", action="store_true",
                        help="Perform hyperparameter tuning")
    
    args = parser.parse_args()
    run_pipeline(data_path=args.data, model_type=args.model, tune=args.tune)


if __name__ == "__main__":
    main()
