import os
import numpy as np
from src.LSTM_Autoencoder.anomaly_analyzer import AnomalyAnalyzer  


# Example usage and utility functions
def demo_single_dataset_analysis():
    """Demonstrate analysis of a single dataset"""
    # Initialize analyzer
    analyzer = AnomalyAnalyzer()

    # Analyze FD004 dataset
    results = analyzer.analyze_dataset(
        dataset_id='FD004',
        sequence_length=30,
        sensors_to_drop=[1, 5, 10, 16, 18, 19],
        epochs=30,
        save_model=True
    )

    # Print model summary
    print("\n📊 Model Summary:")
    summary = analyzer.get_model_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

    return analyzer, results


def demo_load_and_predict():
    """Demonstrate loading a saved model and making predictions"""
    # Initialize analyzer
    analyzer = AnomalyAnalyzer()

    # List available models
    print("Available models:", analyzer.list_available_models())

    # Load a saved model (assuming 'FD004' exists)
    try:
        autoencoder, preprocessor, config = analyzer.load_saved_model('FD004')
        print("Model loaded successfully")

        # Load new data for prediction (using test data as example)
        data = analyzer.data_loader.load_dataset('FD004')

        # Make predictions on test data
        predictions = analyzer.predict_anomalies(data['test'])

        print(f"Predictions made for {len(predictions)} engines")

    except FileNotFoundError:
        print("No saved model found. Run training first.")

    return analyzer


def demo_full_comparison():
    """Demonstrate comparison across all datasets"""
    # Initialize analyzer
    analyzer = AnomalyAnalyzer()

    # Compare all datasets
    results = analyzer.compare_all_datasets()

    return analyzer, results



# Example usage
print("Anomaly Detection Framework")
print("=====================================")

# Option 1: Analyze single dataset
#print("\n1. Single Dataset Analysis:")
#analyzer, results = demo_single_dataset_analysis()

# Option 2: Full comparison (uncomment to use - takes longer)
print("\n3. Full Dataset Comparison:")
analyzer, results = demo_full_comparison()
