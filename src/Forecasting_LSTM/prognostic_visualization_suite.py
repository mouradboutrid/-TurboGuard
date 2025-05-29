import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm


class PrognosticVisualizationSuite:
    """Visualization suite with operational mode support"""

    def __init__(self, save_path=None):
        self.save_path = save_path

    def _save_plot(self, filename):
        """Helper function to save plots"""
        if self.save_path:
            full_path = os.path.join(self.save_path, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {full_path}")

    def plot_dataset_overview(self, individual_datasets, combined_df):
        """Overview with operational mode visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Dataset sizes comparison
        dataset_sizes = {name: len(df) for name, df in individual_datasets.items()}
        colors = ['blue', 'green', 'orange', 'red'][:len(dataset_sizes)]
        axes[0, 0].bar(dataset_sizes.keys(), dataset_sizes.values(), color=colors)
        axes[0, 0].set_title('Dataset Sizes Comparison')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Operational mode distribution
        if 'op_mode' in combined_df.columns:
            mode_counts = combined_df['op_mode'].value_counts()
            axes[0, 1].bar(mode_counts.index.astype(str), mode_counts.values,
                          color=['blue', 'green', 'orange'])
            axes[0, 1].set_title('Operational Mode Distribution')
            axes[0, 1].set_xlabel('Mode Cluster')
            axes[0, 1].set_ylabel('Count')
        else:
            unit_counts = {name: df['unit_id'].nunique() for name, df in individual_datasets.items()}
            axes[0, 1].bar(unit_counts.keys(), unit_counts.values(), color=colors)
            axes[0, 1].set_title('Number of Units per Dataset')
            axes[0, 1].set_ylabel('Number of Units')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Combined RUL distribution
        axes[1, 0].hist(combined_df['RUL'], bins=50, alpha=0.7, color='purple')
        axes[1, 0].set_title('Combined RUL Distribution')
        axes[1, 0].set_xlabel('Remaining Useful Life')
        axes[1, 0].set_ylabel('Frequency')

        # RUL distribution by operational mode if available
        if 'op_mode' in combined_df.columns:
            for mode in combined_df['op_mode'].unique():
                mode_data = combined_df[combined_df['op_mode'] == mode]
                axes[1, 1].hist(mode_data['RUL'], bins=30, alpha=0.6,
                               label=f'Mode {mode}',
                               color=['blue', 'green', 'orange'][int(mode)])
            axes[1, 1].set_title('RUL Distribution by Operational Mode')
        else:
            # Fallback to dataset-wise distribution
            for i, (name, df) in enumerate(individual_datasets.items()):
                axes[1, 1].hist(df['RUL'], bins=30, alpha=0.6,
                               label=name, color=colors[i])
            axes[1, 1].set_title('RUL Distribution by Dataset')

        axes[1, 1].set_xlabel('Remaining Useful Life')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()

        plt.tight_layout()
        self._save_plot('dataset_overview.png')
        plt.show()

    def plot_training_progress(self, history):
        """Plot model training progress"""
        if history is None:
            print("No training history available")
            return

        metrics = list(history.history.keys())
        n_metrics = len([m for m in metrics if not m.startswith('val_')])

        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]  # Ensure axes is iterable

        for i, metric in enumerate([m for m in metrics if not m.startswith('val_')]):
            axes[i].plot(history.history[metric], label=f'Training {metric}', color='blue')
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                axes[i].plot(history.history[val_metric], label=f'Validation {metric}', color='red')
            axes[i].set_title(f'Training {metric.upper()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.upper())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_plot('training_progress.png')
        plt.show()

    def plot_anomaly_results(self, results, modes=None):
        """Anomaly visualization with mode context"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3)

        # Error distributions
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        error_axes = [ax1, ax2, ax3]
        methods = ['mse', 'mae', 'max']

        for i, method in enumerate(methods):
            errors = results[method]['errors']
            threshold = results[method]['threshold']

            if modes is not None:
                # Color points by operational mode
                scatter = error_axes[i].scatter(
                    range(len(errors)),
                    errors,
                    c=modes[:len(errors)],
                    cmap='viridis',
                    alpha=0.6
                )
                plt.colorbar(scatter, ax=error_axes[i], label='Operational Mode')
            else:
                error_axes[i].plot(errors, alpha=0.6)

            error_axes[i].axhline(threshold, color='red', linestyle='--',
                                label=f'Threshold: {threshold:.4f}')
            error_axes[i].set_title(f'{method.upper()} Errors')
            error_axes[i].set_xlabel('Sample Index')
            error_axes[i].set_ylabel('Error Value')
            error_axes[i].legend()
            error_axes[i].grid(True, alpha=0.3)

        # Anomaly comparison
        ax4 = fig.add_subplot(gs[1, :2])
        methods = ['MSE', 'MAE', 'Max Error', 'Ensemble']
        counts = [results['mse']['anomalies'].sum(),
                 results['mae']['anomalies'].sum(),
                 results['max']['anomalies'].sum(),
                 results['ensemble']['anomalies'].sum()]

        bars = ax4.bar(methods, counts, color=['blue', 'green', 'orange', 'red'])
        ax4.set_title('Anomalies Detected by Method')
        ax4.set_ylabel('Number of Anomalies')
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')

        # Error distribution
        ax5 = fig.add_subplot(gs[1, 2])
        for method, color in zip(['mse', 'mae', 'max'], ['blue', 'green', 'orange']):
            ax5.hist(results[method]['errors'], bins=50, alpha=0.5,
                    label=method.upper(), color=color)
            ax5.axvline(results[method]['threshold'], color=color, linestyle='--')
        ax5.set_title('Error Distributions')
        ax5.set_xlabel('Error Value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Summary statistics
        ax6 = fig.add_subplot(gs[2, :])
        total_samples = len(results['mse']['errors'])
        ensemble_rate = (results['ensemble']['anomalies'].sum() / total_samples) * 100

        summary_text = f"""Total Samples: {total_samples:,}
Ensemble Anomalies: {results['ensemble']['anomalies'].sum():,}
Anomaly Rate: {ensemble_rate:.2f}%

Detection Summary:
• MSE: {results['mse']['anomalies'].sum():,} anomalies
• MAE: {results['mae']['anomalies'].sum():,} anomalies
• Max Error: {results['max']['anomalies'].sum():,} anomalies
• Ensemble: {results['ensemble']['anomalies'].sum():,} anomalies"""

        if modes is not None:
            mode_anomalies = []
            for mode in np.unique(modes):
                mask = (modes[:len(results['ensemble']['anomalies'])] == mode)
                mode_count = results['ensemble']['anomalies'][mask].sum()
                mode_rate = (mode_count / mask.sum()) * 100
                mode_anomalies.append(f"Mode {mode}: {mode_count:,} ({mode_rate:.1f}%)")

            summary_text += "\n\nAnomalies by Operational Mode:\n" + "\n".join(["• " + m for m in mode_anomalies])

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Detection Summary')

        plt.tight_layout()
        self._save_plot('anomaly_results.png')
        plt.show()