"""
Model Evaluation and Visualization

This script provides:
- Performance metrics calculation
- Visualization of predictions vs actual values
- Residual plots
- Feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def calculate_metrics(y_true, y_pred):
    """Calculate all performance metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

def plot_predictions_vs_actual(y_true, y_pred, model_name="Model", save_path=None):
    """Plot predicted vs actual values"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Add metrics to plot
    textstr = f'RMSE: {metrics["RMSE"]:.2f} nM\n'
    textstr += f'MAE: {metrics["MAE"]:.2f} nM\n'
    textstr += f'R²: {metrics["R²"]:.4f}'
    
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Actual Binding Affinity (nM)', fontsize=12)
    ax.set_ylabel('Predicted Binding Affinity (nM)', fontsize=12)
    ax.set_title(f'{model_name}: Predictions vs Actual', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig

def plot_residuals(y_true, y_pred, model_name="Model", save_path=None):
    """Plot residuals"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs predicted
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Binding Affinity (nM)', fontsize=12)
    ax1.set_ylabel('Residuals (nM)', fontsize=12)
    ax1.set_title(f'{model_name}: Residuals vs Predicted', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Residuals histogram
    ax2.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residuals (nM)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'{model_name}: Residuals Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig

def plot_feature_importance(model, feature_names, top_n=20, model_name="Model", save_path=None):
    """Plot feature importance (for tree-based models)"""
    try:
        # Try to get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print(f"Model {model_name} does not support feature importance")
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(importance_df)), importance_df['importance'])
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'{model_name}: Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        return None

def create_evaluation_report(y_true, y_pred, model_name="Model", save_dir='results'):
    """Create comprehensive evaluation report"""
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*60}")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if metric == 'R²':
            print(f"  {metric}: {value:.4f}")
        elif metric == 'MAPE':
            print(f"  {metric}: {value:.2f}%")
        else:
            print(f"  {metric}: {value:.2f} nM")
    
    # Create visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Predictions vs Actual
    plot_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_predictions.png')
    plot_predictions_vs_actual(y_true, y_pred, model_name, plot_path)
    
    # Residuals
    residual_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_residuals.png')
    plot_residuals(y_true, y_pred, model_name, residual_path)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df['Model'] = model_name
    metrics_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded successfully")
    print("Use create_evaluation_report() to generate comprehensive reports")

