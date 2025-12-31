import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from datetime import datetime
import json

def load_model_artifacts():
    """Load trained model, scaler, features, and metadata."""
    with open('cascade_classifier_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('cascade_classifier_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('cascade_classifier_features.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    with open('cascade_classifier_metadata.pkl', 'rb') as f:
        model_metadata = pickle.load(f)
    
    return model, scaler, feature_columns, model_metadata


def load_test_data():
    """Load the test data and predictions from training."""
    # You'll need to save these during training, or we can regenerate them
    # For now, assuming they're saved
    try:
        test_results = pd.read_csv('test_results.csv')
        return test_results
    except:
        print("Test results not found. You'll need to run predictions on test set.")
        return None


def create_performance_summary(y_true, y_pred, y_proba):
    """Create summary statistics of model performance."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
    }
    
    return metrics


def plot_confusion_matrix_detailed(y_true, y_pred, save_path='report_confusion_matrix.png'):
    """Create detailed confusion matrix with percentages."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['No Cascade', 'Cascade'],
                yticklabels=['No Cascade', 'Cascade'])
    ax1.set_title('Confusion Matrix - Counts')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', ax=ax2,
                xticklabels=['No Cascade', 'Cascade'],
                yticklabels=['No Cascade', 'Cascade'])
    ax2.set_title('Confusion Matrix - Percentages (%)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_roc_and_pr_curves(y_true, y_proba, save_path='report_roc_pr_curves.png'):
    """Plot both ROC and Precision-Recall curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    ax2.plot(recall, precision, linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_probability_distribution(y_true, y_proba, save_path='report_probability_dist.png'):
    """Plot distribution of predicted probabilities for each class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Split probabilities by true class
    cascade_probs = y_proba[y_true == 1]
    no_cascade_probs = y_proba[y_true == 0]
    
    ax.hist(no_cascade_probs, bins=50, alpha=0.6, label='No Cascade (True)', color='blue')
    ax.hist(cascade_probs, bins=50, alpha=0.6, label='Cascade (True)', color='red')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
    
    ax.set_xlabel('Predicted Probability of Cascade')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predicted Probabilities by True Class')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_columns, top_n=25, save_path='report_feature_importance.png'):
    """Plot feature importance with better visualization."""
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    # Top features
    top_features = feature_importance.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
    ax.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Coefficient Value')
    ax.set_title(f'Top {top_n} Most Important Features\n(Green = Increases Cascade Risk, Red = Decreases)')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_importance


def plot_threshold_analysis(y_true, y_proba, save_path='report_threshold_analysis.png'):
    """Analyze performance at different probability thresholds."""
    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics = {'threshold': [], 'precision': [], 'recall': [], 'f1': []}
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        
        metrics['threshold'].append(thresh)
        metrics['precision'].append(precision_score(y_true, y_pred_thresh, zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred_thresh, zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred_thresh, zero_division=0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(metrics['threshold'], metrics['precision'], label='Precision', marker='o')
    ax.plot(metrics['threshold'], metrics['recall'], label='Recall', marker='s')
    ax.plot(metrics['threshold'], metrics['f1'], label='F1 Score', marker='^')
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold')
    
    ax.set_xlabel('Probability Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance vs. Prediction Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pd.DataFrame(metrics)


def analyze_categorical_impact(feature_importance, feature_columns):
    """Analyze impact of carriers, airports, and times of day."""
    
    # Separate features by category
    carrier_features = feature_importance[feature_importance['feature'].str.startswith('carrier_')]
    airport_features = feature_importance[feature_importance['feature'].str.startswith('airport_')]
    
    # Get hour features (sin/cos represent time cyclically, so use original hour importance if available)
    # Since we use sin/cos encoding, we'll look at the actual hour pattern
    time_features = feature_importance[
        (feature_importance['feature'].str.contains('hour')) | 
        (feature_importance['feature'].str.contains('day_of_week'))
    ]
    
    results = {
        'carriers': carrier_features.copy(),
        'airports': airport_features.copy(),
        'time': time_features.copy()
    }
    
    return results


def load_airport_mapping():
    """Load airport mapping to convert codes to names."""
    try:
        airport_mapping = pd.read_csv('data/airport_mapping_complete.csv')
        # Create a dictionary mapping AIRPORT_ID to DESCRIPTION
        airport_dict = dict(zip(airport_mapping['AIRPORT_ID'].astype(str), 
                               airport_mapping['DESCRIPTION']))
        return airport_dict
    except Exception as e:
        print(f"Warning: Could not load airport mapping: {e}")
        return {}


def create_impact_tables_html(feature_importance, airport_mapping=None):
    """Create HTML tables for top/bottom carriers, airports, and times."""
    
    # Separate by category
    carrier_features = feature_importance[feature_importance['feature'].str.startswith('carrier_')].copy()
    airport_features = feature_importance[feature_importance['feature'].str.startswith('airport_')].copy()
    
    # Clean up feature names
    carrier_features['name'] = carrier_features['feature'].str.replace('carrier_', '')
    airport_features['code'] = airport_features['feature'].str.replace('airport_', '')
    
    # Map airport codes to names
    if airport_mapping:
        airport_features['name'] = airport_features['code'].apply(
            lambda x: airport_mapping.get(x, x)
        )
    else:
        airport_features['name'] = airport_features['code']
    
    # Sort by coefficient (positive = increases risk, negative = decreases risk)
    carrier_features = carrier_features.sort_values('coefficient', ascending=False)
    airport_features = airport_features.sort_values('coefficient', ascending=False)
    
    html = ""
    
    # Carriers Table
    html += """
        <div class="section">
            <h2>Carrier Impact Analysis</h2>
            <p>Carriers are ranked by their impact on cascade risk. Positive coefficients increase risk, negative coefficients decrease risk.</p>
            
            <h3>Top 5 Highest Risk Carriers</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Carrier</th>
                    <th>Coefficient</th>
                    <th>Impact</th>
                </tr>
    """
    
    for rank, (idx, row) in enumerate(carrier_features.head(5).iterrows(), 1):
        # Fix color coding - green for negative (reduces risk), red for positive (increases risk)
        color = "red" if row['coefficient'] > 0 else "green"
        impact_text = f"+{abs(row['coefficient']):.4f} risk increase" if row['coefficient'] > 0 else f"{abs(row['coefficient']):.4f} risk decrease"
        
        html += f"""
                <tr>
                    <td>{rank}</td>
                    <td><strong>{row['name']}</strong></td>
                    <td style="color: {color};">{row['coefficient']:.4f}</td>
                    <td>{impact_text}</td>
                </tr>
        """
    
    html += """
            </table>
            
            <h3>Top 5 Lowest Risk Carriers</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Carrier</th>
                    <th>Coefficient</th>
                    <th>Impact</th>
                </tr>
    """
    
    for rank, (idx, row) in enumerate(carrier_features.tail(5).iloc[::-1].iterrows(), 1):
        color = "red" if row['coefficient'] > 0 else "green"
        impact_text = f"+{abs(row['coefficient']):.4f} risk increase" if row['coefficient'] > 0 else f"{abs(row['coefficient']):.4f} risk decrease"
        
        html += f"""
                <tr>
                    <td>{rank}</td>
                    <td><strong>{row['name']}</strong></td>
                    <td style="color: {color};">{row['coefficient']:.4f}</td>
                    <td>{impact_text}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    """
    
    # Airports Table
    html += """
        <div class="section">
            <h2>Airport Impact Analysis</h2>
            <p>Airports are ranked by their impact on cascade risk. Positive coefficients increase risk, negative coefficients decrease risk.</p>
            
            <h3>Top 5 Highest Risk Airports</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Airport Code</th>
                    <th>Airport Name</th>
                    <th>Coefficient</th>
                    <th>Impact</th>
                </tr>
    """
    
    for rank, (idx, row) in enumerate(airport_features.head(5).iterrows(), 1):
        color = "red" if row['coefficient'] > 0 else "green"
        impact_text = f"+{abs(row['coefficient']):.4f} risk increase" if row['coefficient'] > 0 else f"{abs(row['coefficient']):.4f} risk decrease"
        
        html += f"""
                <tr>
                    <td>{rank}</td>
                    <td><strong>{row['code']}</strong></td>
                    <td>{row['name']}</td>
                    <td style="color: {color};">{row['coefficient']:.4f}</td>
                    <td>{impact_text}</td>
                </tr>
        """
    
    html += """
            </table>
            
            <h3>Top 5 Lowest Risk Airports</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Airport Code</th>
                    <th>Airport Name</th>
                    <th>Coefficient</th>
                    <th>Impact</th>
                </tr>
    """
    
    for rank, (idx, row) in enumerate(airport_features.tail(5).iloc[::-1].iterrows(), 1):
        color = "red" if row['coefficient'] > 0 else "green"
        impact_text = f"+{abs(row['coefficient']):.4f} risk increase" if row['coefficient'] > 0 else f"{abs(row['coefficient']):.4f} risk decrease"
        
        html += f"""
                <tr>
                    <td>{rank}</td>
                    <td><strong>{row['code']}</strong></td>
                    <td>{row['name']}</td>
                    <td style="color: {color};">{row['coefficient']:.4f}</td>
                    <td>{impact_text}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    """
    
    # Time of Day Analysis
    time_features = feature_importance[
        feature_importance['feature'].str.contains('hour_sin') | 
        feature_importance['feature'].str.contains('hour_cos')
    ]
    
    html += """
        <div class="section">
            <h2>Time of Day Impact Analysis</h2>
            <p>Time features use cyclical encoding (sine/cosine). The coefficients below show the temporal patterns:</p>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Coefficient</th>
                    <th>Interpretation</th>
                </tr>
    """
    
    for idx, row in time_features.iterrows():
        if 'hour_sin' in row['feature']:
            interp = "Affects morning (6-12) vs evening (18-24) risk patterns"
        elif 'hour_cos' in row['feature']:
            interp = "Affects midnight-noon vs noon-midnight risk patterns"
        else:
            interp = "General time effect"
        
        color = "red" if row['coefficient'] > 0 else "green"
        html += f"""
                <tr>
                    <td><strong>{row['feature']}</strong></td>
                    <td style="color: {color};">{row['coefficient']:.4f}</td>
                    <td>{interp}</td>
                </tr>
        """
    
    html += """
            </table>
            <p><strong>Note:</strong> Specific hour-by-hour risk requires reconstructing from sine/cosine coefficients. 
            Generally, positive hour_sin indicates higher risk in morning/evening, while hour_cos affects noon periods.</p>
        </div>
    """
    
    return html


def create_html_report(metrics, feature_importance, cm, model_info, output_path='model_report.html'):
    """Generate HTML report with all results."""
    
    # Load airport mapping
    airport_mapping = load_airport_mapping()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Cascade Likelihood Classifier - Model Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 5px;
            }}
            h3 {{
                color: #555;
                margin-top: 20px;
            }}
            .metric-box {{
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-item {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
            }}
            .metric-label {{
                color: #7f8c8d;
                margin-top: 5px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background-color: white;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            .section {{
                background-color: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <h1>Cascade Likelihood Classifier - Model Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Model Overview</h2>
            <p><strong>Model Type:</strong> {model_info['model_type']}</p>
            <p><strong>Purpose:</strong> Predict whether a delayed flight will cause downstream cascading delays</p>
            <p><strong>Total Features:</strong> {model_info['n_features']}</p>
            <p><strong>Training Samples:</strong> {model_info['n_samples']}</p>
            <p><strong>Class Balance:</strong> {model_info['class_balance']}</p>
        </div>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-value">{metrics['Accuracy']:.3f}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{metrics['Precision']:.3f}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{metrics['Recall']:.3f}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{metrics['F1 Score']:.3f}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{metrics['ROC-AUC']:.3f}</div>
                    <div class="metric-label">ROC-AUC</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Confusion Matrix</h2>
            <img src="report_confusion_matrix.png" alt="Confusion Matrix">
            <table>
                <tr>
                    <th></th>
                    <th>Predicted: No Cascade</th>
                    <th>Predicted: Cascade</th>
                </tr>
                <tr>
                    <td><strong>Actual: No Cascade</strong></td>
                    <td>True Negatives: {cm[0][0]}</td>
                    <td>False Positives: {cm[0][1]}</td>
                </tr>
                <tr>
                    <td><strong>Actual: Cascade</strong></td>
                    <td>False Negatives: {cm[1][0]}</td>
                    <td>True Positives: {cm[1][1]}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>ROC and Precision-Recall Curves</h2>
            <img src="report_roc_pr_curves.png" alt="ROC and PR Curves">
        </div>
        
        <div class="section">
            <h2>Probability Distribution</h2>
            <img src="report_probability_dist.png" alt="Probability Distribution">
            <p>This shows how well the model separates cascade vs non-cascade events. Good separation means the model is confident in its predictions.</p>
        </div>
        
        <div class="section">
            <h2>Threshold Analysis</h2>
            <img src="report_threshold_analysis.png" alt="Threshold Analysis">
            <p>Use this to adjust the decision threshold based on your priorities:</p>
            <ul>
                <li><strong>Higher threshold (e.g., 0.7):</strong> Fewer false alarms, but might miss some cascades</li>
                <li><strong>Lower threshold (e.g., 0.3):</strong> Catch more cascades, but more false alarms</li>
                <li><strong>Current threshold: 0.5</strong> (default)</li>
            </ul>
        </div>
    """
    
    # Add categorical impact analysis with airport mapping
    html_content += create_impact_tables_html(feature_importance, airport_mapping)
    
    # Continue with feature importance
    html_content += """
        <div class="section">
            <h2>Top 20 Most Important Features (Overall)</h2>
            <img src="report_feature_importance.png" alt="Feature Importance">
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Coefficient</th>
                    <th>Impact</th>
                </tr>
    """
    
    # Add top 20 features to table - fix color coding here too
    for rank, (idx, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        impact = "Increases Risk" if row['coefficient'] > 0 else "Decreases Risk"
        color = "red" if row['coefficient'] > 0 else "green"
        html_content += f"""
                <tr>
                    <td>{rank}</td>
                    <td>{row['feature']}</td>
                    <td style="color: {color};">{row['coefficient']:.4f}</td>
                    <td>{impact}</td>
                </tr>
        """
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>Key Insights & Recommendations</h2>
            <ul>
                <li><strong>Model Performance:</strong> The model can distinguish between cascade and non-cascade events with {metrics['Accuracy']:.1%} accuracy</li>
                <li><strong>Use Case:</strong> Use this model to triage delayed flights before running expensive cascade simulations</li>
                <li><strong>Threshold Recommendation:</strong> For operational use, consider threshold = 0.5 for balanced performance</li>
                <li><strong>Carrier Insights:</strong> Review the carrier impact analysis to identify airlines with higher/lower cascade risk</li>
                <li><strong>Airport Insights:</strong> Review the airport impact analysis to identify locations with higher/lower cascade risk</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>Model Files</h2>
            <ul>
                <li><code>cascade_classifier_model.pkl</code> - Trained model</li>
                <li><code>cascade_classifier_scaler.pkl</code> - Feature scaler</li>
                <li><code>cascade_classifier_features.pkl</code> - Feature list</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {output_path}")

def save_metrics_json(metrics, feature_importance, model_info, output_path='model_metrics.json'):
    """Save metrics to JSON for programmatic access."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'model_info': model_info,
        'performance_metrics': metrics,
        'top_features': feature_importance.head(20).to_dict('records')
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Metrics JSON saved to: {output_path}")

def generate_report(y_true, y_pred, y_proba, model, feature_columns, model_info):
    """Generate complete model report with all visualizations."""
    print("\n" + "="*60)
    print("GENERATING MODEL REPORT")
    print("="*60)
    
    # Calculate metrics
    print("\n1. Calculating performance metrics...")
    metrics = create_performance_summary(y_true, y_pred, y_proba)
    
    # Generate visualizations
    print("2. Creating confusion matrix...")
    cm = plot_confusion_matrix_detailed(y_true, y_pred)
    
    print("3. Plotting ROC and PR curves...")
    plot_roc_and_pr_curves(y_true, y_proba)
    
    print("4. Analyzing probability distributions...")
    plot_probability_distribution(y_true, y_proba)
    
    print("5. Analyzing feature importance...")
    feature_importance = plot_feature_importance(model, feature_columns)
    
    print("6. Performing threshold analysis...")
    threshold_df = plot_threshold_analysis(y_true, y_proba)
    
    # Generate reports
    print("7. Creating HTML report...")
    create_html_report(metrics, feature_importance, cm, model_info)
    
    print("8. Saving metrics to JSON...")
    save_metrics_json(metrics, feature_importance, model_info)
    
    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - model_report.html (open in browser)")
    print("  - model_metrics.json")
    print("  - report_confusion_matrix.png")
    print("  - report_roc_pr_curves.png")
    print("  - report_probability_dist.png")
    print("  - report_feature_importance.png")
    print("  - report_threshold_analysis.png")


def main():
    """Main function to generate model report."""
    print("Loading model artifacts...")
    model, scaler, feature_columns, model_metadata = load_model_artifacts()
    
    # Load test predictions
    test_results = pd.read_csv('results/MachineLearning/test_results.csv')
    y_true = test_results['y_true']
    y_pred = test_results['y_pred']
    y_proba = test_results['y_proba']
    
    # Create model info from metadata
    model_info = {
        'model_type': 'Logistic Regression',
        'n_features': model_metadata['n_features'],
        'n_samples': model_metadata['n_total'],
        'class_balance': f"{model_metadata['cascade_pct_train']:.1f}% cascade events",
        'training_date': model_metadata['training_date'],
        'train_samples': model_metadata['n_train'],
        'val_samples': model_metadata['n_val'],
        'test_samples': model_metadata['n_test']
    }
    
    # Generate report
    generate_report(y_true, y_pred, y_proba, model, feature_columns, model_info)


if __name__ == "__main__":
    main()