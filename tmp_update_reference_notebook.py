import ast
import json
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference.ipynb"


def code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "id": uuid4().hex[:8],
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def main():
    nb = json.loads(REFERENCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Reset code execution metadata/outputs.
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    # Title
    cells[0]["source"] = ["# Kampala Air Quality and TB Analysis - LSTM Pipeline\n"]

    # Imports
    cells[1]["source"] = [
        "# Import required libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import statsmodels.api as sm\n",
        "import seaborn as sns\n",
        "from scipy import stats\n",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import LSTM, Dense, Dropout\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "from pathlib import Path\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
    ]

    # Local modules
    cells[2]["source"] = [
        "# Import simplified modules\n",
        "import sys\n",
        "sys.path.append('./src')\n",
        "\n",
        "from simplified_data_processor import SimplifiedDataProcessor\n",
        "from intervention_simulator import InterventionSimulator\n",
        "from time_series_modeler import TimeSeriesModeler\n",
        "from visualizer import Visualizer\n",
        "from report_generator import ReportGenerator\n",
        "\n",
        "# Set up plotting\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette('husl')\n",
        "\n",
        "print('✓ All modules imported successfully')\n",
    ]

    # Data loading
    cells[3]["source"] = [
        "# ## Step 1: Load and Process Data\n",
        "\n",
        "# Initialize data processor\n",
        "processor = SimplifiedDataProcessor(data_dir='./data')\n",
        "\n",
        "# Load the merged data\n",
        "data = processor.load_data('./data/merged_climate_tb_data.csv')\n",
        "print(f'✓ Loaded data: {len(data)} records')\n",
    ]

    # Add engineered features after local processor features.
    cells[4]["source"] = [
        "# Add lag features and rolling averages\n",
        "processed_data = processor.add_lag_features()\n",
        "\n",
        "# Additional engineered features for downstream LSTM modeling\n",
        "processed_data['date'] = pd.to_datetime(processed_data['date'], errors='coerce', dayfirst=True)\n",
        "processed_data['temp_humidity_interaction'] = processed_data['avgtemp'] * processed_data['avghumidity']\n",
        "processed_data['temp_sat_gap'] = processed_data['avgtemp'] - processed_data.get('avgtemp_sat', processed_data['avgtemp'])\n",
        "processed_data['humidity_sat_gap'] = processed_data['avghumidity'] - processed_data.get('humidity_sat', processed_data['avghumidity'])\n",
        "processed_data['precip_rolling_4'] = processed_data['precip'].rolling(window=4, min_periods=1).mean()\n",
        "processed_data['windspeed_rolling_4'] = processed_data['windspeed'].rolling(window=4, min_periods=1).mean()\n",
        "processed_data['pm25_temp_ratio'] = processed_data['pm2_5'] / (processed_data['avgtemp'].abs() + 1)\n",
        "processed_data['pm25_humidity_ratio'] = processed_data['pm2_5'] / (processed_data['avghumidity'].abs() + 1)\n",
        "processed_data['tb_pct_change_1'] = processed_data['TB'].pct_change().replace([np.inf, -np.inf], np.nan)\n",
        "processed_data['tb_diff_1'] = processed_data['TB'].diff()\n",
        "processed_data['sin_week'] = np.sin(2 * np.pi * processed_data['week_of_year'] / 52)\n",
        "processed_data['cos_week'] = np.cos(2 * np.pi * processed_data['week_of_year'] / 52)\n",
        "processed_data['sin_month'] = np.sin(2 * np.pi * processed_data['month'] / 12)\n",
        "processed_data['cos_month'] = np.cos(2 * np.pi * processed_data['month'] / 12)\n",
        "\n",
        "print(f'✓ Added features: {processed_data.shape[1]} total columns')\n",
    ]

    # Summary
    cells[5]["source"] = [
        "# Get summary statistics\n",
        "summary = processor.get_summary_statistics()\n",
        "print('\\nData Summary:')\n",
        "print(f\"- Date range: {summary['date_range']}\")\n",
        "print(f\"- Total records: {summary['total_records']}\")\n",
        "print(f\"- Average PM2.5: {summary['pm25_statistics']['mean']:.2f} μg/m³\")\n",
        "print(f\"- Total TB cases: {summary['tb_statistics']['total_cases']:.2f}\")\n",
        "print(f\"- PM2.5-TB correlation: {summary['correlation_pm25_tb']:.3f}\")\n",
    ]

    # Preview columns
    cells[6]["source"] = [
        "# Display first few rows including engineered features\n",
        "preview_cols = [\n",
        "    'date', 'pm2_5', 'TB', 'avgtemp', 'avghumidity',\n",
        "    'temp_humidity_interaction', 'temp_sat_gap', 'humidity_sat_gap',\n",
        "    'precip_rolling_4', 'windspeed_rolling_4', 'pm25_temp_ratio',\n",
        "    'pm25_humidity_ratio', 'tb_pct_change_1', 'tb_diff_1',\n",
        "    'sin_week', 'cos_week', 'sin_month', 'cos_month'\n",
        "]\n",
        "processed_data[preview_cols].head(10)\n",
    ]

    # Time series plot
    cells[8]["source"] = [
        "# Time series plot\n",
        "fig, axes = plt.subplots(3, 1, figsize=(14, 10))\n",
        "\n",
        "axes[0].plot(data['date'], data['pm2_5'], color='blue', alpha=0.7)\n",
        "axes[0].axhline(y=15, color='red', linestyle='--', label='WHO Guideline (24-hour average)')\n",
        "axes[0].axhline(y=5, color='orange', linestyle='--', label='WHO Guideline (annual average)')\n",
        "axes[0].set_title('PM2.5 Levels in Kampala')\n",
        "axes[0].set_ylabel('PM2.5 (μg/m³)')\n",
        "axes[0].legend()\n",
        "axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "axes[1].plot(data['date'], data['TB'], color='darkgreen', alpha=0.7)\n",
        "axes[1].set_title('TB Cases')\n",
        "axes[1].set_ylabel('Number of Cases')\n",
        "axes[1].grid(True, alpha=0.3)\n",
        "\n",
        "ax2_twin = axes[2].twinx()\n",
        "axes[2].plot(data['date'], data['avgtemp'], color='orange', label='Temperature')\n",
        "ax2_twin.plot(data['date'], data['avghumidity'], color='cyan', label='Humidity')\n",
        "axes[2].set_title('Weather Conditions')\n",
        "axes[2].set_ylabel('Temperature (°C)', color='orange')\n",
        "ax2_twin.set_ylabel('Humidity (%)', color='cyan')\n",
        "axes[2].set_xlabel('Date')\n",
        "axes[2].grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
    ]

    # Correlation
    cells[9]["source"] = [
        "# Correlation analysis\n",
        "correlation_vars = [\n",
        "    'pm2_5', 'TB', 'avgtemp', 'avghumidity', 'windspeed', 'precip',\n",
        "    'temp_humidity_interaction', 'temp_sat_gap', 'humidity_sat_gap',\n",
        "    'precip_rolling_4', 'windspeed_rolling_4', 'pm25_temp_ratio',\n",
        "    'pm25_humidity_ratio', 'tb_pct_change_1', 'tb_diff_1'\n",
        "]\n",
        "corr_matrix = data[correlation_vars].corr()\n",
        "\n",
        "plt.figure(figsize=(12, 8))\n",
        "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')\n",
        "plt.title('Correlation Matrix with Engineered Features')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
    ]

    # Scatter
    cells[10]["source"] = [
        "# PM2.5 vs TB scatter plot with regression line\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.scatter(data['pm2_5'], data['TB'], alpha=0.5, color='blue')\n",
        "\n",
        "z = np.polyfit(data['pm2_5'], data['TB'], 1)\n",
        "p = np.poly1d(z)\n",
        "plt.plot(data['pm2_5'], p(data['pm2_5']), 'r--', alpha=0.8, label=f'y={z[0]:.2f}x+{z[1]:.2f}')\n",
        "\n",
        "corr, p_value = stats.pearsonr(data['pm2_5'], data['TB'])\n",
        "plt.title(f'PM2.5 vs TB Cases (r={corr:.3f}, p={p_value:.4f})')\n",
        "plt.xlabel('PM2.5 (μg/m³)')\n",
        "plt.ylabel('TB Cases')\n",
        "plt.legend()\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.show()\n",
    ]

    # ITS preprocessing
    cells[12]["source"] = [
        "# --- 1. Data Preprocessing ---\n",
        "\n",
        "data_itsa = processed_data.copy()\n",
        "data_itsa['date'] = pd.to_datetime(data_itsa['date'], errors='coerce')\n",
        "data_itsa = data_itsa.sort_values('date').set_index('date')\n",
    ]
    cells[13]["source"] = [
        "# --- 2. Handle Missing Data ---\n",
        "\n",
        "print('--- Missing Values Before Handling ---')\n",
        "print(data_itsa.isnull().sum())\n",
        "\n",
        "data_itsa.fillna(value=0, inplace=True)\n",
        "\n",
        "print('\\n--- Missing Values After Handling ---')\n",
        "print(data_itsa.isnull().sum())\n",
    ]
    cells[14]["source"] = [
        "# --- 3. Define the Intervention Point ---\n",
        "\n",
        "intervention_point = len(data_itsa) // 2\n",
        "intervention_date = data_itsa.index[intervention_point]\n",
        "\n",
        "print(f'\\nThe dataset has {len(data_itsa)} data points.')\n",
        "print(f'The intervention is set at index {intervention_point}, which corresponds to the date: {intervention_date.date()}')\n",
    ]
    cells[15]["source"] = [
        "# --- 4. Create Interrupted Time Series (ITS) Variables ---\n",
        "\n",
        "data_itsa['time'] = np.arange(1, len(data_itsa) + 1)\n",
        "data_itsa['intervention'] = (data_itsa.index >= intervention_date).astype(int)\n",
        "data_itsa['time_since_intervention'] = 0\n",
        "post_intervention_indices = data_itsa[data_itsa['intervention'] == 1].index\n",
        "time_since_intervention_values = np.arange(len(post_intervention_indices))\n",
        "data_itsa.loc[post_intervention_indices, 'time_since_intervention'] = time_since_intervention_values\n",
    ]

    # LSTM model
    cells[16]["source"] = [
        "# --- 5. LSTM Modeling for predicted_tb ---\n",
        "\n",
        "feature_cols = [\n",
        "    'TB', 'pm2_5', 'avgtemp', 'avghumidity', 'windspeed', 'precip',\n",
        "    'temp_humidity_interaction', 'temp_sat_gap', 'humidity_sat_gap',\n",
        "    'precip_rolling_4', 'windspeed_rolling_4', 'pm25_temp_ratio',\n",
        "    'pm25_humidity_ratio', 'tb_pct_change_1', 'tb_diff_1',\n",
        "    'pm25_lag_1', 'pm25_lag_2', 'tb_lag_1', 'tb_lag_2', 'tb_lag_3', 'tb_lag_4',\n",
        "    'pm25_ma_7', 'tb_ma_7', 'pm25_ma_14', 'tb_ma_14', 'pm25_ma_30', 'tb_ma_30',\n",
        "    'sin_week', 'cos_week', 'sin_month', 'cos_month', 'time', 'intervention', 'time_since_intervention'\n",
        "]\n",
        "\n",
        "sequence_length = 8\n",
        "lstm_df = data_itsa[feature_cols].copy().fillna(0)\n",
        "\n",
        "feature_scaler = MinMaxScaler()\n",
        "target_scaler = MinMaxScaler()\n",
        "scaled_features = feature_scaler.fit_transform(lstm_df)\n",
        "scaled_target = target_scaler.fit_transform(data_itsa[['TB']])\n",
        "\n",
        "def build_sequences(features, target, seq_len):\n",
        "    X_seq, y_seq = [], []\n",
        "    for i in range(seq_len, len(features)):\n",
        "        X_seq.append(features[i-seq_len:i])\n",
        "        y_seq.append(target[i])\n",
        "    return np.array(X_seq), np.array(y_seq)\n",
        "\n",
        "X_all, y_all = build_sequences(scaled_features, scaled_target, sequence_length)\n",
        "train_end = max(sequence_length + 1, intervention_point)\n",
        "X_train = X_all[:train_end-sequence_length]\n",
        "y_train = y_all[:train_end-sequence_length]\n",
        "\n",
        "lstm_model = Sequential([\n",
        "    LSTM(64, return_sequences=True, input_shape=(sequence_length, X_all.shape[2])),\n",
        "    Dropout(0.2),\n",
        "    LSTM(32),\n",
        "    Dropout(0.2),\n",
        "    Dense(16, activation='relu'),\n",
        "    Dense(1)\n",
        "])\n",
        "lstm_model.compile(optimizer='adam', loss='mse')\n",
        "early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)\n",
        "history = lstm_model.fit(\n",
        "    X_train,\n",
        "    y_train,\n",
        "    epochs=150,\n",
        "    batch_size=16,\n",
        "    validation_split=0.2,\n",
        "    verbose=0,\n",
        "    callbacks=[early_stopping]\n",
        ")\n",
        "\n",
        "pred_scaled = lstm_model.predict(X_all, verbose=0)\n",
        "pred_tb = target_scaler.inverse_transform(pred_scaled).flatten()\n",
        "data_itsa['predicted_tb'] = data_itsa['TB'].astype(float)\n",
        "data_itsa.iloc[sequence_length:, data_itsa.columns.get_loc('predicted_tb')] = pred_tb\n",
        "\n",
        "counterfactual_features = lstm_df.copy()\n",
        "counterfactual_features.loc[counterfactual_features.index >= intervention_date, 'intervention'] = 0\n",
        "counterfactual_features.loc[counterfactual_features.index >= intervention_date, 'time_since_intervention'] = 0\n",
        "scaled_counterfactual = feature_scaler.transform(counterfactual_features.fillna(0))\n",
        "X_counterfactual, _ = build_sequences(scaled_counterfactual, scaled_target, sequence_length)\n",
        "counter_scaled = lstm_model.predict(X_counterfactual, verbose=0)\n",
        "counter_tb = target_scaler.inverse_transform(counter_scaled).flatten()\n",
        "data_itsa['counterfactual_tb'] = data_itsa['TB'].astype(float)\n",
        "data_itsa.iloc[sequence_length:, data_itsa.columns.get_loc('counterfactual_tb')] = counter_tb\n",
        "\n",
        "actual_for_metrics = data_itsa['TB'].iloc[sequence_length:]\n",
        "pred_for_metrics = data_itsa['predicted_tb'].iloc[sequence_length:]\n",
        "rmse = np.sqrt(mean_squared_error(actual_for_metrics, pred_for_metrics))\n",
        "mae = mean_absolute_error(actual_for_metrics, pred_for_metrics)\n",
        "\n",
        "print('\\n--- LSTM Model Summary ---')\n",
        "print(f'Features used: {len(feature_cols)}')\n",
        "print(f'Sequence length: {sequence_length} weeks')\n",
        "print(f'Training windows: {len(X_train)}')\n",
        "print(f\"Best validation loss: {min(history.history['val_loss']):.6f}\")\n",
        "print(f'RMSE: {rmse:.2f}')\n",
        "print(f'MAE: {mae:.2f}')\n",
    ]

    # Clear blank cell 17.
    cells[17] = code_cell(
        """
data_itsa.index = pd.to_datetime(data_itsa.index, errors='coerce')
"""
    )

    # Advanced visualization
    cells[18]["source"] = [
        "# --- 6. Advanced Visualization ---\n",
        "\n",
        "plt.style.use('seaborn-v0_8-whitegrid')\n",
        "fig, ax = plt.subplots(figsize=(16, 8))\n",
        "\n",
        "ax.scatter(data_itsa.index, data_itsa['TB'], color='black', alpha=0.55, label='Observed TB Cases', s=18)\n",
        "\n",
        "pre_intervention_data = data_itsa[data_itsa['intervention'] == 0]\n",
        "post_intervention_data = data_itsa[data_itsa['intervention'] == 1]\n",
        "ax.plot(pre_intervention_data.index, pre_intervention_data['predicted_tb'], color='blue', linewidth=2, label='Pre-intervention Trend')\n",
        "ax.plot(post_intervention_data.index, post_intervention_data['predicted_tb'], color='red', linewidth=2, label='Post-intervention Trend')\n",
        "ax.plot(post_intervention_data.index, post_intervention_data['counterfactual_tb'], color='green', linestyle='--', linewidth=2, label='Counterfactual (No Intervention)')\n",
        "ax.axvline(x=intervention_date, color='purple', linestyle=':', linewidth=2.5, label=f'Intervention: {intervention_date.date()}')\n",
        "\n",
        "ax.set_title('Interrupted Time Series Analysis of TB Cases', fontsize=18, fontweight='bold')\n",
        "ax.set_xlabel('Date', fontsize=12)\n",
        "ax.set_ylabel('Number of TB Cases', fontsize=12)\n",
        "ax.legend(loc='upper left', fontsize=10)\n",
        "plt.xticks(rotation=45)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
    ]

    # Monthly patterns
    cells[20]["source"] = [
        "# Monthly patterns\n",
        "data['month'] = data['date'].dt.month\n",
        "monthly_stats = data.groupby('month').agg({\n",
        "    'pm2_5': ['mean', 'std'],\n",
        "    'TB': ['mean', 'sum'],\n",
        "    'avgtemp': 'mean',\n",
        "    'avghumidity': 'mean'\n",
        "}).round(2)\n",
        "\n",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
        "axes[0, 0].bar(range(1, 13), monthly_stats['pm2_5']['mean'].values, color='blue', alpha=0.7)\n",
        "axes[0, 0].errorbar(range(1, 13), monthly_stats['pm2_5']['mean'].values, yerr=monthly_stats['pm2_5']['std'].values, fmt='none', color='black', alpha=0.5)\n",
        "axes[0, 0].set_title('Average PM2.5 by Month')\n",
        "axes[0, 0].set_xlabel('Month')\n",
        "axes[0, 0].set_ylabel('PM2.5 (μg/m³)')\n",
        "axes[0, 0].set_xticks(range(1, 13))\n",
        "\n",
        "axes[0, 1].bar(range(1, 13), monthly_stats['TB']['sum'].values, color='green', alpha=0.7)\n",
        "axes[0, 1].set_title('Total TB Cases by Month')\n",
        "axes[0, 1].set_xlabel('Month')\n",
        "axes[0, 1].set_ylabel('TB Cases')\n",
        "axes[0, 1].set_xticks(range(1, 13))\n",
        "\n",
        "axes[1, 0].plot(range(1, 13), monthly_stats['avgtemp']['mean'].values, 'o-', color='orange')\n",
        "axes[1, 0].set_title('Average Temperature by Month')\n",
        "axes[1, 0].set_xlabel('Month')\n",
        "axes[1, 0].set_ylabel('Temperature (°C)')\n",
        "axes[1, 0].set_xticks(range(1, 13))\n",
        "axes[1, 0].grid(True, alpha=0.3)\n",
        "\n",
        "axes[1, 1].plot(range(1, 13), monthly_stats['avghumidity']['mean'].values, 'o-', color='cyan')\n",
        "axes[1, 1].set_title('Average Humidity by Month')\n",
        "axes[1, 1].set_xlabel('Month')\n",
        "axes[1, 1].set_ylabel('Humidity (%)')\n",
        "axes[1, 1].set_xticks(range(1, 13))\n",
        "axes[1, 1].grid(True, alpha=0.3)\n",
        "\n",
        "plt.suptitle('Seasonal Patterns in Kampala', fontsize=14, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
    ]

    # Replace tail with coherent later sections.
    tail_cells = [
        markdown_cell("## Step 5: Time Series Modeling"),
        code_cell(
            """
# Prepare data for additional time-series model comparison
data_predict = processed_data.copy()
data_predict['date'] = pd.to_datetime(data_predict['date'], errors='coerce')
data_predict = data_predict.sort_values('date')

model_feature_cols = [
    'pm2_5', 'avgtemp', 'avghumidity', 'windspeed', 'precip',
    'temp_humidity_interaction', 'temp_sat_gap', 'humidity_sat_gap',
    'precip_rolling_4', 'windspeed_rolling_4', 'pm25_temp_ratio',
    'pm25_humidity_ratio', 'tb_pct_change_1', 'tb_diff_1',
    'pm25_lag_1', 'pm25_lag_2', 'tb_lag_1', 'tb_lag_2', 'tb_lag_3', 'tb_lag_4',
    'pm25_ma_7', 'tb_ma_7', 'pm25_ma_14', 'tb_ma_14', 'pm25_ma_30', 'tb_ma_30',
    'sin_week', 'cos_week', 'sin_month', 'cos_month'
]

available_model_features = [col for col in model_feature_cols if col in data_predict.columns]
print(f'Available model features: {len(available_model_features)}')
print(available_model_features)
"""
        ),
        code_cell(
            """
ts_modeler = TimeSeriesModeler(data_predict, target_column='TB')
X_ts, y_ts = ts_modeler.prepare_time_series_data(
    window_size=8,
    features=available_model_features,
    target_type='binary'
)

print('Prepared time-series benchmark arrays:')
print('X shape:', X_ts.shape)
print('y shape:', y_ts.shape)
"""
        ),
        code_cell(
            """
benchmark_results = ts_modeler.train_models(
    X_ts,
    y_ts,
    models_to_train=['random_forest', 'gradient_boosting', 'svm'],
    cv_splits=3
)

if benchmark_results:
    models_performance = pd.DataFrame([
        {
            'Model': model_name,
            'Accuracy': metrics.get('avg_accuracy', np.nan),
            'F1 Score': metrics.get('avg_f1', np.nan),
            'Precision': metrics.get('avg_precision', np.nan),
            'Recall': metrics.get('avg_recall', np.nan)
        }
        for model_name, metrics in benchmark_results.items()
    ]).sort_values('F1 Score', ascending=False)
else:
    models_performance = pd.DataFrame(columns=['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])

models_performance
"""
        ),
        code_cell(
            """
print('--- Model Comparison Summary ---')
if not models_performance.empty:
    print(models_performance.round(3).to_string(index=False))
else:
    print('No benchmark models were trained.')

print(f'\\nLSTM Regression RMSE: {rmse:.2f}')
print(f'LSTM Regression MAE: {mae:.2f}')
"""
        ),
        markdown_cell("## Step 4: Intervention Simulation"),
        code_cell(
            """
# Get baseline metrics
baseline = processor.get_intervention_baseline()
print('Baseline Metrics:')
print(f"- Average PM2.5: {baseline['baseline_pm25']:.2f}")
print(f"- Average TB cases: {baseline['baseline_tb']:.2f}")
print(f"- Total TB cases: {baseline['total_tb_cases']:.2f}")
print(f"- PM2.5-TB correlation: {baseline['correlation']:.3f}")
"""
        ),
        code_cell(
            """
# Simulate intervention scenarios
scenarios = {
    'Traffic Control': {'reduction': 0.25, 'cost_factor': 2.5},
    'Green Spaces': {'reduction': 0.15, 'cost_factor': 3.0},
    'Dust Control': {'reduction': 0.20, 'cost_factor': 4.0},
    'Combined': {'reduction': 0.35, 'cost_factor': 5.0}
}

cr_tb = 1.08
results = []
for scenario_name, params in scenarios.items():
    reduced_pm25 = baseline['baseline_pm25'] * (1 - params['reduction'])
    pm25_change = reduced_pm25 - baseline['baseline_pm25']
    relative_risk = cr_tb ** (pm25_change / 10)
    tb_cases_baseline = baseline['total_tb_cases']
    tb_cases_with_intervention = tb_cases_baseline * relative_risk
    tb_cases_prevented = tb_cases_baseline - tb_cases_with_intervention
    cost = params['cost_factor'] * 1_000_000
    cost_per_case = cost / tb_cases_prevented if tb_cases_prevented > 0 else np.inf
    results.append({
        'Scenario': scenario_name,
        'PM2.5 Reduction (%)': params['reduction'] * 100,
        'Final PM2.5 (μg/m³)': reduced_pm25,
        'TB Cases Prevented': tb_cases_prevented,
        'Cost Factor': params['cost_factor'],
        'Cost per Case Prevented ($)': cost_per_case
    })

results_df = pd.DataFrame(results)
results_df
"""
        ),
        code_cell(
            """
# Visualize intervention impacts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(
    results_df['PM2.5 Reduction (%)'],
    results_df['TB Cases Prevented'],
    s=results_df['Cost Factor'] * 60,
    alpha=0.7,
    c=range(len(results_df)),
    cmap='viridis'
)
for i, txt in enumerate(results_df['Scenario']):
    axes[0].annotate(txt, (results_df['PM2.5 Reduction (%)'].iloc[i], results_df['TB Cases Prevented'].iloc[i]))
axes[0].set_xlabel('PM2.5 Reduction (%)')
axes[0].set_ylabel('TB Cases Prevented')
axes[0].set_title('Intervention Effectiveness')
axes[0].grid(True, alpha=0.3)

axes[1].bar(
    results_df['Scenario'],
    results_df['Cost per Case Prevented ($)'],
    color=['blue', 'green', 'orange', 'red'],
    alpha=0.7
)
axes[1].set_xlabel('Scenario')
axes[1].set_ylabel('Cost per Case Prevented ($)')
axes[1].set_title('Cost-Effectiveness Analysis')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
"""
        ),
        markdown_cell("## Step 6: Generate Report"),
        code_cell(
            """
# Summary Report
print('=' * 60)
print('KAMPALA AIR QUALITY AND TB ANALYSIS - SUMMARY REPORT')
print('=' * 60)
print()
print('1. DATA OVERVIEW')
print('-' * 40)
print(f"   - Analysis Period: {summary['date_range']}")
print(f"   - Total Records: {summary['total_records']}")
print(f"   - Average PM2.5: {summary['pm25_statistics']['mean']:.2f} μg/m³")
print(f"   - Total TB Cases: {summary['tb_statistics']['total_cases']:.2f}")
print()
print('2. KEY FINDINGS')
print('-' * 40)
print(f"   - PM2.5-TB Correlation: {summary['correlation_pm25_tb']:.3f}")
print('   - Seasonal variation observed in both PM2.5 and TB cases')
print(f"   - LSTM RMSE: {rmse:.2f}")
print(f"   - LSTM MAE: {mae:.2f}")
print()
print('3. INTERVENTION RECOMMENDATIONS')
print('-' * 40)
best_scenario = results_df.loc[results_df['TB Cases Prevented'].idxmax()]
print(f"   - Most Effective: {best_scenario['Scenario']}")
print(f"   - PM2.5 Reduction: {best_scenario['PM2.5 Reduction (%)']:.1f}%")
print(f"   - TB Cases Prevented: {best_scenario['TB Cases Prevented']:.0f}")
print()
print('4. BENCHMARK MODEL PERFORMANCE')
print('-' * 40)
if not models_performance.empty:
    best_model = models_performance.iloc[0]
    print(f"   - Best Benchmark Model: {best_model['Model']}")
    print(f"   - Accuracy: {best_model['Accuracy']:.3f}")
    print(f"   - F1 Score: {best_model['F1 Score']:.3f}")
else:
    print('   - No benchmark classification models were trained.')
print()
print('=' * 60)

results_dir = Path('./results')
results_dir.mkdir(exist_ok=True)
results_df.to_csv(results_dir / 'intervention_scenarios.csv', index=False)
models_performance.to_csv(results_dir / 'model_performance.csv', index=False)
print('✓ Results saved to ./results/')
"""
        ),
    ]

    cells = cells[:21] + tail_cells
    nb["cells"] = cells

    # Sanity-check code syntax.
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            ast.parse(source, filename=f"reference_cell_{idx}.py")

    REFERENCE.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {REFERENCE.name} with {len(cells)} cells.")


if __name__ == "__main__":
    main()
