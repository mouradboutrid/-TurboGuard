import os
import tempfile
import json
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
from forecaster_anomaly_predictor_app import AnomalyPredictorApp   
from autoencoder_anomaly_detector_app import AutoencoderAnomalyDetector
from analyzer_app import AutoencoderAnomalyAnalyzer

MODEL_PATHS = {
    'forecasting': {
        'model': "trained_models/forecaster_model/saved_models/lstm_model_20250529_005930.h5",
        'config': "trained_models/forecaster_model/config/analysis_config_20250529_005930.json"
    },
    'autoencoder': {
        'FD001': {
            'autoencoder': "trained_models/autoencoder_models/FD001/autoencoder.keras",
            'encoder': "trained_models/autoencoder_models/FD001/encoder.keras",
            'config': "trained_models/autoencoder_models/FD001/config.json"
        },
        'FD002': {
            'autoencoder': "trained_models/autoencoder_models/FD002/autoencoder.keras",
            'encoder': "trained_models/autoencoder_models/FD002/encoder.keras",
            'config': "trained_models/autoencoder_models/FD002/config.json"
        },
        'FD003': {
            'autoencoder': "trained_models/autoencoder_models/FD003/autoencoder.keras",
            'encoder': "trained_models/autoencoder_models/FD003/encoder.keras",
            'config': "trained_models/autoencoder_models/FD003/config.json"
        },
        'FD004': {
            'autoencoder': "trained_models/autoencoder_models/FD004/autoencoder.keras",
            'encoder': "trained_models/autoencoder_models/FD004/encoder.keras",
            'config': "trained_models/autoencoder_models/FD004/config.json"
        }
    }
}

def create_enhanced_sensor_plot(data, selected_units, selected_sensors, anomaly_data=None, plot_type="line"):
    """Create interactive sensor plots with better anomaly visualization"""
    if data.empty or not selected_units or not selected_sensors:
        return None

    filtered_data = data[data['unit_id'].isin(selected_units)]

    if plot_type == "line":
        fig = make_subplots(
            rows=len(selected_sensors),
            cols=1,
            subplot_titles=[f"{sensor}" for sensor in selected_sensors],
            vertical_spacing=0.08
        )

        colors = px.colors.qualitative.Set1

        for j, sensor in enumerate(selected_sensors):
            for i, unit_id in enumerate(selected_units):
                unit_data = filtered_data[filtered_data['unit_id'] == unit_id]
                if unit_data.empty or sensor not in unit_data.columns:
                    continue

                unit_data = unit_data.sort_values('time_cycle')

                # Normal data line
                fig.add_trace(go.Scatter(
                    x=unit_data['time_cycle'],y=unit_data[sensor],
                    mode='lines',
                    name=f'Unit {unit_id}',
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=(j == 0),  # Only show legend for first subplot
                    legendgroup=f'unit_{unit_id}'
                ), row=j+1, col=1)

                # Add anomaly points if available
                if anomaly_data is not None and 'unit_ids' in anomaly_data and 'sequence_indices' in anomaly_data:
                    unit_anomaly_mask = anomaly_data['unit_ids'] == unit_id
                    if np.any(unit_anomaly_mask):
                        anomaly_indices = anomaly_data['sequence_indices'][unit_anomaly_mask]
                        anomaly_cycles = []
                        anomaly_values = []

                        for idx in anomaly_indices:
                            if idx < len(unit_data):
                                cycle_val = unit_data.iloc[idx]['time_cycle'] if idx < len(unit_data) else None
                                sensor_val = unit_data.iloc[idx][sensor] if idx < len(unit_data) and sensor in unit_data.columns else None
                                if cycle_val is not None and sensor_val is not None:
                                    anomaly_cycles.append(cycle_val)
                                    anomaly_values.append(sensor_val)

                        if anomaly_cycles:
                            fig.add_trace(go.Scatter(
                                x=anomaly_cycles,
                                y=anomaly_values,
                                mode='markers',
                                name=f'Anomalies Unit {unit_id}',
                                marker=dict(
                                    color='red',
                                    size=8,
                                    symbol='x',
                                    line=dict(width=2, color='darkred')
                                ),
                                showlegend=(j == 0),
                                legendgroup=f'anomaly_{unit_id}'
                            ), row=j+1, col=1)

        fig.update_layout(
            height=300 * len(selected_sensors),
            title_text="Sensor Data Analysis with Anomaly Detection",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        for j in range(len(selected_sensors)):
            fig.update_xaxes(title_text="Time Cycle", row=j+1, col=1)
            fig.update_yaxes(title_text="Sensor Value", row=j+1, col=1)

    else:  # Heatmap
        # Create correlation heatmap
        sensor_data = filtered_data[selected_sensors]
        correlation_matrix = sensor_data.corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title="Sensor Correlation Heatmap",
            xaxis_title="Sensors",
            yaxis_title="Sensors",
            height=600
        )

    return fig


def create_anomaly_summary_plots(unit_analysis, model_type="Model"):
    """Create summary plots for anomaly analysis"""
    if not unit_analysis:
        return None, None

    # Prepare data
    units = list(unit_analysis.keys())
    anomaly_rates = [unit_analysis[unit]['anomaly_rate'] for unit in units]
    risk_levels = [unit_analysis[unit]['risk_level'] for unit in units]
    total_sequences = [unit_analysis[unit]['total_sequences'] for unit in units]
    anomaly_counts = [unit_analysis[unit]['anomalies'] for unit in units]

    # Color mapping for risk levels
    risk_colors = {
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red',
        'Critical': 'darkred'
    }

    # Plot 1: Anomaly Rate by Unit
    fig1 = go.Figure()

    for risk in ['Low', 'Medium', 'High', 'Critical']:
        mask = [r == risk for r in risk_levels]
        if any(mask):
            fig1.add_trace(go.Bar(
                x=[units[i] for i in range(len(units)) if mask[i]],
                y=[anomaly_rates[i] for i in range(len(anomaly_rates)) if mask[i]],
                name=f'{risk} Risk',
                marker_color=risk_colors[risk],
                text=[f'{anomaly_rates[i]:.1f}%' for i in range(len(anomaly_rates)) if mask[i]],
                textposition='auto'
            ))

    fig1.update_layout(
        title=f'{model_type} - Anomaly Rate by Unit',
        xaxis_title='Unit ID',
        yaxis_title='Anomaly Rate (%)',
        barmode='group',
        height=500,
        showlegend=True
    )

    # Plot 2: Risk Level Distribution
    risk_counts = {}
    for risk in risk_levels:
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    fig2 = go.Figure(data=[
        go.Pie(
            labels=list(risk_counts.keys()),
            values=list(risk_counts.values()),
            marker_colors=[risk_colors[risk] for risk in risk_counts.keys()],
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{value} units<br>(%{percent})',
            hovertemplate='<b>%{label}</b><br>Units: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
    ])

    fig2.update_layout(
        title=f'{model_type} - Risk Level Distribution',
        height=400,
        showlegend=True
    )

    return fig1, fig2


def display_detailed_results(results, model_name):
    """Display detailed analysis results"""
    st.subheader(f"📊 {model_name} - Detailed Analysis Results")

    if not results:
        st.error("No results to display")
        return

    # Basic Statistics
    col1, col2, col3, col4 = st.columns(4)

    total_sequences = len(results.get('unit_ids', []))
    total_anomalies = np.sum(results['anomaly_results']['ensemble']['anomalies']) if 'ensemble' in results['anomaly_results'] else 0
    anomaly_rate = (total_anomalies / total_sequences * 100) if total_sequences > 0 else 0
    total_units = len(results['unit_analysis'])

    with col1:
        st.metric("Total Sequences", total_sequences)
    with col2:
        st.metric("Total Anomalies", total_anomalies)
    with col3:
        st.metric("Overall Anomaly Rate", f"{anomaly_rate:.2f}%")
    with col4:
        st.metric("Units Analyzed", total_units)

    # Unit Analysis Table
    if results['unit_analysis']:
        st.subheader("Unit-wise Analysis")

        unit_df = pd.DataFrame.from_dict(results['unit_analysis'], orient='index')
        unit_df.index.name = 'Unit ID'
        unit_df = unit_df.reset_index()

        # Style the dataframe
        def color_risk_level(val):
            colors = {
                'Low': 'background-color: #d4edda; color: #155724',
                'Medium': 'background-color: #fff3cd; color: #856404',
                'High': 'background-color: #f8d7da; color: #721c24',
                'Critical': 'background-color: #721c24; color: white'
            }
            return colors.get(val, '')

        styled_df = unit_df.style.applymap(color_risk_level, subset=['risk_level'])
        st.dataframe(styled_df, use_container_width=True)

    # Method-specific results
    if 'anomaly_results' in results:
        st.subheader("🔍 Detection Method Results")

        method_cols = st.columns(len(results['anomaly_results']))

        for i, (method, method_results) in enumerate(results['anomaly_results'].items()):
            with method_cols[i]:
                st.write(f"**{method.title()}**")
                if 'anomalies' in method_results:
                    method_anomalies = np.sum(method_results['anomalies'])
                    method_rate = (method_anomalies / total_sequences * 100) if total_sequences > 0 else 0
                    st.metric(f"{method.title()} Anomalies", method_anomalies)
                    st.write(f"Rate: {method_rate:.2f}%")

                    if 'threshold' in method_results and method_results['threshold'] is not None:
                        st.write(f"Threshold: {method_results['threshold']:.4f}")




# Main Streamlit App
def convert_keys_to_builtin_types(obj):
    if isinstance(obj, dict):
        return {
            (int(k) if isinstance(k, np.integer) else str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k):
            convert_keys_to_builtin_types(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_keys_to_builtin_types(item) for item in obj]
    else:
        return obj

def main():
    st.set_page_config(
        page_title="TurboGuard",
        page_icon="✈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🛫 TurboGuard: Intelligent Predictive Maintenance Solution for Turbofan Engines")
    st.markdown("---")

    # Sidebar for model selection
    st.sidebar.header("🎛Configuration")

    # Model Selection
    model_choice = st.sidebar.selectbox(
        "Select Model Type",
        ["LSTM Forecasting Model", "Autoencoder Ensemble Model"],
        help="Choose between pre-trained LSTM forecasting or the autoencoder model"
    )

    # Dataset selection for autoencoder
    if model_choice == "Autoencoder Ensemble Model":
        dataset_choice = st.sidebar.selectbox(
            "Select Dataset",
            ["FD001", "FD002", "FD003", "FD004"],
            help="Choose which operation mode to use for autoencoder analysis"
        )
    else:
        dataset_choice = None

    # File Upload
    st.sidebar.subheader("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Test Data (CSV/TXT)",
        type=['csv', 'txt'],
        help="Upload data file"
    )

    # Analysis Parameters
    st.sidebar.subheader("⚙️ Analysis Parameters")
    threshold_percentile = st.sidebar.slider(
        "Anomaly Threshold Percentile",
        min_value=80,
        max_value=99,
        value=95,
        help="Percentile threshold for anomaly detection"
    )

    if model_choice == "Autoencoder Ensemble Model":
        detection_methods = st.sidebar.multiselect(
            "Detection Methods",
            ["autoencoder"], #we will add the statistical methods in the next version
            default=["autoencoder"],
            help="Select which detection methods to use in ensemble"
        )
    else:
        detection_methods = None

    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None

    # Main Analysis Section
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_filepath = tmp_file.name

        col1, col2 = st.columns([2, 1])

        with col1:
            st.info(f"📁 File uploaded: {uploaded_file.name}")
            st.info(f"Selected Model: {model_choice}")
            if dataset_choice:
                st.info(f"📊 Dataset: {dataset_choice}")

        with col2:
            analyze_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

        if analyze_button:
            with st.spinner(f"Analyzing with {model_choice}..."):
                try:
                    if model_choice == "LSTM Forecasting Model":
                        # Initialize and run LSTM model
                        predictor = AnomalyPredictorApp()

                        if predictor.load_model_and_config():
                            results = predictor.predict_and_analyze(tmp_filepath, threshold_percentile)
                            if results:
                                st.session_state.analysis_results = results
                                st.session_state.current_model = model_choice
                                st.success("✅ LSTM Analysis completed successfully!")
                            else:
                                st.error("❌ LSTM Analysis failed")

                    else:  # Autoencoder Ensemble Model
                        # Initialize and run autoencoder model
                        analyzer = AutoencoderAnomalyAnalyzer(dataset_choice)

                        if analyzer.load_models():
                            results = analyzer.predict_and_analyze(
                                tmp_filepath,
                                methods=detection_methods,
                                threshold_percentile=threshold_percentile
                            )
                            if results:
                                st.session_state.analysis_results = results
                                st.session_state.current_model = model_choice
                                st.success("✅ Autoencoder Analysis completed successfully!")
                            else:
                                st.error("❌ Autoencoder Analysis failed")

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.error("Please check your model files and data format")

        # Clean up temporary file
        try:
            os.unlink(tmp_filepath)
        except:
            pass

    # Display Results
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        model_name = st.session_state.current_model

        st.markdown("---")
        st.header("Analysis Results")

        # Display detailed results
        display_detailed_results(results, model_name)

        # Create and display summary plots
        if results['unit_analysis']:
            st.markdown("---")
            st.subheader("📊 Summary Visualizations")

            fig1, fig2 = create_anomaly_summary_plots(results['unit_analysis'], model_name)

            if fig1 and fig2:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)

        # Sensor Analysis Section
        if 'test_data' in results and not results['test_data'].empty:
            st.markdown("---")
            st.subheader("🔍 Sensor Data Analysis")

            # Sensor plot controls
            sensor_cols = [col for col in results['test_data'].columns if col.startswith('sensor_')]
            available_units = sorted(results['test_data']['unit_id'].unique())

            col1, col2, col3 = st.columns(3)

            with col1:
                selected_units = st.multiselect(
                    "Select Units to Analyze",
                    available_units,
                    default=available_units[:5] if len(available_units) > 5 else available_units,
                    help="Choose which units to display in the sensor plots"
                )

            with col2:
                selected_sensors = st.multiselect(
                    "Select Sensors",
                    sensor_cols,
                    default=sensor_cols[:3] if len(sensor_cols) > 3 else sensor_cols,
                    help="Choose which sensors to analyze"
                )

            with col3:
                plot_type = st.selectbox(
                    "Plot Type",
                    ["line", "heatmap"],
                    help="Choose visualization type"
                )

            # Create and display sensor plots
            if selected_units and selected_sensors:
                sensor_fig = create_enhanced_sensor_plot(
                    results['test_data'],
                    selected_units,
                    selected_sensors,
                    results,
                    plot_type
                )

                if sensor_fig:
                    st.plotly_chart(sensor_fig, use_container_width=True)

        # Download Results
        st.markdown("---")
        st.subheader("Export Results")

        if st.button("📥 Download Analysis Report", use_container_width=True):
            # Create summary report
            report_data = {
                'model_type': model_name,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'total_sequences': len(results.get('unit_ids', [])),
                'total_anomalies': int(np.sum(results['anomaly_results']['ensemble']['anomalies'])) if 'ensemble' in results['anomaly_results'] else 0,
                'unit_analysis': results['unit_analysis']
            }
            report_data = convert_keys_to_builtin_types(report_data)
            report_json = json.dumps(report_data, indent=2, default=str)

            st.download_button(
                label="📄 Download JSON Report",
                data=report_json,
                file_name=f"anomaly_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    else:
        # Instructions when no analysis has been run
        st.info("Please upload a test data file and click 'Start Analysis' to begin.")

        with st.expander("ℹ️ Instructions & Model Information"):
            st.markdown("""
            ### How to Use This Dashboard

            1. **Select Model Type**: Choose between LSTM Forecasting or Autoencoder Ensemble models
            2. **Upload Data**: Upload your CMAPSS test data file (CSV or TXT format)
            3. **Configure Parameters**: Adjust threshold percentile and detection methods
            4. **Run Analysis**: Click 'Start Analysis' to detect anomalies
            5. **Review Results**: Examine the detailed results and visualizations

            ### Model Information

            **LSTM Forecasting Model:**
            - Uses pre-trained LSTM networks for time series forecasting
            - Detects anomalies based on prediction errors
            - Suitable for temporal pattern analysis

            **Autoencoder Ensemble Model:**
            - Combines multiple detection methods (autoencoder, statistical, wavelet)
            - Uses reconstruction error and statistical analysis
            - Robust ensemble approach for anomaly detection

            ### Data Format
            - CMAPSS dataset format with sensor readings
            - Columns: unit_id, time_cycle, operational_settings, sensor_1 to sensor_21
            - Space or comma separated values
            """)

main()
