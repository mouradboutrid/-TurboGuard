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
from app.forecaster_anomaly_predictor_app import AnomalyPredictorApp   
from app.autoencoder_anomaly_detector_app import AutoencoderAnomalyPredictor

MODEL_PATHS = {
    'forecasting': {
        'model': "saved_models/lstm_model_20250529_005930.h5",
        'config': "saved_models/analysis_config_20250529_005930.json"
    },
    'autoencoder': {
        'FD001': {
            'autoencoder': "models/FD001/autoencoder.keras",
            'encoder': "models/FD001/encoder.keras",
            'config': "models/FD001/config.json"
        },
        'FD002': {
            'autoencoder': "models/FD002/autoencoder.keras",
            'encoder': "models/FD002/encoder.keras",
            'config': "models/FD002/config.json"
        },
        'FD003': {
            'autoencoder': "models/FD003/autoencoder.keras",
            'encoder': "models/FD003/encoder.keras",
            'config': "models/FD003/config.json"
        },
        'FD004': {
            'autoencoder': "models/FD004/autoencoder.keras",
            'encoder': "models/FD004/encoder.keras",
            'config': "models/FD004/config.json"
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
                    x=unit_data['time_cycle'],
                    y=unit_data[sensor],
                    name=f"Unit {unit_id}",
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.8,
                    hovertemplate=f"<b>Unit {unit_id} - {sensor}</b><br>" +
                                "Cycle: %{x}<br>" +
                                "Value: %{y:.4f}<extra></extra>",
                    showlegend=(j == 0)), row=j+1, col=1)

                # Add anomaly points if available
                if anomaly_data and unit_id in anomaly_data:
                    anomaly_positions = anomaly_data[unit_id].get('anomaly_positions', [])
                    if anomaly_positions:
                        anomaly_cycles = [pos for pos in anomaly_positions if pos in unit_data['time_cycle'].values]
                        if anomaly_cycles:
                            anomaly_values = unit_data[unit_data['time_cycle'].isin(anomaly_cycles)][sensor]
                            fig.add_trace(go.Scatter(
                                x=anomaly_cycles,
                                y=anomaly_values,
                                mode='markers',
                                name=f"Unit {unit_id} Anomalies",
                                marker=dict(
                                    color='red',
                                    size=8,
                                    symbol='x',
                                    line=dict(width=2, color='darkred')
                                ),
                                hovertemplate=f"<b>ANOMALY - Unit {unit_id}</b><br>" +
                                            "Cycle: %{x}<br>" +
                                            "Value: %{y:.4f}<extra></extra>",
                                showlegend=(j == 0)
                            ), row=j+1, col=1)

        fig.update_layout(
            height=300 * len(selected_sensors),
            title="Sensor Data with Anomalies",
            showlegend=True,
            hovermode='closest'
        )

        fig.update_xaxes(title_text="Time Cycle")
        fig.update_yaxes(title_text="Normalized Value")

    elif plot_type == "heatmap":
        # Create correlation heatmap
        correlation_data = []
        for unit_id in selected_units:
            unit_data = filtered_data[filtered_data['unit_id'] == unit_id]
            if not unit_data.empty:
                unit_corr = unit_data[selected_sensors].corr()
                correlation_data.append(unit_corr.values)

        if correlation_data:
            avg_corr = np.mean(correlation_data, axis=0)
            fig = go.Figure(data=go.Heatmap(
                z=avg_corr,
                x=selected_sensors,
                y=selected_sensors,
                colorscale='RdBu',
                zmid=0,
                text=np.round(avg_corr, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
            ))
            fig.update_layout(
                title="Sensor Correlation Heatmap",
                height=600,
                width=800
            )

    return fig

def create_anomaly_summary_chart(unit_analysis):
    """Create summary charts for anomaly analysis"""
    if not unit_analysis:
        return None, None

    # Risk level distribution
    risk_counts = {}
    anomaly_rates = []
    unit_ids = []

    for unit_id, analysis in unit_analysis.items():
        risk_level = analysis['risk_level']
        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        anomaly_rates.append(analysis['anomaly_rate'])
        unit_ids.append(unit_id)

    # Risk distribution pie chart
    fig1 = go.Figure(data=[go.Pie(
        labels=list(risk_counts.keys()),
        values=list(risk_counts.values()),
        hole=.3,
        marker_colors=['green', 'yellow', 'orange', 'red']
    )])
    fig1.update_layout(
        title="Risk Level Distribution",
        height=400
    )

    # Anomaly rate bar chart
    colors = ['red' if rate >= 75 else 'orange' if rate >= 50 else 'yellow' if rate >= 25 else 'green'
              for rate in anomaly_rates]

    fig2 = go.Figure(data=[go.Bar(
        x=[f"Unit {uid}" for uid in unit_ids],
        y=anomaly_rates,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>Anomaly Rate: %{y:.1f}%<extra></extra>"
    )])
    fig2.update_layout(
        title="Anomaly Rate by Unit",
        xaxis_title="Unit ID",
        yaxis_title="Anomaly Rate (%)",
        height=400
    )

    return fig1, fig2

def create_method_comparison_chart(anomaly_results):
    """Create comparison chart for different detection methods"""
    if not anomaly_results:
        return None

    methods_data = []
    for method, result in anomaly_results.items():
        if method != 'ensemble' and 'anomalies' in result:
            anomaly_count = np.sum(result['anomalies'])
            methods_data.append({
                'Method': method.title(),
                'Anomalies Detected': anomaly_count,
                'Detection Rate': (anomaly_count / len(result['anomalies'])) * 100
            })

    if methods_data:
        df_methods = pd.DataFrame(methods_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Anomalies Detected',
            x=df_methods['Method'],
            y=df_methods['Anomalies Detected'],
            yaxis='y',
            offsetgroup=1,
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Detection Rate (%)',
            x=df_methods['Method'],
            y=df_methods['Detection Rate'],
            yaxis='y2',
            offsetgroup=2,
            marker_color='orange'
        ))

        fig.update_layout(
            title='Detection Methods Comparison',
            xaxis_title='Detection Method',
            yaxis=dict(title='Number of Anomalies', side='left'),
            yaxis2=dict(title='Detection Rate (%)', side='right', overlaying='y'),
            legend=dict(x=0.7, y=1),
            height=400
        )
        return fig

    return None

# ===== STREAMLIT APP =====
def main():
    st.set_page_config(
        page_title="TurboGuard",
        page_icon="🛬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("✈️TurboGuard: AI-Powered Health Monitoring & Anomaly Detection")
    st.markdown("---")

    # Sidebar for model selection
    st.sidebar.header("🎛️ Configuration")

    model_choice = st.sidebar.selectbox(
        "Select Analysis Model:",
        ["LSTM Forecasting Model", "Autoencoder Ensemble Model"],
        help="Choose between pre-trained LSTM forecasting or autoencoder ensemble models"
    )

    # Model-specific configurations
    if model_choice == "LSTM Forecasting Model":
        st.sidebar.subheader("LSTM Model Settings")
        threshold_percentile = st.sidebar.slider(
            "Anomaly Threshold Percentile",
            min_value=75, max_value=99, value=99, step=1,
            help="Higher values = fewer anomalies detected"
        )
        predictor = AnomalyPredictorApp()

    else:  # Autoencoder Ensemble Model
        st.sidebar.subheader("Autoencoder Mode Settings")
        dataset_choice = st.sidebar.selectbox(
            "Select Dataset Model:",
            ["FD001", "FD002", "FD003", "FD004"],
            help="Choose which model to use : Depending to whish Operation Mode !!!!!"
        )

        detection_methods = st.sidebar.multiselect(
            "Detection Methods:",
            ["autoencoder"
             #"statistical", "wavelet" : for this virsion of the app we wont use this to aprochs
             ],
            default=["autoencoder"],
            help="Select anomaly detection methods to use in ensemble"
        )

        threshold_percentile = st.sidebar.slider(
            "Anomaly Threshold Percentile",
            min_value=80, max_value=99, value=95, step=1
        )

        analyzer = AutoencoderAnomalyAnalyzer(dataset_choice)

    # File upload
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your engine Dataset",
        type=['txt', 'csv'],
        help="Upload your CMAPSS test dataset file"
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load and analyze based on selected model
            if model_choice == "LSTM Forecasting Model":
                st.header("🤖 LSTM Forecasting Model Analysis")

                with st.spinner("Loading LSTM model..."):
                    if predictor.load_model_and_config():
                        with st.spinner("Analyzing data for anomalies..."):
                            results = predictor.predict_and_analyze(tmp_file_path, threshold_percentile)

                            if results:
                                st.success("✅ Analysis completed successfully!")
                                display_results(results)
                            else:
                                st.error("❌ Analysis failed. Please check your data format.")
                    else:
                        st.error("❌ Failed to load LSTM model. Please check model files.")

            else:  # Autoencoder Ensemble Model
                st.header("🧠 Autoencoder Ensemble Model Analysis")

                with st.spinner(f"Loading autoencoder models for {dataset_choice}..."):
                    if analyzer.load_models():
                        with st.spinner("Analyzing data for anomalies..."):
                            results = analyzer.predict_and_analyze(
                                tmp_file_path,
                                methods=detection_methods,
                                threshold_percentile=threshold_percentile
                            )

                            if results:
                                st.success("✅ Analysis completed successfully!")
                                display_results(results, detection_methods)
                            else:
                                st.error("❌ Analysis failed. Please check your data format.")
                    else:
                        st.error(f"❌ Failed to load autoencoder models for {dataset_choice}.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    else:
        st.info("👆 Please upload a CMAPSS dataset file to begin analysis.")

        # Show model information
        if model_choice == "LSTM Forecasting Model":
            st.subheader("📊 LSTM Forecasting Model Info")
            st.write("""
            This model uses LSTM neural networks to forecast sensor values and detect anomalies
            based on prediction errors. It's trained on CMAPSS data and can identify:
            - Unusual sensor behavior patterns
            - Deviations from expected operational conditions
            - Multi-variate anomaly detection
            """)

        else:
            st.subheader("🔍 Autoencoder Ensemble Model Info")
            st.write(f"""
            This model uses pre-trained autoencoders for {dataset_choice} dataset with ensemble detection:
            - **Autoencoder**: Reconstruction error-based detection
            """)

def display_results(results, detection_methods=None):
    """Display analysis results with enhanced visualizations"""
    anomaly_results = results['anomaly_results']
    unit_analysis = results['unit_analysis']
    test_data = results['test_data']
    model_type = results['model_type']

    # Summary metrics
    st.subheader("📊 Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    total_units = len(unit_analysis)
    total_sequences = sum(ua['total_sequences'] for ua in unit_analysis.values())
    total_anomalies = sum(ua['anomalies'] for ua in unit_analysis.values())
    avg_anomaly_rate = np.mean([ua['anomaly_rate'] for ua in unit_analysis.values()])

    col1.metric("Total Units", total_units)
    col2.metric("Total Sequences", total_sequences)
    col3.metric("Total Anomalies", total_anomalies)
    col4.metric("Avg Anomaly Rate", f"{avg_anomaly_rate:.1f}%")

    # Risk level summary
    risk_levels = [ua['risk_level'] for ua in unit_analysis.values()]
    risk_counts = {level: risk_levels.count(level) for level in ['Low', 'Medium', 'High', 'Critical']}

    st.subheader("⚠️ Risk Assessment")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    risk_col1.metric("🟢 Low Risk", risk_counts.get('Low', 0))
    risk_col2.metric("🟡 Medium Risk", risk_counts.get('Medium', 0))
    risk_col3.metric("🟠 High Risk", risk_counts.get('High', 0))
    risk_col4.metric("🔴 Critical Risk", risk_counts.get('Critical', 0))

    # Method comparison for ensemble models
    if detection_methods and len(detection_methods) > 1:
        st.subheader("🔍 Detection Methods Comparison")
        method_fig = create_method_comparison_chart(anomaly_results)
        if method_fig:
            st.plotly_chart(method_fig, use_container_width=True)

    # Anomaly summary charts
    st.subheader("📈 Anomaly Analysis Charts")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        risk_fig, _ = create_anomaly_summary_chart(unit_analysis)
        if risk_fig:
            st.plotly_chart(risk_fig, use_container_width=True)

    with chart_col2:
        _, anomaly_fig = create_anomaly_summary_chart(unit_analysis)
        if anomaly_fig:
            st.plotly_chart(anomaly_fig, use_container_width=True)

    # Detailed unit analysis
    st.subheader("🔧 Detailed Unit Analysis")

    # Unit selection for detailed view
    high_risk_units = [uid for uid, ua in unit_analysis.items() if ua['risk_level'] in ['High', 'Critical']]
    all_units = list(unit_analysis.keys())

    default_units = high_risk_units[:5] if high_risk_units else all_units[:5]

    selected_units = st.multiselect(
        "Select units for detailed analysis:",
        options=all_units,
        default=default_units,
        help="High-risk units are recommended for detailed analysis"
    )

    if selected_units:
        # Sensor selection
        sensor_columns = [col for col in test_data.columns if col.startswith('sensor_')]

        selected_sensors = st.multiselect(
            "Select sensors to visualize:",
            options=sensor_columns,
            default=sensor_columns[:4],
            help="Select sensors to display in the detailed plots"
        )

        if selected_sensors:
            # Plot type selection
            plot_type = st.radio(
                "Visualization Type:",
                ["line", "heatmap"],
                help="Line plot shows time series, heatmap shows correlations"
            )

            # Create and display sensor plots
            sensor_fig = create_enhanced_sensor_plot(
                test_data, selected_units, selected_sensors,
                unit_analysis, plot_type
            )

            if sensor_fig:
                st.plotly_chart(sensor_fig, use_container_width=True)

    # Detailed results table
    st.subheader("📋 Unit Analysis Table")

    # Convert unit analysis to DataFrame for better display
    unit_df_data = []
    for unit_id, analysis in unit_analysis.items():
        unit_df_data.append({
            'Unit ID': unit_id,
            'Risk Level': analysis['risk_level'],
            'Anomaly Rate (%)': f"{analysis['anomaly_rate']:.1f}",
            'Total Sequences': analysis['total_sequences'],
            'Anomalies': analysis['anomalies'],
            'Anomaly Positions': str(analysis['anomaly_positions'][:5]) + ('...' if len(analysis['anomaly_positions']) > 5 else '')
        })

    unit_df = pd.DataFrame(unit_df_data)

    # Sort by risk level and anomaly rate
    risk_order = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
    unit_df['Risk_Order'] = unit_df['Risk Level'].map(risk_order)
    unit_df = unit_df.sort_values(['Risk_Order', 'Anomaly Rate (%)'], ascending=[False, False])
    unit_df = unit_df.drop('Risk_Order', axis=1)

    st.dataframe(unit_df, use_container_width=True)

    # Export results
    st.subheader("💾 Export Results")

    if st.button("📄 Generate Analysis Report"):
        report_data = {
            'model_type': model_type,
            'analysis_summary': {
                'total_units': total_units,
                'total_sequences': total_sequences,
                'total_anomalies': total_anomalies,
                'average_anomaly_rate': avg_anomaly_rate
            },
            'risk_distribution': risk_counts,
            'unit_analysis': unit_analysis
        }

        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="📥 Download JSON Report",
            data=report_json,
            file_name=f"cmapss_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # CSV export
        csv_buffer = io.StringIO()
        unit_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📊 Download CSV Report",
            data=csv_buffer.getvalue(),
            file_name=f"cmapss_unit_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

main()
