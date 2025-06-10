flowchart TD
    Start([🚀 Start TurboGuard]) --> InitApp[Initialize Application]
    InitApp --> DisplayUI[Display Main Interface]
    DisplayUI --> ModelSelect{Select Analysis Model}
    
    %% Model Selection Branch
    ModelSelect -->|LSTM Forecasting| LoadLSTM[Load LSTM Model & Config]
    ModelSelect -->|Autoencoder Ensemble| SelectDataset[Select Dataset Type<br/>FD001/FD002/FD003/FD004]
    
    SelectDataset --> LoadAE[Load Autoencoder Models]
    LoadLSTM --> CheckLSTM{Model Files<br/>Available?}
    LoadAE --> CheckAE{Model Files<br/>Available?}
    
    CheckLSTM -->|No| ErrorLSTM[❌ Display Error Message]
    CheckAE -->|No| ErrorAE[❌ Display Error Message]
    ErrorLSTM --> ModelSelect
    ErrorAE --> ModelSelect
    
    CheckLSTM -->|Yes| LSTMReady[✅ LSTM Model Ready]
    CheckAE -->|Yes| AEReady[✅ Autoencoder Models Ready]
    
    %% Configuration Phase
    LSTMReady --> ConfigLSTM[Configure LSTM Parameters<br/>• Threshold Percentile]
    AEReady --> ConfigAE[Configure AE Parameters<br/>• Detection Methods<br/>• Threshold Percentile]
    
    ConfigLSTM --> WaitUpload[Wait for Data Upload]
    ConfigAE --> WaitUpload
    
    %% Data Upload Phase
    WaitUpload --> UploadFile{Upload CMAPSS<br/>Dataset?}
    UploadFile -->|No| WaitUpload
    UploadFile -->|Yes| ValidateFile[Validate File Format]
    ValidateFile --> FileValid{File Valid?}
    FileValid -->|No| FileError[❌ Display Format Error]
    FileError --> WaitUpload
    FileValid -->|Yes| SaveTemp[Save Temporary File]
    
    %% Analysis Trigger
    SaveTemp --> StartAnalysis[🔄 Start Analysis Process]
    StartAnalysis --> ModelType{Which Model<br/>Selected?}
    
    %% LSTM Analysis Branch
    ModelType -->|LSTM| PreprocessLSTM[Preprocess Test Data<br/>• Load & Clean Data<br/>• Normalize by Op Mode<br/>• Remove Low-Variance Sensors]
    PreprocessLSTM --> CreateSeqLSTM[Create Sequences<br/>• Generate Time Windows<br/>• Extract Features<br/>• Prepare Model Input]
    CreateSeqLSTM --> PredictLSTM[LSTM Prediction<br/>• Forward Pass<br/>• Generate Forecasts]
    PredictLSTM --> DetectLSTM[Detect Anomalies<br/>• Calculate MSE/MAE/MAX Errors<br/>• Apply Threshold<br/>• Ensemble Voting]
    DetectLSTM --> AnalyzeLSTM[Analyze Unit Results<br/>• Calculate Anomaly Rates<br/>• Assign Risk Levels]
    
    %% Autoencoder Analysis Branch  
    ModelType -->|Autoencoder| PreprocessAE[Preprocess Test Data<br/>• Load & Validate Data<br/>• Match Model Config<br/>• Normalize Features]
    PreprocessAE --> CreateSeqAE[Create Sequences<br/>• Generate Time Windows<br/>• Ensure Correct Dimensions]
    CreateSeqAE --> DetectMethods{Select Detection<br/>Methods}
    
    DetectMethods --> ReconstructionAE[Autoencoder Detection<br/>• Compute Reconstruction Error<br/>• Apply Threshold]
    DetectMethods --> StatisticalAE[Statistical Detection<br/>• Z-Score Analysis<br/>• IQR-based Detection]
    DetectMethods --> WaveletAE[Wavelet Detection<br/>• Decompose Signals<br/>• Analyze Detail Coefficients]
    
    ReconstructionAE --> EnsembleAE[Ensemble Detection<br/>• Combine Method Results<br/>• Majority Voting]
    StatisticalAE --> EnsembleAE
    WaveletAE --> EnsembleAE
    
    EnsembleAE --> AnalyzeAE[Analyze Unit Results<br/>• Calculate Anomaly Rates<br/>• Assign Risk Levels]
    
    %% Results Processing
    AnalyzeLSTM --> ProcessResults[Process Analysis Results<br/>• Calculate Summary Metrics<br/>• Generate Risk Assessment]
    AnalyzeAE --> ProcessResults
    
    ProcessResults --> CreateViz[Create Visualizations<br/>• Risk Distribution Charts<br/>• Anomaly Rate Plots<br/>• Method Comparison]
    CreateViz --> DisplayResults[📊 Display Results Dashboard<br/>• Summary Metrics<br/>• Risk Assessment<br/>• Unit Analysis Table]
    
    %% Interactive Features
    DisplayResults --> UserAction{User Action}
    UserAction -->|View Details| SelectUnits[Select Units for<br/>Detailed Analysis]
    SelectUnits --> SelectSensors[Select Sensors<br/>to Visualize]
    SelectSensors --> CreatePlots[Create Enhanced Plots<br/>• Time Series with Anomalies<br/>• Correlation Heatmaps]
    CreatePlots --> ShowPlots[Display Interactive Plots]
    ShowPlots --> UserAction
    
    UserAction -->|Export| GenerateReport[Generate Analysis Report<br/>• JSON Format<br/>• CSV Format]
    GenerateReport --> DownloadReport[📥 Download Report Files]
    DownloadReport --> UserAction
    
    UserAction -->|New Analysis| CleanupTemp[Cleanup Temporary Files]
    CleanupTemp --> WaitUpload
    
    UserAction -->|Exit| Cleanup[Cleanup Resources]
    Cleanup --> End([🔚 End])
    
    %% Error Handling
    PreprocessLSTM -.->|Error| ProcessError[❌ Handle Processing Error]
    PreprocessAE -.->|Error| ProcessError
    PredictLSTM -.->|Error| ProcessError
    ReconstructionAE -.->|Error| ProcessError
    ProcessError --> DisplayError[Display Error Message]
    DisplayError --> WaitUpload
    
    %% Styling
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#1b5e20
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#0d47a1
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#c62828
    classDef success fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#1b5e20
    classDef analysis fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    
    class Start,End startEnd
    class InitApp,DisplayUI,SaveTemp,PreprocessLSTM,PreprocessAE,CreateSeqLSTM,CreateSeqAE,ProcessResults,CreateViz,SelectUnits,SelectSensors,CreatePlots,ShowPlots,GenerateReport,DownloadReport,CleanupTemp,Cleanup process
    class ModelSelect,CheckLSTM,CheckAE,UploadFile,FileValid,ModelType,DetectMethods,UserAction decision
    class ErrorLSTM,ErrorAE,FileError,ProcessError,DisplayError error
    class LSTMReady,AEReady,LoadLSTM,LoadAE success
    class PredictLSTM,DetectLSTM,AnalyzeLSTM,ReconstructionAE,StatisticalAE,WaveletAE,EnsembleAE,AnalyzeAE,DisplayResults analysis
