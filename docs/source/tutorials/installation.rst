Installation
============

This guide will walk you through installing TurboGuard and all its dependencies on your system.

System Requirements
-------------------

Before installing TurboGuard, ensure your system meets these requirements:

**Operating System**
- Windows
- macOS 
- Linux

**Python**
- Python 3.8 or higher
- pip package manager

**Hardware**
- Minimum 6GB RAM (8GB recommended)
- GPU support optional but recommended for training

Installation Methods
--------------------

Method 1: Install from GitHub (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone the Repository**

   .. code-block:: bash

      git clone https://github.com/mouradboutrid/TurboGuard.git
      cd TurboGuard

2. **Create Virtual Environment**

   .. code-block:: bash

      # Using venv
      python -m venv turboguard_env
      
      # Activate on Windows
      turboguard_env\Scripts\activate
      
      # Activate on macOS/Linux
      source turboguard_env/bin/activate

3. **Install Dependencies**

   .. code-block:: bash

      pip install -r requirements.txt

Method 2: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For contributors or advanced users who want to modify the code:

.. code-block:: bash

   git clone https://github.com/mouradboutrid/TurboGuard.git
   cd TurboGuard
   pip install -e .

This installs TurboGuard in "editable" mode, so changes to the source code are immediately reflected.

Required Dependencies
---------------------

TurboGuard requires the following main packages:

**Core Dependencies**

.. code-block:: text

   tensorflow>=2.8.0
   numpy>=1.21.0
   pandas>=1.3.0
   scikit-learn>=1.0.0
   matplotlib>=3.5.0
   seaborn>=0.11.0

**Dashboard Dependencies**

.. code-block:: text

   streamlit>=1.12.0
   plotly>=5.10.0
   altair>=4.2.0

**Utility Dependencies**

.. code-block:: text

   tqdm>=4.64.0
   pyyaml>=6.0
   joblib>=1.1.0

GPU Support (Optional)
----------------------

To enable GPU acceleration for model training:

**NVIDIA GPU Setup**

1. Install CUDA Toolkit (11.2 or later):
   
   Download from `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_

2. Install cuDNN:
   
   Download from `NVIDIA cuDNN <https://developer.nvidia.com/cudnn>`_

3. Install TensorFlow GPU support:

   .. code-block:: bash

      pip install tensorflow[and-cuda]

**Verify GPU Installation**

.. code-block:: python

   import tensorflow as tf
   print("GPU Available: ", tf.config.list_physical_devices('GPU'))

Verify Installation
-------------------

Test your installation with these verification steps:

**1. Import Test**

.. code-block:: python

   # Test core imports
   import tensorflow as tf
   import pandas as pd
   import numpy as np
   import streamlit as st
   
   print("✅ All core dependencies imported successfully!")

**2. TurboGuard Import Test**

.. code-block:: python

   from src.LSTM_AutoEncoder.data_loader import CMAPSSDataLoader
   from src.LSTM_AutoEncoder.lstm_autoencoder import LSTMAutoencoder
   from src.Forecasting_LSTM.forecasting_lstm import ForecastingLSTM
   
   print("✅ TurboGuard modules imported successfully!")

**3. Dashboard Test**

.. code-block:: bash

   streamlit run app/app.py

If successful, you should see:

.. code-block:: text

   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501

Common Installation Issues
--------------------------

**Issue 1: TensorFlow Installation Fails**

*Error*: ``ERROR: Could not find a version that satisfies the requirement tensorflow``

*Solution*:
- Ensure Python version is 3.8-3.11
- Update pip: ``pip install --upgrade pip``
- Try: ``pip install tensorflow --upgrade``

**Issue 2: CUDA/GPU Issues**

*Error*: ``Could not load dynamic library 'libcudart.so.11.0'``

*Solution*:
- Verify CUDA installation
- Check CUDA version compatibility with TensorFlow
- Install matching cuDNN version

**Issue 3: Memory Issues During Installation**

*Error*: ``MemoryError`` during package installation

*Solution*:
- Close other applications
- Install packages one by one
- Use: ``pip install --no-cache-dir -r requirements.txt``

**Issue 4: Streamlit Port Already in Use**

*Error*: ``OSError: [Errno 48] Address already in use``

*Solution*:
- Use different port: ``streamlit run app/app.py --server.port 8502``
- Kill existing process on port 8501

Next Steps
----------

Once installation is complete:

1. ✅ **Continue to** :doc:`quickstart` to launch your first TurboGuard session
2. 📊 **Explore** the interactive dashboard 
3. 🤖 **Build** your first model in :doc:`first_model`

Need Help?
----------

If you encounter issues not covered here:

- 🐛 **Report bugs**: `GitHub Issues <https://github.com/mouradboutrid/-TurboGuard/issues>`_
- 💬 **Ask questions**: Create a discussion on GitHub
- 📖 **Check docs**: Refer to our detailed API documentation

Congratulations! You're ready to start using TurboGuard! 🎉
