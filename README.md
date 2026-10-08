FLAME 2026 — Live Neural Network
A small neural network trained in Google Colab:

Celsius → 3 hidden neurons → 1 output neuron → Fahrenheit

Run locally
pip install -r requirements.txt
streamlit run app.py
Deployment
TensorFlow is used only during training.

The Streamlit application loads learned parameters from model.json.

The deployment environment only needs:

streamlit
numpy