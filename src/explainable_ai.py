import shap
import matplotlib.pyplot as plt

def explain_model_predictions(model, X):
    """Explain model predictions using SHAP."""
    explainer = shap.Explainer(model)
    plt.switch_backend('Agg')  # Use a non-interactive backend
    shap.summary_plot(shap_values, X)
    shap.summary_plot(shap_values, X)
