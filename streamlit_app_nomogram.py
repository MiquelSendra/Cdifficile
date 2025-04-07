import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from the new CSV file
results_df = pd.read_csv("logistic_regression_selected_columns_translated.csv", sep= ";", index_col=0)

# Load optimal features from the CSV file
optimal_features_df = pd.read_csv("optimal_features_logistic_regression.csv", sep= ";", index_col=0)

# Streamlit app@
def main():
    st.title("Outcome Prediction Nomogram")

    st.sidebar.header("Input Patient Data")

    # Add an illustration to the right side of the app
    st.sidebar.image("c_difficile_illustration_app.jpg", use_container_width=True)

    # Filter results_df to include only optimal features
    optimal_features = optimal_features_df.values.flatten()
    filtered_results_df = results_df.loc[optimal_features]

    # Fix the issue with repeated outcome labels in the title
    feature_outcome_map = {}
    for outcome in ["Death", "Reinfection"]:
        for feature in filtered_results_df.index:
            if feature not in feature_outcome_map:
                feature_outcome_map[feature] = set()
            feature_outcome_map[feature].add(outcome)

    # Create input fields with combined outcome information
    inputs = {}
    # Set default values for inputs
    for feature, outcomes in feature_outcome_map.items():
        outcome_text = " ".join(sorted(outcomes))  # Ensure unique and sorted outcomes
        default_value = 1 if feature == "Vancomycin" else 0
        inputs[feature] = st.sidebar.selectbox(
            f"{feature}", options=[0, 1], index=default_value, key=f"{feature}_key"
        )

    # Automatically calculate predictions on page load
    filtered_results_df = filtered_results_df[~filtered_results_df.index.duplicated(keep='first')]

    predictions = []
    for outcome in ["Death", "Reinfection"]:
        coefs = filtered_results_df[f"{outcome}_coefficients"]
        intercept = results_df.loc["const", f"{outcome}_coefficients"]
        linear_predictor = intercept + sum(
            coefs[feature] * inputs[feature] for feature in filtered_results_df.index
        )

        # Convert to probability using logistic function
        probability = 1 / (1 + np.exp(-linear_predictor))
        predictions.append({"Outcome": outcome, "Probability": probability})

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Display predictions DataFrame in the Streamlit app
    st.subheader("Prediction Results using the 5 most relevant variables")
    st.dataframe(predictions_df)

    # Calculate error bars using a normalized approach to reduce the impact of multiple variables
    yerr = []
    for outcome in ["Death", "Reinfection"]:
        coefs = filtered_results_df[f"{outcome}_coefficients"]
        std_errors = filtered_results_df[f"{outcome}_Std_Error"]

        # Normalize the contribution of each variable to reduce the overall error
        total_weight = sum(abs(coefs[feature]) for feature in filtered_results_df.index)
        normalized_variance = sum(
            ((std_errors[feature] * inputs[feature] * abs(coefs[feature]) / total_weight)**2)
            for feature in filtered_results_df.index
        )
        combined_std_error = np.sqrt(normalized_variance)
        yerr.append(combined_std_error)

    # Adjust the plot size and x positions
    fig, ax = plt.subplots(figsize=(4, 4))  # Smaller figure size
    x_positions = [0.25, 0.75]  # Evenly distributed x positions
    ax.errorbar(
        x_positions,
        predictions_df["Probability"],
        yerr=yerr,
        fmt="o",
        capsize=5,
        label="Predicted Probability",
        color="blue"
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(predictions_df["Outcome"])
    ax.set_xlim(0, 1)
    ax.set_title("Predicted Probabilities")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Display results_df at the bottom
    st.subheader("Results of Logistic regretion for all variables in the study")
    st.dataframe(results_df)

if __name__ == "__main__":
    main()