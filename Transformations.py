import pandas as pd
import numpy as np
from itertools import product
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut

# Load dataset
merged_residuals = pd.read_csv("merged_residuals.csv")

# Models selected 
selected_models = {
    "model_1": ["PC.ae.C36.5", "Cer.d18.1.24.1.", "Cer.d18.1.22.0.", "HexCer.d18.1.26.1."],
    "model_2": ["PC.ae.C36.5", "Cer.d18.1.22.0.", "HexCer.d18.1.26.1.", "CE.18.3."],
    "model_3": ["PC.ae.C36.5", "Cer.d18.1.24.1.", "Cer.d18.1.22.0.", "HexCer.d18.1.26.1.", "CE.18.3."],
    "model_4": ["PC.ae.C36.5", "Cer.d18.1.24.1.", "Cer.d18.1.24.0.", "HexCer.d18.1.26.1.", "CE.22.5."],
    "model_5": ["PC.ae.C36.5", "PC.aa.C42.6", "Cer.d18.1.24.1.", "Cer.d18.1.24.0.", "Cer.d18.2.18.1.", "HexCer.d18.1.26.1."],
    "model_6": ["PC.ae.C36.5", "PC.aa.C42.6", "Cer.d18.1.24.1.", "Cer.d18.1.22.0.", "HexCer.d18.1.26.1.", "CE.18.3."],
    "model_7": ["PC.ae.C36.5", "PC.aa.C36.1", "Cer.d18.1.24.1.", "Cer.d18.2.20.0.", "HexCer.d18.1.26.1.", "CE.18.3."],
    "model_8": ["PC.aa.C42.5", "PC.ae.C36.5", "Cer.d18.1.24.1.", "Cer.d18.1.22.0.", "HexCer.d18.1.26.1.", "CE.18.3."],
}

# Transformations for each metabolite in each model
for met in merged_residuals.columns:
    if met not in ["ID", "Group"]:
        merged_residuals[f"{met}_sq"] = merged_residuals[met] ** 2
        merged_residuals[f"{met}_log10"] = np.log10(merged_residuals[met] + 1)
        merged_residuals[f"{met}_sqrt"] = np.sqrt(np.maximum(merged_residuals[met], 0))

# LDA with LOOCV
def run_lda_loocv(df, feature_set):
    X = df[feature_set].values
    y = df["Group"].values
    loo = LeaveOneOut()
    accuracies = []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        
        accuracies.append(y_pred[0] == y_test[0])
    
    mean_accuracy = np.mean(accuracies)
    sd_accuracy = np.std(accuracies)
    
    return {"mean_accuracy": mean_accuracy, "sd_accuracy": sd_accuracy}

# Initialize list of results
performance_results = []

# Iterate over models
for model_name, metabolites in selected_models.items():
    print(f"\n PROCESSING MODEL: {model_name}\n")
    
    metabolite_options = [[met, f"{met}_sq", f"{met}_log10", f"{met}_sqrt"] for met in metabolites]
    
    for combination in product(*metabolite_options):
        feature_set = list(combination)
        combination_name = "_".join(feature_set)
        
        print(f"    Combination: {combination_name}")
        
        results = run_lda_loocv(merged_residuals, feature_set)
        
        performance_results.append({
            "Model": model_name,
            "Combination": combination_name,
            "MeanAccuracy": results["mean_accuracy"],
            "SDAccuracy": results["sd_accuracy"]
        })

# Save results
df_results = pd.DataFrame(performance_results)
df_results.to_csv("lda_results_loocv.csv", index=False)


print(df_results)
