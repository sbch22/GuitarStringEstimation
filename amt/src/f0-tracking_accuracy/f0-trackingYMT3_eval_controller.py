import subprocess
import json
import os

# List of models
modelNames = ["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"]

# dict to store results
modelScores = {}

for model_name in modelNames:
    print(f"Starting evaluation for model: {model_name}")

    # start F0-tracking evaluation script
    result = subprocess.run(
        ["python3", "f0-trackingYMT3_eval.py", model_name],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error while processing model {model_name}: {result.stderr}")
    else:
        print(f"Completed evaluation for model: {model_name}")

        # load from json
        output_file = f"results_{model_name.replace(' ', '_')}.json"
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                modelScores[model_name] = json.load(f)
        else:
            print(f"Warning: No results file found for {model_name}. Skipping...")

# save results
with open('model_scores_overall.json', 'w') as f:
    json.dump(modelScores, f)


with open('model_scores_overall.txt', 'w') as txt_file:
    for model_name, scores in modelScores.items():
        txt_file.write(f"{model_name}:\n")
        txt_file.write(f"  Precision: {scores['Precision']}\n")
        txt_file.write(f"  Recall: {scores['Recall']}\n")
        txt_file.write(f"  F-Score: {scores['F-Score']}\n")
        txt_file.write("\n")


print("All models processed. Results saved in 'model_scores_overall.txt'.")