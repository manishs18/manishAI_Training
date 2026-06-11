import pandas as pd
from pathlib import Path

def save_ticket_eval(results, output_path="outputs/ticket_eval.csv"):
    df = pd.DataFrame(results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    accuracy = (df["pred_label"] == df["ground_truth"]).mean()
    print(f"Accuracy: {accuracy:.2f}")
    return df, accuracy