import pandas as pd

def save_ticket_eval(results, output_path="outputs/ticket_eval.csv"):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    accuracy = (df["pred_label"] == df["ground_truth"]).mean()
    print(f"Accuracy: {accuracy:.2f}")
    return df, accuracy