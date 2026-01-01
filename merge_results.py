import pandas as pd
import os

files = {
    "Base/Base+RAG": "evaluation/results_base.csv",
    "Finetuned": "evaluation/results_ft.csv",
    "Finetuned+RAG": "evaluation/results_ft_rag.csv"
}

dfs = []
for name, path in files.items():
    if os.path.exists(path):
        print(f"Loading {name} from {path}")
        df = pd.read_csv(path)
        dfs.append(df)
    else:
        print(f"Warning: {path} not found!")

if dfs:
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv("evaluation/results_final.csv", index=False)
    print("Combined results saved to evaluation/results_final.csv")
    
    # Calculate and print summary
    summary = final_df.groupby("config")[["score_mcq", "score_numeric", "score_explanation"]].mean()
    print("\nFinal Summary:")
    print(summary)
    
    # Also save summary
    summary.to_csv("evaluation/results_summary.csv")
else:
    print("No results found to merge.")
