import pandas as pd

def build_change_table(original: list[str], current: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"Original": original, "Current": current})
    ct = df.groupby(["Original", "Current"]).size().reset_index(name="Count")
    total = ct["Count"].sum()
    ct["Percentage (%)"] = ct["Count"] / total * 100.0
    return ct.sort_values(["Original", "Current"]).reset_index(drop=True)
