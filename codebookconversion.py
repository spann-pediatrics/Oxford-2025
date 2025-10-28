import pandas as pd

# Read your markdown file
md_path = "Data_dictionary.md"

# Read as a table (pandas can infer tables from markdown-style formatting)
df = pd.read_csv(md_path, sep="|", engine="python", skiprows=2).dropna(how="all", axis=1)

# Clean up column names (strip spaces, remove extra border columns)
df.columns = df.columns.str.strip()
df = df.rename(columns={'Keep ':'Keep', ' Variable ':'Variable'})
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Save to Excel
df.to_excel("Data_dictionary_converted.xlsx", index=False)
print("✅ Saved as Data_dictionary_converted.xlsx")
