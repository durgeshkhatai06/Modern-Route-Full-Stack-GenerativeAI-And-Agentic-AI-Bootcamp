import json
import pathlib
import re


NOTEBOOK_PATH = pathlib.Path(
    r"F:\Modern-Route-Full-Stack-GenerativeAI-And-Agentic-AI-Bootcamp\Assignments\Text Feature Engineering\Text_Feature_Engineering.ipynb"
)


def indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" "))]


def describe_import(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("import "):
        modules = stripped.replace("import ", "", 1)
        return f"Import {modules} so this notebook can use it."
    if stripped.startswith("from "):
        match = re.match(r"from\s+(.+?)\s+import\s+(.+)", stripped)
        if match:
            module, names = match.groups()
            return f"Import {names} from {module} for this step."
    return "Import the required library for the next operations."


def describe_line(line: str) -> str:
    stripped = line.strip()

    if not stripped:
        return ""
    if stripped.startswith("#"):
        return ""
    if stripped.startswith("import ") or stripped.startswith("from "):
        return describe_import(line)
    if stripped.startswith("%"):
        return f"Run the notebook magic command `{stripped}`."
    if stripped.startswith("def "):
        name = stripped.split("(")[0].replace("def ", "", 1)
        return f"Define the `{name}` function."
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return "Start a multiline string block."
    if stripped == ")" or stripped == "]" or stripped == "}" or stripped in {"],", "),", "},"}:
        return "Close the current expression."
    if stripped.startswith("return "):
        return "Return the computed value from this function."
    if stripped == "return":
        return "Return from this function."
    if stripped.startswith("for "):
        return "Loop through the items needed for this operation."
    if stripped.startswith("if "):
        return "Check this condition before running the next statement."
    if stripped.startswith("elif "):
        return "Check the next condition if the previous one was not met."
    if stripped == "else:":
        return "Handle the fallback case."
    if stripped.startswith("try:"):
        return "Start a protected block so errors can be handled safely."
    if stripped.startswith("except "):
        return "Handle any exception raised in the try block."
    if stripped.startswith("with "):
        return "Open a managed context for this resource."
    if stripped.startswith("print("):
        return "Display this information in the notebook output."
    if stripped.startswith("df.head(") or stripped == "df.head()":
        return "Show the first few rows of the dataset."
    if stripped.startswith("plt.show(") or stripped == "plt.show()":
        return "Render the current plot in the notebook."
    if stripped.startswith("plt.tight_layout(") or stripped == "plt.tight_layout()":
        return "Adjust the layout so plot elements fit neatly."
    if stripped.startswith("plt.savefig("):
        return "Save the current figure to an image file."
    if stripped.startswith("ax.legend(") or stripped == "ax.legend()":
        return "Display the legend for the chart."
    if stripped.startswith("ax.grid("):
        return "Add grid lines to make the chart easier to read."
    if stripped.startswith("ax.set_") or stripped.startswith("plt.suptitle("):
        return "Set a label or title for the visualization."
    if stripped.startswith("warnings.filterwarnings"):
        return "Suppress warning messages to keep the output clean."
    if stripped.startswith(("plt.", "ax.", "fig,", "fig ", "bars", "colors", "titles", "width = ", "x = np.arange")):
        return "Configure or build the current visualization step."
    if stripped.startswith("fig, ax") or stripped.startswith("fig, axes"):
        return "Create the matplotlib figure and axes objects."
    if stripped.startswith("df = "):
        return "Load the dataset into the DataFrame variable `df`."
    if stripped.startswith("STOPWORDS ="):
        return "Create the custom stopword set used during preprocessing."
    if stripped.startswith("LEMMA_MAP ="):
        return "Create the rule-based lemmatization dictionary."
    if stripped.startswith("results = []"):
        return "Create an empty list to collect model evaluation results."
    if stripped.startswith("configs = ["):
        return "List the model and feature configurations to evaluate."
    if stripped.startswith("summary = pd.DataFrame("):
        return "Build a DataFrame that summarizes the comparison."
    if stripped.startswith("matrices = "):
        return "Group the feature matrices in one dictionary for reuse."
    if stripped.startswith("custom_reviews = ["):
        return "Define a few custom reviews for final testing."
    if stripped.startswith("all_tokens = []"):
        return "Create a list that will store every cleaned token."
    if stripped.startswith("word_freq = Counter("):
        return "Count how often each token appears in the corpus."
    if stripped.startswith("vocab = "):
        return "Build the sorted vocabulary list."
    if stripped.startswith("top20 = "):
        return "Select the twenty most frequent words."
    if stripped.startswith("sk_vec = "):
        return "Create a CountVectorizer object for comparison."
    if stripped.startswith("ohe_vec = "):
        return "Create the vectorizer used for one-hot style features."
    if stripped.startswith("bow_vec = ") or stripped.startswith("bow_cv  = "):
        return "Create the Bag of Words vectorizer."
    if stripped.startswith("tfidf_vec = ") or stripped.startswith("tfidf_cv = "):
        return "Create the TF-IDF vectorizer."
    if stripped.startswith("le = "):
        return "Create the label encoder for sentiment labels."
    if stripped.startswith("X_text = "):
        return "Select the cleaned review text as the model input."
    if stripped.startswith("model.fit("):
        return "Train the current model on the training features."
    if stripped.startswith("preds = ") or stripped.startswith("best_preds = "):
        return "Generate predictions for the current data."
    if stripped.startswith("report = "):
        return "Generate the classification report dictionary."
    if stripped.startswith("results_df = "):
        return "Convert the collected results into a DataFrame."
    if stripped.startswith("best_idx"):
        return "Find the row index of the best-performing result."
    if stripped.startswith("best_row"):
        return "Read the best-performing result row."

    assignment = re.match(r"([A-Za-z_][A-Za-z0-9_\[\]\"']*)\s*=\s*(.+)", stripped)
    if assignment:
        variable = assignment.group(1)
        return f"Store the result of this step in `{variable}`."

    return f"Explain the purpose of: {stripped}"


AUTO_COMMENT_PREFIXES = (
    "# Import ",
    "# Run the notebook magic command ",
    "# Define the `",
    "# Start a multiline string block.",
    "# Close the current expression.",
    "# Return ",
    "# Loop through ",
    "# Check ",
    "# Handle ",
    "# Display this information in the notebook output.",
    "# Suppress warning messages to keep the output clean.",
    "# Configure or build the current visualization step.",
    "# Create the matplotlib figure and axes objects.",
    "# Load the dataset into the DataFrame variable `df`.",
    "# Create the custom stopword set used during preprocessing.",
    "# Create the rule-based lemmatization dictionary.",
    "# Create an empty list to collect model evaluation results.",
    "# List the model and feature configurations to evaluate.",
    "# Build a DataFrame that summarizes the comparison.",
    "# Group the feature matrices in one dictionary for reuse.",
    "# Define a few custom reviews for final testing.",
    "# Create a list that will store every cleaned token.",
    "# Count how often each token appears in the corpus.",
    "# Build the sorted vocabulary list.",
    "# Select the twenty most frequent words.",
    "# Create a CountVectorizer object for comparison.",
    "# Create the vectorizer used for one-hot style features.",
    "# Create the Bag of Words vectorizer.",
    "# Create the TF-IDF vectorizer.",
    "# Create the label encoder for sentiment labels.",
    "# Select the cleaned review text as the model input.",
    "# Train the current model on the training features.",
    "# Generate predictions for the current data.",
    "# Generate the classification report dictionary.",
    "# Convert the collected results into a DataFrame.",
    "# Find the row index of the best-performing result.",
    "# Read the best-performing result row.",
    "# Store the result of this step in `",
    "# Show the first few rows of the dataset.",
    "# Render the current plot in the notebook.",
    "# Adjust the layout so plot elements fit neatly.",
    "# Save the current figure to an image file.",
    "# Display the legend for the chart.",
    "# Add grid lines to make the chart easier to read.",
    "# Set a label or title for the visualization.",
    "# Explain the purpose of: ",
)


def is_auto_comment(line: str) -> bool:
    stripped = line.strip()
    return any(stripped.startswith(prefix) for prefix in AUTO_COMMENT_PREFIXES)


def comment_line(original_line: str) -> list[str]:
    stripped = original_line.strip()
    if not stripped or stripped.startswith("#"):
        return [original_line]

    comment = describe_line(original_line)
    indent = indent_of(original_line)
    return [f"{indent}# {comment}\n", original_line]


def transform_source(source: list[str]) -> list[str]:
    cleaned_source = [line for line in source if not is_auto_comment(line)]
    new_source: list[str] = []
    for line in cleaned_source:
        new_source.extend(comment_line(line))
    return new_source


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["source"] = transform_source(cell.get("source", []))
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
