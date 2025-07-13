# Final Production Version 9.0
# This version correctly handles nested markdown code blocks.
import os
import argparse
from typing import Set

# --- CONSTANTS ---
DEFAULT_IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".vscode", ".idea", 
    "dist", "build", "venv", ".venv", "env", ".env", "target", "out"
}
DEFAULT_IGNORE_FILES = {
    ".DS_Store", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", 
    ".gitignore", "*.pyc", "*.log", "*.lock", "*.swp", "*.env"
}
LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "jsx",
    ".tsx": "tsx", ".html": "html", ".css": "css", ".scss": "scss",
    ".java": "java", ".kt": "kotlin", ".rs": "rust", ".go": "go",
    ".c": "c", ".cpp": "cpp", ".cs": "csharp", ".sh": "shell",
    ".bat": "batch", ".ps1": "powershell", ".rb": "ruby", ".php": "php",
    ".sql": "sql", ".md": "markdown", ".json": "json", ".yml": "yaml",
    ".yaml": "yaml", "dockerfile": "dockerfile", "makefile": "makefile",
}

# --- CORE FUNCTIONS ---
def get_language_from_extension(file_path: str) -> str:
    _, extension = os.path.splitext(file_path)
    filename = os.path.basename(file_path).lower()
    return LANGUAGE_MAP.get(extension.lower(), "") if extension else LANGUAGE_MAP.get(filename, "")

def generate_tree(dir_path: str, ignore_set: Set[str], prefix: str = "") -> str:
    tree_str = ""
    try:
        items = sorted([item for item in os.listdir(dir_path) if item not in ignore_set])
    except FileNotFoundError:
        return ""

    for i, item in enumerate(items):
        path = os.path.join(dir_path, item)
        connector = "L-- " if i == (len(items) - 1) else "|-- "
        tree_str += prefix + connector + item + "\n"
        if os.path.isdir(path):
            new_prefix = prefix + "    " if i == (len(items) - 1) else "|   "
            tree_str += generate_tree(path, ignore_set, new_prefix)
    return tree_str

def aggregate_project_files(project_path: str, ignore_set: Set[str]) -> str:
    content_list = []
    for root, dirs, files in os.walk(project_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in ignore_set]
        sorted_files = sorted([f for f in files if f not in ignore_set])
        
        for filename in sorted_files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, project_path).replace("\\", "/")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
                if not file_content.strip():
                    continue
                language = get_language_from_extension(filename)
                
                content_list.append("---\n")
                content_list.append(f"### File: `{relative_path}`\n\n")
                
                # Using four backticks to create robust code blocks.
                content_list.append(f"````{language}\n")
                content_list.append(file_content.strip())
                content_list.append(f"\n````\n\n")

            except Exception as e:
                print(f"[ERROR] Skipping file due to error: {relative_path} ({e})")
    return "".join(content_list)

def main():
    parser = argparse.ArgumentParser(
        description="Consolidates a project into a single file, handling nested markdown. (v9.0)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("project_path", nargs="?", default=".", help="Path to project directory.")
    parser.add_argument("-o", "--output", default="project_context.md", help="Output file name.")
    parser.add_argument("-i", "--ignore", nargs='*', default=[], help="Additional items to ignore.")
    
    args = parser.parse_args()
    
    ignore_set = DEFAULT_IGNORE_DIRS.union(DEFAULT_IGNORE_FILES).union(set(args.ignore))
    
    try:
        script_name = os.path.basename(__file__)
        ignore_set.add(script_name)
    except NameError:
        pass
        
    ignore_set.add(args.output)
    
    project_path = os.path.abspath(args.project_path)
    if not os.path.isdir(project_path):
        print(f"[ERROR] Project path does not exist: '{project_path}'")
        return

    print(f"--- Running Project Packer (v9.0) ---")
    
    tree_output = generate_tree(project_path, ignore_set)
    files_output = aggregate_project_files(project_path, ignore_set)
    
    output_filename = args.output
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Project Context: {os.path.basename(project_path)}\n\n")
            f.write("## Project Structure\n\n")
            f.write("```\n")
            f.write(f"{os.path.basename(project_path)}\n")
            f.write(tree_output)
            f.write("```\n\n")
            f.write("## File Contents\n\n")
            f.write(files_output)
        print(f"\n✅ Done! Context saved to '{output_filename}'")
    except Exception as e:
        print(f"\n[ERROR] Failed to write output file: {e}")

if __name__ == "__main__":
    main()