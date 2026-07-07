import os
import re
import json
import py_compile
from pathlib import Path

# Paths
BASE_DIR = Path("/home/luke/ai_training/ultralytics/learning_lab/experiments/04_basic_yolo")

# Output lists
approved = []
rejected = {} # map ex_id -> list of reasons

def add_error(ex_id, reason):
    if ex_id not in rejected:
        rejected[ex_id] = []
    rejected[ex_id].append(reason)

def check_python_file(ex_id, file_path):
    if not file_path.exists():
        add_error(ex_id, f"Missing Python script: {file_path.name}")
        return False
    try:
        py_compile.compile(str(file_path), doraise=True)
        return True
    except py_compile.PyCompileError as e:
        add_error(ex_id, f"Python compilation failed for {file_path.name}: {str(e)}")
        return False
    except Exception as e:
        add_error(ex_id, f"Error compiling {file_path.name}: {str(e)}")
        return False

def strip_code_blocks(text):
    # Strip triple-backtick code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Strip inline code blocks
    text = re.sub(r'`[^`\n]+`', '', text)
    return text

def check_latex_math(ex_id, file_name, text):
    stripped = strip_code_blocks(text)
    
    # Count double dollars
    double_dollars = stripped.count("$$")
    if double_dollars % 2 != 0:
        add_error(ex_id, f"Unmatched double dollars '$$' in {file_name}")
        
    # Count single dollars
    # We replace $$ first, then check remaining $
    single_dollars = stripped.replace("$$", "").count("$")
    if single_dollars % 2 != 0:
        add_error(ex_id, f"Unmatched single dollar '$' in {file_name}")

def verify_markdown(ex_id, ex_dir, md_path, is_th=False):
    if not md_path.exists():
        add_error(ex_id, f"Missing markdown file: {md_path.name}")
        return
        
    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception as e:
        add_error(ex_id, f"Failed to read markdown {md_path.name}: {e}")
        return

    # Check LaTeX
    check_latex_math(ex_id, md_path.name, content)
    
    # Check for escaped pipes in Wikilinks: [[target\|alias]] is invalid, should be [[target|alias]]
    # Note: Escaped pipes are okay inside Markdown tables (as cell separators), 
    # but not inside normal lists or text.
    # Let's check for any [[...\|...]] in the text.
    escaped_pipes = re.findall(r'\[\[[^\]]*\\\|[^\]]*\]\]', content)
    if escaped_pipes:
        for link in escaped_pipes:
            add_error(ex_id, f"Escaped pipe character '\\|' in Wikilink '{link}' in {md_path.name}. Must use unescaped pipe '|'.")
            
    # Check Wikilinks
    # Strip the backslash from search if it exists (but we already flagged it as an error)
    wikilinks = re.findall(r'\[\[([^\]]+)\]\]', content)
    
    has_plan = False
    has_journal = False
    
    for link in wikilinks:
        target = link.split('|')[0].split('\\|')[0].strip()
        if target == "YOLO_Learning_Plan":
            has_plan = True
        elif target == "learning_journal":
            has_journal = True
        elif target.startswith("EX"):
            # Check if target exercise folder exists
            target_dir = BASE_DIR / target
            # Strip _TH suffix if it's there
            clean_target = target.replace("_TH", "")
            target_dir = BASE_DIR / clean_target
            if not target_dir.exists():
                add_error(ex_id, f"Invalid Wikilink target in {md_path.name}: {target} (Folder {clean_target} does not exist)")
        elif target.startswith("Journal/") or target.startswith("examples/") or target.startswith("notebooks/") or target.startswith("Concepts/"):
            # These are other valid paths in vault
            pass
        else:
            add_error(ex_id, f"Unknown Wikilink target in {md_path.name}: {target}")
            
    if not has_plan:
        add_error(ex_id, f"Missing [[YOLO_Learning_Plan]] tag/link in {md_path.name}")
    if not has_journal:
        add_error(ex_id, f"Missing [[learning_journal]] tag/link in {md_path.name}")
        
    # Check any file:/// links
    file_links = re.findall(r'\[[^\]]*\]\((file:///[^\)]+)\)', content)
    for link in file_links:
        clean_path = link.replace("file:///", "/")
        path_obj = Path(clean_path)
        if not path_obj.exists():
            add_error(ex_id, f"Dead file link in {md_path.name}: {link}")

def verify_notebook(ex_id, nb_path, is_th=False):
    if not nb_path.exists():
        add_error(ex_id, f"Missing notebook file: {nb_path.name}")
        return
        
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        add_error(ex_id, f"Invalid JSON syntax in {nb_path.name}: {e}")
        return
    except Exception as e:
        add_error(ex_id, f"Failed to read/parse {nb_path.name}: {e}")
        return

    # Check cells
    cells = data.get("cells", [])
    if not cells:
        add_error(ex_id, f"Notebook {nb_path.name} has no cells")
        return
        
    has_code = False
    has_audit_or_plot = False
    
    for cell in cells:
        cell_type = cell.get("cell_type")
        source_list = cell.get("source", [])
        source_str = "".join(source_list)
        
        if cell_type == "code":
            has_code = True
            # Check for shape printing, debug info, plot, visualize, cv2.imwrite, or assert
            if any(keyword in source_str for keyword in ["print", "shape", "debug", "plot", "visualize", "cv2.imwrite", "imshow", "plt.", "assert"]):
                has_audit_or_plot = True
                
        # Also check LaTeX math in markdown cells of notebook
        if cell_type == "markdown":
            check_latex_math(ex_id, f"{nb_path.name} cell", source_str)
            # Check Wikilinks in notebook markdown cells (if any)
            wikilinks = re.findall(r'\[\[([^\]]+)\]\]', source_str)
            for link in wikilinks:
                target = link.split('|')[0].split('\\|')[0].strip()
                if target not in ["YOLO_Learning_Plan", "learning_journal"]:
                    if target.startswith("EX"):
                        clean_target = target.replace("_TH", "")
                        target_dir = BASE_DIR / clean_target
                        if not target_dir.exists():
                            add_error(ex_id, f"Invalid Wikilink target in {nb_path.name}: {target}")
                            
            # Check file:/// links in notebook markdown cells
            file_links = re.findall(r'\[[^\]]*\]\((file:///[^\)]+)\)', source_str)
            for link in file_links:
                clean_path = link.replace("file:///", "/")
                path_obj = Path(clean_path)
                if not path_obj.exists():
                    add_error(ex_id, f"Dead file link in {nb_path.name}: {link}")
                    
    if not has_code:
        add_error(ex_id, f"Notebook {nb_path.name} has no code cells")
    if not has_audit_or_plot:
        add_error(ex_id, f"Notebook {nb_path.name} does not contain code audits or visualization plotting cells")

def main():
    # Audit exercises EX51 to EX64
    for i in range(51, 65):
        ex_prefix = f"EX{i}_"
        # Find matching directory
        matching_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith(ex_prefix)]
        if not matching_dirs:
            add_error(ex_prefix[:-1], "Directory not found")
            continue
            
        ex_dir = matching_dirs[0]
        ex_id = ex_dir.name
        
        # Verify markdown files
        md_name = f"{ex_id}.md"
        md_path = ex_dir / md_name
        verify_markdown(ex_id, ex_dir, md_path, is_th=False)
        
        md_th_name = f"{ex_id}_TH.md"
        md_th_path = ex_dir / md_th_name
        verify_markdown(ex_id, ex_dir, md_th_path, is_th=True)
        
        # Verify Jupyter notebooks
        verify_notebook(ex_id, ex_dir / "explanation.ipynb", is_th=False)
        verify_notebook(ex_id, ex_dir / "explanation_TH.ipynb", is_th=True)
        
        # Verify Python scripts
        check_python_file(ex_id, ex_dir / "solution.py")
        check_python_file(ex_id, ex_dir / "skeleton.py")
        
        # If no errors for this exercise, approve it
        if ex_id not in rejected:
            approved.append(ex_id)
            
    print("--- AUDIT RESULTS ---")
    print(f"APPROVED EXERCISES ({len(approved)}):")
    for app in approved:
        print(f" - {app}")
        
    print(f"\nREJECTED EXERCISES ({len(rejected)}):")
    for rej, reasons in rejected.items():
        print(f" - {rej}:")
        for reason in reasons:
            print(f"    * {reason}")

if __name__ == "__main__":
    main()
