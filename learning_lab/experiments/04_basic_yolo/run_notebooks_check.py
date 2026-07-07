import os
import sys
import json
import traceback
from pathlib import Path

# Set matplotlib backend to Agg to prevent opening windows
import matplotlib
matplotlib.use('Agg')

BASE_DIR = Path("/home/luke/ai_training/ultralytics/learning_lab/experiments/04_basic_yolo")

def execute_notebook_code(nb_path):
    print(f"Executing notebook: {nb_path.name}")
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        return False, f"JSON Load Error: {e}"
        
    # Set up execution environment
    # Add the notebook folder to sys.path so it can find its local 'solution.py'
    nb_dir = str(nb_path.parent)
    if nb_dir not in sys.path:
        sys.path.insert(0, nb_dir)
        
    # Clear local cache of 'solution' module to ensure we load the local one
    if 'solution' in sys.modules:
        del sys.modules['solution']
        
    global_env = {
        '__name__': '__main__',
        '__file__': str(nb_path.parent / "solution.py"),
    }
    
    # Run each code cell sequentially in the same environment
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
            
        source_lines = cell.get("source", [])
        source_code = "".join(source_lines)
        
        # Strip out IPython magic commands
        clean_lines = []
        for line in source_code.splitlines():
            if not line.strip().startswith("%") and not line.strip().startswith("!"):
                clean_lines.append(line)
        clean_code = "\n".join(clean_lines)
        
        if not clean_code.strip():
            continue
            
        try:
            # Change working directory to the notebook's directory so paths resolve correctly
            old_cwd = os.getcwd()
            os.chdir(nb_dir)
            
            # Execute
            exec(clean_code, global_env)
            
            # Restore CWD
            os.chdir(old_cwd)
        except Exception as e:
            # Restore CWD
            os.chdir(old_cwd)
            tb = traceback.format_exc()
            return False, f"Runtime Error in Cell #{i+1}:\n{e}\nTraceback:\n{tb}"
            
    # Clean up sys.path
    if nb_dir in sys.path:
        sys.path.remove(nb_dir)
        
    return True, "Success"

def main():
    success_count = 0
    failure_count = 0
    failures = []
    
    # Audit exercises EX51 to EX64
    for i in range(51, 65):
        ex_prefix = f"EX{i}_"
        matching_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith(ex_prefix)]
        if not matching_dirs:
            continue
            
        ex_dir = matching_dirs[0]
        
        # Check both notebooks
        for filename in ["explanation.ipynb", "explanation_TH.ipynb"]:
            nb_path = ex_dir / filename
            if not nb_path.exists():
                failures.append((nb_path, "Notebook does not exist"))
                failure_count += 1
                continue
                
            ok, msg = execute_notebook_code(nb_path)
            if ok:
                success_count += 1
                print(f"  Passed execution: {nb_path.relative_to(BASE_DIR)}")
            else:
                failures.append((nb_path, msg))
                failure_count += 1
                print(f"  [ERROR] Execution failed: {nb_path.relative_to(BASE_DIR)}\n{msg}")
                
    print("\n--- RUNTIME EXECUTION SUMMARY ---")
    print(f"Total Successful Notebook Runs: {success_count}")
    print(f"Total Failed Notebook Runs: {failure_count}")
    if failure_count > 0:
        print("\nFailures Detail:")
        for path, err in failures:
            print(f" - {path.name}:\n{err}\n")
        sys.exit(1)
    else:
        print("All notebooks executed successfully without any runtime or syntax errors!")

if __name__ == "__main__":
    main()
