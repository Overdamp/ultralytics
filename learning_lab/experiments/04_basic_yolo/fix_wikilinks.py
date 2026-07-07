import re
from pathlib import Path

BASE_DIR = Path("/home/luke/ai_training/ultralytics/learning_lab/experiments/04_basic_yolo")

def replace_escaped_pipes(match):
    content = match.group(1)
    # Replace all occurrences of escaped pipe \| with a normal pipe |
    fixed_content = content.replace(r'\|', '|')
    return f'[[{fixed_content}]]'

def main():
    modified_count = 0
    # Walk all .md files
    for md_path in BASE_DIR.glob("**/*.md"):
        if md_path.suffix == '.bak':
            continue
        content = md_path.read_text(encoding="utf-8")
        
        # Check if there are any Wikilinks with escaped pipes
        if '[[' in content and r'\|' in content:
            new_content = re.sub(r'\[\[(.*?)\]\]', replace_escaped_pipes, content)
            if new_content != content:
                md_path.write_text(new_content, encoding="utf-8")
                print(f"Fixed Wikilinks in: {md_path.relative_to(BASE_DIR)}")
                modified_count += 1
                
    print(f"Successfully fixed Wikilinks in {modified_count} files.")

if __name__ == "__main__":
    main()
