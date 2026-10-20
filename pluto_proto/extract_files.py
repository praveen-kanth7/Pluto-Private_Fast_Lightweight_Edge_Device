import json
import os
import re

transcript_path = r"C:\Users\prave\.gemini\antigravity-ide\brain\18752502-1d4d-45af-97c4-e025d5414eae\.system_generated\logs\transcript_full.jsonl"

target_files = ['mfcc.cpp', 'mfcc.h', 'kws_preprocess.cpp', 'kws_preprocess.h']
found_content = {t: None for t in target_files}

print("Parsing transcript...")
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            entry = json.loads(line)
        except Exception:
            continue
            
        if 'content' not in entry:
            continue
            
        content = entry['content']
        if not isinstance(content, str):
            continue
            
        # Check if it's a VIEW_FILE output
        if "File Path: " in content and "The following code has been modified" in content:
            for t in target_files:
                if t in content and found_content[t] is None:
                    print(f"Found contents for {t}")
                    # Extract the actual code
                    # It's after "The following code has been modified... leading space.\n"
                    # format: "1: #include ..."
                    lines = content.split('\n')
                    code_lines = []
                    for l in lines:
                        match = re.match(r'^\d+: (.*)', l)
                        if match:
                            code_lines.append(match.group(1))
                        elif re.match(r'^\d+:$', l):
                            code_lines.append("")
                    if code_lines:
                        found_content[t] = "\n".join(code_lines)

for t in target_files:
    if found_content[t]:
        out_path = os.path.join("main", t)
        print(f"Writing {out_path}...")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(found_content[t])
    else:
        print(f"Could not find {t}")
