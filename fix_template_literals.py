#!/usr/bin/env python3
"""
Fix template literal issues in web_interface.py
Converts multi-line template literals to single line strings
"""

import re

def fix_template_literals(content):
    """Fix multi-line template literals in JavaScript code"""
    
    # Pattern to match template literals with line breaks
    # This is complex because we need to handle nested quotes and escapes
    lines = content.split('\n')
    fixed_lines = []
    in_template = False
    template_start_line = -1
    template_content = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if we're starting a template literal
        if not in_template and '`' in line and not line.strip().endswith('`'):
            # This might be the start of a multi-line template
            if 'innerHTML = `' in line or '.innerHTML = `' in line or 'textContent = `' in line:
                in_template = True
                template_start_line = i
                template_content = [line]
                i += 1
                continue
        
        # If we're in a template literal, collect lines until we find the closing backtick
        if in_template:
            template_content.append(line)
            if '`;' in line:
                # End of template literal found
                in_template = False
                
                # Convert the multi-line template to a single line
                # Extract the assignment part
                first_line = template_content[0]
                assignment_part = first_line[:first_line.index('`') + 1]
                
                # Extract the content between backticks
                full_template = '\n'.join(template_content)
                start_idx = full_template.index('`') + 1
                end_idx = full_template.rindex('`')
                content_between = full_template[start_idx:end_idx]
                
                # Clean up the content - remove extra whitespace and newlines
                cleaned_content = ' '.join(content_between.split())
                
                # Replace backticks with single quotes and escape any single quotes in the content
                cleaned_content = cleaned_content.replace("'", "\\'")
                
                # Create the fixed line
                fixed_line = assignment_part[:-1] + "'" + cleaned_content + "';"
                fixed_lines.append(fixed_line)
                
                template_content = []
                i += 1
                continue
        
        # Normal line, just append
        if not in_template:
            fixed_lines.append(line)
        
        i += 1
    
    return '\n'.join(fixed_lines)

def main():
    # Read the file
    with open('web_interface.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix template literals
    fixed_content = fix_template_literals(content)
    
    # Write back
    with open('web_interface.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ Fixed template literal issues in web_interface.py")
    print("🔍 Please restart your Flask server to see the changes")

if __name__ == "__main__":
    main()