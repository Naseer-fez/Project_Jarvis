import re

files_to_patch = [
    r"d:\AI\Jarvis\core\memory\sqlite_storage.py",
    r"d:\AI\Jarvis\core\memory\code_indexer_service.py"
]

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find `await conn.execute` that doesn't have `async with`
    lines = content.split('\n')
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.search(r'^(\s*)await conn\.(execute|executemany|executescript)\(', line)
        if match:
            indent = match.group(1)
            # Find the end of the statement by matching parentheses
            statement_lines = [line]
            open_parens = line.count('(') - line.count(')')
            j = i + 1
            while open_parens > 0 and j < len(lines):
                statement_lines.append(lines[j])
                open_parens += lines[j].count('(') - lines[j].count(')')
                j += 1
            
            # replace `await conn.` with `async with conn.`
            statement_lines[0] = statement_lines[0].replace('await conn.', 'async with conn.')
            # add `:` to the last line
            statement_lines[-1] = statement_lines[-1] + ":"
            
            # add the block
            new_lines.extend(statement_lines)
            new_lines.append(indent + "    pass")
            
            i = j - 1
        else:
            new_lines.append(line)
        i += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))

for f in files_to_patch:
    process_file(f)
    print(f"Patched {f}")
