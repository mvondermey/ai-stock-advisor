with open('src/main.py', 'r') as f:
    content = f.read()

# Find the end of the main function
lines = content.split('\n')
end_idx = -1
for i, line in enumerate(lines):
    if line.strip() == 'if __name__ == "__main__":':
        end_idx = i
        break

if end_idx > 0:
    # Keep only up to the main block
    clean_content = '\n'.join(lines[:end_idx]) + '\n\nif __name__ == "__main__":\n    main()'
    with open('src/main.py', 'w') as f:
        f.write(clean_content)
    print('Cleaned main.py successfully')
else:
    print('Could not find main block')







