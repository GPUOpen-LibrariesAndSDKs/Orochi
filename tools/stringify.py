#!/usr/bin/env python3
"""Stringify GPU kernel source files into C++ string literals.

Reads kernel source files and converts them into C++ const char* variables
that can be compiled directly into the binary.

Usage:
    python3 stringify.py <kernel_file>
"""
import os
import sys


def print_file(filename, output, api, base_dir='./'):
    """Recursively read a kernel file, inlining includes and escaping for C++."""
    with open(filename) as fh:
        for line in fh.readlines():
            line = line.strip('\r\n').strip()

            if line.startswith('//'):
                continue

            # Inline .inl includes
            if '#include' in line and ('inl.cl' in line or 'inl.metal' in line or 'inl.cu' in line):
                _, tail = os.path.split(line)
                tail = base_dir + tail.replace('>', '')
                output = print_file(tail, output, api, base_dir)

            if '#include' in line and api != 'hip':
                continue

            # Escape for C++ string literal
            escaped = '"' + line.replace('"', '\\"').replace("'", "\\'") + '\\n"'
            output += escaped + '\n'

    return output


def stringify(filename, string_name, api, base_dir='./'):
    """Convert a kernel file to a C++ string literal variable."""
    print('static const char* ' + string_name + '= \\')
    output = print_file(filename, '', api, base_dir)
    print(output + ';')


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel_file>", file=sys.stderr)
        sys.exit(1)

    files = [sys.argv[1]]
    api = 'hip'

    # Process Math files first, then the rest
    for math_first in (True, False):
        for source_file in files:
            if ('Math.' in source_file) != math_first:
                continue
            if not any(ext in source_file for ext in ('.cl', '.cu', '.metal', '.h')):
                continue
            string_name = source_file.replace('.cl', '').replace('.cu', '').replace('.metal', '').replace('.h', '')
            string_name = api + '_' + string_name.split('/')[-1]
            stringify('./' + source_file, string_name, api)


if __name__ == '__main__':
    main()
