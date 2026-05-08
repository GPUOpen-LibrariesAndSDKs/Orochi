#!/usr/bin/env python
"""Generate kernel argument headers from HIP kernel source files.

Reads #include directives from a kernel header and produces a C++ array
of argument file references for runtime kernel compilation.

Usage:
    python genArgs.py <kernel_header.h>
"""
from __future__ import print_function

import os
import sys


def gen_args(filename, api, includes):
    """Parse a kernel header and emit argument array declarations."""
    with open(filename) as f:
        base_name = os.path.basename(filename).split('.')[0]

        print('#if !defined(ORO_PP_LOAD_FROM_STRING)')
        print('\tstatic const char** ' + base_name + 'Args = 0;')
        print('#else')
        print('\tstatic const char* ' + base_name + 'Args[] = {')

        includes += base_name + 'Includes[] = {'

        for line in f.readlines():
            line = line.strip('\r\n')

            if '#include' not in line:
                continue
            if '#include' in line and 'inl.' + api in line:
                continue
            if api in ('cl', 'metal') and '.cu' in line:
                continue
            if '"' in line and '#include' in line:
                continue

            header = os.path.basename(line.split('<')[1].split('>')[0])
            includes += '"' + line.split('<')[1].split('>')[0] + '",'
            name = header.split('.' + api)[0]
            name = name.split('.h')[0]
            name = api + '_' + name
            print(name + ',')

        print(api + '_' + base_name + '};')
        print('#endif')

    return includes


def main():
    if len(sys.argv) < 2:
        print("Usage: {} <kernel_header.h>".format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    files = [sys.argv[1]]
    api = 'hip'

    print('#pragma once')
    print('namespace ' + api + ' {')

    includes = 'static const char* '
    for source_file in files:
        includes = gen_args(source_file, api, includes)
    includes += '};'

    print(includes)
    print('}\t//namespace ' + api)


if __name__ == '__main__':
    main()
