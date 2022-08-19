#!/usr/bin/env python
from __future__ import print_function

import sys
import os

def genArgs( fileName, api, includes ):
    with open(fileName) as f:
        iName = os.path.basename( fileName ).split('.')[0]
        
        print( '#if !defined(ORO_PP_LOAD_FROM_STRING)' )
        print( '	static const char** '+iName+'Args = 0;' )
        print( '#else' )
        print( '	static const char* '+iName+'Args[] = {' )
        includes += iName +'Includes[] = {'
        for line in f.readlines():
            a = line.strip('\r\n')
            if a.find('#include') == -1:
                continue
            if a.find('#include') != -1 and a.find('inl.' + api) != -1:
                continue
            if (api == 'cl' or api == 'metal') and a.find('.cu') != -1:
                continue
            if (a.find('"') != -1 and a.find('#include') != -1):
                continue

            filename = os.path.basename(a.split('<')[1].split('>')[0])
            includes += '"' + a.split('<')[1].split('>')[0] + '",'
            name = filename.split('.' + api)[0]
            name = name.split('.h')[0]
            name = api + '_'+name
            print ( name + ',' )
        print( api + '_'+iName+'};' )
        print( '#endif' )
        return includes

argvs = sys.argv

files = []
if len(argvs) >= 2:
    files.append( argvs[1] )

print( '#pragma once' )


api = 'hip'

# Visit each file
print( 'namespace ' + api + ' {')

includes = 'static const char* '
for s in files:
    includes = genArgs(s, api, includes)
includes += '};'
print( includes )
print( '}\t//namespace ' + api)
