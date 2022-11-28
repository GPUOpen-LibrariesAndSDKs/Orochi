rm -rf cache
cd ../UnitTest/bitcodes && ./generate_bitcodes_gfx1102.sh
cd ../../scripts
../dist/bin/Release/Unittest64 --gtest_filter=-*link*:*getErrorString* --gtest_output=xml:../result.xml
