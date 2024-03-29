rm -rf cache
cd ../UnitTest/bitcodes
./generate_bitcodes.sh
cd ../../scripts
../dist/bin/Release/Unittest64 --gtest_filter=-*link_bundledBc* --gtest_output=xml:../result.xml
