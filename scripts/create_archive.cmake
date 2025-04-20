
# create_archive.cmake
# Create a raw Zstd-compressed "archive" from a single file.

# Variables expected:
#   INPUT_FILE    – path to the file to compress
#   OUTPUT_FILE   – path to the compressed file to generate
#   DO_COMPRESS: ON/OFF


if(DO_COMPRESS)
	file(ARCHIVE_CREATE
		OUTPUT            "${OUTPUT_FILE}"
		PATHS             "${INPUT_FILE}"
		FORMAT            raw
		COMPRESSION       Zstd
		COMPRESSION_LEVEL 19
	)
endif()



