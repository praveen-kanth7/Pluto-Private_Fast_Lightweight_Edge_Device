idf_component_register(
    SRCS "main.cpp" "kws_preprocess.cpp"
    INCLUDE_DIRS "."
    EMBED_FILES "ds_cnn_int8.tflite"
)
