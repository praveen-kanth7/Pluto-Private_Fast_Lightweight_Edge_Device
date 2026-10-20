idf_component_register(
    SRCS "main.cpp"
         "audio/audio_capture.cpp"
         "dsp/fft480.cpp"
         "dsp/mel_filterbank.cpp"
         "dsp/mfcc_processor.cpp"
         "ml/wake_word_model.cpp"
         "ml/inference_engine.cpp"
         "app/wake_word_detector.cpp"
         "diagnostics/pipeline_diagnostic.cpp"
    INCLUDE_DIRS "."
    EMBED_FILES "ds_cnn_int8.tflite"
)
