import os

def extract_wav():
    if not os.path.exists('spiffs.bin'):
        print('Error: spiffs.bin not found!')
        return
    with open('spiffs.bin', 'rb') as f:
        data = f.read()
    
    # Use rfind to find the LAST written WAV file instead of a ghost/deleted one!
    start_idx = data.rfind(b'RIFF')
    if start_idx == -1:
        print('Error: Could not find WAV file in SPIFFS image.')
        return
    
    wav_data = data[start_idx : start_idx + 32044]
    with open('mic_test.wav', 'wb') as f:
        f.write(wav_data)
    print('SUCCESS! Extracted latest 1-second trigger audio to mic_test.wav.')

if __name__ == '__main__':
    extract_wav()
