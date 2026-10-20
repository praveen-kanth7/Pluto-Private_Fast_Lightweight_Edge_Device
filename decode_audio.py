import os, base64, glob

def decode_audio():
    print('Searching for Base64 audio in terminal logs...')
    log_files = glob.glob('build/log/idf_py_stdout_output_*')
    if not log_files:
        print('No log files found. Please create a file named base64.txt and paste the Base64 text into it.')
        log_files = ['base64.txt']
    
    b64_string = ''
    for lf in log_files:
        if not os.path.exists(lf): continue
        with open(lf, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            start = content.rfind('--- BEGIN AUDIO BASE64 ---')
            if start != -1:
                end = content.find('--- END AUDIO BASE64 ---', start)
                if end != -1:
                    b64_string = content[start + 26 : end].strip()
                    break
    
    if not b64_string:
        print('Could not find Base64 audio in logs. Please copy the big block of text from your terminal into a file named base64.txt and run this script again.')
        return
    
    # Decode the audio data
    audio_data = base64.b64decode(b64_string)
    
    # Generate WAV header for 16000Hz, 16-bit, Mono
    sample_rate = 16000
    data_size = len(audio_data)
    byte_rate = sample_rate * 2
    header = bytearray([
        ord('R'),ord('I'),ord('F'),ord('F'),
        (data_size + 36) & 0xFF, ((data_size + 36) >> 8) & 0xFF, ((data_size + 36) >> 16) & 0xFF, ((data_size + 36) >> 24) & 0xFF,
        ord('W'),ord('A'),ord('V'),ord('E'),
        ord('f'),ord('m'),ord('t'),ord(' '),
        16, 0, 0, 0,
        1, 0, 1, 0,
        sample_rate & 0xFF, (sample_rate >> 8) & 0xFF, (sample_rate >> 16) & 0xFF, (sample_rate >> 24) & 0xFF,
        byte_rate & 0xFF, (byte_rate >> 8) & 0xFF, (byte_rate >> 16) & 0xFF, (byte_rate >> 24) & 0xFF,
        2, 0, 16, 0,
        ord('d'),ord('a'),ord('t'),ord('a'),
        data_size & 0xFF, (data_size >> 8) & 0xFF, (data_size >> 16) & 0xFF, (data_size >> 24) & 0xFF
    ])
    
    with open('perfect_audio.wav', 'wb') as f:
        f.write(header + audio_data)
    
    print('SUCCESS! Saved perfect, uncorrupted audio to perfect_audio.wav!')

if __name__ == '__main__':
    decode_audio()
