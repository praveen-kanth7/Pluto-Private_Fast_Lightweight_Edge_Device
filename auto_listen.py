import serial, sys, base64, time

def listen():
    print('Connecting to ESP32 on COM13...')
    try:
        ser = serial.Serial('COM13', 115200, timeout=1)
    except Exception as e:
        print(f'Error opening COM13: {e}')
        return
    print('Connected! Waiting for you to say 'Hey Pluto'...')
    
    recording = False
    b64_string = ''
    
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            print(f'ESP32: {line}')
            
            if '--- BEGIN AUDIO BASE64 ---' in line:
                print('Detected trigger! Downloading audio...')
                recording = True
                continue
            
            if '--- END AUDIO BASE64 ---' in line:
                print('Audio download complete! Decoding...')
                break
                
            if recording:
                b64_string += line
                
        except KeyboardInterrupt:
            print('Stopped.')
            break
    
    ser.close()
    
    if not b64_string:
        return
        
    audio_data = base64.b64decode(b64_string)
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
    print('\\n=========================================')
    print('SUCCESS! Audio saved to perfect_audio.wav')
    print('=========================================')

if __name__ == '__main__':
    listen()
