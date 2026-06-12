from faster_whisper import WhisperModel
m = WhisperModel('base', device='cpu', compute_type='int8')
print('Model loaded OK')

