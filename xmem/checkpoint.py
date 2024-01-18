import os

# https://drive.google.com/drive/folders/1QYsog7zNzcxGXTGBzEhMUg8QVJwZB6D1
# FILE_IDS = {
#     'XMem': '11vNugAezixkE7Ng3Ok43oUxeNAjMd7OK',
#     'XMem-s012': '1ZClOHqhvWtoPipQy_dCvj9cAIhzqsRKz',
#     'XMem-s2': '19vxZb262riEYHLGA6tt98O9yXnRbdclS',
#     'XMem-s01': '1i71itBQ0Ip7HRJYxNdrngsnlTOk57M5U',
#     'XMem-s0': '1PjQ3jZbUmu5wXWj47rp1BfIJ8fKoaG-U',
#     'XMem-no-sensory': '1F4PpXuZMc_VmDL6XgDSpYXJIiUjJVRju',
# }
# BASE_URL = 'https://docs.google.com/uc?export=download&confirm=t&id={}'
FILE_IDS = {
    'XMem': 'XMem.pth',
    'XMem-s012': 'XMem-s012.pth',
    'XMem-s2': 'XMem-s2.pth',
    'XMem-s01': 'XMem-s01.pth',
    'XMem-s0': 'XMem-s0.pth',
    'XMem-no-sensory': 'XMem-no-sensory.pth',
}
BASE_URL = 'https://github.com/beasteers/XMem/releases/download/v1/{}'

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.getenv("MODEL_DIR") or os.path.join(ROOT_DIR, 'saves')

def ensure_checkpoint(key='XMem', path=None, file_id=None):
    file_id = file_id or FILE_IDS[key]
    path = path or os.path.join(MODEL_DIR, f'{key}.pth')
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print("No checkpoint found. Downloading...")
        def show_progress(i, size, total):
            print(f'downloading checkpoint to {path}: {i * size / total:.2%}', end="\r")
        
        import urllib.request
        urllib.request.urlretrieve(BASE_URL.format(file_id), path, show_progress)
    return path

