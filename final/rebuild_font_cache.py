import os
from matplotlib import get_cachedir

cache_dir = get_cachedir()
font_cache_file = os.path.join(cache_dir, 'fontlist-v330.json')

print(f"Clearing: {font_cache_file}")
if os.path.exists(font_cache_file):
    os.remove(font_cache_file)
    print("Font cache removed.")
else:
    print("No font cache found.")
