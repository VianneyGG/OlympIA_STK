import os
import sys
import ctypes
import site
from pathlib import Path

ENV_BIN = r"C:\Users\LeV\miniconda3\envs\OlympIA_STK_py311\Library\bin"
STK_BIN = r"C:\Program Files\SuperTuxKart 1.4"

print("Python:", sys.version)
print("Adding DLL directories:")
for p in [ENV_BIN, STK_BIN]:
    if os.path.isdir(p):
        os.add_dll_directory(p)
        print(" -", p)
    else:
        print(" - MISSING:", p)

deps = [
    'libzlib.dll','libpng16.dll','libcurl.dll','libmbedcrypto.dll',
    'libjpeg-62.dll','libfreetype.dll','libharfbuzz.dll','libastcenc.dll','SDL2.dll'
]

print("\nChecking dependent DLLs:")
for d in deps:
    try:
        ctypes.CDLL(d)
        print(f" OK  {d}")
    except OSError as e:
        print(f" ERR {d}: {e}")

print("\nAttempting to load pystk2 extension explicitly:")
sp = [p for p in sys.path if p.endswith('site-packages')]
sp = sp[0] if sp else site.getsitepackages()[1]
pyd = Path(sp) / 'pystk2.cp311-win_amd64.pyd'
print("pyd path:", pyd)
try:
    ctypes.CDLL(str(pyd))
    print(" pystk2.pyd loaded OK via ctypes")
except OSError as e:
    print(" pystk2.pyd load error:", e)

print("\nNow try regular import:")
try:
    import pystk2  # noqa: F401
    print(" import pystk2 OK")
except Exception as e:
    print(" import pystk2 FAILED:", repr(e))
