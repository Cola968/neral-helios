"""Protect server-side JSON with the current Windows user's DPAPI key.

Legacy JSON stays readable. Other platforms rely on filesystem protection.
Browser storage is separate and is not encrypted by this module.
"""
import base64
import ctypes
import json
import os

def _protect(raw, decrypt=False):
    from ctypes import wintypes
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(raw)
    source = Blob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    fn = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    fn.restype = wintypes.BOOL
    if not fn(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise OSError('Lokaler Speicher konnte nicht entschlüsselt/verschlüsselt werden.')
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(ctypes.cast(target.data, ctypes.c_void_p))

def dumps(value, **kwargs):
    text = json.dumps(value, **kwargs)
    if os.name == 'nt':
        return 'dpapi1:' + base64.b64encode(_protect(text.encode('utf-8'))).decode('ascii')
    return text

def loads(text):
    if text.startswith('dpapi1:'):
        if os.name != 'nt':
            raise ValueError('Dieser Speicher gehört zu einem Windows-Benutzer.')
        text = _protect(base64.b64decode(text[7:], validate=True), True).decode('utf-8')
    return json.loads(text)
