# -*- mode: python ; coding: utf-8 -*-

import os
import glob

# Securely collect config files, strictly excluding .env secrets
config_files = []
for f in glob.glob('config/*'):
    if not f.endswith('.env') and not f.endswith('.env.example') and os.path.isfile(f):
        config_files.append((f, 'config'))

added_files = [
    ('dashboard/templates', 'dashboard/templates'),
    ('dashboard/static', 'dashboard/static'),
    ('workflows', 'workflows'),
    ('plugins', 'plugins'),
] + config_files

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'sqlite3',
        'torch',
        'transformers',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Fix tesseract bundling by using Tree to maintain directory structure
a.datas += Tree('bin/tesseract', prefix='bin/tesseract')

# Exclude config/.env from the bundle
for item in a.datas.copy():
    if item[0].endswith('.env'):
        a.datas.remove(item)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='jarvis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='jarvis',
)
