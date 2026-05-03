#!/usr/bin/env python3
"""FirstPass v2 — Single launch script"""
import os, sys, subprocess, webbrowser, time, threading

def install(pkgs):
    subprocess.run([sys.executable,'-m','pip','install','--break-system-packages','-q']+pkgs)

def check_deps():
    missing=[]
    for pkg in ['xgboost','sklearn','pandas','numpy','openpyxl']:
        try: __import__(pkg.replace('-','_'))
        except ImportError: missing.append(pkg)
    if missing: print(f'Installing: {missing}'); install(missing)

def open_browser():
    time.sleep(2.5); webbrowser.open('http://localhost:8765')

if __name__=='__main__':
    print('='*56)
    print('  FirstPass v2 — RFT Intelligence')
    print('='*56)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check_deps()
    if not os.path.exists('model.pkl'):
        print('\nFirst-time model training (~60s)…')
        import model as m; m.train_and_save()
    print('\n✅  Model ready')
    print('🚀  http://localhost:8765\n')
    threading.Thread(target=open_browser,daemon=True).start()
    import server; server.run()
