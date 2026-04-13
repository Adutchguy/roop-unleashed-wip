"""
setup_3drecon.py
----------------
Verifies that the dependencies for the "🧊 3D source pose matching" feature
are available.  No model downloads are required — this feature uses only
the 3D landmarks that insightface already provides.

Run from your app/ directory:

    python setup_3drecon.py

What this checks
----------------
- scipy     (for rotation math — likely already installed)
- opencv    (for solvePnP / warpAffine — already installed)
- numpy     (already installed)
- insightface landmark_3d_68 availability (enabled by default in globals.py)

If all checks pass you can immediately enable
"🧊 3D source pose matching" in the UI settings panel.
"""

import sys
import importlib

OK = True

for pkg, import_name in [("numpy", "numpy"), ("cv2", "cv2"), ("scipy", "scipy")]:
    try:
        importlib.import_module(import_name)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg} — installing …")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        OK = True   # try again implicitly next run

if OK:
    print("\n✅ All dependencies satisfied.")
    print("   Enable '🧊 3D source pose matching' in the UI settings panel.")
    print("   No model downloads required — uses insightface landmarks only.")
