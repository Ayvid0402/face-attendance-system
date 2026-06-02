# check_setup.py - verify all libraries are installed correctly

libraries = [
    "cv2",
    "face_recognition",
    "numpy",
    "pandas",
    "PIL",
    "streamlit",
    "dlib",
]

print("Checking all libraries...\n")
all_good = True

for lib in libraries:
    try:
        __import__(lib)
        print(f"  installed - {lib}")
    except ImportError:
        print(f"  NOT installed - {lib}")
        all_good = False

print()
if all_good:
    print("All libraries ready! Lets build the system!")
else:
    print("Some libraries missing!")