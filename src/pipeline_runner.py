import subprocess
import sys

scripts = [
    'src/phase2_extract.py',
    'src/convert_to_chunks.py', 
    'src/embeding.py',
    'src/store_to_vector.py',
]
print("="*50)
print("BookRDR - Add New Books Pipeline")
print("="*50)

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"FAILED at {script} — stopping pipeline")
        break
    
    print(f"✅ {script} complete")

print("\n" + "="*50)
print("\nFull pipeline done!")
print("="* 50)

# Orchestrator Script