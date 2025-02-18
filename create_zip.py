import zipfile
import os

def create_project_zip():
    # Files to include
    files_to_zip = [
        'testarcade.py',
        'shaders/compute_shader.glsl',
        'requirements.md'
    ]
    
    # Create zip file
    with zipfile.ZipFile('project.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            if os.path.exists(file):
                zipf.write(file)
            else:
                print(f"Warning: {file} not found")

if __name__ == "__main__":
    create_project_zip()
    print("project.zip created successfully") 