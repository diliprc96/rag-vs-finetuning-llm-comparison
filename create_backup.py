import shutil
import os
import datetime

def create_backup():
    # Define what to backup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"rag_vs_finetuning_backup_{timestamp}"
    root_dir = os.getcwd()
    
    # Create a temporary directory for the backup
    backup_dir = os.path.join(root_dir, backup_filename)
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    os.makedirs(backup_dir)
    
    # Folders to copy
    folders = [
        "evaluation",
        "finetuning",
        "data_extraction",
        "rag_pipeline",
        "context",
        "results" # Includes adapter (approx 200MB)
    ]
    
    print(f"Backing up to {backup_filename}...")
    
    for folder in folders:
        src = os.path.join(root_dir, folder)
        dst = os.path.join(backup_dir, folder)
        if os.path.exists(src):
            print(f"Copying {folder}...")
            shutil.copytree(src, dst)
        else:
            print(f"Warning: {folder} not found.")

    # Files to copy
    files = [
        "requirements.txt",
        ".env",
        "README.md",
        "analyze_dataset.py"
    ]
    
    for file in files:
        src = os.path.join(root_dir, file)
        if os.path.exists(src):
            print(f"Copying {file}...")
            shutil.copy(src, backup_dir)
            
    # Zip it
    print("Creating zip archive...")
    shutil.make_archive(backup_filename, 'zip', root_dir, backup_filename)
    
    # Cleanup temp dir
    shutil.rmtree(backup_dir)
    
    print(f"Backup created successfully: {os.path.join(root_dir, backup_filename + '.zip')}")

if __name__ == "__main__":
    create_backup()
