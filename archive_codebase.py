#!/usr/bin/env python3
"""
Script to archive the entire codebase into a zip file and
then extract all text into a single file for backup/sharing purposes.
"""

import os
import sys
import zipfile
import shutil
import glob
import datetime

def create_archive_and_extract_text(source_dir, exclude_dirs=None, exclude_extensions=None):
    """
    Create a zip archive of the codebase and extract all text content.
    
    Args:
        source_dir: Directory to archive
        exclude_dirs: List of directories to exclude (relative to source_dir)
        exclude_extensions: List of file extensions to exclude (.zip, .pyc, etc)
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'venv', '.ipynb_checkpoints', 'node_modules']
    
    if exclude_extensions is None:
        exclude_extensions = ['.zip', '.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll', 
                             '.exe', '.bin', '.dat', '.pkl', '.joblib', '.model', 
                             '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff',
                             '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac']
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output file paths
    zip_path = os.path.join(source_dir, f"codebase_archive_{timestamp}.zip")
    txt_path = os.path.join(source_dir, f"codebase_text_{timestamp}.txt")
    
    print(f"Starting archive process...")
    print(f"Source directory: {source_dir}")
    print(f"Output zip file: {zip_path}")
    print(f"Output text file: {txt_path}")
    
    # Create zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through directory
        for root, dirs, files in os.walk(source_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not any(os.path.join(root, d).startswith(os.path.join(source_dir, ex_dir)) for ex_dir in exclude_dirs)]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip the zip file itself and the output txt file
                if file_path == zip_path or file_path == txt_path:
                    continue
                
                # Skip excluded extensions
                if any(file.endswith(ext) for ext in exclude_extensions):
                    continue
                
                # Add file to zip
                rel_path = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, rel_path)
                print(f"Added: {rel_path}")
    
    print(f"\nZip archive created: {zip_path}")
    
    # Now extract text content from all files in the zip
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(f"CODEBASE TEXT EXTRACTION - {timestamp}\n")
        txt_file.write("="*80 + "\n\n")
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            # Get list of file names in the zip
            file_list = zipf.namelist()
            file_list.sort()  # Sort for more organized output
            
            for file_name in file_list:
                try:
                    # Skip directories
                    if file_name.endswith('/'):
                        continue
                    
                    # Extract file content
                    content = zipf.read(file_name)
                    
                    # Try to decode as text
                    try:
                        text_content = content.decode('utf-8')
                        
                        # Write file header and content to txt file
                        txt_file.write(f"\n\n{'='*40}\n")
                        txt_file.write(f"FILE: {file_name}\n")
                        txt_file.write(f"{'='*40}\n\n")
                        txt_file.write(text_content)
                        txt_file.write("\n")
                        
                        print(f"Extracted text from: {file_name}")
                    except UnicodeDecodeError:
                        txt_file.write(f"\n\n{'='*40}\n")
                        txt_file.write(f"FILE: {file_name} (BINARY - CONTENT NOT SHOWN)\n")
                        txt_file.write(f"{'='*40}\n\n")
                        print(f"Skipped binary file: {file_name}")
                        
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    txt_file.write(f"\n\nError processing {file_name}: {e}\n")
    
    print(f"\nText extraction complete!")
    print(f"Text file created: {txt_path}")
    print(f"Zip archive: {zip_path}")
    
    # Return paths for reference
    return zip_path, txt_path

if __name__ == "__main__":
    # Get source directory (default to current directory)
    if len(sys.argv) > 1:
        source_directory = sys.argv[1]
    else:
        source_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Create archive and extract text
    zip_file, txt_file = create_archive_and_extract_text(source_directory)
    
    print(f"\nProcess complete!")
    print(f"Archive saved to: {zip_file}")
    print(f"Text extraction saved to: {txt_file}")
