import os
import glob

def remove_files(file_extensions):
    current_dir = os.getcwd()
    
    for ext in file_extensions:
        # 使用 glob 来查找所有匹配的文件
        files = glob.glob(f"{current_dir}/**/*{ext}", recursive=True)
        
        for file in files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

if __name__ == "__main__":
    extensions_to_remove = ['.csv', '.png', '.svg']
    remove_files(extensions_to_remove)
    print("File removal process completed.")
