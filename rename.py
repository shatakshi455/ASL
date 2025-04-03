import os

def rename_images(folder_path):
    count = 1  # Start numbering from 1
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(old_path):
            name, ext = os.path.splitext(filename)

            # Ensure there are at least 6 characters in the name for safe manipulation
            if len(name) > 5:
                # First five characters remain the same
                first_part = name[:5]
                # Add underscore after the 6th character
                second_part = name[5:]
                # Add the number starting from 1
                new_name = f"{first_part}_{second_part[:1]}{count}{ext}"
                
                new_path = os.path.join(folder_path, new_name)
                
                os.rename(old_path, new_path)
                print(f'Renamed: {filename} -> {new_name}')
                
                # Increment the counter
                count += 1

# Change this to the path of your folder
folder_path = "P/"
rename_images(folder_path)
