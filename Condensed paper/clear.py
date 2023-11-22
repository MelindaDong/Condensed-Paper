# clear the content of other_images folder
import os

def clear_images(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return
    
    # if the folder is empty
    if len(os.listdir(folder_path)) == 0:
        print(f"The folder '{folder_path}' is empty.")
        return

    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if it is a file
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)
            #print(f"Removed: {file_path}")

    print(f"All images in '{folder_path}' have been removed.")


#------------------------------------------------------------
# clear the images in other_images folder and math_images folder
clear_images('other_images')
clear_images('math_images')

# remove image_discription_df.csv
if os.path.exists('image_discription_df.csv'):
    os.remove('image_discription_df.csv')

if os.path.exists('math_ocr_text.txt'):
    os.remove('math_ocr_text.txt')


print("ready for the next run.")
