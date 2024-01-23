'''
for folder in train and test
    for files/folders in folder
        remove MaterialsAndParts folder and contents
        remove Vessels and contents
'''
# import os
# import shutil

# folders_to_remove = ['MaterialsAndParts', 'Vessels']

# for split in ['Train', 'Test']:
#     if not os.path.exists(split):
#         print(f"{split} folder does not exist.")
#         continue

#     for root, dirs, files in os.walk(split):
#         for folder in dirs:
#             if folder in folders_to_remove:
#                 folder_to_remove = os.path.join(root, folder)
#                 shutil.rmtree(folder_to_remove)
#                 print(f"Removed {folder_to_remove}")

# categories = {}
# for each folder in Train or Test
#   for each file in SemanticMaps\FullImage
#       if file in categories.keys()
#           categories[file] += 1
#       else
#           categories[file] = 0
