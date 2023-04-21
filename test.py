import os
import shutil

# source_dir = "/media/group1/data/yangchu/EventSOT/EventSOT_2.1_annoting"
# target_dir = "/home/test4/code/EventBenchmark/lib/pytracking/data/EventSOT2/aedat4"

# Recursively iterate through the source directory
# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         if file.endswith(".aedat4"):
#             # Construct the full path of the source file and the target file
#             source_file = os.path.join(root, file)
#             target_file = os.path.join(target_dir, file)
#             # Copy the file to the target directory
#             shutil.copy(source_file, target_file)


source_dir = "/media/group1/data/yangchu/EventSOT/EventSOT_2.1_annoting"
target_dir = "/home/test4/code/EventBenchmark/lib/pytracking/data/EventSOT2/default"

sequences = os.listdir(target_dir)
for seq in sequences:

    # Construct the target directory path
    target_subdir = os.path.join(target_dir, seq)

    if not os.path.exists(target_subdir):
        raise
        os.makedirs(target_subdir)

    # Copy the "img" directory to the target directory's subdirectory
    source_img_dir = os.path.join(source_dir,seq, "img")
    target_img_dir = os.path.join(target_subdir, "img")
    shutil.copytree(source_img_dir, target_img_dir)