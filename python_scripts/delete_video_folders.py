import os
import argparse
import shutil


def delete_videos_folders(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            if dir_name == "videos":
                videos_folder_path = os.path.join(root, dir_name)
                print("Deleting folder:", videos_folder_path)
                shutil.rmtree(videos_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()
    # 调用函数，指定要删除的文件夹路径
    delete_videos_folders(args.folder)
