import os


if __name__ == "__main__":
    data_dir = "./recording_data"
    # rename all the file names in the directory
    for i, file_name in enumerate(os.listdir(data_dir)):
        os.rename(
            os.path.join(data_dir, file_name),
            os.path.join(data_dir, f"record_{i}.pickle")
        )
