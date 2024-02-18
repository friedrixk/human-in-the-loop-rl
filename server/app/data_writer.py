import csv
import datetime
import os


def get_dir_path():
    # Get the absolute path of the script
    script_path = os.path.abspath(__file__)
    # Get the parent directory of the script
    parent_directory = os.path.dirname(script_path)
    # Directory path where you want to create the CSV file
    directory_path = os.path.join(parent_directory, "..")
    dir_path = os.path.join(directory_path, "user_data")
    return dir_path


def write_amount_feedback_frames(data, filename):
    dir_path = get_dir_path()
    path = os.path.join(dir_path, filename)
    # if the file does not exist, abort:
    if not os.path.exists(path):
        print("File does not exist: " + path)
        return
    else:
        # timestamp with for logging
        time = datetime.datetime.now().strftime("%H:%M:%S")
        data.append(time)
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)


def write_user_data_feedback(data, filename):
    """
    Method that writes data to a CSV file. The filename includes a time stamp to identify the data.

    :param data: data that is written to the CSV file.
    """
    dir_path = get_dir_path()
    path = os.path.join(dir_path, filename)
    # if the file does not exist, abort:
    if not os.path.exists(path):
        print("File does not exist: " + path)
        return
    else:
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)


def write_user_data_interactions(data, filename):
    """
    Method that writes data to a CSV file. The filename includes a time stamp to identify the data.

    :param data: data that is written to the CSV file.
    """
    dir_path = get_dir_path()
    path = os.path.join(dir_path, filename)
    # if the file does not exist, abort:
    if not os.path.exists(path):
        print("File does not exist: " + path)
        return
    else:
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)


def generate_feedback_file():
    """
    Method that generates a .csv file name with a time stamp.
    @return:
    """
    dir_path = get_dir_path()
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "user_data_feedback_" + time_stamp + ".csv"
    path = os.path.join(dir_path, filename)
    header = ["time", "proc_id", "episode_id", 'buffer_id', 'type', "feedback", 'mean reward', 'mean loss', 'entropy',
              'buffer_counter']
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
    return filename


def generate_interactions_file():
    """
    Method that generates a .csv file name with a time stamp.
    @return:
    """
    dir_path = get_dir_path()
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "user_data_interactions_" + time_stamp + ".csv"
    path = os.path.join(dir_path, filename)
    header = ["time", "component", "type", "e_idx", "p_idx", "step_idx", "buffer_idx", "details"]
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
    return filename
