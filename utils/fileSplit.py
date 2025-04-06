import platform

def fileSplit(filePath):
    """
    Splits a file path into a list of components based on the operating system.

    Input:
    - filePath (str): The file path to split.

    Output:
    - list: A list of path components.

    Functionality:
    - Splits the input file path into components using the appropriate separator
      for the current operating system (Windows or Linux).
    """
    if platform.system().lower() == 'windows':
        fileSplit = filePath.split("\\", maxsplit=-1)
    if platform.system().lower() == 'linux':
        fileSplit = filePath.split(r"/", maxsplit=-1)
    return fileSplit

if __name__ == '__main__':
    path = "G:\Files\DeepFlow\DeepFlow_algorithm_pytorch\data"
    fileSplit(path)