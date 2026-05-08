import os
import tarfile
import zipfile


def is_zip_path(img_or_path):
    """judge if this is a zip path"""
    return '.zip@' in img_or_path


class ZipReader(object):
    """A class to read zipped files"""
    zip_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path not in zip_bank:
            if path.endswith(('.tar.gz', '.tgz')):
                zfile = tarfile.open(path, 'r:gz')
            else:
                zfile = zipfile.ZipFile(path, 'r')
            zip_bank[path] = zfile
        return zip_bank[path]

    @staticmethod
    def normalize_member_path(path_img):
        if path_img.startswith('sentence_frames-512x512/'):
            return 'frames_512x512/' + path_img[len('sentence_frames-512x512/'):]
        return path_img

    @staticmethod
    def split_zip_style_path(path):
        pos_at = path.index('@')
        assert pos_at != -1, "character '@' is not found from the given path '%s'" % path

        zip_path = path[0: pos_at]
        folder_path = path[pos_at + 1:]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path

    @staticmethod
    def is_directory_root(path):
        return os.path.isdir(path)

    @staticmethod
    def list_folder(path):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        if ZipReader.is_directory_root(zip_path):
            folder_dir = os.path.join(zip_path, folder_path)
            if not os.path.isdir(folder_dir):
                return []
            return sorted(
                name for name in os.listdir(folder_dir)
                if os.path.isdir(os.path.join(folder_dir, name))
            )

        zfile = ZipReader.get_zipfile(zip_path)
        folder_path = ZipReader.normalize_member_path(folder_path)
        folder_list = []
        names = zfile.getnames() if hasattr(zfile, 'getnames') else zfile.namelist()
        for file_folder_name in names:
            file_folder_name = str.strip(file_folder_name, '/')
            if file_folder_name.startswith(folder_path) and \
               len(os.path.splitext(file_folder_name)[-1]) == 0 and \
               file_folder_name != folder_path:
                if len(folder_path) == 0:
                    folder_list.append(file_folder_name)
                else:
                    folder_list.append(file_folder_name[len(folder_path)+1:])

        return folder_list

    @staticmethod
    def list_files(path, extension=None):
        if extension is None:
            extension = ['.*']
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        if ZipReader.is_directory_root(zip_path):
            folder_dir = os.path.join(zip_path, folder_path)
            if not os.path.isdir(folder_dir):
                return []
            extension = {ext.lower() for ext in extension}
            return sorted(
                name for name in os.listdir(folder_dir)
                if os.path.isfile(os.path.join(folder_dir, name)) and (
                    '.*' in extension or os.path.splitext(name)[-1].lower() in extension
                )
            )

        zfile = ZipReader.get_zipfile(zip_path)
        folder_path = ZipReader.normalize_member_path(folder_path)
        file_lists = []
        names = zfile.getnames() if hasattr(zfile, 'getnames') else zfile.namelist()
        for file_folder_name in names:
            file_folder_name = str.strip(file_folder_name, '/')
            if file_folder_name.startswith(folder_path) and \
                    str.lower(os.path.splitext(file_folder_name)[-1]) in extension:
                if len(folder_path) == 0:
                    file_lists.append(file_folder_name)
                else:
                    file_lists.append(file_folder_name[len(folder_path)+1:])

        return file_lists

    @staticmethod
    def read(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)

        if ZipReader.is_directory_root(zip_path):
            with open(os.path.join(zip_path, path_img), 'rb') as handle:
                return handle.read()

        zfile = ZipReader.get_zipfile(zip_path)
        path_img = ZipReader.normalize_member_path(path_img)
        if hasattr(zfile, 'extractfile'):
            handle = zfile.extractfile(path_img)
            if handle is None:
                raise KeyError(path_img)
            data = handle.read()
        else:
            data = zfile.read(path_img)
        return data