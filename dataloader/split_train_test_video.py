from __future__ import print_function
import os
import glob
import pickle


class UCF101_splitter():
    def __init__(self, path, split):
        self.path = path
        self.split = split

    def get_action_index(self):
        self.action_label = {}
        with open(os.path.join(self.path, 'classInd.txt')) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label, action = line.split(' ')
            if action not in self.action_label.keys():
                self.action_label[action] = label

    def split_video(self):
        self.get_action_index()
        for path, subdir, files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist' + self.split:
                    train_video = self.file2_dic(self.path + filename)
                if filename.split('.')[0] == 'testlist' + self.split:
                    test_video = self.file2_dic(self.path + filename)
        print('==> (Training video, Validation video):(', len(train_video),
              len(test_video), ')')
        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)

        return self.train_video, self.test_video

    def file2_dic(self, fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic = {}
        for line in content:
            video = line.split('/', 1)[1].split(' ', 1)[0]
            key = video.split('_', 1)[1].split('.', 1)[0]
            label = self.action_label[line.split('/')[0]]
            dic[key] = int(label)
        return dic

    def name_HandstandPushups(self, dic):
        dic2 = {}
        for video in dic:
            n, g = video.split('_', 1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_' + g
            else:
                videoname = video
            dic2[videoname] = dic[video]
        return dic2


class HmdbSplitter(object):
    """Helper class to split the HMDB video files according to a split.

    Since processed flow from two-stream repo discards class folders, have
    to regenerate this from the split files.
    """

    def __init__(self, split_path, split):
        self.split_path = split_path
        self.split = split
        self.action_label = self.make_action_label(split_path)

    def _filename_to_class(self, filename):
        """Helper to extract class from filename"""
        return filename.split("_test_")[0]

    def make_action_label(self, split_path):
        """Dictionary from action name to label."""
        classes = set(
            [self._filename_to_class(file) for file in os.listdir(split_path)])
        classes = sorted(list(classes))
        return {category: i for i, category in enumerate(classes)}

    def split_video(self):
        """Build up a map of paths to class label used by this split for train / val"""
        train_videos = {}
        test_videos = {}

        # List of split files per category
        split_files = glob.glob(
            os.path.join(self.split_path, "*{}*".format(self.split)))

        # Process each category's split file
        for category_file in split_files:
            label = self.action_label[self._filename_to_class(
                category_file.split('/')[-1])]
            with open(category_file, 'r') as f:
                # 0 -> not included, 1 -> training, 2 -> test
                for line in f:
                    parts = line.split()
                    video = parts[0].split(".avi")[0]
                    split_type = int(parts[1])
                    if split_type == 1:
                        train_videos[video] = label
                    elif split_type == 2:
                        test_videos[video] = label

        print('==> (Training video, Validation video):(', len(train_videos),
              len(test_videos), ')')

        return train_videos, test_videos


if __name__ == '__main__':
    path = '/vision/vision_users/azou/data/hmdb51-splits/'
    split = '1'
    splitter = HmdbSplitter(path, split)
    train_video, test_video = splitter.split_video()
    print(len(train_video), len(test_video))
