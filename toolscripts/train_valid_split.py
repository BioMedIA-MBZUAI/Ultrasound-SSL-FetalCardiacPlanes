import os, glob
import random
import shutil
from tqdm.auto import tqdm


def print_info(string, writepath):
    print(string)
    print(string, file=open(writepath, "a"))


def split_image_folder(source_root, target_root, portion=2, ratio= [0.8, 0.2]):
    random.seed(73)
    assert len(ratio) == portion
    assert sum(ratio) == 1
    os.makedirs(target_root, exist_ok = True)
    class_folder = [n for n in os.listdir(source_root)]

    for cf in class_folder:
        ipath = os.path.join(source_root, cf, "*.png")
        imnames = [os.path.basename(x) for x in glob.glob(ipath)]
        random.shuffle(imnames)
        lsize = [round(len(imnames)*r) for r in ratio ] ## floor will be cleaner
        begin = 0; outnames = []
        # print(lsize)
        for i in range(portion):
            outnames.append(imnames[begin:begin+lsize[i]])
            begin += lsize[i]
        ## handle leftouts due to rounding
        for j in range(begin, len(imnames)):
            outnames[j%portion].append(imnames[j])
        print_info(f"ClassFolder: {cf}, Count: {len(imnames)}", target_root+"/split-info.txt")
        ipath_dir =  os.path.dirname(ipath)
        for o, grp in tqdm(enumerate(outnames)):
            print_info(f'Group-{o+1}:',len(grp), target_root+"/split-info.txt")

            opath = os.path.join(target_root, f"group{o+1}",cf)
            os.makedirs(opath, exist_ok = True)
            for g in grp:
                shutil.copy(os.path.join(ipath_dir, g), opath)

    return None



def split_image_folder_by_nameid(source_root, target_root,
                                    ratio= [0.8, 0.2] ):
    """
    Written just for two folder logic - train, valid

    """

    def parse_study_id(string):
        str_list = [ x for x in string.split("-") if "Study" in x ]
        return str_list[0]

    def grouper(name_list):
        out_dic = {}
        for name in name_list:
            id = parse_study_id(name)
            if id in out_dic:
                out_dic[id]["count"] += 1
                out_dic[id]["files"].append(name)
            else:
                out_dic[id] = {}
                out_dic[id]["count"] = 1
                out_dic[id]["files"] = [name]
        out_dic = dict(sorted(out_dic.items(), key = lambda item: item[1]["count"],
                                reverse=True ))
        return out_dic

    random.seed(73)
    assert sum(ratio) == 1
    assert len(ratio) <= 2

    os.makedirs(target_root, exist_ok = True)
    class_folder = [n for n in os.listdir(source_root)]


    for cf in class_folder:
        ipath = os.path.join(source_root, cf, "*.png")
        imnames = [os.path.basename(x) for x in glob.glob(ipath)]
        total_count = len(imnames)
        grouped_dic = grouper(imnames)

        ##TODO: rewrite logic for N splits when life bestows time
        curr_count = 0
        valid_dic = {}
        while curr_count < total_count * ratio[-1]:
            k = random.sample(grouped_dic.keys(), 1)[0]
            item = grouped_dic.pop(k)
            valid_dic[k] = item
            curr_count += item["count"]
        print_info(f'Class:{cf},Tot:{total_count} Train:{total_count - curr_count} Valid:{curr_count}',
                    target_root+"/split-info.txt")

        ipath_dir =  os.path.dirname(ipath)
        for o, dic in enumerate([grouped_dic, valid_dic]):
            for k in tqdm(dic.keys()):
                grp = dic[k]["files"]
                opath = os.path.join(target_root, f"group{o+1}",cf)
                os.makedirs(opath, exist_ok = True)
                for g in grp:
                    shutil.copy(os.path.join(ipath_dir, g), opath)

    return None




if __name__ == "__main__":

    split_image_folder_by_nameid(
        "/home/joseph.benjamin/WERK/UltraSound/FetalDataImg",
        "/home/joseph.benjamin/WERK/data/study-split/",
        ratio=[0.8, 0.2]
    )




