import os, glob
import random
import shutil
from tqdm.auto import tqdm

def split_image_folder(source_root, target_root, portion=2, ratio= [0.8, 0.2]):
    random.seed(73)
    assert len(ratio) == portion
    assert sum(ratio) == 1
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
        print(f"ClassFolder: {cf}, Count: {len(imnames)}")
        ipath_dir =  os.path.dirname(ipath)
        for o, grp in tqdm(enumerate(outnames)):
            print(f'Group-{o+1}:',len(grp))
            opath = os.path.join(target_root, f"group{o+1}",cf)
            os.makedirs(opath, exist_ok = True)
            for g in grp:
                shutil.copy(os.path.join(ipath_dir, g), opath)

    return None








split_image_folder(
    "/home/joseph.benjamin/WERK/UltraSound/FetalDataImg",
    "/home/joseph.benjamin/WERK/data/fetal-split",
    portion=2,
    ratio=[0.8, 0.2]
)




