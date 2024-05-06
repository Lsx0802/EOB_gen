import os
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
from glob import glob
from scipy.ndimage import zoom
import numpy as np

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  #获取原图size
    originSpacing = itkimage.GetSpacing()  #获取原图spacing
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int64)   #spacing格式转换
    resampler.SetReferenceImage(itkimage)  #指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  #得到重新采样后的图像
    return itkimgResampled
    


DATA_DIR = "/home/sxli/data/"
assert os.path.isdir(DATA_DIR), f"{DATA_DIR} is not a directory."


# def download_and_unzip():
#     zip_dir = os.path.join(DATA_DIR, "database.zip")
#     url = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/637218e573e9f0047faa00fc/download"

#     if not os.path.exists(zip_dir):
#         print("Downloading ACDC dataset... (about 2GB)")
#         os.system(f"curl -L {url} -o {zip_dir}")
#     if not os.path.exists(os.path.join(DATA_DIR, "database/testing/patient150/patient150_4d.nii.gz")):
#         print("Unzipping ACDC dataset...")
#         os.system(f"unzip -o {zip_dir} -d {DATA_DIR}")

def preprocess():
    patient_fns = glob(os.path.join(DATA_DIR, "EOB", "data_ori/*"))

    resize_shape = (6,32,128,128)
    all_dat = np.zeros(shape=(len(patient_fns),)+resize_shape, dtype=np.float32)

    resize_shape_=(32,128,128)
    count=0
    dat=np.zeros(shape=(6,)+resize_shape_, dtype=np.float32)
    for pat_fn in tqdm(patient_fns):
        # pat_idx = int((pat_fn.split("/")[-1]).split(" ")[0])
        pat_idx=count

        # dat1=sitk.GetArrayFromImage(sitk.ReadImage(pat_fn+"/CE1.mha"))
        # dat2=sitk.GetArrayFromImage(sitk.ReadImage(pat_fn+"/CE2.mha"))
        # dat3=sitk.GetArrayFromImage(sitk.ReadImage(pat_fn+"/CE3.mha"))
        # dat4=sitk.GetArrayFromImage(sitk.ReadImage(pat_fn+"/CE4.mha"))
        # dat5=sitk.GetArrayFromImage(sitk.ReadImage(pat_fn+"/CE5.mha"))
        # dat6=sitk.GetArrayFromImage(sitk.ReadImage(pat_fn+"/20.mha"))

        for i in range(1,6):
            dat_=sitk.ReadImage(pat_fn+"/CE%01d.mha"%int(i))
            dat_=sitk.GetArrayFromImage(dat_)
            dat_=np.resize(dat_,(32,128,128))
            dat_=dat_-dat_.min()
            dat_=dat_/dat_.max()
            dat[i-1]=dat_
        dat_=sitk.ReadImage(pat_fn+"/20.mha")
        dat_=sitk.GetArrayFromImage(dat_)
        dat_=np.resize(dat_,(32,128,128))
        dat_=dat_-dat_.min()
        dat_=dat_/dat_.max()
        dat[-1]=dat_

        all_dat[pat_idx-1] = dat
    count=count+1


    all_dat = np.transpose(all_dat , (0,1,4,3,2))

    np.random.seed(0)

    rand_idx = np.random.permutation(135)

    trn_dat = all_dat[rand_idx[:27*3]]
    val_dat = all_dat[rand_idx[27*3:27*4]]
    tst_dat = all_dat[rand_idx[27*4:]]

    np.save(os.path.join("data", "trn_dat.npy"), trn_dat)
    np.save(os.path.join("data", "val_dat.npy"), val_dat)
    np.save(os.path.join("data", "tst_dat.npy"), tst_dat)

if __name__ == "__main__":
    # download_and_unzip()
    preprocess()