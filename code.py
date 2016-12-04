# encoding:utf-8
import dicom
import os
import numpy
from matplotlib import pyplot


# 单个dcm文件列表
def getFileList(PathDicom):
    listfilesDCM = []
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():
                listfilesDCM.append(os.path.join(dirName, filename))
    return listfilesDCM


def getConstPixelDims(PathDicom):
    listfilesDCM = getFileList(PathDicom)
    RefDs = dicom.read_file(listfilesDCM[0])
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(listfilesDCM))
    ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    for filenameDCM in listfilesDCM:
        ds = dicom.read_file(filenameDCM)
        ArrayDicom[:, :, listfilesDCM.index(filenameDCM)] = ds.pixel_array
    return ArrayDicom


def plotpicture(PathDicom):
    listfilesDCM = getFileList(PathDicom)
    ArrayDicom=getConstPixelDims(PathDicom)
    RefDs = dicom.read_file(listfilesDCM[0])
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(listfilesDCM))
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
    x = numpy.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = numpy.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = numpy.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])
    pyplot.figure(dpi=50)
    pyplot.axes().set_aspect('equal', 'datalim')
    pyplot.set_cmap(pyplot.gray())
    pyplot.pcolormesh(y,x,numpy.flipud(ArrayDicom[:, :, 56]))
    pyplot.show()
    # pyplot.figure(dpi=50)
    # pyplot.axes().set_aspect('equal', 'datalim')
    # pyplot.set_cmap(pyplot.gray())
    # pyplot.pcolormesh(z, x, numpy.flipud(ArrayDicom[:, 20, :]))
    # pyplot.show()
if __name__ == "__main__":
    PathDicom = "/Users/xufly/prostateX/DOI/ProstateX-0000/1.3.6.1.4.1.14519.5.2.1.7311.5101.158323547117540061132729905711/1.3.6.1.4.1.14519.5.2.1.7311.5101.193930593885836732899229869284"
    print getConstPixelDims(PathDicom).shape
    plotpicture(PathDicom)

