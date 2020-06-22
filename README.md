# Superpixel-Segmentation
C++/opencv实现SLIC与VCells超像素分割

SuperpixelSegmentation ss(fileName);
\r\n
（1）ss.SLIC(600, 20, 10);
（2）ss.VCells(7, 300, 5, 10);
ss.SaveContour(saveFileName);// 保存结果
