# Superpixel-Segmentation
C++/opencv实现SLIC与VCells超像素分割<br>

SuperpixelSegmentation ss(fileName);<br>
（1）ss.SLIC(600, 20, 10);<br>
（2）ss.VCells(7, 300, 5, 10);<br>
ss.SaveContour(saveFileName);// 保存结果
