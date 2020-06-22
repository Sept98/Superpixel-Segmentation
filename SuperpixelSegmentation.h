#pragma once
#pragma warning (disable: 26451)
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\opencv.hpp>
#include<unordered_set>
#include<queue>

// cv::Point的hash，用于unordered_set中
template<>class std::hash<cv::Point>
{
public:
	size_t operator()(const cv::Point& p) const
	{
		return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
	}
};

// 超像素类
class Superpixel
{
public:
	cv::Point center;
	int color[3]{ 0,0,0 };
	std::vector<cv::Point> points;
	int index;
public:
	Superpixel(cv::Point p) :index(-1), center(p) {}
	Superpixel(int i) :index(i), center(cv::Point(-1, -1)) {}
};

class SuperpixelSegmentation
{
private:
	std::string fileName;
	cv::Mat srcImg, proImg;
	size_t width, height, size;
private:
	const int Laplacian[3][3]{ { 0,1,0 },{ 1,-4,1 },{ 0,1,0 } };
	const int dx4[4]{ 0,0,1,-1 };
	const int dy4[4]{ 1,-1,0,0 };
	const int dx8[8]{ -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8]{ 0, -1, -1, -1, 0, 1, 1, 1 };
	std::vector<Superpixel> superpixels;
	int* labels;
private:
	// Laplacian算子计算梯度
	inline int CalGradient(int x, int y)
	{
		int grad = 0;
		for (int i = -1; i <= 1; ++i)
		{
			for (int j = -1; j <= 1; ++j)
			{
				cv::Vec3b v = proImg.ptr<cv::Vec3b>(y + j)[x + i];
				int a = ((int)v[0] + (int)v[1] + (int)v[2]) / 3;
				grad += a * Laplacian[i + 1][j + 1];
			}
		}
		return grad;
	}

	// 计算像素之间的距离
	inline double CalSLICDist(int index, int off_x, int off_y, int x, int y, const int& stepPow, const int& ncPow)
	{
		// 直线距离
		double dist1 = (double)(off_x * off_x + off_y * off_y) / stepPow;

		// 颜色距离
		cv::Vec3b v = proImg.ptr<cv::Vec3b>(y)[x];
		double dist2 = pow((superpixels[index].color[0] - (int)v[0]), 2);
		dist2 += pow((superpixels[index].color[1] - (int)v[1]), 2);
		dist2 += pow((superpixels[index].color[2] - (int)v[2]), 2);
		dist2 /= ncPow;

		return dist1 + dist2;
	}
private:
	// 生成初始六边形
	void GenerateInitNet(int size)
	{
		double grid_width = size * 1.5;
		double grid_height = size * 0.866025;

		int hex_even = ceil((ceil(height / grid_height) - 1) / 2) + 1;
		int hex_odd = floor((ceil(height / grid_height) - 1) / 2) + 1;

		for (int x = 0; x < this->width; ++x)
		{
			int i = floor(x * 1.0 / grid_width);
			for (int y = 0; y < this->height; ++y)
			{
				int j = floor(y * 1.0 / grid_height);

				int segIndex;
				if ((i + j) % 2 == 0)
				{
					if (CalDist2(cv::Point(x, y), grid_width * i, grid_height * j) <=
						CalDist2(cv::Point(x, y), grid_width * (i + 1), grid_height * (j + 1)))
					{
						segIndex = GetHexIndex(i, j, hex_even, hex_odd);
					}
					else
					{
						segIndex = GetHexIndex(i + 1, j + 1, hex_even, hex_odd);
					}
				}
				else
				{
					if (CalDist2(cv::Point(x, y), grid_width * (i + 1), grid_height * j) <=
						CalDist2(cv::Point(x, y), grid_width * i, grid_height * (j + 1)))
					{
						segIndex = GetHexIndex(i + 1, j, hex_even, hex_odd);
					}
					else
					{
						segIndex = GetHexIndex(i, j + 1, hex_even, hex_odd);
					}
				}

				int curIndex = superpixels.size();
				while (curIndex <= segIndex)
				{
					superpixels.push_back(Superpixel(curIndex));
					curIndex += 1;
				}

				labels[y * this->width + x] = segIndex;
			}
		}

		for (int i = 0; i < superpixels.size(); ++i)
		{
			superpixels[i].color[0] = superpixels[i].color[1] = superpixels[i].color[2] = 0;
			superpixels[i].points.clear();
		}
		for (int i = 0; i < this->width; ++i)
		{
			for (int j = 0; j < this->height; ++j)
			{
				int cluster = labels[j * this->width + i];
				cv::Vec3b v = proImg.ptr<cv::Vec3b>(j)[i];
				superpixels[cluster].color[0] += (int)v[0];
				superpixels[cluster].color[1] += (int)v[1];
				superpixels[cluster].color[2] += (int)v[2];
				superpixels[cluster].points.push_back(cv::Point(i, j));
			}
		}

		for (int i = 0; i < superpixels.size(); ++i)
		{
			if (superpixels[i].points.empty())continue;
			superpixels[i].color[0] /= superpixels[i].points.size();
			superpixels[i].color[1] /= superpixels[i].points.size();
			superpixels[i].color[2] /= superpixels[i].points.size();
		}
	}

	// 获得超像素索引
	inline int GetHexIndex(int i, int j, int hex_even, int hex_odd)
	{
		int past = (i >> 1) * (hex_even + hex_odd);
		if (i % 2 == 1) past += hex_even;
		return past + (j >> 1);
	}

	// 计算两点间的距离平方
	inline double CalDist2(const cv::Point& p, double x, double y)
	{
		return (x - p.x) * (x - p.x) + (y - p.y) * (y - p.y);
	}

	// 计算NK距离平方
	inline int CalVCellsDist(int x, int y, Superpixel s, const std::vector<cv::Point> omega, const int& weight)
	{
		cv::Vec3b v = proImg.ptr<cv::Vec3b>(y)[x];
		int d1 = pow((s.color[0] - (int)v[0]), 2);
		d1 += pow((s.color[1] - (int)v[1]), 2);
		d1 += pow((s.color[2] - (int)v[2]), 2);
		int d2 = 0;
		for (cv::Point c : omega)
		{
			int m = c.x + x;
			int n = c.y + y;
			if (m >= width || m < 0 || n >= height || n < 0) continue;
			if (labels[n * width + m] == s.index) d2++;
		}
		d2 = omega.size() - d2;
		return d1 + 2 * weight * d2;
	}

	// 是否为边界
	inline bool IsEdge(int x, int y)
	{
		if (x < 0 || x >= width || y + 1 < 0 || y + 1 >= height)return true;
		if (labels[y * this->width + x] != labels[(y + 1) * this->width + x])return true;
		if (x < 0 || x >= width || y - 1 < 0 || y - 1 >= height)return true;
		if (labels[y * this->width + x] != labels[(y - 1) * this->width + x])return true;
		if (x + 1 < 0 || x + 1 >= width || y < 0 || y >= height)return true;
		if (labels[y * this->width + x] != labels[y * this->width + (x + 1)])return true;
		if (x - 1 < 0 || x - 1 >= width || y < 0 || y >= height)return true;
		if (labels[y * this->width + x] != labels[y * this->width + (x - 1)])return true;
		return false;
	}
private:
	// 初始化参数
	void Reset()
	{
		superpixels.clear();
		this->labels = new int[this->size];
		for (size_t i = 0; i < this->size; ++i) labels[i] = -1;
	}

	// 不连续超像素的连续化
	void Continue()
	{
		for (auto i = superpixels.begin(); i != superpixels.end();)
		{
			if (i->points.empty())i = superpixels.erase(i);
			else ++i;
		}

		for (size_t i = 0; i < superpixels.size(); ++i)
		{
			std::vector<std::vector<cv::Point>>points;// 分成几个部分
			std::unordered_set<cv::Point> all;// 该超像素内的所有点
			for (size_t j = 0; j < superpixels[i].points.size(); ++j)
			{
				all.insert(superpixels[i].points[j]);
			}

			// 包含最多点的部分的下标与点数量
			int index = 0, number = 0;

			// 判断分成几部分
			while (!all.empty())
			{
				std::vector<cv::Point> part;
				std::queue<cv::Point> que;
				que.push(*(all.begin()));
				while (!que.empty())
				{
					cv::Point p = que.front();
					que.pop();
					if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height)continue;
					if (all.find(p) == all.end())continue;
					part.push_back(p);
					all.erase(p);

					que.push(cv::Point(p.x - 1, p.y));
					que.push(cv::Point(p.x + 1, p.y));
					que.push(cv::Point(p.x, p.y - 1));
					que.push(cv::Point(p.x, p.y + 1));
				}
				points.push_back(part);
				if (part.size() > number)
				{
					index = points.size() - 1;
					number = part.size();
				}
			}
			superpixels[i].points = points[index];

			for (size_t j = 0; j < points.size(); ++j)
			{
				if (j == index) continue;

				bool isMove = false;
				for (size_t b = 0; b < points[j].size(); ++b)
				{
					for (size_t k = 0; k < 8; ++k)
					{
						int m = points[j][b].x + dx8[k];
						int n = points[j][b].y + dy8[k];
						if (m < 0 || m >= width || n < 0 || n >= height)continue;
						if (labels[n * width + m] != superpixels[i].index && labels[n * width + m] != -1)
						{
							isMove = true;
							for (int a = 0; a < points[j].size(); ++a)
							{
								cv::Point p = points[j][a];
								labels[p.y * width + p.x] = labels[n * width + m];
								superpixels[labels[n * width + m]].points.push_back(p);
							}
						}
						if (isMove)break;
					}
					if (isMove)break;
				}
			}
		}

		for (size_t i = 0; i < superpixels.size(); ++i)
		{
			superpixels[i].index = i;

			// 重新标记超像素内部的像素
			for (cv::Point p : superpixels[i].points)
			{
				labels[p.y * this->width + p.x] = i;
			}
		}
	}
public:
	SuperpixelSegmentation(std::string _fileName)
	{
		this->fileName = _fileName;
		this->srcImg = cv::imread(this->fileName);
		this->width = this->srcImg.cols;
		this->height = this->srcImg.rows;
		this->size = this->width * this->height;
	}

	~SuperpixelSegmentation()
	{
		delete[] labels;
	}

	void SLIC(int num, int nc, int iterTimes)
	{
		this->Reset();

		this->proImg = srcImg.clone();
		cv::cvtColor(this->proImg, this->proImg, CV_RGB2Lab);

		int ncPow = nc * nc;
		int step = sqrt((this->width * this->height) / num);
		int stepPow = step * step;

		// 初始化聚类中心
		for (int m = step / 2; m < width; m += step)
		{
			for (int n = step / 2; n < height; n += step)
			{
				int minGrad = INT_MAX, maxI = 0, maxJ = 0;
				for (int i = -1; i <= 1; ++i)
				{
					if (m + i < 0 || m + i >= width)continue;
					for (int j = -1; j <= 1; ++j)
					{
						if (n + j < 0 || n + j >= height)continue;
						int temp = CalGradient(m + i, n + j);
						if (temp < minGrad) { minGrad = temp; maxI = i; maxJ = j; }
					}
				}
				//Superpixel c(cv::Point(m + maxI, n + maxJ));
				Superpixel c(cv::Point(m, n));
				c.index = superpixels.size();
				c.color[0] = proImg.at<cv::Vec3b>(c.center.y, c.center.x)[0];
				c.color[1] = proImg.at<cv::Vec3b>(c.center.y, c.center.x)[1];
				c.color[2] = proImg.at<cv::Vec3b>(c.center.y, c.center.x)[2];
				superpixels.push_back(c);
				//for (int i = -step; i <= step; ++i)
				//{
				//	int x = superpixels[superpixels.size()-1].center.x + i;
				//	if (x < 0 || x >= this->width) continue;
				//	for (int j = -step; j <= step; ++j)
				//	{
				//		int y = superpixels[superpixels.size() - 1].center.y + j;
				//		if (y < 0 || y >= this->height) continue;
				//		int index = y * this->width + x;
				//		this->labels[index] = superpixels.size() - 1;
				//	}
				//}
			}
		}

		double* distances = new double[this->size]();
		while (iterTimes-- > 0)
		{
			for (size_t i = 0; i < this->size; ++i) distances[i] = INT_MAX;
			for (size_t k = 0; k < superpixels.size(); ++k)
			{
				for (int i = -step; i <= step; ++i)
				{
					int x = superpixels[k].center.x + i;
					if (x < 0 || x >= this->width) continue;
					for (int j = -step; j <= step; ++j)
					{
						int y = superpixels[k].center.y + j;
						if (y < 0 || y >= this->height) continue;
						double temp = CalSLICDist(k, i, j, x, y, stepPow, ncPow);
						if (temp < distances[y * this->width + x])
						{
							distances[y * this->width + x] = temp;
							this->labels[y * this->width + x] = k;
						}
					}
				}
			}

			for (size_t i = 0; i < superpixels.size(); ++i)
			{
				superpixels[i].color[0] = superpixels[i].color[1] = superpixels[i].color[2] = 0;
				superpixels[i].center.x = superpixels[i].center.y = 0;
				superpixels[i].points.clear();
			}
			for (size_t i = 0; i < this->width; ++i)
			{
				for (size_t j = 0; j < this->height; ++j)
				{
					int cluster = labels[j * this->width + i];

					superpixels[cluster].points.push_back(cv::Point(i, j));
					superpixels[cluster].center.x += i;
					superpixels[cluster].center.y += j;
					superpixels[cluster].color[0] += proImg.ptr<cv::Vec3b>(j)[i][0];
					superpixels[cluster].color[1] += proImg.ptr<cv::Vec3b>(j)[i][1];
					superpixels[cluster].color[2] += proImg.ptr<cv::Vec3b>(j)[i][2];
				}
			}

			for (size_t i = 0; i < superpixels.size(); ++i)
			{
				if (superpixels[i].points.empty())continue;
				superpixels[i].color[0] /= superpixels[i].points.size();
				superpixels[i].color[1] /= superpixels[i].points.size();
				superpixels[i].color[2] /= superpixels[i].points.size();
				superpixels[i].center.x /= superpixels[i].points.size();
				superpixels[i].center.y /= superpixels[i].points.size();
			}
		}



		// 释放资源
		delete[] distances;

		Continue();
	}

	void VCells(int size, int weight, int radius, int iterTimes)
	{
		this->Reset();
		this->proImg = srcImg.clone();
		// 初始化六边形格网
		GenerateInitNet(size);

		// 搜索范围确认
		std::vector<cv::Point>omega;
		for (int i = -radius; i <= radius; ++i)
		{
			for (int j = -radius; j <= radius; ++j)
			{
				if (i * i + j * j <= radius * radius)
				{
					omega.push_back(cv::Point(i, j));
				}
			}
		}

		int* distances = new int[this->size]();
		memset(distances, -1, this->size * 4);

		while (iterTimes-- > 0)
		{
			for (int i = 0; i < this->width; ++i)
			{
				for (int j = 0; j < this->height; ++j)
				{
					if (!IsEdge(i, j)) continue;
					int curIndex = labels[j * width + i];
					int index = curIndex;
					if (distances[j * width + i] == -1)
					{
						distances[j * width + i] = CalVCellsDist(i, j, superpixels[curIndex], omega, weight);
					}
					for (int k = 0; k < 4; ++k)
					{
						int m = i + dx4[k];
						int n = j + dy4[k];
						if (m < 0 || m >= width || n < 0 || n >= height)continue;
						if (labels[n * width + m] != -1 && labels[n * width + m] != curIndex)
						{
							int res = CalVCellsDist(i, j, superpixels[labels[n * width + m]], omega, weight);
							if (res < distances[j * width + i])
							{
								index = labels[n * width + m];
								distances[j * width + i] = res;
							}
						}
					}

					if (index != curIndex) labels[j * width + i] = index;
				}
			}

			for (int i = 0; i < superpixels.size(); ++i)
			{
				superpixels[i].color[0] = superpixels[i].color[1] = superpixels[i].color[2] = 0;
				superpixels[i].points.clear();
			}

			for (size_t i = 0; i < this->width; ++i)
			{
				for (size_t j = 0; j < this->height; ++j)
				{
					int cluster = labels[j * this->width + i];

					superpixels[cluster].points.push_back(cv::Point(i, j));
					superpixels[cluster].color[0] += proImg.ptr<cv::Vec3b>(j)[i][0];
					superpixels[cluster].color[1] += proImg.ptr<cv::Vec3b>(j)[i][1];
					superpixels[cluster].color[2] += proImg.ptr<cv::Vec3b>(j)[i][2];
				}
			}

			for (int i = 0; i < superpixels.size(); ++i)
			{
				if (superpixels[i].points.empty())continue;
				superpixels[i].color[0] /= superpixels[i].points.size();
				superpixels[i].color[1] /= superpixels[i].points.size();
				superpixels[i].color[2] /= superpixels[i].points.size();
			}
		}

		delete[] distances;

		Continue();
	}

	void SaveContour(std::string saveFileName)
	{
		bool* isTaken = new bool[this->size];
		for (size_t i = 0; i < this->size; ++i)isTaken[i] = false;

		std::vector<cv::Point> contours;
		for (int i = 0; i < width; ++i)
		{
			for (int j = 0; j < height; ++j)
			{
				int nr_p = 0;
				for (size_t k = 0; k < 8; ++k)
				{
					int x = i + dx8[k];
					int y = j + dy8[k];
					if (x >= 0 && x < width && y >= 0 && y < height)
					{
						if (isTaken[y * width + x] == false && labels[j * width + i] != labels[y * width + x])
						{
							nr_p += 1;
						}
					}
				}

				// 属于轮廓的部分
				if (nr_p >= 2)
				{
					contours.push_back(cv::Point(i, j));
					isTaken[j * width + i] = true;
				}
			}
		}

		cv::Mat drawImg = srcImg.clone();

		for (cv::Point p : contours)
		{
			drawImg.ptr<cv::Vec3b>(p.y)[p.x][2] = 255;
			drawImg.ptr<cv::Vec3b>(p.y)[p.x][1] = 0;
			drawImg.ptr<cv::Vec3b>(p.y)[p.x][0] = 0;
		}
		cv::imwrite(saveFileName, drawImg);
		delete[] isTaken;
	}
};

//#include "SuperpixelSegmentation.h"
//int main()
//{
//	std::string fileName = "C:\\Users\\zze\\Desktop\\118035.jpg";
//	std::string saveFileName = "C:\\Users\\zze\\Desktop\\aaa.jpg";
//
//	SuperpixelSegmentation ss(fileName);
//	ss.SLIC(600, 10, 10);
//	ss.SaveContour(saveFileName);
//	//std::getchar();
//	return 0;
//}