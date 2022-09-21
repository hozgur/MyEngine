#pragma once
#include "my.h"
#include "mypy.h"
#include "mytensor.h"

class MyEngine : public myEngine
{
public:
	MyEngine(const char* path) :myEngine(path)
	{

	}
	std::string replace(std::string str, const std::string& from, const std::string& to) {
		size_t start_pos = str.find(from);
		if (start_pos == std::string::npos)
			return str;
		std::string newstr = str.replace(start_pos, from.length(), to);
		return newstr;
	}
	int c = 0;
	int id = 0;
	int width = 178;
	int height = 218;
	bool canRun = true;
	bool OnStart() override
	{
		if (!myPy::init())
		{
			debug << "Py Error!\n";
			exit(1);
		}
		AddWindow(1600, 1000);
		myPy::dofile(myfs::path("user/DeepLearning/AutoEncoder/test1.py"));
		return true;
	}

	void OnKey(uint8_t key, bool pressed) override
	{
		if (pressed && key == myKey::S) canRun = false;
		if (pressed && key == myKey::C) canRun = true;
	}

	void run()
	{
		int stat = myPy::dofunction("runBatch", {});
		if (stat < 0) { canRun = false; return; }

		int id = myPy::getglobal<int>("inpId");
		int ido = myPy::getglobal<int>("outId");
		int idc = myPy::getglobal<int>("codeId");
		myTensor<float>* tensIn = (myTensor<float>*) getObject(id);
		myTensor<float>* tensOut = (myTensor<float>*) getObject(ido);
		//tensor<float>* tensCode = (tensor<float>*) GetMyObject(idc);
		if (tensIn != nullptr)
		{
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 8; j++)
				{
					// Input
					for (int y = 0; y < height; y++)
					{
						float* tData = tensIn->getData() + (y + (i * 8 + j) * height) * width;
						myColor* p = GetLinePointer(y + i * height) + j * width;
						for (int x = 0; x < width; x++)
						{
							int t = (int)(tData[x] * 255);
							if (t < 0) t = 0;
							if (t > 255) t = 255;
							p[x] = myColor(t, t, t);
						}
					}
					// output
					for (int y = 0; y < height; y++)
					{
						float* tData = tensOut->getData() + (y + (i * 8 + j) * height) * width;
						myColor* p = GetLinePointer(y + i * height + 2 * height) + j * width;
						for (int x = 0; x < width; x++)
						{
							int t = (int)(tData[x] * 255);
							if (t < 0) t = 0;
							if (t > 255) t = 255;
							p[x] = myColor(t, t, t);
						}
					}

					// code
			/*		for (int y = 0; y < 4; y++)
					{
						float* tData = tensCode->getData(1, i * 16*16 + j * 16 + y * 4);
						for (int x = 0; x < 4; x++)
						{
							int t = tData[x] * 255;
							int x1 = j * 16 + x * 4;
							int y1 = 16 * i + (16 * 28) + y * 4;
							FillRect({ x1,y1 }, { 4, 4 }, Color(t, t, t));
						}
					}*/
				}
			}
		}
	}

	void draw()
	{
		int x = 500;
		int y = 10;
		int width = 200;
		int height = 200;
		//DrawRect({ x,y }, { width,height }, Color::DarkRed);
		double xx = (mouseX - 500) / 100;
		double yy = (mouseY - 10) / 100;
		if (xx < 0) xx = 0;
		if (yy < 0) yy = 0;
		FillCircle({ xx * 100 + 500,yy * 100 + 10 }, 3, myColor::Blue);
		myPy::dofunction("Forward2", { xx,yy });
		int id = myPy::getglobal<int>("fwdId");
		myTensor<float>* tensDraw = (myTensor<float>*) getObject(id);
		for (int y = 0; y < 28; y++)
		{
			float* tData = tensDraw->getData() + y * width;
			for (int x = 0; x < 28; x++)
			{
				int t = (int)(tData[x] * 255);
				if (t < 0) t = 0;
				if (t > 255) t = 255;
				FillRect({ x * 3 + 500,y * 3 + 500 }, { 3,3 }, myColor(t, t, t));
			}
		}

	}

	void OnIdle() override
	{
		if (canRun)
			run();
	}

	void OnDraw() override
	{
		//Clear(Color::Black);

		/*draw();*/
	}

	void OnUpdate() override
	{

	}
};