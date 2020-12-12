#pragma once
#include <math.h>
#include "myimage.h"
namespace My
{
	namespace Math
	{
		const double pi = 3.14159265358979323846;
		const double pi2 = 2 * pi;		
		double Gauss(double x, double c)
		{
			return (1 / (sqrt(pi2*c))) * exp(-x * x / (2 * c ));
		}

		double Gauss2(double x)
		{
			const double c = 0.1;
			return Gauss(x, c) / Gauss(0, c);
		}
	}
}