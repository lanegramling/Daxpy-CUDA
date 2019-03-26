#include <iostream>
using namespace std;

//Test


// Device code: Computes Z = aX + Y
__global__
void daxpy(double a, const double* X, const double* Y,
	int arraySize, double* Z)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < arraySize)
		Z[i] = a * X[i] + Y[i];
}

// Host code
void doTheKernelLaunch(double h_a, double* h_X, double* h_Y,
	int arraySize, double* h_Z)
{
	// Now on with the show...
	size_t size = arraySize * sizeof(double);

	// Allocate vectors in device memory
	double* d_X;
	cudaMalloc((void**)&d_X, size);
	double* d_Y;
	cudaMalloc((void**)&d_Y, size);
	double* d_Z;
	cudaMalloc((void**)&d_Z, size);

	// Copy vectors from host memory to device memory
	cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, h_Y, size, cudaMemcpyHostToDevice);

	// Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =
		(arraySize + threadsPerBlock - 1) / threadsPerBlock;
	daxpy<<<blocksPerGrid, threadsPerBlock>>>(h_a, d_X, d_Y, arraySize, d_Z);

	// Copy result from device memory to host memory
	// h_Z will contain the result in host memory
	cudaMemcpy(h_Z, d_Z, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_Z);
}

double* do_daxpy(int arraySize)
{
	double a = 2.0;
	double* X = new double[arraySize];
	double* Y = new double[arraySize];
	double* Z = new double[arraySize];
	for (int i=0 ; i<arraySize ; i++)
	{
		X[i] = 1000.0;
		Y[i] =   10.0;
	}
	doTheKernelLaunch(a, X, Y, arraySize, Z);
	for (int i=0 ; i<arraySize ; i++)
		cout << Z[i] << " = " << a << " * " << X[i] << "  +  " << Y[i] << '\n';
	delete [] X;
	delete [] Y;
	return Z;
}

int main()
{
	// report versions
	int driverVersion, runtimeVersion;
	cudaError_t dv = cudaDriverGetVersion(&driverVersion);
	cudaError_t rv = cudaRuntimeGetVersion(&runtimeVersion);
	cout << "Driver version: " << driverVersion << "; Runtime version: "
	     << runtimeVersion << "\n\n";

	double* Z = do_daxpy(20);
	// ...
	delete [] Z;
	return 0;
}
