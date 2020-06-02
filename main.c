#include <stdio.h>

#include "lrp.h"

int main(int argc, char** argv) {
	int nTest = 100;
	int sum = 0;
	int sum_lrp = 0;
	int i, res;

	cudaSetDevice(0);

	for(i = 0; i<nTest; i++){
		printf("--------- Iter: %d -----------\n", i);
		res = -1;
		res = lrp();
		sum += res;
	}
	printf("Average total time: %d us", sum / (nTest - 2));
	return 0;
}





