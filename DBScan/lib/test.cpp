/*
* @Author: huzhu
* @Date:   2019-10-23 16:39:06
* @Last Modified by:   huzhu
* @Last Modified time: 2019-10-23 17:05:08
*/
#include <iostream>

#include<time.h>

using namespace std;
int main(){
clock_t startTime,endTime;
	startTime = clock();
	for(int i = 0; i < 2000; i++){
		for (int j = 0; j < 2000; j++){
			int k = 1 * 2;
		}
	}
	endTime = clock();
	cout<<(double)(endTime-startTime)/CLOCKS_PER_SEC<<endl;
	return 0;
}