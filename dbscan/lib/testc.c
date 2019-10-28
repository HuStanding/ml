/*
* @Author: huzhu
* @Date:   2019-10-23 16:39:06
* @Last Modified by:   huzhu
* @Last Modified time: 2019-10-27 12:43:51
*/

int fibonacci(int n){
	if(n == 1 || n == 2){return 1;}
	else{
		return fibonacci(n - 2) + fibonacci(n - 1);
	}
}