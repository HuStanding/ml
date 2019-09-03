/*
* @Author: huzhu
* @Date:   2019-09-01 17:28:19
* @Last Modified by:   huzhu
* @Last Modified time: 2019-09-02 15:59:19
*/

#include <iostream>
#include <string>
using namespace std;

int ic,dc,rc;  //插入删除修改

/**
 * @param a
 * @param b
 * @return 字符串之间的距离
 */
int get_distance(string a,string b){
    int row = a.size();
    int col = b.size();
    int **dp = new int*[row+1];//动态创建一个二维数组
    for(int i = 0;i < row + 1;i++){
        dp[i] = new int[col + 1]();   //全部初始化为0
    }
    for (int i=1; i<=col; ++i)    
        dp[0][i] = ic*i;
    for (int i=1; i<=row; ++i)    
        dp[i][0] = dc*i;
    for (int i=1; i<=row; ++i) {
        for (int j=1; j<=col; ++j) {
            int case1 = dp[i-1][j]+dc;
            int case2 = dp[i][j-1]+ic;
            int case3 = dp[i-1][j-1];
            if(a[i-1] != b[j-1])  
                case3 += rc;
            dp[i][j] = min(min(case1, case2), case3);
        }
    }
    for(int i = 0; i < row + 1; i++){
        delete[] dp[i];
    } 
    delete[] dp;
    return dp[row][col];
}

int main(){
    string a,b;
    ic = dc = rc = 1;
    a = "abcdefg";
    b = "bcdef";
    cout << get_distance(a, b) << endl;
    return 0;
}