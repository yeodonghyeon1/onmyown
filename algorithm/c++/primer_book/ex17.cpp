//입의 개수 읽기

#include <iostream>
// #include  "Salse_time.h"
//cin에서 정수를 읽어 합을 출력하는 프로그램을 만든다
int main(){
    int sum = 0;
    int value = 0;
    while(std::cin >> value){
        sum += value;
    }
    std::cout << sum << std::endl;
    system("pause");
    return 0; 

}