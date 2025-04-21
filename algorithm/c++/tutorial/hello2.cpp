#include <iostream>

using std::cout;
using std::endl;

int main(int argc, char * argv[]){
    int i = 0;
    int j { 3 };
    int n = { 10 };
    int a ( 2 );

    //constexpr: "컴파일타임에 평가되어야 함을 의미". 이것은 주로 상수를 지정하고 읽기 전용 메모리에 데이터를 배치할 수 있게하며 성능을 위해 사용된다. constexpr의 값은 컴파일 타임에 계산된다.
    constexpr int var = 10;

    std::cout << "Hello World" << std::endl;
    getchar();
    return EXIT_SUCCESS;
}