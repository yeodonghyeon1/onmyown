#include <iostream>
#include "hello_header.h"

#ifndef A
#define A

namespace A {
    int abc(int a, int b){
        return a + b;
    }
}
#endif

namespace {
    int abcd(int a, int b){
        return a + b;
    }
    void printHello() {  // Hello World를 출력하는 함수
        std::cout << "hi" << std::endl << "my name is " << "Psi" << std::endl;
    }
}


int main()
{
    std::cout << "Hello World" << std::endl;
    printf("hello world");
    HelloSpace::printHello();
    std::cout << A::abc(1,2) << std::endl;
    std::cout << abcd(1,2) << std::endl;
    int a =2;
    int b = 3;
    b = a;
    a= 4;
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    printHello();
    getchar();
    return 0;
}



/*
namespace는 함수 객체 충돌 방지를 위해 사용
main 위에 있어야함.
std == stdio 같은 역할 기본 c++ 라이브러리
HelloSpace::printHello() 이런 식으로 호출
헤더 나누는 거는 C랑 비슷함
#ifndef랑 #define #endif 3개로 namespace 구분
<< 이거는 출력 연산자
namespace 이름 안붙여서 쓸수도 있음 하지만 파일 내부에서만 사용 가능
같은 파일 내 namespace에 소속된 함수랑 이름 동일하면 에러남
std::endl  하고 나서도  << 붙여서 이어 출력 가능


*/