#include <iostream>

// /*
// * 안녕
// * 가나다 
// * 아악
// */



//while을 사용해 50부터 100까지 수를 더하는 프로그램을 만든다.
void first(){
    int num = 0;
    int iter = 50;
    while( iter <= 100){
        num += iter;
        iter++;
    }
    std::cout << num << std::endl; 


}


//피연사에 1을 더하는 ++연산자와 더불어 1을 빼는 감소 연산자가 있다. 감소 연산자를 사용해 두 정수의 범위 내에 있는 수를 출력하는 프로그램을 만든다.
void second(){
    int num = 10;
    while( num >= 0){
        std::cout << num-- << std::endl;
    }
}

// 사용자에게 입력받은 두 정수의 범위 내에 있는 수를 출력하는 프로그램을 만든다.
void third(){
    int input, input2;
    std::cin >> input >> input2;  // 콤마(,) 대신 >> 연산자 사용
    while( input <= input2){
        std::cout<< "add input: " << input++ << std::endl;
    }
}

class one{
    public:
        one();
        int a = 0;

        void test(){
            std::cout << "one" << std::endl;
        }
};



int main()
{
    third();

    system("pause");
    return 0;
}