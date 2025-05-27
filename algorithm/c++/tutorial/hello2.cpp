#include <iostream>
#include <memory>
using std::cout;
using std::endl;

class one{
    public:
        int a= 3;
        static int b;
        one(){};


};

int main(int argc, char * argv[]){
    int i = 0;
    int j { 3 };
    int n = { 10 };
    int a ( 2 );
    // one::b에서 에러가 발생하는 이유는 static 멤버 변수 b가 선언만 되고 정의되지 않았기 때문입니다.
    // static 멤버 변수는 클래스 외부에서 별도로 정의해야 합니다.
    // 클래스 외부에 'int one::b = 2;'와 같이 정의해야 합니다.
    one::b = 2;  // 이 코드는 one::b가 정의된 후에만 작동합니다
    std::unique_ptr<one> test = std::make_unique<one>();
    std::cout << test->b << std::endl;

    //constexpr: "컴파일타임에 평가되어야 함을 의미". 이것은 주로 상수를 지정하고 읽기 전용 메모리에 데이터를 배치할 수 있게하며 성능을 위해 사용된다. constexpr의 값은 컴파일 타임에 계산된다.
    constexpr int var = 10;

    std::cout << "Hello World" << std::endl;
    getchar();
    return EXIT_SUCCESS;
}