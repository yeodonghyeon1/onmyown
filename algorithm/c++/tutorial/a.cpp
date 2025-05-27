#include "test.hpp"


namespace crash{
    int crash1(int& temp){
        cout << "hi2" << endl;
        temp = 3;
        cout << "this is temp : " << temp << endl;
    };

    int crash2(double& temp2, float*p){
        temp2 = 3.3;
        cout << "this is temp2 : " << temp2 << endl;
        cout << "this is p : " << p << endl;
    }
};



int main(){
    int inum;
    // std::cin >> inum;
    // std::cout << "input number: ";

    // std::cout << "input number: " << inum << std::flush;
    int a;
    float* p;
    double b;
    double& ref = b;
    float c;

    long d;
    p = &c;  
    a = 10;
    c = 0.2;

    b = 12.2;
    d = 2;
    // crash::crash1();
    // crash1();
    crash::crash1(a);
    crash::crash2(ref, p);
    cout << "hello"; 
    cout << " my name is" << endl;
    cout << a << " " << b  <<" " << c <<" " << d << endl;
    getchar();
    getchar();

    return 0;

}