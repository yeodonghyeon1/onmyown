#include <iostream>


using namespace std;


class hashtable{   
    public:
        int a;
        int b();
        void one();
        hashtable();

    private:
        void two();
};

hashtable::hashtable(){
    std::cout << "first" << std::endl;
    a = 3;
}

int hashtable::b(){
    int c;
    return c; 
}

void hashtable::one(){
    std::cout << "secend" << std::endl;
    two();
}

void hashtable::two(){
    std::cout << "thrid" << std::endl;
}


int main(){
    hashtable hs;
    hs.one();
    getchar();
}