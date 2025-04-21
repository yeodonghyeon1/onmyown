#include <iostream>
#include <vector>

/*

[ Hash 함수(해시 함수) ]
해시 함수에서 중요한 것은 고유한 인덱스 값을 설정하는 것이다. 해시 테이블에 사용되는 대표적인 해시 함수로는 아래의 3가지가 있다. 

1.Division Method: 나눗셈을 이용하는 방법으로 입력값을 테이블의 크기로 나누어 계산한다.( 주소 = 입력값 % 테이블의 크기) 테이블의 크기를 
소수로 정하고 2의 제곱수와 먼 값을 사용해야 효과가 좋다고 알려져 있다.
2.Digit Folding: 각 Key의 문자열을 ASCII 코드로 바꾸고 값을 합한 데이터를 테이블 내의 주소로 사용하는 방법이다.
3.Multiplication Method: 숫자로 된 Key값 K와 0과 1사이의 실수 A, 보통 2의 제곱수인 m을 사용하여 다음과 같은 계산을 해준다. h(k)=(kAmod1) × m
4.Univeral Hashing: 다수의 해시함수를 만들어 집합 H에 넣어두고, 무작위로 해시함수를 선택해 해시값을 만드는 기법이다.
출 처: https://mangkyu.tistory.com/102 [MangKyu's Diary:티스토리]

[ 해시(Hash)값이 충돌하는 경우]
 1.분리 연결법(Separate Chaining)
 2.개방 주소법(Open Addressing)

*/

using namespace std;

class Node{
    public:
        string key;
        int hash_key;
        Node();
        Node(string, int);
        Node under_data(string, int);
};

Node::Node(){
    
}
Node::Node(string key, int hash_key){
    this->key = key;
    this->hash_key = hash_key;
} 

class hashtable{   
    public:
        hashtable(int vec);
        void push(string, string);
        int hash_func(string);
        string search(string);
        vector<string> table;
        Node top_node;
        Node node;
        int vec;
};

//table 동적 할당
hashtable::hashtable(int vec){
    table.resize(vec);
    this->vec = vec;
    node = top_node;
}   

//해시 함수
int hashtable::hash_func(string k){
    int add_ascll = 0;
    for(int i = 0; i< k.size(); i++){
        add_ascll += int(k[i]);
    }
    return add_ascll;
}

//값 입력
void hashtable::push(string k, string v){
    int hash_key = hash_func(k);
    hash_key = hash_key % vec;
    hash_key = int(hash_key);
    std::cout << hash_key << std::endl;    

    //만약 같은 키 값이 나올 시 
    if(table[hash_key] != ""){
        node = node.under_data(k, hash_key);
    }
    else{
        table[hash_key] = v;
        top_node.hash_key = hash_key;
        top_node.key = k;
    }
}

string hashtable::search(string k){
    
}


int main(){
    hashtable hs(100);
    // std::string temp = "abc";
    hs.push("a", "1");
    // if(hs.table[3] == ""){
    //     std::cout << hs.table[97] << std::endl;    
    // }
    getchar();
}