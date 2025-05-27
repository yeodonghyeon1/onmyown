#include <iostream>
#include <vector>
#include <thread>
#include "block.h"
#include <mutex>
#include <memory>
using namespace std;


//현재 하고 있는 기능
/*

맵 생성. 
맵 크기 조절.
맵 출력
시작 기능
블럭 일정 시간마다 내리기.(blockAction)
블럭 보드에 추가(blockAction)


//내가 생각하는 테트리스 요소

맵 생성. 
맵 크기 조절.
맵 출력
시작 기능

*/
class tetris{
    public:
        bool successfully_bind_block = true;
        std::shared_ptr<Block> current_block;
        vector<vector<int>> map;
        tetris(){
            x = 10;
            y = 20;
        }
    
        void set_x(int new_x){
            x= new_x;
        }
        void set_y(int new_y){
            y= new_y;
        }

        void run(){
                map = gridmap();
                t1 = std::thread(&tetris::down_block, this);
                int block_number = 0;
                while(true){
                    if(successfully_bind_block){
                        current_block = selete_block(block_number);
                        insert_block(map, current_block);
                        block_number++;

                        if(block_number == 2){
                            block_number = 0;
                        }       
                    }             
                    show_map(map);
                    
                    system("cls");
                }
                system("pause"); 
        }

        ~tetris(){
            if(t1.joinable()) t1.join();
        }


    private:
        int x, y;
        std::thread t1;
        std::mutex mtx;
        bool bind_block = false;
        int check_bind_block = 0;

        //144번 블럭 검사 144번 블럭 이동 또는 바인드 1초 동안 1000초 288번 for문
        //144번 만에 하고 싶음
        void down_block(){
            mtx.lock();
            while(true){
                for(int i = y-1; i > 1; --i){
                    for(int j = x-1; j > 1; --j){
                        if(map[i][j] == 1){
                            if((map[i+1][j] == -1 || map[i+1][j]== 2)){
                                bind_block = true;
                            }
                        }
                    }   
                }

                for(int i = y-1; i > 1; --i){
                    for(int j = x-1; j > 1; --j){
                        if(map[i][j] == 1){
                            if(bind_block){
                                map[i][j] = 2;
                            }
                            else{
                                map[i][j] = 0;
                                map[i+1][j] = 1;     
                            }
                        }
                    }
                }

                if(bind_block){
                    successfully_bind_block = true;
                    bind_block = false;
                }

                mtx.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        std::shared_ptr<Block> selete_block(int block_number){
            switch(block_number) {
                case 0: {
                    auto b = std::make_shared<I_Mino>();
                    return b;
                }
                case 1: {
                    auto b = std::make_shared<O_Mino>();
                    return b;
                }
                default:
                    return nullptr;
            }
        }

        void insert_block(vector<vector<int>> &map, auto b){
            vector<vector<vector<int>>> block = b->create_block();
            //블럭 생성 송간 (4,4) 크기 맞춰준 겁니다
            int insert_window_front = ((x/ 2) - 2);
            int insert_window_back = ((x/ 2) + 2);
            int insert_window_up = 1;
            int insert_window_down = 5;
            
            int block_x=0;
            int block_y=0;
            //블럭 생성 공간입니다
            for(int i = insert_window_up; i < insert_window_down; ++i){
                block_x = 0;
                for(int j = insert_window_front; j < insert_window_back; ++j){
                    map[i][j] = block[0][block_y][block_x];
                    block_x++;
                }   
                block_y++;
            }
            successfully_bind_block = false;
        }
      
        vector<vector<int>> gridmap(){
            vector<vector<int>> v2(y, vector<int>(x, -1));
            for(int i =0; i< y; ++i){
                for(int j =0; j< x; ++j){
                    if(i == 0) break;
                    if(i == (y-1)) break;
                    if(j == 0) continue;
                    if(j == (x-1)) continue;
                    v2[i][j] = 0;
                }
            }
            return v2;
        }

        void show_map(vector<vector<int>> map){
            for(int i = 0; i< y; ++i){
                for(int j =0; j< x; ++j){
                    //벽면
                    if(map[i][j] == -1){
                        std::cout << "■ ";
                    }
                    //빈 공간
                    else if(map[i][j] == 0){
                        std::cout << "  ";
                    }
                    //현재 블럭
                    else if(map[i][j] == 1){
                        std::cout << "□ ";
                    }
                    //고정 블럭
                    else if(map[i][j] == 2){
                        std::cout << "▣ ";
                    }
                }
                std::cout << std::endl;  
            }
        }
};

int main(){
    tetris te;
    te.run();
}