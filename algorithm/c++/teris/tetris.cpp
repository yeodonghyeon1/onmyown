#include <iostream>
#include <vector>
#include <thread>
#include "block.h"
#include <mutex>
#include <memory>
#include <conio.h>
#include <algorithm>
#include <atomic>
using namespace std;
#define UP 72
#define DOWN 80
#define LEFT 75
#define RIGHT 77

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
        bool game_state = true;
        struct block_window{
            int insert_window_front;
            int insert_window_back;
            int insert_window_up;
            int insert_window_down;
        };
     
        std::shared_ptr<block_window> bw;


        tetris(){
            x = 10;
            y = 20;
            bw = std::make_shared<block_window>();
            bw->insert_window_front = ((x/2 -2));
            bw->insert_window_back = ((x/2 +2));
            bw->insert_window_up = 1; 
            bw->insert_window_down = 5; 
            
        }
    
        void set_x(int new_x){
            x= new_x;
        }
        void set_y(int new_y){
            y= new_y;
        }

        void run(){
            map = gridmap();
            t1 = std::thread(&tetris::down_block_and_bind, this);
            t2 = std::thread(&tetris::key_event, this);
            
            int block_number = 0;
            while(true){
                if(successfully_bind_block){
                    current_block = selete_block(block_number);
                    insert_block(map);
                    block_number++;

                    if(block_number == 7){
                        block_number = 0;
                    }       
                }             
                show_map(map);
                system("cls");
                if(!game_state) break;
            }
        }

        ~tetris(){
            if(t1.joinable()) t1.join();
            if(t2.joinable()) t2.join();

        }


    private:
        
        int x, y;
        std::thread t1;
        std::thread t2;
        std::mutex mtx;
        std::atomic_bool running = true;
        int check_bind_block = 0;
        bool bind_block = false;

        void clear_block(){

            int line_clear_count = 0;
            int line_clear = 0;
            for(int i = y -2; i > 0; --i){
                for(int j = x -2; j > 0; --j){
                    if(map[i][j] == 2){
                        line_clear_count++;
                    }
                } 
                if(line_clear_count == (x -2)){
                    line_clear++;
                    for(int n = x -2; n > 0; --n){
                        map[i][n] = 0;
                    }
                }
                line_clear_count = 0;
            }
            
            while(line_clear != 0){
                for(int i = y -2; i > 1; --i){
                    for(int j = x -2; j > 0; --j){
                        if(map[i][j] == 0){
                            line_clear_count++;
                        }
                    }
                    if(line_clear_count == (x-2)){
                        for(int n = x-2; n>0; --n){
                            map[i][n] = map[i-1][n];
                            map[i-1][n] = 0;
                        }   
                    }
                    line_clear_count = 0;
                }
                line_clear--;
            }

            if(line_clear == true){
                line_clear = false;
            }

                

            
        }


        void key_event(){
            char c;
            bool can_move = false;
            int lotation_number = 0;
            while(running) {
                if (_kbhit()) {        //키보드 입력 확인 (true / false)
                    c = _getch();      // 방향키 입력시 224 00이 들어오게 되기에 앞에 있는 값 224를 없앰
                    if (c == -32) {    // -32로 입력되면
                        c = _getch();  // 새로 입력값을 판별하여 상하좌우 출력
                        {
                            int move_number = 0;

                            std::lock_guard<std::mutex> lock(mtx);
                            int down = std::clamp(bw->insert_window_down, 0, y-1);
                            int up = std::clamp(bw->insert_window_up, 0, y-1);
                            int back = std::clamp(bw->insert_window_back, 0, x-1);
                            int front = std::clamp(bw->insert_window_front, 0, x-1);
                            
                            switch (c) {
                            case LEFT:
                                for(int i = up; i < down; ++i){
                                    for(int j = front; j < back; ++j){ 
                                        if(map[i][j] == 1 && (map[i][j-1] == 0 || map[i][j-1] == 1)){
                                            move_number++;
                                            if(move_number == 4){
                                                can_move = true;
                                                move_number = 0;
                                            }
                                        }
                                    }
                                }
                                if(can_move){
                                    for(int i = up; i < down; ++i){
                                        for(int j = front; j < back; ++j){ 
                                                if(map[i][j] == 1 && (map[i][j-1] == 0 || map[i][j-1] == 1)){
                                                    map[i][j] = 0;
                                                    map[i][j-1] = 1;
                                            }
                                        }
                                    }
                                }
                                if(can_move){
                                    bw->insert_window_front--;
                                    bw->insert_window_back--;
                                    can_move = false;
                                }
                                break;
                            case RIGHT:
                                for(int i = up; i < down; ++i){
                                    for(int j = back-1; j > front-1; --j){ 
                                        if(map[i][j] == 1 && (map[i][j+1] == 0 || map[i][j+1] == 1)){
                                            move_number++;
                                            if(move_number == 4){
                                                can_move = true;
                                                move_number = 0;
                                            }
                                        }
                                    }
                                }
                                if(can_move){
                                    for(int i = up; i < down; ++i){
                                        for(int j = back-1; j > front-1; --j){ 
                                            if(map[i][j] == 1 && (map[i][j+1] == 0 || map[i][j+1] == 1)){
                                                map[i][j] = 0;
                                                map[i][j+1] = 1;
                                            }
                                        }
                                    }
                                }
                                if(can_move){
                                    bw->insert_window_front++;
                                    bw->insert_window_back++;
                                    can_move = false;
                                }
                                break;
                            case UP: {
                                bool result = block_rotation(lotation_number);
                                if(result){ 
                                    lotation_number++;
                                }
                                if(lotation_number == 4){
                                    lotation_number = 0;
                                }
                                break;
                            }
                            case DOWN:
                                for(int i = down-1; i >= up-1; --i){
                                    for(int j = back-1; j >= front-1; --j){
                                        if(map[i][j] == 1){
                                            if((map[i+1][j] == -1 || map[i+1][j]== 2)){
                                                bind_block = true;
                                            }
                                        }
                                    }   
                                }
                                for(int i = down-1; i > up-1; --i){
                                    for(int j = front; j < back; ++j){
                                        if(!bind_block){
                                            if(map[i][j] == 1 && map[i+1][j] == 0){
                                                map[i][j] = 0;
                                                map[i+1][j] = 1;
                                            }
                                        }
                                    }
                                }
                                if(!bind_block){
                                    bw->insert_window_up++;
                                    bw->insert_window_down++;
                                }
                                break;
                            // 스페이스바 입력 처리 (동작은 비워둠)
                            case ' ':
                                // 스페이스바 동작 구현 예정
                                break;
                            }
                        }
                    }
                }
            }
        }
 
        bool block_rotation(int lotation_number){
            int down = std::clamp(bw->insert_window_down, 0, y-1);
            int up = std::clamp(bw->insert_window_up, 0, y-1);
            int back = std::clamp(bw->insert_window_back, 0, x-1);
            int front = std::clamp(bw->insert_window_front, 0, x-1);
            vector<vector<vector<int>>> block = current_block->create_block();
            int block_x=0;
            int block_y=0;
            int rotation_allow = 0;
            vector<vector<int>> temp_map = map;

             for(int i = up; i < down; ++i){
                block_x = 0;
                for(int j = front; j < back; ++j){
                        if(block[lotation_number][block_y][block_x] == 1){
                            if(map[i][j] == 1 || map[i][j] == 0){
                                rotation_allow++;
                            }
                        }
                        block_x++;
                    }
                block_y++;
            }

            block_x=0;
            block_y=0;
            for(int i = up; i < down; ++i){
                block_x = 0;
                for(int j = front; j < back; ++j){
                    if(rotation_allow == 4){
                        if(map[i][j] == 1 || map[i][j] == 0){
                            map[i][j] = block[lotation_number][block_y][block_x];
                        }
                    }
                    block_x++;

                }
                block_y++;
            }

            if(rotation_allow ==4)
                return true;
            else
                return false;
        }

        void down_block_and_bind(){
            while(running){
                mtx.lock();


                //전체 보드 순회 확인 
                // for(int i = y-1; i > 0; --i){
                //     for(int j = x-1; j > 0; --j){
                //         if(map[i][j] == 1){
                //             if((map[i+1][j] == -1 || map[i+1][j]== 2)){
                //                 bind_block = true;
                //             }
                //         }
                //     }   
                // }
                //3  7 (3 4 5 6)
                //1  5 (1 2 3 4)


                //윈도우 크기만큼만 확인 (18*8*2 = 288번 순회 -> 4*4*2 32번 순회 개선)
                int down = std::clamp(bw->insert_window_down, 0, y-1);
                int up = std::clamp(bw->insert_window_up, 0, y-1);
                int back = std::clamp(bw->insert_window_back, 0, x-1);
                int front = std::clamp(bw->insert_window_front, 0, x-1);


                for(int i = down-1; i >= up; --i){
                    for(int j = back-1; j >= front; --j){
                        if(map[i][j] == 1){
                            if((map[i+1][j] == -1 || map[i+1][j]== 2)){
                                bind_block = true;
                            }
                        }
                    }   
                }
                for(int i = down-1; i >= up; --i){
                    for(int j = back-1; j >= front; --j){
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
                    clear_block();
                }
                else{
                    bw->insert_window_up++;
                    bw->insert_window_down++;
                }
                mtx.unlock();

                std::this_thread::sleep_for(std::chrono::milliseconds(500));

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
                case 2: {
                    auto b = std::make_shared<T_Mino>();
                    return b;
                }
                case 3: {
                    auto b = std::make_shared<L_Mino>();
                    return b;
                }
                case 4: {
                    auto b = std::make_shared<J_Mino>();
                    return b;
                }
                case 5: {
                    auto b = std::make_shared<S_Mino>();
                    return b;
                }
                case 6: {
                    auto b = std::make_shared<Z_Mino>();
                    return b;
                }
                default:
                    return nullptr;
            }
        }

        void insert_block(vector<vector<int>> &map){
            vector<vector<vector<int>>> block = current_block->create_block();
            
            bw->insert_window_front = ((x/2 -2));
            bw->insert_window_back = ((x/2 +2));
            bw->insert_window_up = 1; 
            bw->insert_window_down = 5; 
            int block_x=0;
            int block_y=0;
            //블럭 생성 공간입니다

            for(int i = bw->insert_window_up; i < bw->insert_window_down; ++i){
                block_x = 0;
                for(int j = bw->insert_window_front; j < bw->insert_window_back; ++j){
                    if(block[3][block_y][block_x] == 1){
                        if(map[i][j] == 2){
                            game_over();
                        }
                    }
                    map[i][j] = block[3][block_y][block_x];
                    block_x++;

                }   
                block_y++;
            }
            successfully_bind_block = false;
        }

        void game_over(){
            game_state = false;
            running = false;
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
    system("pause"); 

}