import random
import numpy as np
from collections import deque
import json
import os
from datetime import datetime

# 게임 설정
GRID_WIDTH = 10
GRID_HEIGHT = 20

# 테트리스 블록 모양
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]],  # J
    [[1, 1, 0], [0, 1, 1]],  # S
    [[0, 1, 1], [1, 1, 0]]   # Z
]

# 블록 색상
COLORS = [
    '#00FFFF',  # Cyan (I)
    '#FFFF00',  # Yellow (O)
    '#FF00FF',  # Magenta (T)
    '#FFA500',  # Orange (L)
    '#0000FF',  # Blue (J)
    '#00FF00',  # Green (S)
    '#FF0000'   # Red (Z)
]

class AITrainer:
    def __init__(self):
        self.move_history = deque(maxlen=1000)
        self.best_moves = []
        
    def record_move(self, state, move, score):
        self.move_history.append({
            'state': state,
            'move': move,
            'score': score,
            'timestamp': datetime.now().isoformat()
        })
    
    def analyze_moves(self):
        if len(self.move_history) < 10:
            return None
        
        successful_patterns = []
        for i in range(len(self.move_history) - 1):
            if self.move_history[i+1]['score'] > self.move_history[i]['score']:
                successful_patterns.append(self.move_history[i]['move'])
        
        return successful_patterns

    def save_data(self, filename='tetris_ai_data.json'):
        with open(filename, 'w') as f:
            json.dump(list(self.move_history), f)

    def load_data(self, filename='tetris_ai_data.json'):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.move_history = deque(json.load(f), maxlen=1000)

class Tetris:
    def __init__(self):
        self.ai_trainer = AITrainer()
        self.reset_game()

    def reset_game(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.held_piece = None
        self.can_hold = True
        self.next_pieces = [self.new_piece() for _ in range(3)]
        self.ghost_piece = None
        self.game_over = False
        self.paused = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.combo = 0
        self.update_ghost_piece()

    def new_piece(self):
        shape_idx = random.randint(0, len(SHAPES) - 1)
        return {
            'shape': SHAPES[shape_idx],
            'color': COLORS[shape_idx],
            'x': GRID_WIDTH // 2 - len(SHAPES[shape_idx][0]) // 2,
            'y': 0
        }

    def hold_piece(self):
        if not self.can_hold:
            return
        
        if self.held_piece is None:
            self.held_piece = {
                'shape': self.current_piece['shape'],
                'color': self.current_piece['color']
            }
            self.current_piece = self.next_pieces.pop(0)
            self.next_pieces.append(self.new_piece())
        else:
            temp = self.current_piece
            self.current_piece = {
                'shape': self.held_piece['shape'],
                'color': self.held_piece['color'],
                'x': GRID_WIDTH // 2 - len(self.held_piece['shape'][0]) // 2,
                'y': 0
            }
            self.held_piece = {
                'shape': temp['shape'],
                'color': temp['color']
            }
        
        self.can_hold = False
        self.update_ghost_piece()

    def update_ghost_piece(self):
        if self.current_piece is None:
            return
            
        self.ghost_piece = {
            'shape': self.current_piece['shape'],
            'color': '#808080',  # Gray
            'x': self.current_piece['x'],
            'y': self.current_piece['y']
        }
        
        while self.valid_move(self.ghost_piece, self.ghost_piece['x'], self.ghost_piece['y'] + 1):
            self.ghost_piece['y'] += 1

    def valid_move(self, piece, x, y):
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    new_x = x + j
                    new_y = y + i
                    if (new_x < 0 or new_x >= GRID_WIDTH or 
                        new_y >= GRID_HEIGHT or 
                        (new_y >= 0 and self.grid[new_y][new_x])):
                        return False
        return True

    def rotate_piece(self):
        shape = self.current_piece['shape']
        rotated = list(zip(*shape[::-1]))
        
        # T-스핀 감지
        is_t_spin = False
        if len(shape) == 3 and len(shape[0]) == 3:  # T 모양 블록
            corners_filled = 0
            x, y = self.current_piece['x'], self.current_piece['y']
            corners = [(x, y), (x+2, y), (x, y+2), (x+2, y+2)]
            for corner_x, corner_y in corners:
                if (corner_x < 0 or corner_x >= GRID_WIDTH or 
                    corner_y < 0 or corner_y >= GRID_HEIGHT or
                    (corner_y < GRID_HEIGHT and corner_x < GRID_WIDTH and self.grid[corner_y][corner_x])):
                    corners_filled += 1
            is_t_spin = corners_filled >= 3

        if self.valid_move({'shape': rotated, 'x': self.current_piece['x'], 'y': self.current_piece['y']}, 
                          self.current_piece['x'], self.current_piece['y']):
            self.current_piece['shape'] = rotated
            if is_t_spin:
                self.score += 400 * self.level  # T-스핀 보너스

        self.update_ghost_piece()

    def clear_lines(self):
        lines_to_clear = []
        for i, row in enumerate(self.grid):
            if all(cell != 0 for cell in row):
                lines_to_clear.append(i)
        
        cleared = len(lines_to_clear)
        if cleared > 0:
            for line in lines_to_clear:
                del self.grid[line]
                self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
            
            self.lines_cleared += cleared
            base_score = [100, 300, 500, 800][cleared-1]  # 1~4줄 클리어 점수
            self.score += base_score * self.level * (self.combo + 1)
            self.combo += 1
            self.level = self.lines_cleared // 10 + 1
        else:
            self.combo = 0

    def move_left(self):
        if self.valid_move(self.current_piece, self.current_piece['x'] - 1, self.current_piece['y']):
            self.current_piece['x'] -= 1
            self.update_ghost_piece()
            return True
        return False

    def move_right(self):
        if self.valid_move(self.current_piece, self.current_piece['x'] + 1, self.current_piece['y']):
            self.current_piece['x'] += 1
            self.update_ghost_piece()
            return True
        return False

    def move_down(self):
        if self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y'] + 1):
            self.current_piece['y'] += 1
            return True
        return False

    def hard_drop(self):
        while self.move_down():
            self.score += 2

    def lock_piece(self):
        # 현재 상태 기록
        state = np.array(self.grid)
        move = {
            'x': self.current_piece['x'],
            'y': self.current_piece['y'],
            'shape': self.current_piece['shape']
        }
        
        # 블록 배치
        for i, row in enumerate(self.current_piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    self.grid[self.current_piece['y'] + i][self.current_piece['x'] + j] = self.current_piece['color']
        
        self.clear_lines()
        self.current_piece = self.next_pieces.pop(0)
        self.next_pieces.append(self.new_piece())
        self.can_hold = True
        self.update_ghost_piece()
        
        # AI 트레이너에 기록
        self.ai_trainer.record_move(state.tolist(), move, self.score)
        
        if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
            self.game_over = True
            self.ai_trainer.save_data()

    def get_state(self):
        return {
            'grid': self.grid,
            'current_piece': self.current_piece,
            'held_piece': self.held_piece,
            'next_pieces': self.next_pieces,
            'ghost_piece': self.ghost_piece,
            'score': self.score,
            'level': self.level,
            'lines_cleared': self.lines_cleared,
            'combo': self.combo,
            'game_over': self.game_over,
            'paused': self.paused
        }
