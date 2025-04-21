from flask import Flask, render_template, jsonify, request, session
import os
from tetris import Tetris, AITrainer

app = Flask(__name__)
app.secret_key = os.urandom(24)
ai_trainer = AITrainer()

# 전역 게임 인스턴스 저장소
games = {}

def create_new_game_state():
    game = Tetris()
    return game

@app.route('/')
def index():
    # 세션 ID를 사용하여 게임 인스턴스 관리
    session_id = session.get('session_id')
    if not session_id:
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
    
    if session_id not in games:
        games[session_id] = create_new_game_state()
    
    return render_template('index.html')

@app.route('/api/game/state')
def get_game_state():
    session_id = session.get('session_id')
    if not session_id or session_id not in games:
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
        games[session_id] = create_new_game_state()
    
    return jsonify(games[session_id].get_state())

@app.route('/api/game/update', methods=['POST'])
def update_game_state():
    session_id = session.get('session_id')
    if not session_id or session_id not in games:
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
        games[session_id] = create_new_game_state()
    
    game = games[session_id]
    data = request.get_json()
    action = data.get('action')
    
    if action == 'move_left':
        game.move_left()
    elif action == 'move_right':
        game.move_right()
    elif action == 'move_down':
        if not game.move_down():
            game.lock_piece()
    elif action == 'rotate':
        game.rotate_piece()
    elif action == 'hard_drop':
        game.hard_drop()
        game.lock_piece()
    elif action == 'hold':
        game.hold_piece()
    elif action == 'reset':
        game.reset_game()
    elif action == 'pause':
        game.paused = not game.paused
    
    return jsonify(game.get_state())

@app.route('/api/ai/analysis')
def get_ai_analysis():
    patterns = ai_trainer.analyze_moves()
    return jsonify({
        'patterns': patterns if patterns else []
    })

@app.route('/api/ai/save', methods=['POST'])
def save_ai_data():
    try:
        ai_trainer.save_data()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ai/load', methods=['POST'])
def load_ai_data():
    try:
        ai_trainer.load_data()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 