#!/usr/bin/env python3
"""
Multi-Agent Debate Frontend
A Flask web application with WebSocket support for real-time debate visualization.
"""

import os
import sys
import json
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import argparse
from mmengine import DictAction

# Add project root to path
root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
load_dotenv(verbose=True)

# Import project modules
from src.config import config
from src.infrastructures.models import model_manager
from src.agents import acp
from src.logger import logger

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'debate_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
debate_config = None
initialized = False


def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "multi_agent_debate.py"), help="config file path")

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


class DebateFrontend:
    def __init__(self):
        self.config = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the debate system."""
        try:
            args = parse_args()
    
            config.init_config(args.config, args)
            logger.init_logger(config)
            logger.info(f"| Config: {config.pretty_text}")
            
            # Initialize model manager
            logger.info("🧠 Initializing model manager...")
            await model_manager.initialize(use_local_proxy=config.use_local_proxy)
            logger.info(f"✅ Model manager initialized: {model_manager.list()}")
            
            # Initialize agents
            logger.info("🤖 Initializing agents...")
            await acp.initialize(config.agent_names)
            await acp.copy(acp.get_info("simple_chat"), name=config.alice_agent.name, description=config.alice_agent.description)
            await acp.copy(acp.get_info("simple_chat"), name=config.bob_agent.name, description=config.bob_agent.description)
            logger.info(f"✅ Agents initialized: {acp.list()}")
            
            self.initialized = True
            logger.info("🎉 Debate system initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize debate system: {e}")
            raise
    
    async def start_debate(self, topic, agents=None):
        """Start a debate and emit real-time updates."""
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"🎯 Starting debate on: {topic}")
            
            # Emit debate start event
            socketio.emit('debate_started', {
                'topic': topic,
                'agents': agents or ["alice", "bob"],
                'timestamp': datetime.now().isoformat()
            })
            
            # Get debate manager and start debate with streaming
            debate_manager_info = acp.get_info("debate_manager")
            if debate_manager_info and hasattr(debate_manager_info, 'instance'):
                debate_manager = debate_manager_info.instance
                
                # Start debate with streaming
                async for event in debate_manager.start_debate(topic, [], agents or ["alice", "bob"]):
                    # Emit real-time events to frontend
                    event_type = event.get('type', 'unknown')
                    agent_name = event.get('agent', 'unknown')
                    content = event.get('content', '')
                    
                    if event_type == 'agent_thinking':
                        socketio.emit('agent_thinking', {
                            'agent': agent_name,
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                    elif event_type == 'agent_response':
                        socketio.emit('agent_response', {
                            'agent': agent_name,
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                    elif event_type == 'agent_decline':
                        socketio.emit('agent_decline', {
                            'agent': agent_name,
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                    elif event_type == 'agent_exit':
                        socketio.emit('agent_exit', {
                            'agent': agent_name,
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                    elif event_type == 'agent_error':
                        socketio.emit('agent_error', {
                            'agent': agent_name,
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    logger.info(f"📡 Emitted event: {event_type} from {agent_name}")
            else:
                # Fallback to regular acp call
                input_data = {
                    "name": "debate_manager",
                    "input": {
                        "topic": topic,
                        "files": [],
                        "agents": agents or ["alice", "bob"]
                    }
                }
                result = await acp.ainvoke(**input_data)
            
            # Emit debate completed event
            socketio.emit('debate_completed', {
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"✅ Debate completed")
            
        except Exception as e:
            logger.error(f"❌ Error during debate: {e}")
            socketio.emit('debate_error', {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

# Global debate frontend instance
debate_frontend = DebateFrontend()

@app.route('/')
def index():
    """Main page."""
    return render_template('debate.html')

@app.route('/api/status')
def api_status():
    """API status endpoint."""
    return jsonify({
        'status': 'running',
        'initialized': debate_frontend.initialized,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/start_debate', methods=['POST'])
def api_start_debate():
    """Start a debate via API."""
    try:
        data = request.get_json()
        topic = data.get('topic', 'Let\'s have a debate!')
        agents = data.get('agents', ['alice', 'bob'])
        
        # Start debate in background thread
        def run_debate():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(debate_frontend.start_debate(topic, agents))
            loop.close()
        
        thread = threading.Thread(target=run_debate)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'topic': topic,
            'agents': agents,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error starting debate: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("🔌 Client connected")
    emit('connected', {'message': 'Connected to debate server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("🔌 Client disconnected")

@socketio.on('request_status')
def handle_status_request():
    """Handle status request."""
    emit('status_response', {
        'initialized': debate_frontend.initialized,
        'timestamp': datetime.now().isoformat()
    })

def initialize_debate_system():
    """Initialize the debate system in background."""
    def init_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(debate_frontend.initialize())
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
        finally:
            loop.close()
    
    thread = threading.Thread(target=init_loop)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    # Initialize debate system
    initialize_debate_system()
    
    # Start Flask app
    logger.info("🚀 Starting Debate Frontend Server...")
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)
