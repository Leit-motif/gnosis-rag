from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

class ConversationMemory:
    def __init__(self, storage_dir: str = "data/conversations", max_history: int = 10, context_window: int = 5):
        """
        Initialize conversation memory manager
        
        Args:
            storage_dir: Directory to store conversation history
            max_history: Maximum number of interactions to store per session
            context_window: Number of recent interactions to include in context
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.context_window = context_window
        self.sessions: Dict[str, List[Dict]] = {}
        
        # Load existing sessions
        self._load_sessions()
    
    def _load_sessions(self):
        """Load all existing session files"""
        for file in self.storage_dir.glob("*.json"):
            session_id = file.stem
            try:
                with open(file, "r", encoding="utf-8") as f:
                    self.sessions[session_id] = json.load(f)
            except Exception as e:
                print(f"Error loading session {session_id}: {e}")
    
    def _save_session(self, session_id: str):
        """Save a session to disk"""
        file_path = self.storage_dir / f"{session_id}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.sessions[session_id], f, indent=2)
        except Exception as e:
            print(f"Error saving session {session_id}: {e}")
    
    def add_interaction(self, session_id: str, user_message: str, assistant_message: str):
        """
        Add a new interaction to a session
        
        Args:
            session_id: Unique identifier for the conversation session
            user_message: The user's message
            assistant_message: The assistant's response
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_message": assistant_message
        }
        
        self.sessions[session_id].append(interaction)
        
        # Trim history if needed
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
        
        # Save to disk
        self._save_session(session_id)
    
    def get_context_window(self, session_id: str) -> str:
        """
        Get recent conversation context for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            String containing recent conversation history
        """
        if session_id not in self.sessions:
            return ""
            
        history = self.sessions[session_id][-self.context_window:]
        context = []
        
        for interaction in history:
            context.extend([
                f"User: {interaction['user_message']}",
                f"Assistant: {interaction['assistant_message']}"
            ])
        
        return "\n\n".join(context)
    
    def clear_session(self, session_id: str):
        """Clear a session's history"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            file_path = self.storage_dir / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()
    
    def clear_old_sessions(self, days: int = 7):
        """Clear sessions older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        for session_id in list(self.sessions.keys()):
            history = self.sessions[session_id]
            if not history:
                continue
                
            last_interaction = datetime.fromisoformat(history[-1]["timestamp"])
            if last_interaction < cutoff:
                self.clear_session(session_id) 